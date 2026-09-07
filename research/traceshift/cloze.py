"""Deterministic teacher-forced cloze scoring + frozen 2-of-3 recall gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .model_loader import LoadedModel
from .paths import default_cloze_probes
from .yaml_io import load_yaml


@dataclass
class CandidateScore:
    candidate: str
    logprob_sum: float
    n_tokens: int
    token_ids: list[int]


@dataclass
class ProbeResult:
    probe_id: str
    fact_id: str
    prefix: str
    expected_answer: str
    ranking: list[CandidateScore]
    winner: str
    expected_wins: bool
    tie_against_expected: bool


@dataclass
class FactRecallResult:
    fact_id: str
    entity: str
    expected_answer: str
    probe_results: list[ProbeResult]
    wins: int
    n_probes: int
    pass_threshold: int
    status: str  # PASS | FAIL


@dataclass
class ClozeEvaluation:
    fact_results: list[FactRecallResult]
    candidates: list[str]
    pass_threshold: int
    n_probes_per_fact: int
    scoring: str = "teacher_forced_token_logprob_sum"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scoring": self.scoring,
            "candidates": self.candidates,
            "pass_threshold": self.pass_threshold,
            "n_probes_per_fact": self.n_probes_per_fact,
            "notes": self.notes,
            "facts": [
                {
                    **{k: v for k, v in asdict(fr).items() if k != "probe_results"},
                    "probes": [asdict(p) for p in fr.probe_results],
                }
                for fr in self.fact_results
            ],
        }


def load_cloze_spec(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path else default_cloze_probes()
    return load_yaml(p)


def _candidate_token_ids(tokenizer: Any, prefix: str, candidate: str) -> tuple[torch.Tensor, int]:
    """Return full input_ids for ``prefix + ' ' + candidate`` and the candidate start index.

    A single ASCII space is inserted between prefix and candidate so completions
    are scored as ordinary continuations of the frozen cloze prefixes (which do
    not end with trailing whitespace).
    """
    if not prefix:
        raise ValueError("Empty cloze prefix")
    if candidate == "":
        raise ValueError("Empty candidate")

    # Prefer encoding the joined string once for correct BPE boundaries.
    joined = prefix + " " + candidate
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    joined_ids = tokenizer(joined, add_special_tokens=False)["input_ids"]

    # Candidate tokens = suffix of joined not explained by prefix alone.
    # If BPE merges across the boundary, fall back to explicit candidate encoding
    # after a leading space (GPT/Qwen-style).
    if len(joined_ids) > len(prefix_ids) and joined_ids[: len(prefix_ids)] == prefix_ids:
        cand_ids = joined_ids[len(prefix_ids) :]
        full_ids = joined_ids
        cand_start = len(prefix_ids)
    else:
        # Boundary merge: encode " {candidate}" and append to prefix ids.
        spaced = tokenizer(" " + candidate, add_special_tokens=False)["input_ids"]
        if not spaced:
            raise RuntimeError(
                f"Tokenizer produced empty ids for candidate {candidate!r} "
                f"after prefix {prefix!r}"
            )
        full_ids = prefix_ids + spaced
        cand_ids = spaced
        cand_start = len(prefix_ids)

    if not cand_ids:
        raise RuntimeError(
            f"No candidate tokens for {candidate!r} under prefix {prefix!r}. "
            "Report tokenization issue; do not silently change the protocol."
        )
    return torch.tensor(full_ids, dtype=torch.long), cand_start


@torch.inference_mode()
def score_candidate_logprob_sum(
    loaded: LoadedModel,
    prefix: str,
    candidate: str,
) -> CandidateScore:
    """Teacher-forced sum of token log-probs for ``candidate`` given ``prefix``."""
    model = loaded.model
    tokenizer = loaded.tokenizer
    device = loaded.device

    input_ids, cand_start = _candidate_token_ids(tokenizer, prefix, candidate)
    input_ids = input_ids.unsqueeze(0).to(device)
    outputs = model(input_ids=input_ids)
    # logits[t] predicts token at t+1
    log_probs = F.log_softmax(outputs.logits[0, :-1, :], dim=-1)
    target = input_ids[0, 1:]
    # Candidate tokens occupy positions cand_start .. end-1 in input_ids,
    # which correspond to log_probs indices cand_start-1 .. end-2 when cand_start>=1.
    if cand_start < 1:
        raise RuntimeError("Candidate start index < 1; cannot score first token safely.")
    tok_logprobs = []
    token_ids = []
    for pos in range(cand_start, input_ids.size(1)):
        # logprob of token at `pos` from logits at `pos-1`
        lp = log_probs[pos - 1, target[pos - 1]].item()
        tok_logprobs.append(lp)
        token_ids.append(int(target[pos - 1].item()))

    return CandidateScore(
        candidate=candidate,
        logprob_sum=float(sum(tok_logprobs)),
        n_tokens=len(tok_logprobs),
        token_ids=token_ids,
    )


def score_probe(
    loaded: LoadedModel,
    *,
    probe_id: str,
    fact_id: str,
    prefix: str,
    expected_answer: str,
    candidates: list[str],
) -> ProbeResult:
    scores = [
        score_candidate_logprob_sum(loaded, prefix, c) for c in candidates
    ]
    # Rank by logprob_sum descending; stable sort for determinism among equals.
    ranking = sorted(scores, key=lambda s: (-s.logprob_sum, s.candidate))
    best = ranking[0].logprob_sum
    top = [s for s in ranking if s.logprob_sum == best]
    tie_against_expected = False
    if len(top) == 1:
        winner = top[0].candidate
        expected_wins = winner == expected_answer
    else:
        # Ties against the expected answer count as a loss (frozen rule).
        names = {s.candidate for s in top}
        if expected_answer in names and len(names) > 1:
            tie_against_expected = True
            expected_wins = False
            # Report lexicographically first among tied as nominal winner for logging.
            winner = sorted(names)[0]
        elif expected_answer in names and len(names) == 1:
            winner = expected_answer
            expected_wins = True
        else:
            winner = sorted(names)[0]
            expected_wins = False

    return ProbeResult(
        probe_id=probe_id,
        fact_id=fact_id,
        prefix=prefix,
        expected_answer=expected_answer,
        ranking=ranking,
        winner=winner,
        expected_wins=expected_wins,
        tie_against_expected=tie_against_expected,
    )


def evaluate_fact(
    loaded: LoadedModel,
    *,
    fact_id: str,
    entity: str,
    expected_answer: str,
    probes: list[dict[str, Any]],
    candidates: list[str],
    pass_threshold: int,
) -> FactRecallResult:
    results = []
    for item in probes:
        results.append(
            score_probe(
                loaded,
                probe_id=str(item["probe_id"]),
                fact_id=fact_id,
                prefix=str(item["prefix"]),
                expected_answer=expected_answer,
                candidates=candidates,
            )
        )
    wins = sum(1 for r in results if r.expected_wins)
    status = "PASS" if wins >= pass_threshold else "FAIL"
    return FactRecallResult(
        fact_id=fact_id,
        entity=entity,
        expected_answer=expected_answer,
        probe_results=results,
        wins=wins,
        n_probes=len(results),
        pass_threshold=pass_threshold,
        status=status,
    )


def evaluate_all_cloze(
    loaded: LoadedModel,
    cloze_path: Path | str | None = None,
) -> ClozeEvaluation:
    """Run all frozen cloze probes with the configured candidate set and 2-of-3 gate."""
    loaded.eval_mode()
    spec = load_cloze_spec(cloze_path)
    candidates = list(spec.get("candidate_completions") or [])
    if not candidates:
        raise ValueError("cloze_probes.yaml missing candidate_completions")
    pass_threshold = int(spec.get("recall_pass_threshold", 2))
    n_probes = int(spec.get("n_probes_per_fact", 3))
    probes_map = spec.get("probes") or {}

    notes: list[str] = []
    # Document scoring choice: protocol allows sum/mean; we use sum (standard TF).
    notes.append(
        "Scoring uses sum of teacher-forced token log-probabilities "
        "(frozen retrieval_rule allows sum/mean; mean is not applied)."
    )

    fact_results: list[FactRecallResult] = []
    for fact_id in sorted(probes_map.keys()):
        block = probes_map[fact_id]
        items = block.get("items") or []
        if len(items) != n_probes:
            notes.append(
                f"{fact_id}: expected {n_probes} probes, found {len(items)}"
            )
        fact_results.append(
            evaluate_fact(
                loaded,
                fact_id=str(block.get("fact_id", fact_id)),
                entity=str(block.get("entity", "")),
                expected_answer=str(block.get("expected_answer")),
                probes=items,
                candidates=candidates,
                pass_threshold=pass_threshold,
            )
        )

    return ClozeEvaluation(
        fact_results=fact_results,
        candidates=candidates,
        pass_threshold=pass_threshold,
        n_probes_per_fact=n_probes,
        notes=notes,
    )


def format_evaluation_report(
    evaluation: ClozeEvaluation,
    repro: Optional[dict[str, Any]] = None,
) -> str:
    lines: list[str] = []
    lines.append("=== TraceShift cloze evaluation ===")
    if repro:
        lines.append(f"model_id: {repro.get('model_id')}")
        lines.append(f"tokenizer_id: {repro.get('tokenizer_id')}")
        lines.append(f"device: {repro.get('device')}")
        lines.append(f"dtype: {repro.get('dtype')}")
        lines.append(f"software: {repro.get('software')}")
    lines.append(f"scoring: {evaluation.scoring}")
    lines.append(f"candidates: {evaluation.candidates}")
    lines.append(
        f"gate: {evaluation.pass_threshold}-of-{evaluation.n_probes_per_fact}"
    )
    for note in evaluation.notes:
        lines.append(f"note: {note}")
    lines.append("")
    for fr in evaluation.fact_results:
        lines.append(
            f"## {fr.fact_id} ({fr.entity}) expected={fr.expected_answer!r} "
            f"→ {fr.status} ({fr.wins}/{fr.n_probes})"
        )
        for pr in fr.probe_results:
            lines.append(f"  probe {pr.probe_id}: {pr.prefix!r}")
            lines.append(
                f"    winner={pr.winner!r} expected_wins={pr.expected_wins} "
                f"tie_loss={pr.tie_against_expected}"
            )
            for rank, cs in enumerate(pr.ranking, start=1):
                lines.append(
                    f"    {rank}. {cs.candidate!r}  "
                    f"logprob_sum={cs.logprob_sum:.6f}  n_tokens={cs.n_tokens}  "
                    f"ids={cs.token_ids}"
                )
        lines.append("")
    return "\n".join(lines)
