#!/usr/bin/env python3
"""TraceShift frozen-spec integrity validator (pre-experiment, no model I/O).

Validates internal consistency of frozen YAML under research/data/frozen and
research/configs/experiment.yaml. Does not load models, install packages,
run AI-Engram, or modify scientific content.

Usage (from research/ or repo root):
  python scripts/validate_frozen_experiment.py
  python scripts/validate_frozen_experiment.py --research-root /path/to/research
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any


FACT_IDS = ("F01", "F02", "F03", "F04")
EXPECTED_ALPHA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
EXPECTED_DOMAIN_ANSWER = {
    "F01": "tennis",
    "F02": "tennis",
    "F03": "swimming",
    "F04": "acting",
}
EXPECTED_LORA = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}
EXPECTED_FT_HPARAMS = {
    "num_epochs": 3,
    "learning_rate": 2.0e-4,
    "optimizer": "adamw",
    "lr_scheduler_type": "cosine",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 256,
    "weight_decay": 0.0,
    "warmup_steps": 5,
}


# ---------------------------------------------------------------------------
# Minimal YAML subset loader (stdlib only; matches this project's frozen files)
# ---------------------------------------------------------------------------

def _strip_comment(line: str) -> str:
    """Remove YAML comments. '#' only starts a comment when preceded by whitespace."""
    in_single = in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            # Only toggle when this looks like a quoted scalar start/end, not an
            # apostrophe inside a plain scalar (e.g. Curie's / NASA's).
            if in_single:
                in_single = False
            elif i == 0 or line[i - 1].isspace() or line[i - 1] in ":-[{,":
                in_single = True
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if ch == "#" and not in_single and not in_double:
            if i == 0 or line[i - 1].isspace():
                return line[:i].rstrip()
    return line.rstrip()


def _parse_scalar(raw: str) -> Any:
    s = raw.strip()
    if s == "" or s == "null" or s == "~":
        return None
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        parts = []
        buf = []
        depth = 0
        in_q = None
        for ch in inner:
            if in_q:
                buf.append(ch)
                if ch == in_q:
                    in_q = None
                continue
            if ch in ("'", '"'):
                in_q = ch
                buf.append(ch)
                continue
            if ch == "[":
                depth += 1
                buf.append(ch)
                continue
            if ch == "]":
                depth -= 1
                buf.append(ch)
                continue
            if ch == "," and depth == 0:
                parts.append(_parse_scalar("".join(buf)))
                buf = []
                continue
            buf.append(ch)
        if buf or parts:
            parts.append(_parse_scalar("".join(buf)))
        return parts
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if re.fullmatch(r"-?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?", s) or re.fullmatch(
        r"-?\d+[eE][+-]?\d+", s
    ):
        return float(s)
    return s


def load_yaml(path: Path) -> Any:
    """Load a restricted YAML subset used by TraceShift frozen specs."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Preprocess folded/literal blocks into single logical lines.
    logical: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        if not raw.strip() or raw.lstrip().startswith("#"):
            i += 1
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        content = _strip_comment(raw)
        if not content.strip():
            i += 1
            continue
        # key: > or key: |
        m = re.match(r"^(\s*)([^:#][^:]*?):\s*([>|])\s*$", content)
        if m:
            key = m.group(2).strip()
            fold = m.group(3)
            block_indent = None
            chunks: list[str] = []
            i += 1
            while i < len(lines):
                br = lines[i]
                if not br.strip():
                    chunks.append("")
                    i += 1
                    continue
                if br.lstrip().startswith("#") and (
                    block_indent is None or len(br) - len(br.lstrip(" ")) < (block_indent or 0)
                ):
                    break
                bi = len(br) - len(br.lstrip(" "))
                if block_indent is None:
                    if bi <= indent:
                        break
                    block_indent = bi
                if bi < block_indent:
                    break
                chunks.append(br[block_indent:])
                i += 1
            if fold == ">":
                # Folded: join non-empty with spaces; blank lines -> paragraph break.
                parts: list[str] = []
                para: list[str] = []
                for c in chunks:
                    if c.strip() == "":
                        if para:
                            parts.append(" ".join(para))
                            para = []
                        parts.append("")
                    else:
                        para.append(c.strip())
                if para:
                    parts.append(" ".join(para))
                value = "\n".join(parts).strip()
            else:
                value = "\n".join(chunks).rstrip("\n")
            logical.append((indent, f"{key}: {value!r}"))
            continue

        stripped = content.strip()
        # Plain multiline continuation of previous list/scalar item (no '-' / no ':').
        if (
            logical
            and indent > logical[-1][0]
            and not stripped.startswith("- ")
            and stripped != "-"
            and not re.match(r"^[^\"'#][^:]*:\s*", stripped)
        ):
            prev_ind, prev_ln = logical[-1]
            logical[-1] = (prev_ind, f"{prev_ln} {stripped}")
            i += 1
            continue

        logical.append((indent, stripped))
        i += 1

    def parse_block(start: int, min_indent: int) -> tuple[Any, int]:
        if start >= len(logical):
            return None, start
        indent, line = logical[start]

        # Sequence at this level
        if line.startswith("- ") or line == "-":
            items = []
            j = start
            while j < len(logical):
                ind, ln = logical[j]
                if ind < min_indent:
                    break
                if ind > min_indent and items:
                    # continuation belongs to previous item — shouldn't happen at seq head
                    break
                if ind != min_indent or not (ln.startswith("- ") or ln == "-"):
                    break
                item_body = "" if ln == "-" else ln[2:].strip()
                j += 1
                if item_body == "":
                    # nested complex value
                    if j < len(logical) and logical[j][0] > min_indent:
                        val, j = parse_block(j, logical[j][0])
                        items.append(val)
                    else:
                        items.append(None)
                elif re.match(r"^[^:]+:\s*", item_body) and not item_body.startswith("["):
                    # inline mapping on dash line, possibly with nested children
                    key, _, rest = item_body.partition(":")
                    key = key.strip()
                    rest = rest.strip()
                    node: dict[str, Any] = {}
                    if rest != "":
                        node[key] = _parse_scalar(rest)
                    else:
                        if j < len(logical) and logical[j][0] > min_indent:
                            node[key], j = parse_block(j, logical[j][0])
                        else:
                            node[key] = None
                    # further keys at child indent
                    child_indent = None
                    while j < len(logical):
                        ci, cl = logical[j]
                        if ci <= min_indent:
                            break
                        if cl.startswith("- "):
                            break
                        if child_indent is None:
                            child_indent = ci
                        if ci != child_indent:
                            break
                        if ":" not in cl:
                            break
                        k, _, r = cl.partition(":")
                        k = k.strip()
                        r = r.strip()
                        j += 1
                        if r != "":
                            node[k] = _parse_scalar(r)
                        else:
                            if j < len(logical) and logical[j][0] > child_indent:
                                node[k], j = parse_block(j, logical[j][0])
                            else:
                                node[k] = None
                    items.append(node)
                else:
                    # scalar item; may have nested mapping if next lines more indented
                    # Unusual in our files — treat as scalar.
                    items.append(_parse_scalar(item_body))
            return items, j

        # Mapping
        mapping: dict[str, Any] = {}
        j = start
        while j < len(logical):
            ind, ln = logical[j]
            if ind < min_indent:
                break
            if ind > min_indent:
                break
            if ln.startswith("- "):
                break
            if ":" not in ln:
                raise ValueError(f"YAML parse error in {path}: expected key: value, got {ln!r}")
            key, _, rest = ln.partition(":")
            key = key.strip()
            rest = rest.strip()
            j += 1
            if rest != "":
                # Handle Python-repr style from folded preprocess
                if (rest.startswith("'") and rest.endswith("'")) or (
                    rest.startswith('"') and rest.endswith('"')
                ):
                    try:
                        mapping[key] = ast_literal(rest)
                    except Exception:
                        mapping[key] = _parse_scalar(rest)
                else:
                    mapping[key] = _parse_scalar(rest)
            else:
                if j < len(logical) and logical[j][0] > min_indent:
                    mapping[key], j = parse_block(j, logical[j][0])
                else:
                    mapping[key] = None
        return mapping, j

    def ast_literal(s: str) -> Any:
        import ast

        return ast.literal_eval(s)

    if not logical:
        return {}
    root_indent = logical[0][0]
    value, idx = parse_block(0, root_indent)
    if idx != len(logical):
        # Multiple top-level keys: parse as mapping spanning all
        if not isinstance(value, dict):
            value = {}
            idx = 0
        guard = 0
        while idx < len(logical):
            before = idx
            more, idx = parse_block(idx, root_indent)
            if idx == before:
                raise ValueError(
                    f"YAML parse stuck at line content {logical[idx]!r} in {path}"
                )
            if isinstance(more, dict):
                value.update(more)
            else:
                raise ValueError(f"Unexpected trailing YAML structure in {path}: {more!r}")
            guard += 1
            if guard > len(logical) + 5:
                raise ValueError(f"YAML parse guard exceeded in {path}")
    return value


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class ValidationReport:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.checks_passed: list[str] = []
        self.counts: dict[str, Any] = {}

    def ok(self, name: str, detail: str = "") -> None:
        msg = name if not detail else f"{name}: {detail}"
        self.checks_passed.append(msg)

    def fail(self, message: str) -> None:
        self.errors.append(message)

    @property
    def ok_count(self) -> int:
        return len(self.checks_passed)

    @property
    def error_count(self) -> int:
        return len(self.errors)


def _require(report: ValidationReport, cond: bool, err: str, ok_name: str, ok_detail: str = "") -> bool:
    if cond:
        report.ok(ok_name, ok_detail)
        return True
    report.fail(err)
    return False


def _float_eq(a: Any, b: float, tol: float = 1e-12) -> bool:
    try:
        return math.isclose(float(a), b, rel_tol=0.0, abs_tol=tol)
    except (TypeError, ValueError):
        return False


def validate_facts(data: dict, report: ValidationReport) -> dict[str, dict]:
    facts_list = data.get("facts")
    if not isinstance(facts_list, list):
        report.fail("facts.yaml: missing list under 'facts'")
        return {}
    by_id = {}
    for f in facts_list:
        if not isinstance(f, dict) or "id" not in f:
            report.fail(f"facts.yaml: invalid fact entry: {f!r}")
            continue
        by_id[f["id"]] = f
    _require(
        report,
        set(by_id) == set(FACT_IDS),
        f"facts.yaml: expected IDs {FACT_IDS}, found {tuple(by_id)}",
        "facts.ids",
        "F01–F04 present",
    )
    required = ("id", "entity", "domain", "group", "description", "used_in_tennis_finetuning")
    for fid in FACT_IDS:
        f = by_id.get(fid)
        if not f:
            continue
        missing = [k for k in required if k not in f or f[k] is None or f[k] == ""]
        _require(
            report,
            not missing,
            f"facts.yaml {fid}: missing required fields {missing}",
            f"facts.{fid}.fields",
        )
    expectations = {
        "F01": ("in_domain", True),
        "F02": ("in_domain", True),
        "F03": ("control", False),
        "F04": ("control", False),
    }
    for fid, (group, ft_flag) in expectations.items():
        f = by_id.get(fid, {})
        _require(
            report,
            f.get("group") == group,
            f"facts.yaml {fid}: group expected {group!r}, got {f.get('group')!r}",
            f"facts.{fid}.group",
            group,
        )
        _require(
            report,
            f.get("used_in_tennis_finetuning") is ft_flag,
            f"facts.yaml {fid}: used_in_tennis_finetuning expected {ft_flag}, "
            f"got {f.get('used_in_tennis_finetuning')!r}",
            f"facts.{fid}.ft_flag",
            str(ft_flag),
        )
    report.counts["facts"] = len(by_id)
    return by_id


def validate_extraction_sets(
    data: dict, report: ValidationReport, kind: str
) -> dict[str, list[dict]]:
    sets = data.get("sets")
    if not isinstance(sets, dict):
        report.fail(f"extraction_sets_{kind}.yaml: missing 'sets' mapping")
        return {}
    _require(
        report,
        set(sets) == set(FACT_IDS),
        f"extraction_sets_{kind}.yaml: expected keys {FACT_IDS}, found {tuple(sets)}",
        f"extract.{kind}.ids",
    )
    out: dict[str, list[dict]] = {}
    for fid in FACT_IDS:
        block = sets.get(fid)
        if not isinstance(block, dict):
            report.fail(f"extraction_sets_{kind}.yaml: {fid} missing/invalid")
            continue
        _require(
            report,
            block.get("fact_id") == fid,
            f"extraction_sets_{kind}.yaml {fid}: fact_id mismatch {block.get('fact_id')!r}",
            f"extract.{kind}.{fid}.fact_id",
        )
        items = block.get("items")
        if not isinstance(items, list):
            report.fail(f"extraction_sets_{kind}.yaml {fid}: items must be a list")
            continue
        _require(
            report,
            len(items) == 5,
            f"extraction_sets_{kind}.yaml {fid}: expected 5 sentences, got {len(items)}",
            f"extract.{kind}.{fid}.count",
            "5",
        )
        cleaned = []
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                report.fail(f"extraction_sets_{kind}.yaml {fid} item[{i}]: not a mapping")
                continue
            text = it.get("text")
            iid = it.get("id")
            if not isinstance(text, str) or not text.strip():
                report.fail(f"extraction_sets_{kind}.yaml {fid} item[{i}]: empty text")
            else:
                cleaned.append(it)
            if not iid:
                report.fail(f"extraction_sets_{kind}.yaml {fid} item[{i}]: missing id")
            elif not str(iid).startswith(fid):
                report.fail(
                    f"extraction_sets_{kind}.yaml {fid}: id {iid!r} does not start with fact id"
                )
        out[fid] = cleaned
    report.counts[f"extract_{kind}_sentences"] = sum(len(v) for v in out.values())
    return out


def validate_reference(
    ref: dict,
    explicit: dict[str, list[dict]],
    lexical: dict[str, list[dict]],
    report: ValidationReport,
) -> None:
    corpora = ref.get("corpora")
    if not isinstance(corpora, dict):
        report.fail("reference_corpus.yaml: missing 'corpora'")
        return

    def check_variant(name: str, targets: dict[str, list[dict]]) -> None:
        block = corpora.get(name)
        if not isinstance(block, dict):
            report.fail(f"reference_corpus.yaml: missing corpora.{name}")
            return
        docs = block.get("documents")
        if not isinstance(docs, list):
            report.fail(f"reference_corpus.yaml corpora.{name}: documents must be a list")
            return
        expected_items = []
        for fid in FACT_IDS:
            expected_items.extend(targets.get(fid, []))
        _require(
            report,
            len(docs) == 20 and len(expected_items) == 20,
            f"reference_corpus.yaml {name}: expected 20 docs from targets, "
            f"got docs={len(docs)} targets={len(expected_items)}",
            f"ref.{name}.count",
            "20",
        )
        # Match by id then text; order F01→F04 as frozen
        by_id_doc = {d.get("doc_id"): d for d in docs if isinstance(d, dict)}
        missing = []
        mismatched = []
        for it in expected_items:
            iid = it.get("id")
            d = by_id_doc.get(iid)
            if d is None:
                missing.append(iid)
                continue
            if d.get("text") != it.get("text"):
                mismatched.append(iid)
            if d.get("fact_id") != it.get("id", "")[:3] and d.get("fact_id") not in FACT_IDS:
                mismatched.append(f"{iid}:fact_id")
            # fact_id should match prefix of item id (F01 from F01_EX01)
            expected_fid = str(iid).split("_")[0]
            if d.get("fact_id") != expected_fid:
                report.fail(
                    f"reference_corpus.yaml {name} {iid}: fact_id {d.get('fact_id')!r} "
                    f"!= {expected_fid!r}"
                )
        extra = set(by_id_doc) - {it.get("id") for it in expected_items}
        _require(
            report,
            not missing and not mismatched and not extra,
            f"reference_corpus.yaml {name}: missing={missing} mismatched={mismatched} "
            f"extra={sorted(extra)}",
            f"ref.{name}.identity",
            "all target sentences present, no extras",
        )

    check_variant("explicit", explicit)
    check_variant("lexical_control", lexical)
    report.counts["reference_explicit_docs"] = len(
        (corpora.get("explicit") or {}).get("documents") or []
    )
    report.counts["reference_lexical_docs"] = len(
        (corpora.get("lexical_control") or {}).get("documents") or []
    )


def validate_cloze(data: dict, facts: dict[str, dict], report: ValidationReport) -> list[str]:
    """Return probe prefix texts for disjointness checks."""
    probes = data.get("probes")
    if not isinstance(probes, dict):
        report.fail("cloze_probes.yaml: missing 'probes'")
        return []
    _require(
        report,
        set(probes) == set(FACT_IDS),
        f"cloze_probes.yaml: expected keys {FACT_IDS}, found {tuple(probes)}",
        "cloze.ids",
    )
    candidates = data.get("candidate_completions")
    _require(
        report,
        isinstance(candidates, list) and len(candidates) >= 4,
        "cloze_probes.yaml: candidate_completions missing or too short",
        "cloze.candidates",
        f"n={len(candidates) if isinstance(candidates, list) else 0}",
    )
    thr = data.get("recall_pass_threshold")
    npp = data.get("n_probes_per_fact")
    _require(
        report,
        thr == 2 and npp == 3,
        f"cloze_probes.yaml: expected 2-of-3 gate "
        f"(recall_pass_threshold=2, n_probes_per_fact=3), got {thr=}, {npp=}",
        "cloze.gate",
        "2-of-3",
    )
    rule = data.get("retrieval_rule")
    _require(
        report,
        isinstance(rule, dict) and rule.get("method") == "teacher_forced_token_logprob",
        "cloze_probes.yaml: retrieval_rule.method missing/incorrect",
        "cloze.retrieval_rule",
    )
    prefixes: list[str] = []
    for fid in FACT_IDS:
        block = probes.get(fid, {})
        if not isinstance(block, dict):
            report.fail(f"cloze_probes.yaml: {fid} invalid")
            continue
        _require(
            report,
            block.get("fact_id") == fid,
            f"cloze_probes.yaml {fid}: fact_id mismatch",
            f"cloze.{fid}.fact_id",
        )
        expected = block.get("expected_answer")
        domain = (facts.get(fid) or {}).get("domain")
        want = EXPECTED_DOMAIN_ANSWER[fid]
        _require(
            report,
            expected == want and domain == want,
            f"cloze_probes.yaml {fid}: expected_answer {expected!r} inconsistent "
            f"with facts domain {domain!r} / required {want!r}",
            f"cloze.{fid}.answer",
            str(expected),
        )
        items = block.get("items")
        if not isinstance(items, list):
            report.fail(f"cloze_probes.yaml {fid}: items must be a list")
            continue
        _require(
            report,
            len(items) == 3,
            f"cloze_probes.yaml {fid}: expected 3 probes, got {len(items)}",
            f"cloze.{fid}.count",
            "3",
        )
        for it in items:
            if isinstance(it, dict) and isinstance(it.get("prefix"), str) and it["prefix"].strip():
                prefixes.append(it["prefix"].strip())
            else:
                report.fail(f"cloze_probes.yaml {fid}: empty/missing prefix in {it!r}")
    report.counts["cloze_probes"] = len(prefixes)
    return prefixes


def validate_ft_datasets(
    primary: dict, control: dict, report: ValidationReport
) -> tuple[list[str], list[str]]:
    pd = primary.get("dataset") or {}
    cd = control.get("dataset") or {}
    p_ex = pd.get("examples") if isinstance(pd.get("examples"), list) else []
    c_ex = cd.get("examples") if isinstance(cd.get("examples"), list) else []

    _require(
        report,
        len(p_ex) == 40 and pd.get("n_examples") == 40,
        f"finetuning_dataset_spec.yaml: expected 40 examples, got len={len(p_ex)} "
        f"n_examples={pd.get('n_examples')!r}",
        "ft.primary.count",
        "40",
    )
    _require(
        report,
        len(c_ex) == 40 and cd.get("n_examples") == 40,
        f"control_finetuning_dataset_spec.yaml: expected 40 examples, got len={len(c_ex)} "
        f"n_examples={cd.get('n_examples')!r}",
        "ft.control.count",
        "40",
    )

    fed = [e for e in p_ex if isinstance(e, dict) and e.get("fact_id") == "F01"]
    nad = [e for e in p_ex if isinstance(e, dict) and e.get("fact_id") == "F02"]
    # Also count by subject name if fact_id present
    fed_subj = [
        e
        for e in p_ex
        if isinstance(e, dict) and e.get("subject") == "Roger Federer"
    ]
    nad_subj = [
        e for e in p_ex if isinstance(e, dict) and e.get("subject") == "Rafael Nadal"
    ]
    _require(
        report,
        len(fed) == 20 and len(nad) == 20 and len(fed_subj) == 20 and len(nad_subj) == 20,
        f"finetuning_dataset_spec.yaml: expected 20 Federer + 20 Nadal, "
        f"got fact_id F01/F02={len(fed)}/{len(nad)}, subject={len(fed_subj)}/{len(nad_subj)}",
        "ft.primary.balance",
        "20/20",
    )

    curie = [e for e in c_ex if isinstance(e, dict) and e.get("subject") == "Marie Curie"]
    arm = [e for e in c_ex if isinstance(e, dict) and e.get("subject") == "Neil Armstrong"]
    _require(
        report,
        len(curie) == 20 and len(arm) == 20,
        f"control_finetuning_dataset_spec.yaml: expected 20 Curie + 20 Armstrong, "
        f"got {len(curie)}/{len(arm)}",
        "ft.control.balance",
        "20/20",
    )

    # Schema / metadata for disjointness assumptions
    for label, ds in (("primary", pd), ("control", cd)):
        hos = ds.get("held_out_statement")
        _require(
            report,
            isinstance(hos, str) and "held out" in hos.lower(),
            f"{label} FT dataset: held_out_statement missing or incomplete",
            f"ft.{label}.held_out_statement",
        )
        _require(
            report,
            ds.get("format") == "plain_text_examples",
            f"{label} FT dataset: format expected plain_text_examples, got {ds.get('format')!r}",
            f"ft.{label}.format",
        )
        bad = [
            e
            for e in (ds.get("examples") or [])
            if not isinstance(e, dict) or not e.get("id") or not isinstance(e.get("text"), str)
        ]
        _require(
            report,
            not bad,
            f"{label} FT dataset: {len(bad)} examples missing id/text",
            f"ft.{label}.schema",
        )

    _require(
        report,
        isinstance(pd.get("balance_rules"), list) and len(pd.get("balance_rules") or []) > 0,
        "primary FT dataset: balance_rules missing",
        "ft.primary.balance_rules",
    )
    _require(
        report,
        isinstance(cd.get("exclusion_rules"), list) and len(cd.get("exclusion_rules") or []) > 0,
        "control FT dataset: exclusion_rules missing",
        "ft.control.exclusion_rules",
    )
    _require(
        report,
        cd.get("matched_to") == "tennis_biography_ft" and cd.get("size_match_target") == 40,
        "control FT dataset: matched_to/size_match_target incorrect",
        "ft.control.match_meta",
    )

    p_texts = [e["text"].strip() for e in p_ex if isinstance(e, dict) and e.get("text")]
    c_texts = [e["text"].strip() for e in c_ex if isinstance(e, dict) and e.get("text")]
    report.counts["ft_primary"] = len(p_texts)
    report.counts["ft_control"] = len(c_texts)
    return p_texts, c_texts


def validate_config(cfg: dict, report: ValidationReport) -> None:
    model = cfg.get("model") or {}
    _require(
        report,
        model.get("model_id") == "Qwen/Qwen3-1.7B"
        and model.get("base_model_id") == "Qwen/Qwen3-1.7B",
        f"experiment.yaml model_id/base_model_id expected Qwen/Qwen3-1.7B, "
        f"got {model.get('model_id')!r} / {model.get('base_model_id')!r}",
        "config.base_model",
        "Qwen/Qwen3-1.7B",
    )

    ae = cfg.get("ai_engram") or {}
    _require(
        report,
        str(ae.get("version")) == "0.9.0"
        and str(model.get("ai_engram_version")) == "0.9.0",
        f"experiment.yaml AI-Engram version expected 0.9.0, "
        f"got ai_engram.version={ae.get('version')!r} "
        f"model.ai_engram_version={model.get('ai_engram_version')!r}",
        "config.ai_engram_version",
        "0.9.0",
    )

    mods = ((ae.get("target_modules") or {}).get("modules")) or []
    expected_mods = [f"model.layers.{i}.mlp.down_proj" for i in range(28)]
    _require(
        report,
        mods == expected_mods,
        f"experiment.yaml ai_engram.target_modules.modules must be layers 0–27 "
        f"mlp.down_proj (28 entries); got n={len(mods)}",
        "config.target_modules",
        "layers 0–27 down_proj",
    )
    lir = (ae.get("target_modules") or {}).get("layer_index_range")
    _require(
        report,
        lir == [0, 27],
        f"experiment.yaml layer_index_range expected [0, 27], got {lir!r}",
        "config.layer_index_range",
    )

    grid = (cfg.get("intervention") or {}).get("alpha_grid")
    grid_ok = (
        isinstance(grid, list)
        and len(grid) == len(EXPECTED_ALPHA_GRID)
        and all(_float_eq(a, b) for a, b in zip(grid, EXPECTED_ALPHA_GRID))
    )
    _require(
        report,
        grid_ok,
        f"experiment.yaml intervention.alpha_grid expected {EXPECTED_ALPHA_GRID}, got {grid!r}",
        "config.alpha_grid",
    )

    alpha_usage = ae.get("alpha_usage") or {}
    cal = cfg.get("alpha_calibration") or {}
    step_ids = [
        s.get("id")
        for s in (cal.get("steps") or [])
        if isinstance(s, dict)
    ]
    _require(
        report,
        alpha_usage.get("hold_fixed_for_primary_base_vs_ft") is True
        and alpha_usage.get("calibrate_on") == "M_base"
        and "8_primary_alphas_held_fixed" in step_ids
        and cal.get("model_stage") == "M_base",
        "experiment.yaml: base-alpha / fixed-alpha rule incomplete "
        "(need ai_engram.alpha_usage.hold_fixed_for_primary_base_vs_ft, "
        "calibrate_on=M_base, alpha_calibration step 8_primary_alphas_held_fixed)",
        "config.fixed_base_alpha_rule",
    )

    ft = cfg.get("fine_tuning") or {}
    lora = ft.get("lora") or {}
    lora_ok = (
        ft.get("method") == "lora"
        and lora.get("r") == EXPECTED_LORA["r"]
        and lora.get("alpha") == EXPECTED_LORA["alpha"]
        and _float_eq(lora.get("dropout"), EXPECTED_LORA["dropout"])
        and lora.get("bias") == EXPECTED_LORA["bias"]
        and lora.get("task_type") == EXPECTED_LORA["task_type"]
        and lora.get("target_modules") == EXPECTED_LORA["target_modules"]
    )
    _require(
        report,
        lora_ok,
        f"experiment.yaml LoRA config mismatch. got method={ft.get('method')!r} lora={lora!r} "
        f"expected {EXPECTED_LORA}",
        "config.lora",
        "r=16 α=32 dropout=0.05 q/k/v/o_proj",
    )

    hp = ft.get("hyperparameters") or {}
    hp_errs = []
    for k, v in EXPECTED_FT_HPARAMS.items():
        got = hp.get(k)
        if isinstance(v, float):
            if not _float_eq(got, v):
                hp_errs.append(f"{k}: {got!r}!={v}")
        else:
            if got != v:
                hp_errs.append(f"{k}: {got!r}!={v}")
    _require(
        report,
        not hp_errs,
        f"experiment.yaml fine_tuning.hyperparameters mismatch: {hp_errs}",
        "config.ft_hyperparameters",
    )

    _require(
        report,
        ft.get("identical_hparams_for_primary_and_control") is True
        and ft.get("only_difference_between_runs") == "training_corpus"
        and (ft.get("runs") or {}).get("primary", {}).get("uses_hparams")
        == "fine_tuning.hyperparameters"
        and (ft.get("runs") or {}).get("control", {}).get("uses_hparams")
        == "fine_tuning.hyperparameters",
        "experiment.yaml: primary/control must share identical training settings "
        "(identical_hparams_for_primary_and_control, uses_hparams pointers)",
        "config.primary_control_identical",
    )

    seeds = cfg.get("seeds") or {}
    _require(
        report,
        ft.get("seed") == 42 and seeds.get("finetuning") == 42 and seeds.get("global") == 42,
        f"experiment.yaml seed expected 42, got fine_tuning.seed={ft.get('seed')!r} "
        f"seeds.finetuning={seeds.get('finetuning')!r} seeds.global={seeds.get('global')!r}",
        "config.seed",
        "42",
    )


def validate_disjointness(
    ft_texts: list[str],
    held_out_texts: list[str],
    report: ValidationReport,
) -> None:
    """Exact equality or substring overlap between FT examples and extraction/probe texts."""
    problems = []
    held = [h.strip() for h in held_out_texts if isinstance(h, str) and h.strip()]
    for i, t in enumerate(ft_texts):
        t = t.strip()
        if not t:
            continue
        for h in held:
            if t == h:
                problems.append(f"FT[{i}] exact match with held-out text: {h[:80]!r}")
            elif h in t:
                problems.append(
                    f"FT[{i}] contains held-out as substring: held={h[:80]!r} ft={t[:80]!r}"
                )
            elif t in h:
                problems.append(
                    f"FT[{i}] is substring of held-out: ft={t[:80]!r} held={h[:80]!r}"
                )
    _require(
        report,
        not problems,
        "Dataset disjointness violated:\n  - " + "\n  - ".join(problems[:20])
        + (f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""),
        "disjointness.ft_vs_extract_probe",
        f"checked {len(ft_texts)} FT texts vs {len(held)} held-out strings",
    )
    report.counts["ft_texts_checked"] = len(ft_texts)
    report.counts["held_out_strings"] = len(held)


def run_validation(research_root: Path) -> ValidationReport:
    report = ValidationReport()
    frozen = research_root / "data" / "frozen"
    cfg_path = research_root / "configs" / "experiment.yaml"

    paths = {
        "facts": frozen / "facts.yaml",
        "explicit": frozen / "extraction_sets_explicit.yaml",
        "lexical": frozen / "extraction_sets_lexical.yaml",
        "reference": frozen / "reference_corpus.yaml",
        "cloze": frozen / "cloze_probes.yaml",
        "ft_primary": frozen / "finetuning_dataset_spec.yaml",
        "ft_control": frozen / "control_finetuning_dataset_spec.yaml",
        "config": cfg_path,
    }
    for name, p in paths.items():
        if not p.is_file():
            report.fail(f"Missing required file: {p}")
    if report.errors:
        return report

    try:
        facts_data = load_yaml(paths["facts"])
        explicit_data = load_yaml(paths["explicit"])
        lexical_data = load_yaml(paths["lexical"])
        ref_data = load_yaml(paths["reference"])
        cloze_data = load_yaml(paths["cloze"])
        ft_primary = load_yaml(paths["ft_primary"])
        ft_control = load_yaml(paths["ft_control"])
        cfg = load_yaml(paths["config"])
    except Exception as exc:
        report.fail(f"YAML parse failure: {exc}")
        return report

    facts = validate_facts(facts_data, report)
    explicit = validate_extraction_sets(explicit_data, report, "explicit")
    lexical = validate_extraction_sets(lexical_data, report, "lexical")
    validate_reference(ref_data, explicit, lexical, report)
    prefixes = validate_cloze(cloze_data, facts, report)
    p_texts, c_texts = validate_ft_datasets(ft_primary, ft_control, report)
    validate_config(cfg, report)

    held_out: list[str] = []
    for block in list(explicit.values()) + list(lexical.values()):
        for it in block:
            if isinstance(it.get("text"), str):
                held_out.append(it["text"])
    held_out.extend(prefixes)
    validate_disjointness(p_texts + c_texts, held_out, report)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate frozen TraceShift experiment specs (no model/GPU)."
    )
    parser.add_argument(
        "--research-root",
        type=Path,
        default=None,
        help="Path to research/ directory (default: auto-detect from script location)",
    )
    args = parser.parse_args(argv)

    if args.research_root is not None:
        research_root = args.research_root.resolve()
    else:
        script_dir = Path(__file__).resolve().parent
        # research/scripts/validate_frozen_experiment.py -> research/
        research_root = script_dir.parent

    if not (research_root / "data" / "frozen").is_dir():
        print(
            f"ERROR: research root not found or incomplete: {research_root}",
            file=sys.stderr,
        )
        return 2

    report = run_validation(research_root)

    print("=== TraceShift frozen-spec validation ===")
    print(f"research_root: {research_root}")
    print()
    if report.counts:
        print("Counts:")
        for k, v in report.counts.items():
            print(f"  {k}: {v}")
        print()
    print(f"Checks passed: {report.ok_count}")
    if report.errors:
        print(f"Errors: {report.error_count}")
        print()
        print("FAILURES (actionable):")
        for e in report.errors:
            print(f"  ✗ {e}")
        print()
        print("RESULT: FAIL")
        return 1

    print("RESULT: PASS — frozen specification is internally consistent.")
    print(
        "Note: this does not validate model/AI-Engram runtime compatibility."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
