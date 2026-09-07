"""BASE-model AI-Engram alpha calibration (behavioral dose selection).

Implements the frozen TraceShift protocol:
  baseline recall → collect Engram once (explicit) → alpha sweep → select
  smallest selective alpha (or nonselective fallback).

This is intervention-dose selection only — not a scientific effect-size analysis.
Fine-tuned recalibration and response matrices are out of scope.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from .cloze import ClozeEvaluation, evaluate_all_cloze, load_cloze_spec
from .engram import (
    EXPECTED_ENGRAM_VERSION,
    EXPECTED_MODEL_ID,
    apply_intervention,
    extract_engram,
    load_experiment_config,
    resolve_extraction_bundle,
    require_ai_engram_version,
)
from .model_loader import LoadedModel, reproducibility_record
from .paths import research_root

PRIMARY_EXTRACTION_VARIANT = "explicit"
FACT_IDS = ("F01", "F02", "F03", "F04")
EXPECTED_ALPHA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]


@dataclass
class ProbeGateSummary:
    probe_id: str
    winner: str
    expected_answer: str
    expected_wins: bool


@dataclass
class BaselineFactRecord:
    fact_id: str
    entity: str
    expected_answer: str
    status: str  # PASS | FAIL
    wins: int
    n_probes: int
    probes: list[ProbeGateSummary]


@dataclass
class AlphaSweepRow:
    alpha: float
    fact_statuses: dict[str, str]  # fact_id -> PASS|FAIL
    target_failed: bool
    other_pass_to_fail_count: int
    other_pass_to_fail: list[str]
    selective_ok: bool


@dataclass
class FactCalibrationResult:
    fact_id: str
    status: str  # selective | nonselective | baseline_recall_failed | no_erasure_in_grid
    selected_alpha: float | None
    reason: str
    baseline: BaselineFactRecord
    engram_collected: bool
    sweep: list[AlphaSweepRow] = field(default_factory=list)


@dataclass
class CalibrationRunResult:
    model_id: str
    ai_engram_version: str
    extraction_variant: str
    model_stage: str
    alpha_grid: list[float]
    baseline_recall: list[BaselineFactRecord]
    facts: list[FactCalibrationResult]
    runtime: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "ai_engram_version": self.ai_engram_version,
            "extraction_variant": self.extraction_variant,
            "model_stage": self.model_stage,
            "alpha_grid": self.alpha_grid,
            "baseline_recall": [asdict(b) for b in self.baseline_recall],
            "facts": [
                {
                    "fact_id": f.fact_id,
                    "status": f.status,
                    "selected_alpha": f.selected_alpha,
                    "reason": f.reason,
                    "engram_collected": f.engram_collected,
                    "baseline": asdict(f.baseline),
                    "target_failure_status": (
                        None
                        if not f.sweep
                        else {
                            "at_selected_alpha": (
                                next(
                                    (
                                        r.target_failed
                                        for r in f.sweep
                                        if f.selected_alpha is not None
                                        and abs(r.alpha - f.selected_alpha) < 1e-12
                                    ),
                                    None,
                                )
                            ),
                            "any_in_grid": any(r.target_failed for r in f.sweep),
                        }
                    ),
                    "collateral_pass_to_fail_at_selected": (
                        None
                        if f.selected_alpha is None
                        else next(
                            (
                                {
                                    "count": r.other_pass_to_fail_count,
                                    "facts": r.other_pass_to_fail,
                                }
                                for r in f.sweep
                                if abs(r.alpha - f.selected_alpha) < 1e-12
                            ),
                            None,
                        )
                    ),
                    "sweep": [asdict(r) for r in f.sweep],
                }
                for f in self.facts
            ],
            "runtime": self.runtime,
            "notes": self.notes,
            "scientific_interpretation": (
                "none — calibration is behavioral dose/eligibility only"
            ),
        }


def load_alpha_grid(cfg: dict[str, Any] | None = None) -> list[float]:
    cfg = cfg or load_experiment_config()
    grid = (cfg.get("intervention") or {}).get("alpha_grid")
    if not isinstance(grid, list) or not grid:
        raise ValueError("intervention.alpha_grid missing in experiment.yaml")
    out = [float(a) for a in grid]
    if len(out) != len(EXPECTED_ALPHA_GRID) or any(
        abs(a - b) > 1e-12 for a, b in zip(out, EXPECTED_ALPHA_GRID)
    ):
        raise ValueError(
            f"Frozen alpha grid mismatch: config={out}, expected={EXPECTED_ALPHA_GRID}. "
            "Do not change the grid in code."
        )
    return out


def _baseline_from_evaluation(evaluation: ClozeEvaluation) -> list[BaselineFactRecord]:
    records: list[BaselineFactRecord] = []
    for fr in evaluation.fact_results:
        probes = [
            ProbeGateSummary(
                probe_id=p.probe_id,
                winner=p.winner,
                expected_answer=p.expected_answer,
                expected_wins=p.expected_wins,
            )
            for p in fr.probe_results
        ]
        records.append(
            BaselineFactRecord(
                fact_id=fr.fact_id,
                entity=fr.entity,
                expected_answer=fr.expected_answer,
                status=fr.status,
                wins=fr.wins,
                n_probes=fr.n_probes,
                probes=probes,
            )
        )
    return records


def _statuses_from_evaluation(evaluation: ClozeEvaluation) -> dict[str, str]:
    return {fr.fact_id: fr.status for fr in evaluation.fact_results}


def _wrap_model_for_cloze(canonical: LoadedModel, model: Any) -> LoadedModel:
    """Score cloze with an intervened model copy while reusing tokenizer/device metadata."""
    return LoadedModel(
        model=model,
        tokenizer=canonical.tokenizer,
        model_id=canonical.model_id,
        tokenizer_id=canonical.tokenizer_id,
        device=canonical.device,
        dtype=canonical.dtype,
        source_path=canonical.source_path,
        config_path=canonical.config_path,
        software=dict(canonical.software),
    )


def count_pass_to_fail(
    baseline: dict[str, str],
    after: dict[str, str],
    *,
    exclude_fact: str,
) -> tuple[int, list[str]]:
    hits: list[str] = []
    for fid, base_status in baseline.items():
        if fid == exclude_fact:
            continue
        if base_status == "PASS" and after.get(fid) == "FAIL":
            hits.append(fid)
    return len(hits), hits


def select_alpha_from_sweep(
    sweep: list[AlphaSweepRow],
) -> tuple[str, float | None, str]:
    """Apply frozen selection rule. Does not extend the grid.

    Returns (status, selected_alpha, reason).
    """
    selective_candidate: AlphaSweepRow | None = None
    erasure_candidate: AlphaSweepRow | None = None
    for row in sorted(sweep, key=lambda r: r.alpha):
        if row.target_failed and erasure_candidate is None:
            erasure_candidate = row
        if row.selective_ok and selective_candidate is None:
            selective_candidate = row
            break

    if selective_candidate is not None:
        return (
            "selective",
            float(selective_candidate.alpha),
            "smallest alpha with target gate FAIL and zero PASS→FAIL collateral",
        )
    if erasure_candidate is not None:
        return (
            "nonselective",
            float(erasure_candidate.alpha),
            "no selective alpha in grid; smallest alpha causing target failure retained; "
            "not a primary selective-intervention result unless research-log override",
        )
    return (
        "no_erasure_in_grid",
        None,
        "no alpha in the frozen grid caused target recall failure; grid was not extended",
    )


def run_baseline_recall(loaded: LoadedModel) -> tuple[list[BaselineFactRecord], dict[str, str]]:
    evaluation = evaluate_all_cloze(loaded)
    records = _baseline_from_evaluation(evaluation)
    return records, _statuses_from_evaluation(evaluation)


def calibrate_base_model(
    loaded: LoadedModel,
    *,
    config_path: Path | str | None = None,
    seed: int | None = None,
) -> CalibrationRunResult:
    """Full BASE-model calibration on an already-loaded canonical model.

    Mutates nothing permanent on ``loaded.model`` (interventions use copies).
    """
    cfg = load_experiment_config(config_path)
    require_ai_engram_version(EXPECTED_ENGRAM_VERSION)
    alpha_grid = load_alpha_grid(cfg)
    seeds = cfg.get("seeds") or {}
    if seed is None:
        seed = int(seeds.get("global") if seeds.get("global") is not None else 42)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if loaded.model_id != EXPECTED_MODEL_ID:
        raise ValueError(
            f"Calibration requires frozen base model {EXPECTED_MODEL_ID!r}, "
            f"got {loaded.model_id!r}"
        )

    notes = [
        "PRIMARY calibration uses explicit extraction only (not lexical_control).",
        "BASE-model alphas are fixed primary doses for later base and FT matrices.",
        "No scientific effect-size interpretation is attached to calibration.",
    ]

    baseline_records, baseline_status = run_baseline_recall(loaded)
    fact_results: list[FactCalibrationResult] = []

    for base_rec in baseline_records:
        fid = base_rec.fact_id
        if base_rec.status != "PASS":
            fact_results.append(
                FactCalibrationResult(
                    fact_id=fid,
                    status="baseline_recall_failed",
                    selected_alpha=None,
                    reason="fact failed frozen 2-of-3 baseline recall; not calibrated",
                    baseline=base_rec,
                    engram_collected=False,
                    sweep=[],
                )
            )
            continue

        # Collect Engram once on the canonical base model.
        extracted = extract_engram(
            loaded.model,
            loaded.tokenizer,
            fid,
            PRIMARY_EXTRACTION_VARIANT,
            model_id=loaded.model_id,
            config_path=config_path,
        )

        sweep_rows: list[AlphaSweepRow] = []
        for alpha in alpha_grid:
            edited = apply_intervention(
                loaded.model, extracted.engram, float(alpha), inplace=False
            )
            eval_loaded = _wrap_model_for_cloze(loaded, edited)
            after_eval = evaluate_all_cloze(eval_loaded)
            after_status = _statuses_from_evaluation(after_eval)
            target_failed = after_status.get(fid) == "FAIL"
            n_collat, collat = count_pass_to_fail(
                baseline_status, after_status, exclude_fact=fid
            )
            selective_ok = bool(target_failed and n_collat == 0)
            sweep_rows.append(
                AlphaSweepRow(
                    alpha=float(alpha),
                    fact_statuses=after_status,
                    target_failed=target_failed,
                    other_pass_to_fail_count=n_collat,
                    other_pass_to_fail=collat,
                    selective_ok=selective_ok,
                )
            )
            # Drop edited model promptly (no permanent mutation of canonical).
            del edited, eval_loaded, after_eval

        status, selected, reason = select_alpha_from_sweep(sweep_rows)
        fact_results.append(
            FactCalibrationResult(
                fact_id=fid,
                status=status,
                selected_alpha=selected,
                reason=reason,
                baseline=base_rec,
                engram_collected=True,
                sweep=sweep_rows,
            )
        )

    repro = reproducibility_record(loaded)
    runtime = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "device": repro.get("device"),
        "dtype": repro.get("dtype"),
        "model_id": repro.get("model_id"),
        "tokenizer_id": repro.get("tokenizer_id"),
        "software": repro.get("software"),
        "source_path": repro.get("source_path"),
    }

    return CalibrationRunResult(
        model_id=loaded.model_id,
        ai_engram_version=EXPECTED_ENGRAM_VERSION,
        extraction_variant=PRIMARY_EXTRACTION_VARIANT,
        model_stage="M_base",
        alpha_grid=alpha_grid,
        baseline_recall=baseline_records,
        facts=fact_results,
        runtime=runtime,
        notes=notes,
    )


def primary_alphas_table(result: CalibrationRunResult) -> dict[str, Any]:
    """Compact primary-alpha map for later matrix stages (no interpretation)."""
    return {
        "model_id": result.model_id,
        "model_stage": result.model_stage,
        "extraction_variant": result.extraction_variant,
        "ai_engram_version": result.ai_engram_version,
        "alphas": {
            f.fact_id: {
                "status": f.status,
                "alpha": f.selected_alpha,
                "reason": f.reason,
            }
            for f in result.facts
        },
        "note": (
            "PRIMARY fixed alphas for base and FT intervention matrices; "
            "nonselective / baseline_recall_failed / no_erasure_in_grid are not "
            "primary selective-intervention results unless research-logged."
        ),
    }


def write_calibration_outputs(
    result: CalibrationRunResult,
    *,
    raw_dir: Path | None = None,
    tables_dir: Path | None = None,
) -> dict[str, Path]:
    raw_dir = raw_dir or (research_root() / "results" / "raw" / "alpha_calibration")
    tables_dir = tables_dir or (research_root() / "results" / "tables")
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    full_json = raw_dir / "base_alpha_calibration.json"
    alphas_yaml = raw_dir / "base_alphas.yaml"
    summary_md = tables_dir / "alpha_calibration_summary.md"
    summary_csv = tables_dir / "alpha_calibration_summary.csv"

    full_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    with alphas_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(primary_alphas_table(result), f, sort_keys=False)

    # Human-readable summary (markdown + csv; existing results/tables convention)
    lines = [
        "# Alpha calibration summary (BASE model)",
        "",
        f"- model_id: `{result.model_id}`",
        f"- AI-Engram: `{result.ai_engram_version}`",
        f"- extraction_variant: `{result.extraction_variant}`",
        f"- alpha_grid: `{result.alpha_grid}`",
        f"- device/dtype: `{result.runtime.get('device')}` / `{result.runtime.get('dtype')}`",
        f"- seed: `{result.runtime.get('seed')}`",
        "",
        "Dose/eligibility only — not a scientific effect-size result.",
        "",
        "| fact_id | baseline | status | selected_alpha | reason |",
        "|---------|----------|--------|----------------|--------|",
    ]
    csv_lines = ["fact_id,baseline,status,selected_alpha,reason"]
    for f in result.facts:
        alpha_s = "" if f.selected_alpha is None else str(f.selected_alpha)
        reason = f.reason.replace("|", "/")
        lines.append(
            f"| {f.fact_id} | {f.baseline.status} | {f.status} | {alpha_s or '—'} | {reason} |"
        )
        csv_lines.append(
            f"{f.fact_id},{f.baseline.status},{f.status},{alpha_s},"
            f"\"{f.reason.replace(chr(34), chr(39))}\""
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_csv.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    return {
        "full_json": full_json,
        "base_alphas_yaml": alphas_yaml,
        "summary_md": summary_md,
        "summary_csv": summary_csv,
    }


def result_schema() -> dict[str, Any]:
    """Document the machine-readable calibration result schema (no fabricated values)."""
    return {
        "files": {
            "results/raw/alpha_calibration/base_alpha_calibration.json": (
                "full run record"
            ),
            "results/raw/alpha_calibration/base_alphas.yaml": "compact primary alpha map",
            "results/tables/alpha_calibration_summary.md": "human summary",
            "results/tables/alpha_calibration_summary.csv": "tabular summary",
        },
        "top_level_fields": [
            "model_id",
            "ai_engram_version",
            "extraction_variant",
            "model_stage",
            "alpha_grid",
            "baseline_recall",
            "facts",
            "runtime",
            "notes",
            "scientific_interpretation",
        ],
        "fact_statuses": [
            "selective",
            "nonselective",
            "baseline_recall_failed",
            "no_erasure_in_grid",
        ],
        "selection_rule": {
            "choose": "smallest_alpha_in_grid",
            "require_target_erasure": True,
            "require_no_offtarget_pass_to_fail": True,
            "fallback": "smallest_target_erasure_as_nonselective",
            "grid_extension": "forbidden_in_this_pipeline",
        },
        "primary_extraction_variant": PRIMARY_EXTRACTION_VARIANT,
    }


def validate_calibration_pipeline_without_model(
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Config/control-flow validation with no weight download or Engram execution."""
    cfg = load_experiment_config(config_path)
    ver = require_ai_engram_version()
    grid = load_alpha_grid(cfg)
    cloze = load_cloze_spec()
    cal = cfg.get("alpha_calibration") or {}

    # Resolve explicit bundles for all facts (PRIMARY path).
    bundles = {
        fid: {
            "n_forget": len(b.forget_texts),
            "n_total": len(b.total_texts),
            "reference_id": b.reference_corpus_id,
        }
        for fid, b in (
            (fid, resolve_extraction_bundle(fid, PRIMARY_EXTRACTION_VARIANT, cfg=cfg))
            for fid in FACT_IDS
        )
    }

    # Unit-test selection logic without model outputs.
    demo_selective = [
        AlphaSweepRow(0.0, {"F01": "PASS", "F02": "PASS"}, False, 0, [], False),
        AlphaSweepRow(0.2, {"F01": "FAIL", "F02": "PASS"}, True, 0, [], True),
        AlphaSweepRow(0.4, {"F01": "FAIL", "F02": "FAIL"}, True, 1, ["F02"], False),
    ]
    st, al, _ = select_alpha_from_sweep(demo_selective)
    assert st == "selective" and al == 0.2

    demo_nonselective = [
        AlphaSweepRow(0.0, {"F01": "PASS", "F02": "PASS"}, False, 0, [], False),
        AlphaSweepRow(0.2, {"F01": "FAIL", "F02": "FAIL"}, True, 1, ["F02"], False),
        AlphaSweepRow(0.4, {"F01": "FAIL", "F02": "FAIL"}, True, 1, ["F02"], False),
    ]
    st2, al2, _ = select_alpha_from_sweep(demo_nonselective)
    assert st2 == "nonselective" and al2 == 0.2

    demo_none = [
        AlphaSweepRow(0.0, {"F01": "PASS"}, False, 0, [], False),
        AlphaSweepRow(2.0, {"F01": "PASS"}, False, 0, [], False),
    ]
    st3, al3, _ = select_alpha_from_sweep(demo_none)
    assert st3 == "no_erasure_in_grid" and al3 is None

    # Schema round-trip on an empty pending template (no fabricated alphas).
    schema = result_schema()

    return {
        "ai_engram_version": ver,
        "installed_matches_frozen": ver == EXPECTED_ENGRAM_VERSION,
        "model_id_frozen": EXPECTED_MODEL_ID,
        "alpha_grid": grid,
        "alpha_calibration_model_stage": cal.get("model_stage"),
        "cloze_n_facts": len(cloze.get("probes") or {}),
        "cloze_pass_threshold": cloze.get("recall_pass_threshold"),
        "primary_extraction_variant": PRIMARY_EXTRACTION_VARIANT,
        "explicit_bundles": bundles,
        "selection_logic_self_checks": {
            "selective": {"status": st, "alpha": al},
            "nonselective": {"status": st2, "alpha": al2},
            "no_erasure": {"status": st3, "alpha": al3},
        },
        "result_schema": schema,
        "engram_api_connected": True,
        "model_calibration_executed": False,
    }


def write_pending_status(raw_dir: Path | None = None) -> Path:
    """Record that calibration code exists but has not been executed (no fabricated alphas)."""
    raw_dir = raw_dir or (research_root() / "results" / "raw" / "alpha_calibration")
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "STATUS.yaml"
    payload = {
        "status": "pending_model_weights",
        "model_id": EXPECTED_MODEL_ID,
        "ai_engram_version": EXPECTED_ENGRAM_VERSION,
        "extraction_variant_primary": PRIMARY_EXTRACTION_VARIANT,
        "alpha_grid": EXPECTED_ALPHA_GRID,
        "message": (
            "Alpha-calibration pipeline is implemented. Execution awaits local "
            "Qwen/Qwen3-1.7B weights. No calibration alphas have been fabricated."
        ),
        "schema": result_schema(),
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path
