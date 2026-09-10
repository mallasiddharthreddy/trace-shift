"""FINE-TUNED-model intervention–response matrix (primary tennis LoRA).

Scientific definition (matches BASE matrix measurement; model state differs):

    M_FT[i, j] = 1 - cos(
        unmodified_FT_trace_j,
        post_intervention_FT_trace_j | intervene_i,
    )

Engrams are collected from the primary fine-tuned model (not reused from BASE).
Intervention alphas are the fixed BASE-calibrated alphas (no FT recalibration).
Extraction variant: explicit only (same frozen texts as the base matrix).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from .base_matrix import (
    DEFAULT_ALPHAS_PATH,
    FACT_IDS,
    PRIMARY_EXTRACTION_VARIANT,
    BaseMatrixError,
    FactAlphaSpec,
    MatrixCell,
    _fact_labels,
    eligible_row_facts,
    load_calibrated_base_alphas,
)
from .engram import (
    EXPECTED_ENGRAM_VERSION,
    EXPECTED_MODEL_ID,
    ExtractedEngram,
    TraceVector,
    apply_intervention,
    engram_to_trace,
    extract_engram,
    load_experiment_config,
    resolve_extraction_bundle,
    require_ai_engram_version,
)
from .finetune import load_finetune_run_spec
from .metrics import primary_response
from .model_loader import (
    LoadedModel,
    ModelWeightsUnavailableError,
    find_local_snapshot,
    load_model,
    reproducibility_record,
)
from .paths import research_root

MODEL_STATE = "primary_finetuned"
FT_MATRIX_STAGE = "M_FT"


class FtMatrixError(RuntimeError):
    """FT matrix pipeline configuration / eligibility failure."""


class FinetunedCheckpointUnavailableError(FileNotFoundError):
    """Primary LoRA adapter / checkpoint is not present locally."""


@dataclass
class FtMatrixResult:
    model_id: str
    model_state: str
    ai_engram_version: str
    extraction_variant: str
    model_stage: str
    fact_ids: list[str]
    matrix: list[list[float | None]]
    cosine_similarity_matrix: list[list[float | None]]
    row_alphas: dict[str, float]
    row_eligibility: dict[str, str]
    diagonal_convention: str
    primary_metric: str
    baseline_trace_ids: dict[str, str]
    source_checkpoint_path: str
    alpha_source: str
    runtime: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    cells: list[MatrixCell] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_state": self.model_state,
            "ai_engram_version": self.ai_engram_version,
            "extraction_variant": self.extraction_variant,
            "model_stage": self.model_stage,
            "fact_ids": self.fact_ids,
            "definition": (
                "M_FT[i,j] = response of fact j after intervention on fact i "
                "in the primary fine-tuned model "
                "(primary = cosine_distance = 1 - cosine_similarity of "
                "unmodified FT trace_j vs post-intervention FT trace_j)"
            ),
            "primary_metric": self.primary_metric,
            "diagonal_convention": self.diagonal_convention,
            "matrix_cosine_distance": self.matrix,
            "matrix_cosine_similarity": self.cosine_similarity_matrix,
            "row_alphas": self.row_alphas,
            "row_eligibility": self.row_eligibility,
            "baseline_trace_ids": self.baseline_trace_ids,
            "source_checkpoint_path": self.source_checkpoint_path,
            "alpha_source": self.alpha_source,
            "cells": [asdict(c) for c in self.cells],
            "runtime": self.runtime,
            "notes": self.notes,
            "scientific_interpretation": "none — matrix values only; no significance claims",
        }


def resolve_primary_ft_adapter_path(
    config_path: Path | str | None = None,
) -> Path:
    """Resolve the primary tennis LoRA ``final_adapter`` directory from config."""
    spec = load_finetune_run_spec("primary", config_path=config_path)
    adapter = (spec.output_dir / spec.final_adapter_dirname).resolve()
    return adapter


def primary_ft_adapter_available(
    config_path: Path | str | None = None,
) -> tuple[bool, Path, str]:
    """Return (available, path, reason). Does not download or train."""
    path = resolve_primary_ft_adapter_path(config_path)
    if not path.is_dir():
        return False, path, f"adapter directory missing: {path}"
    # PEFT writes adapter_config.json into the adapter dir
    cfg = path / "adapter_config.json"
    if not cfg.is_file():
        # Also accept adapter at output_dir root (framework variant)
        root_cfg = path.parent / "adapter_config.json"
        if root_cfg.is_file():
            return True, path.parent.resolve(), "adapter at checkpoint root"
        return False, path, f"adapter_config.json not found under {path}"
    return True, path, "ok"


def load_primary_finetuned_model(
    config_path: Path | str | None = None,
    *,
    device: Optional[str] = None,
) -> LoadedModel:
    """Load base Qwen3-1.7B + primary LoRA adapter (local files only).

    Raises ``ModelWeightsUnavailableError`` or ``FinetunedCheckpointUnavailableError``.
    Does not download weights or train.
    """
    available, adapter_path, reason = primary_ft_adapter_available(config_path)
    if not available:
        raise FinetunedCheckpointUnavailableError(
            f"Primary fine-tuned checkpoint unavailable ({reason}). "
            "Run primary LoRA fine-tuning first. No matrix values fabricated."
        )

    base = load_model(config_path, device=device, local_files_only=True)
    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(base.model, str(adapter_path))
    # Base already GPU-placed via device_map when device=cuda; avoid redundant .to()
    # that can conflict with accelerate device_map or spike host RAM.
    if base.device.type != "cuda":
        peft_model.to(base.device)
    peft_model.eval()

    software = dict(base.software)
    try:
        import importlib.metadata

        software["peft"] = importlib.metadata.version("peft")
    except Exception:  # noqa: BLE001
        software["peft"] = "unknown"

    return LoadedModel(
        model=peft_model,
        tokenizer=base.tokenizer,
        model_id=base.model_id,
        tokenizer_id=base.tokenizer_id,
        device=base.device,
        dtype=base.dtype,
        source_path=str(adapter_path),
        config_path=base.config_path,
        software=software,
    )


def _annotate_ft_engram(
    extracted: ExtractedEngram,
    *,
    checkpoint_path: str,
) -> ExtractedEngram:
    """Record that this Engram was collected from the primary FT model state."""
    meta = dict(extracted.metadata or {})
    meta["model_state"] = MODEL_STATE
    meta["checkpoint_path"] = checkpoint_path
    meta["engram_source"] = "fine_tuned_model_extraction"
    meta["not_reused_from_base"] = True
    extracted.metadata = meta
    return extracted


def collect_ft_engrams_once(
    loaded: LoadedModel,
    *,
    fact_ids: tuple[str, ...] = FACT_IDS,
    variant: str = PRIMARY_EXTRACTION_VARIANT,
    config_path: Path | str | None = None,
) -> dict[str, ExtractedEngram]:
    """Collect Engrams once from the fine-tuned model (explicit variant).

    Must not reuse BASE-model Engram objects. Same frozen forget/total texts.
    """
    if variant != PRIMARY_EXTRACTION_VARIANT:
        raise FtMatrixError(
            f"Primary FT matrix requires variant={PRIMARY_EXTRACTION_VARIANT!r}, "
            f"got {variant!r}"
        )
    if loaded.model_id != EXPECTED_MODEL_ID:
        raise FtMatrixError(
            f"Expected frozen model {EXPECTED_MODEL_ID!r}, got {loaded.model_id!r}"
        )
    out: dict[str, ExtractedEngram] = {}
    for fid in fact_ids:
        extracted = extract_engram(
            loaded.model,
            loaded.tokenizer,
            fid,
            PRIMARY_EXTRACTION_VARIANT,
            model_id=loaded.model_id,
            config_path=config_path,
        )
        out[fid] = _annotate_ft_engram(
            extracted, checkpoint_path=loaded.source_path
        )
    return out


def compute_ft_matrix(
    loaded: LoadedModel,
    *,
    alphas_path: Path | str | None = None,
    config_path: Path | str | None = None,
    allow_override: bool = False,
    override_statuses: tuple[str, ...] = (),
    seed: int | None = None,
) -> FtMatrixResult:
    """Compute M_FT with FT Engrams + fixed BASE-calibrated alphas."""
    require_ai_engram_version(EXPECTED_ENGRAM_VERSION)
    if loaded.model_id != EXPECTED_MODEL_ID:
        raise FtMatrixError(
            f"Expected frozen model {EXPECTED_MODEL_ID!r}, got {loaded.model_id!r}"
        )

    cfg = load_experiment_config(config_path)
    seeds = cfg.get("seeds") or {}
    if seed is None:
        seed = int(seeds.get("global") if seeds.get("global") is not None else 42)
    torch.manual_seed(seed)

    alpha_path = Path(alphas_path) if alphas_path else DEFAULT_ALPHAS_PATH
    try:
        alpha_specs = load_calibrated_base_alphas(alpha_path)
    except BaseMatrixError as exc:
        raise FtMatrixError(str(exc)) from exc

    override_record: list[str] = []
    try:
        row_facts = eligible_row_facts(
            alpha_specs,
            allow_override=allow_override,
            override_statuses=override_statuses,
            override_record=override_record,
        )
    except BaseMatrixError as exc:
        raise FtMatrixError(str(exc)) from exc

    if not row_facts:
        raise FtMatrixError(
            "No eligible selective row facts with BASE-calibrated alphas. "
            "Complete BASE alpha calibration first. Do not invent alphas."
        )

    # FT Engrams once → baseline FT traces + row interventions (no BASE Engram reuse)
    ft_engrams = collect_ft_engrams_once(loaded, config_path=config_path)
    baseline_traces: dict[str, TraceVector] = {
        fid: engram_to_trace(ft_engrams[fid]) for fid in FACT_IDS
    }
    row_engrams = {fid: ft_engrams[fid] for fid in row_facts}

    n = len(FACT_IDS)
    dist_mat: list[list[float | None]] = [[None] * n for _ in range(n)]
    sim_mat: list[list[float | None]] = [[None] * n for _ in range(n)]
    cells: list[MatrixCell] = []
    row_alphas = {fid: float(alpha_specs[fid].alpha) for fid in row_facts}  # type: ignore[arg-type]
    row_eligibility = {fid: alpha_specs[fid].status for fid in FACT_IDS}
    index = {fid: i for i, fid in enumerate(FACT_IDS)}

    for i_fact in FACT_IDS:
        i = index[i_fact]
        if i_fact not in row_facts:
            for j_fact in FACT_IDS:
                cells.append(
                    MatrixCell(
                        row_fact=i_fact,
                        col_fact=j_fact,
                        value=None,
                        cosine_similarity=None,
                        is_diagonal=i_fact == j_fact,
                    )
                )
            continue

        alpha_i = row_alphas[i_fact]
        edited = apply_intervention(
            loaded.model,
            row_engrams[i_fact].engram,
            alpha_i,
            inplace=False,
        )
        for j_fact in FACT_IDS:
            j = index[j_fact]
            is_diag = i_fact == j_fact
            if is_diag:
                dist_mat[i][j] = None
                sim_mat[i][j] = None
                cells.append(
                    MatrixCell(
                        row_fact=i_fact,
                        col_fact=j_fact,
                        value=None,
                        cosine_similarity=None,
                        is_diagonal=True,
                    )
                )
                continue

            post_extracted = extract_engram(
                edited,
                loaded.tokenizer,
                j_fact,
                PRIMARY_EXTRACTION_VARIANT,
                model_id=loaded.model_id,
                config_path=config_path,
            )
            post_extracted = _annotate_ft_engram(
                post_extracted,
                checkpoint_path=f"{loaded.source_path}|intervene={i_fact}",
            )
            post_trace = engram_to_trace(post_extracted)
            stats = primary_response(baseline_traces[j_fact], post_trace)
            dist_mat[i][j] = stats["cosine_distance"]
            sim_mat[i][j] = stats["cosine_similarity"]
            cells.append(
                MatrixCell(
                    row_fact=i_fact,
                    col_fact=j_fact,
                    value=stats["cosine_distance"],
                    cosine_similarity=stats["cosine_similarity"],
                    is_diagonal=False,
                )
            )
            del post_extracted, post_trace
        del edited
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    notes = [
        "Primary FT matrix uses explicit extraction only (lexical_control excluded).",
        "Engrams collected from the primary fine-tuned model — not reused from BASE.",
        "Intervention alphas are fixed BASE-calibrated alphas (no FT recalibration).",
        "Baseline FT traces computed once and reused across rows.",
        "Each row uses an independent fresh intervention copy; canonical FT model untouched.",
        "Default rows: selective only (same eligibility as base matrix).",
    ]
    notes.extend(override_record)

    repro = reproducibility_record(loaded)
    return FtMatrixResult(
        model_id=loaded.model_id,
        model_state=MODEL_STATE,
        ai_engram_version=EXPECTED_ENGRAM_VERSION,
        extraction_variant=PRIMARY_EXTRACTION_VARIANT,
        model_stage=FT_MATRIX_STAGE,
        fact_ids=list(FACT_IDS),
        matrix=dist_mat,
        cosine_similarity_matrix=sim_mat,
        row_alphas=row_alphas,
        row_eligibility=row_eligibility,
        diagonal_convention=(
            "null (NaN): fact not compared to itself for primary interaction analysis"
        ),
        primary_metric=(
            "cosine_distance = 1 - cosine_similarity("
            "baseline_FT_j, post_FT_j | intervene_i)"
        ),
        baseline_trace_ids={
            fid: (
                f"ft_trace:{fid}:{PRIMARY_EXTRACTION_VARIANT}:"
                f"{MODEL_STATE}:dim={baseline_traces[fid].vector.numel()}"
            )
            for fid in FACT_IDS
        },
        source_checkpoint_path=loaded.source_path,
        alpha_source=str(alpha_path),
        runtime={
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "device": repro.get("device"),
            "dtype": repro.get("dtype"),
            "software": repro.get("software"),
            "alphas_path": str(alpha_path),
            "model_state": MODEL_STATE,
        },
        notes=notes,
        cells=cells,
    )


def write_ft_matrix_outputs(
    result: FtMatrixResult,
    *,
    raw_dir: Path | None = None,
    tables_dir: Path | None = None,
) -> dict[str, Path]:
    raw_dir = raw_dir or (research_root() / "results" / "raw" / "ft_matrix")
    tables_dir = tables_dir or (research_root() / "results" / "tables")
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    json_path = raw_dir / "ft_matrix.json"
    md_path = tables_dir / "ft_matrix.md"

    payload = result.to_dict()
    payload["fact_labels"] = _fact_labels()
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    labels = payload["fact_labels"]
    lines = [
        "# Fine-tuned intervention–response matrix (`M_FT`)",
        "",
        f"- model_id: `{result.model_id}`",
        f"- model_state: `{result.model_state}`",
        f"- AI-Engram: `{result.ai_engram_version}`",
        f"- extraction_variant: `{result.extraction_variant}`",
        f"- primary_metric: `{result.primary_metric}`",
        f"- diagonal: {result.diagonal_convention}",
        f"- source_checkpoint: `{result.source_checkpoint_path}`",
        f"- alpha_source: `{result.alpha_source}` (BASE-calibrated; not recalibrated)",
        "",
        "Definition: `M_FT[i,j]` = response of fact **j** after intervening on fact **i** "
        "in the primary fine-tuned model.",
        "",
        "## Fact labels",
        "",
        "| fact_id | label |",
        "|---------|-------|",
    ]
    for fid in result.fact_ids:
        lines.append(f"| {fid} | {labels.get(fid, '')} |")

    lines.extend(
        [
            "",
            "## Intervention alphas (rows; BASE-calibrated)",
            "",
            "| row fact | label | status | alpha |",
            "|----------|-------|--------|-------|",
        ]
    )
    for fid in result.fact_ids:
        status = result.row_eligibility.get(fid, "")
        alpha = result.row_alphas.get(fid)
        alpha_s = "—" if (fid not in result.row_alphas or alpha is None) else str(alpha)
        lines.append(
            f"| {fid} | {labels.get(fid, '')} | {status} | {alpha_s} |"
        )

    lines.extend(
        [
            "",
            "## Matrix values (cosine distance)",
            "",
            "Rows = intervened fact *i*; columns = measured fact *j*. "
            "`—` = diagonal or ineligible row.",
            "",
        ]
    )
    header = "| i \\ j | " + " | ".join(result.fact_ids) + " |"
    sep = "|-------|" + "|".join(["------"] * len(result.fact_ids)) + "|"
    lines.append(header)
    lines.append(sep)
    for i, i_fact in enumerate(result.fact_ids):
        cells_s = []
        for j, _ in enumerate(result.fact_ids):
            v = result.matrix[i][j]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cells_s.append("—")
            else:
                cells_s.append(f"{v:.6f}")
        lines.append(f"| {i_fact} | " + " | ".join(cells_s) + " |")

    lines.extend(
        [
            "",
            "No scientific significance claims or delta interpretation in this table.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"ft_matrix_json": json_path, "ft_matrix_md": md_path}


def result_schema() -> dict[str, Any]:
    return {
        "files": {
            "results/raw/ft_matrix/ft_matrix.json": "metadata + matrices",
            "results/tables/ft_matrix.md": "human-readable matrix",
        },
        "definition": (
            "M_FT[i,j] = response of fact j after intervention on fact i "
            "in the primary fine-tuned model"
        ),
        "primary_metric": "cosine_distance = 1 - cosine_similarity",
        "similarity_convention": (
            "global float32 flattened-projection cosine matching prior compare_engrams "
            "(same as base matrix / metrics.primary_response)"
        ),
        "diagonal": "null",
        "fact_order": list(FACT_IDS),
        "extraction_variant": PRIMARY_EXTRACTION_VARIANT,
        "model_state": MODEL_STATE,
        "alpha_policy": "fixed BASE-calibrated alphas; no FT recalibration",
        "engram_policy": "collected from fine-tuned model; not reused from BASE",
        "default_row_eligibility": ["selective"],
    }


def validate_ft_matrix_pipeline_without_model(
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Validate wiring without weights / FT checkpoint / Engram execution."""
    require_ai_engram_version()
    cfg = load_experiment_config(config_path)
    bundles = {
        fid: {
            "n_forget": len(b.forget_texts),
            "n_total": len(b.total_texts),
            "ref": b.reference_corpus_id,
        }
        for fid, b in (
            (fid, resolve_extraction_bundle(fid, PRIMARY_EXTRACTION_VARIANT, cfg=cfg))
            for fid in FACT_IDS
        )
    }

    from .metrics import cosine_distance_vectors, cosine_similarity_vectors

    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([1.0, 0.0, 0.0])
    c = torch.tensor([0.0, 1.0, 0.0])
    assert abs(cosine_similarity_vectors(a, b) - 1.0) < 1e-6
    assert abs(cosine_distance_vectors(a, b) - 0.0) < 1e-6
    assert abs(cosine_similarity_vectors(a, c) - 0.0) < 1e-6
    assert abs(cosine_distance_vectors(a, c) - 1.0) < 1e-6

    alphas_path = DEFAULT_ALPHAS_PATH
    alphas_status = "present" if alphas_path.is_file() else "missing"
    eligibility_note: Any = None
    if alphas_path.is_file():
        try:
            specs = load_calibrated_base_alphas(alphas_path)
            rows = eligible_row_facts(specs)
            eligibility_note = {
                "selective_rows": rows,
                "specs": {k: asdict(v) for k, v in specs.items()},
            }
        except BaseMatrixError as exc:
            eligibility_note = {"error": str(exc)}

    # Same eligibility probe as base matrix
    fake = {
        "F01": FactAlphaSpec("F01", "baseline_recall_failed", None),
        "F02": FactAlphaSpec("F02", "no_erasure_in_grid", None),
        "F03": FactAlphaSpec("F03", "selective", 0.4),
        "F04": FactAlphaSpec("F04", "selective", 0.6),
    }
    assert eligible_row_facts(fake) == ["F03", "F04"]

    adapter_ok, adapter_path, adapter_reason = primary_ft_adapter_available(config_path)
    weights_local = find_local_snapshot(EXPECTED_MODEL_ID) is not None

    pending_reasons: list[str] = []
    if not weights_local:
        pending_reasons.append("base_model_weights_unavailable")
    if not adapter_ok:
        pending_reasons.append("primary_ft_checkpoint_unavailable")
    if alphas_status == "missing":
        pending_reasons.append("base_alphas_missing")

    return {
        "ai_engram_version": EXPECTED_ENGRAM_VERSION,
        "model_id": EXPECTED_MODEL_ID,
        "model_state": MODEL_STATE,
        "extraction_variant": PRIMARY_EXTRACTION_VARIANT,
        "fact_order": list(FACT_IDS),
        "explicit_bundles": bundles,
        "metric_self_check_ok": True,
        "calibrated_alphas_file": str(alphas_path),
        "calibrated_alphas_status": alphas_status,
        "alpha_policy": "fixed BASE-calibrated; no FT recalibration",
        "eligibility_probe": eligibility_note,
        "eligibility_matches_base_matrix": True,
        "primary_ft_adapter_path": str(adapter_path),
        "primary_ft_adapter_available": adapter_ok,
        "primary_ft_adapter_reason": adapter_reason,
        "base_weights_available_locally": weights_local,
        "result_schema": result_schema(),
        "pending_reasons": pending_reasons,
        "matrix_executed": False,
        "engram_executed": False,
        "training_executed": False,
    }


def write_pending_status(
    *,
    report: dict[str, Any] | None = None,
    raw_dir: Path | None = None,
) -> Path:
    raw_dir = raw_dir or (research_root() / "results" / "raw" / "ft_matrix")
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "STATUS.yaml"
    payload: dict[str, Any] = {
        "status": "pending_weights_and_or_primary_ft_checkpoint_and_or_base_alphas",
        "model_id": EXPECTED_MODEL_ID,
        "model_state": MODEL_STATE,
        "ai_engram_version": EXPECTED_ENGRAM_VERSION,
        "extraction_variant": PRIMARY_EXTRACTION_VARIANT,
        "definition": (
            "M_FT[i,j] = response of fact j after intervention on fact i "
            "in the primary fine-tuned model"
        ),
        "primary_metric": "cosine_distance = 1 - cosine_similarity",
        "alpha_policy": "fixed BASE-calibrated alphas; no FT recalibration",
        "engram_policy": "FT-model Engrams only; not reused from BASE",
        "message": (
            "FT matrix pipeline is implemented. Execution awaits local "
            "Qwen/Qwen3-1.7B weights, primary LoRA checkpoint, and "
            "base_alphas.yaml. No matrix values have been fabricated."
        ),
        "schema": result_schema(),
        "matrix_executed": False,
    }
    if report is not None:
        payload["last_dry_validate"] = {
            k: report[k]
            for k in (
                "primary_ft_adapter_available",
                "primary_ft_adapter_path",
                "calibrated_alphas_status",
                "base_weights_available_locally",
                "pending_reasons",
                "matrix_executed",
            )
            if k in report
        }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path
