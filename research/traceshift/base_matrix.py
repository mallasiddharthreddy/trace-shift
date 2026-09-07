"""BASE-model intervention–response matrix pipeline (explicit extraction).

Scientific definition (do not reinterpret):

    M_base[i, j] = response of fact j after intervention on fact i

Primary cell statistic (prior TraceShift / AI-Engram matrix convention):

    cosine_distance = 1 - global_cosine_similarity(
        unmodified_BASE_trace_j,
        post_intervention_trace_j,
    )

where global cosine similarity matches the previous ``compare_engrams`` formula
on float32 flattened projections (equivalently on the concatenated TraceVector).

Diagonal cells are null/NaN (a fact is not compared to itself for the primary
interaction analysis). Only ``selective`` calibrated facts are rows by default.
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
from .metrics import primary_response
from .model_loader import LoadedModel, reproducibility_record
from .paths import research_root

PRIMARY_EXTRACTION_VARIANT = "explicit"
FACT_IDS = ("F01", "F02", "F03", "F04")
DEFAULT_ALPHAS_PATH = (
    research_root() / "results" / "raw" / "alpha_calibration" / "base_alphas.yaml"
)


class BaseMatrixError(RuntimeError):
    """Base matrix pipeline configuration / eligibility failure."""


@dataclass
class FactAlphaSpec:
    fact_id: str
    status: str
    alpha: float | None
    reason: str = ""


@dataclass
class MatrixCell:
    row_fact: str
    col_fact: str
    value: float | None  # cosine_distance; None on diagonal / skipped
    cosine_similarity: float | None
    is_diagonal: bool


@dataclass
class BaseMatrixResult:
    model_id: str
    ai_engram_version: str
    extraction_variant: str
    model_stage: str
    fact_ids: list[str]
    matrix: list[list[float | None]]  # rows×cols cosine_distance; None=diagonal/skip
    cosine_similarity_matrix: list[list[float | None]]
    row_alphas: dict[str, float]
    row_eligibility: dict[str, str]
    diagonal_convention: str
    primary_metric: str
    baseline_trace_ids: dict[str, str]
    runtime: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    cells: list[MatrixCell] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "ai_engram_version": self.ai_engram_version,
            "extraction_variant": self.extraction_variant,
            "model_stage": self.model_stage,
            "fact_ids": self.fact_ids,
            "definition": (
                "M_base[i,j] = response of fact j after intervention on fact i "
                "(primary = cosine_distance = 1 - cosine_similarity of "
                "unmodified BASE trace_j vs post-intervention trace_j)"
            ),
            "primary_metric": self.primary_metric,
            "diagonal_convention": self.diagonal_convention,
            "matrix_cosine_distance": self.matrix,
            "matrix_cosine_similarity": self.cosine_similarity_matrix,
            "row_alphas": self.row_alphas,
            "row_eligibility": self.row_eligibility,
            "baseline_trace_ids": self.baseline_trace_ids,
            "cells": [asdict(c) for c in self.cells],
            "runtime": self.runtime,
            "notes": self.notes,
            "scientific_interpretation": "none — matrix values only; no significance claims",
        }


def load_calibrated_base_alphas(
    path: Path | str | None = None,
) -> dict[str, FactAlphaSpec]:
    """Load PRIMARY fixed alphas from alpha-calibration output (no recalibration).

    Default path: ``intervention.calibrated_alphas_path`` from experiment.yaml when
    present, else ``results/raw/alpha_calibration/base_alphas.yaml``.
    """
    if path is None:
        try:
            cfg = load_experiment_config()
            rel = (cfg.get("intervention") or {}).get("calibrated_alphas_path")
            if rel:
                cand = Path(rel)
                p = cand if cand.is_absolute() else (research_root() / cand)
            else:
                p = DEFAULT_ALPHAS_PATH
        except Exception:  # noqa: BLE001
            p = DEFAULT_ALPHAS_PATH
    else:
        p = Path(path)
    if not p.is_file():
        raise BaseMatrixError(
            f"Calibrated BASE alphas not found at {p}. "
            "Run BASE alpha calibration first and write base_alphas.yaml. "
            "Do not invent alpha values."
        )
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    alphas = (data or {}).get("alphas") or {}
    out: dict[str, FactAlphaSpec] = {}
    for fid in FACT_IDS:
        block = alphas.get(fid) or {}
        out[fid] = FactAlphaSpec(
            fact_id=fid,
            status=str(block.get("status") or "missing"),
            alpha=(
                None
                if block.get("alpha") is None
                else float(block.get("alpha"))
            ),
            reason=str(block.get("reason") or ""),
        )
    return out


def eligible_row_facts(
    alpha_specs: dict[str, FactAlphaSpec],
    *,
    allow_statuses: tuple[str, ...] = ("selective",),
    allow_override: bool = False,
    override_statuses: tuple[str, ...] = (),
    override_record: list[str] | None = None,
) -> list[str]:
    """Default: only ``selective``. Missing alpha / failed / no_erasure refused.

    Non-default statuses require ``allow_override=True`` and are recorded.
    """
    rows: list[str] = []
    for fid in FACT_IDS:
        spec = alpha_specs.get(fid) or FactAlphaSpec(fid, "missing", None)
        if spec.status in ("baseline_recall_failed", "no_erasure_in_grid"):
            if allow_override and spec.status in override_statuses:
                if spec.alpha is None:
                    raise BaseMatrixError(
                        f"{fid}: override requested for status={spec.status} "
                        "but alpha is missing; refusing to invent an alpha"
                    )
                rows.append(fid)
                if override_record is not None:
                    override_record.append(
                        f"{fid}: override include status={spec.status} alpha={spec.alpha}"
                    )
                continue
            # skipped by default
            continue
        if spec.status == "nonselective":
            if allow_override and "nonselective" in override_statuses:
                if spec.alpha is None:
                    raise BaseMatrixError(
                        f"{fid}: nonselective override but alpha missing"
                    )
                rows.append(fid)
                if override_record is not None:
                    override_record.append(
                        f"{fid}: override include nonselective alpha={spec.alpha}"
                    )
            continue
        if spec.status in allow_statuses:
            if spec.alpha is None:
                raise BaseMatrixError(
                    f"{fid}: status={spec.status} but selected_alpha is missing"
                )
            rows.append(fid)
            continue
        if spec.status == "missing":
            raise BaseMatrixError(
                f"{fid}: no calibration entry; refuse to invent an alpha"
            )
    return rows


def collect_base_engrams_once(
    loaded: LoadedModel,
    *,
    fact_ids: tuple[str, ...] = FACT_IDS,
    variant: str = PRIMARY_EXTRACTION_VARIANT,
    config_path: Path | str | None = None,
) -> dict[str, ExtractedEngram]:
    """Collect unmodified BASE Engrams once per fact (explicit variant).

    Used both for baseline TraceVectors and as the intervention Engrams for
    eligible rows — collect once, apply many times (AI-Engram 0.9.0).
    """
    if variant != PRIMARY_EXTRACTION_VARIANT:
        raise BaseMatrixError(
            f"Primary base matrix requires variant={PRIMARY_EXTRACTION_VARIANT!r}, "
            f"got {variant!r}"
        )
    out: dict[str, ExtractedEngram] = {}
    for fid in fact_ids:
        out[fid] = extract_engram(
            loaded.model,
            loaded.tokenizer,
            fid,
            PRIMARY_EXTRACTION_VARIANT,
            model_id=loaded.model_id,
            config_path=config_path,
        )
    return out


def extract_baseline_traces(
    loaded: LoadedModel,
    *,
    fact_ids: tuple[str, ...] = FACT_IDS,
    variant: str = PRIMARY_EXTRACTION_VARIANT,
    config_path: Path | str | None = None,
    collected: dict[str, ExtractedEngram] | None = None,
) -> dict[str, TraceVector]:
    """Extract unmodified BASE traces once for all facts (explicit variant).

    If ``collected`` is provided, derive traces from those Engrams (no recollect).
    """
    if collected is None:
        collected = collect_base_engrams_once(
            loaded, fact_ids=fact_ids, variant=variant, config_path=config_path
        )
    return {fid: engram_to_trace(collected[fid]) for fid in fact_ids}


def compute_base_matrix(
    loaded: LoadedModel,
    *,
    alphas_path: Path | str | None = None,
    config_path: Path | str | None = None,
    allow_override: bool = False,
    override_statuses: tuple[str, ...] = (),
    seed: int | None = None,
) -> BaseMatrixResult:
    """Compute M_base using fixed calibrated alphas (no recalibration)."""
    require_ai_engram_version(EXPECTED_ENGRAM_VERSION)
    if loaded.model_id != EXPECTED_MODEL_ID:
        raise BaseMatrixError(
            f"Expected frozen model {EXPECTED_MODEL_ID!r}, got {loaded.model_id!r}"
        )

    cfg = load_experiment_config(config_path)
    seeds = cfg.get("seeds") or {}
    if seed is None:
        seed = int(seeds.get("global") if seeds.get("global") is not None else 42)
    torch.manual_seed(seed)

    alpha_specs = load_calibrated_base_alphas(alphas_path)
    override_record: list[str] = []
    row_facts = eligible_row_facts(
        alpha_specs,
        allow_override=allow_override,
        override_statuses=override_statuses,
        override_record=override_record,
    )
    if not row_facts:
        raise BaseMatrixError(
            "No eligible selective row facts with calibrated alphas. "
            "Complete BASE alpha calibration first."
        )

    # Collect BASE Engrams once → baseline traces + row interventions (no recollect)
    base_engrams = collect_base_engrams_once(loaded, config_path=config_path)
    baseline_traces = extract_baseline_traces(
        loaded, collected=base_engrams, config_path=config_path
    )
    row_engrams = {fid: base_engrams[fid] for fid in row_facts}

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
                j = index[j_fact]
                is_diag = i_fact == j_fact
                cells.append(
                    MatrixCell(
                        row_fact=i_fact,
                        col_fact=j_fact,
                        value=None,
                        cosine_similarity=None,
                        is_diagonal=is_diag,
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
        # Re-extract all column facts j under the same frozen explicit texts
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

    notes = [
        "Primary matrix uses explicit extraction only (lexical_control excluded).",
        "Baseline traces computed once and reused across rows.",
        "Each row uses an independent fresh intervention copy; canonical BASE untouched.",
        "BASE Engrams collected once; reused for baseline traces and apply_engram "
        "(no recollect during alpha application).",
        "Default rows: selective only.",
    ]
    notes.extend(override_record)

    repro = reproducibility_record(loaded)
    return BaseMatrixResult(
        model_id=loaded.model_id,
        ai_engram_version=EXPECTED_ENGRAM_VERSION,
        extraction_variant=PRIMARY_EXTRACTION_VARIANT,
        model_stage="M_base",
        fact_ids=list(FACT_IDS),
        matrix=dist_mat,
        cosine_similarity_matrix=sim_mat,
        row_alphas=row_alphas,
        row_eligibility=row_eligibility,
        diagonal_convention="null (NaN): fact not compared to itself for primary interaction analysis",
        primary_metric="cosine_distance = 1 - cosine_similarity(baseline_j, post_j | intervene_i)",
        baseline_trace_ids={
            fid: f"base_trace:{fid}:{PRIMARY_EXTRACTION_VARIANT}:dim={baseline_traces[fid].vector.numel()}"
            for fid in FACT_IDS
        },
        runtime={
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "device": repro.get("device"),
            "dtype": repro.get("dtype"),
            "software": repro.get("software"),
            "alphas_path": str(Path(alphas_path) if alphas_path else DEFAULT_ALPHAS_PATH),
        },
        notes=notes,
        cells=cells,
    )


def _fact_labels() -> dict[str, str]:
    """Human labels from frozen facts.yaml (entity — domain)."""
    facts_path = research_root() / "data" / "frozen" / "facts.yaml"
    data = yaml.safe_load(facts_path.read_text(encoding="utf-8")) or {}
    out: dict[str, str] = {}
    for block in data.get("facts") or []:
        fid = str(block.get("id") or "")
        entity = str(block.get("entity") or "")
        domain = str(block.get("domain") or "")
        out[fid] = f"{entity} — {domain}"
    return out


def write_base_matrix_outputs(
    result: BaseMatrixResult,
    *,
    raw_dir: Path | None = None,
    tables_dir: Path | None = None,
) -> dict[str, Path]:
    raw_dir = raw_dir or (research_root() / "results" / "raw" / "base_matrix")
    tables_dir = tables_dir or (research_root() / "results" / "tables")
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    json_path = raw_dir / "base_matrix.json"
    md_path = tables_dir / "base_matrix.md"

    payload = result.to_dict()
    payload["fact_labels"] = _fact_labels()
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    labels = payload["fact_labels"]
    # Human-readable markdown table (cosine distances); no significance claims.
    lines = [
        "# Base intervention–response matrix (`M_base`)",
        "",
        f"- model_id: `{result.model_id}`",
        f"- AI-Engram: `{result.ai_engram_version}`",
        f"- extraction_variant: `{result.extraction_variant}`",
        f"- primary_metric: `{result.primary_metric}`",
        f"- diagonal: {result.diagonal_convention}",
        "",
        "Definition: `M_base[i,j]` = response of fact **j** after intervening on fact **i**.",
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
            "## Intervention alphas (rows)",
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
        cells = []
        for j, _ in enumerate(result.fact_ids):
            v = result.matrix[i][j]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append("—")
            else:
                cells.append(f"{v:.6f}")
        lines.append(f"| {i_fact} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "No scientific significance claims or tennis-vs-control interpretation in this table.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"base_matrix_json": json_path, "base_matrix_md": md_path}


def result_schema() -> dict[str, Any]:
    return {
        "files": {
            "results/raw/base_matrix/base_matrix.json": "metadata + matrices",
            "results/tables/base_matrix.md": "human-readable matrix",
        },
        "definition": "M_base[i,j] = response of fact j after intervention on fact i",
        "primary_metric": "cosine_distance = 1 - cosine_similarity",
        "similarity_convention": (
            "global float32 flattened-projection cosine matching prior compare_engrams"
        ),
        "diagonal": "null",
        "fact_order": list(FACT_IDS),
        "extraction_variant": PRIMARY_EXTRACTION_VARIANT,
        "default_row_eligibility": ["selective"],
    }


def validate_base_matrix_pipeline_without_model(
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Validate wiring without weights / without fabricating matrix values."""
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

    # Metric self-check
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([1.0, 0.0, 0.0])
    c = torch.tensor([0.0, 1.0, 0.0])
    from .metrics import cosine_distance_vectors, cosine_similarity_vectors

    assert abs(cosine_similarity_vectors(a, b) - 1.0) < 1e-6
    assert abs(cosine_distance_vectors(a, b) - 0.0) < 1e-6
    assert abs(cosine_similarity_vectors(a, c) - 0.0) < 1e-6
    assert abs(cosine_distance_vectors(a, c) - 1.0) < 1e-6

    alphas_path = DEFAULT_ALPHAS_PATH
    alphas_status = "present" if alphas_path.is_file() else "missing"
    eligibility_note = None
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

    # Refusal checks for bad statuses
    fake = {
        "F01": FactAlphaSpec("F01", "baseline_recall_failed", None),
        "F02": FactAlphaSpec("F02", "no_erasure_in_grid", None),
        "F03": FactAlphaSpec("F03", "selective", 0.4),
        "F04": FactAlphaSpec("F04", "selective", 0.6),
    }
    rows_ok = eligible_row_facts(fake)
    assert rows_ok == ["F03", "F04"]

    return {
        "ai_engram_version": EXPECTED_ENGRAM_VERSION,
        "model_id": EXPECTED_MODEL_ID,
        "extraction_variant": PRIMARY_EXTRACTION_VARIANT,
        "explicit_bundles": bundles,
        "metric_self_check_ok": True,
        "calibrated_alphas_file": str(alphas_path),
        "calibrated_alphas_status": alphas_status,
        "eligibility_probe": eligibility_note,
        "result_schema": result_schema(),
        "matrix_executed": False,
    }


def write_pending_status(raw_dir: Path | None = None) -> Path:
    raw_dir = raw_dir or (research_root() / "results" / "raw" / "base_matrix")
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "STATUS.yaml"
    payload = {
        "status": "pending_model_weights_and_or_calibrated_alphas",
        "model_id": EXPECTED_MODEL_ID,
        "ai_engram_version": EXPECTED_ENGRAM_VERSION,
        "extraction_variant": PRIMARY_EXTRACTION_VARIANT,
        "definition": "M_base[i,j] = response of fact j after intervention on fact i",
        "primary_metric": "cosine_distance = 1 - cosine_similarity",
        "message": (
            "Base matrix pipeline is implemented. Execution awaits local "
            "Qwen/Qwen3-1.7B weights and calibrated base_alphas.yaml. "
            "No matrix values have been fabricated."
        ),
        "schema": result_schema(),
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path
