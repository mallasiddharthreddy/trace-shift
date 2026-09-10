"""Lexical-control alternative-hypothesis experiment for TraceShift.

Secondary sensitivity analysis: does the primary pre/post-FT interaction pattern
depend mainly on explicit lexical/domain cues in extraction text?

Procedure (same models, BASE-calibrated alphas, metrics, fact order):
  1. Compute M_base_lexical with variant=lexical_control
  2. Compute M_FT_lexical with separately extracted FT Engrams (lexical_control)
  3. Delta_M_lexical = M_FT_lexical - M_base_lexical
  4. Descriptively compare to primary explicit Delta-M (no second inferential test)

Alphas: reused from BASE explicit calibration — NOT recalibrated on lexical text.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import yaml

from .base_matrix import (
    DEFAULT_ALPHAS_PATH,
    FACT_IDS,
    BaseMatrixError,
    MatrixCell,
    eligible_row_facts,
    load_calibrated_base_alphas,
)
from .delta_analysis import (
    DEFAULT_BASE_MATRIX_PATH,
    DEFAULT_FT_MATRIX_PATH,
    OBSERVED_CONTROL_PAIRS,
    OBSERVED_IN_DOMAIN_PAIRS,
    DeltaAnalysisError,
    compute_delta_m,
    load_matrix_json,
    primary_contrast,
)
from .engram import (
    EXPECTED_ENGRAM_VERSION,
    EXPECTED_MODEL_ID,
    ExtractedEngram,
    TraceVector,
    apply_intervention,
    assert_variant_isolation,
    engram_to_trace,
    extract_engram,
    load_experiment_config,
    resolve_extraction_bundle,
    require_ai_engram_version,
)
from .metrics import primary_response
from .model_loader import (
    LoadedModel,
    find_local_snapshot,
    load_model,
    reproducibility_record,
)
from .paths import research_root

LEXICAL_VARIANT = "lexical_control"
EXPLICIT_VARIANT = "explicit"
EXPECTED_LEXICAL_REF_ID = "shared_reference_lexical_control"
EXPECTED_EXPLICIT_REF_ID = "shared_reference_explicit"

DEFAULT_OUTPUT_JSON = (
    research_root()
    / "results"
    / "raw"
    / "alternative_hypotheses"
    / "lexical_control.json"
)
DEFAULT_OUTPUT_MD = research_root() / "results" / "tables" / "lexical_control.md"
DEFAULT_STATUS = (
    research_root()
    / "results"
    / "raw"
    / "alternative_hypotheses"
    / "lexical_control_STATUS.yaml"
)


class LexicalControlError(RuntimeError):
    """Lexical-control experiment configuration / isolation failure."""


@dataclass
class LexicalMatrixResult:
    model_id: str
    model_stage: str
    model_state: str
    extraction_variant: str
    reference_corpus_id: str
    ai_engram_version: str
    fact_ids: list[str]
    matrix: list[list[float | None]]
    cosine_similarity_matrix: list[list[float | None]]
    row_alphas: dict[str, float]
    row_eligibility: dict[str, str]
    baseline_trace_ids: dict[str, str]
    diagonal_convention: str
    primary_metric: str
    alpha_source: str
    source_checkpoint_path: str | None = None
    notes: list[str] = field(default_factory=list)
    cells: list[MatrixCell] = field(default_factory=list)
    runtime: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_stage": self.model_stage,
            "model_state": self.model_state,
            "extraction_variant": self.extraction_variant,
            "reference_corpus_id": self.reference_corpus_id,
            "ai_engram_version": self.ai_engram_version,
            "fact_ids": self.fact_ids,
            "matrix_cosine_distance": self.matrix,
            "matrix_cosine_similarity": self.cosine_similarity_matrix,
            "row_alphas": self.row_alphas,
            "row_eligibility": self.row_eligibility,
            "baseline_trace_ids": self.baseline_trace_ids,
            "diagonal_convention": self.diagonal_convention,
            "primary_metric": self.primary_metric,
            "alpha_source": self.alpha_source,
            "alpha_policy": (
                "fixed BASE-calibrated alphas from explicit calibration; "
                "NOT recalibrated on lexical_control inputs"
            ),
            "source_checkpoint_path": self.source_checkpoint_path,
            "cells": [asdict(c) for c in self.cells],
            "runtime": self.runtime,
            "notes": self.notes,
        }


def _software() -> dict[str, str]:
    out = {
        "python": f"{__import__('sys').version_info.major}."
        f"{__import__('sys').version_info.minor}."
        f"{__import__('sys').version_info.micro}",
    }
    for pkg in ("torch", "transformers", "peft", "ai-engram", "pyyaml"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "not-installed"
    return out


def assert_lexical_isolation(
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Assert frozen lexical targets/refs; no explicit text mixed in.

    Resolves 4 lexical + 4 explicit bundles (8 total) for isolation checks.
    """
    cfg = load_experiment_config(config_path)
    lexical_bundles: dict[str, Any] = {}
    explicit_bundles: dict[str, Any] = {}
    lexical_texts: set[str] = set()
    explicit_texts: set[str] = set()

    for fid in FACT_IDS:
        lex = resolve_extraction_bundle(fid, LEXICAL_VARIANT, cfg=cfg)
        exp = resolve_extraction_bundle(fid, EXPLICIT_VARIANT, cfg=cfg)
        assert_variant_isolation(lex.variant, LEXICAL_VARIANT)
        assert_variant_isolation(exp.variant, EXPLICIT_VARIANT)

        if lex.variant != LEXICAL_VARIANT:
            raise LexicalControlError(f"{fid}: expected lexical_control bundle")
        if exp.variant != EXPLICIT_VARIANT:
            raise LexicalControlError(f"{fid}: expected explicit bundle for isolation")
        if lex.reference_corpus_id != EXPECTED_LEXICAL_REF_ID:
            raise LexicalControlError(
                f"{fid}: lexical ref id {lex.reference_corpus_id!r} != "
                f"{EXPECTED_LEXICAL_REF_ID!r}"
            )
        if exp.reference_corpus_id != EXPECTED_EXPLICIT_REF_ID:
            raise LexicalControlError(
                f"{fid}: explicit ref id {exp.reference_corpus_id!r} != "
                f"{EXPECTED_EXPLICIT_REF_ID!r}"
            )
        if len(lex.forget_texts) != 5 or len(lex.total_texts) != 20:
            raise LexicalControlError(
                f"{fid}/lexical: expected 5 forget + 20 total, got "
                f"{len(lex.forget_texts)}/{len(lex.total_texts)}"
            )
        # No explicit reference texts inside lexical total
        if set(lex.total_texts) & set(exp.total_texts):
            raise LexicalControlError(
                f"{fid}: lexical reference shares text with explicit reference"
            )
        if set(lex.forget_texts) & set(exp.forget_texts):
            raise LexicalControlError(
                f"{fid}: lexical targets share text with explicit targets"
            )
        # Paths must point at frozen lexical / explicit set files
        if "lexical" not in Path(lex.target_set_path).name:
            raise LexicalControlError(
                f"{fid}: lexical target path unexpected: {lex.target_set_path}"
            )
        if "explicit" not in Path(exp.target_set_path).name:
            raise LexicalControlError(
                f"{fid}: explicit target path unexpected: {exp.target_set_path}"
            )

        lexical_texts.update(lex.forget_texts)
        lexical_texts.update(lex.total_texts)
        explicit_texts.update(exp.forget_texts)
        explicit_texts.update(exp.total_texts)

        lexical_bundles[fid] = {
            "n_forget": len(lex.forget_texts),
            "n_total": len(lex.total_texts),
            "ref": lex.reference_corpus_id,
            "target_set_path": lex.target_set_path,
            "forget_ids": lex.forget_ids,
        }
        explicit_bundles[fid] = {
            "n_forget": len(exp.forget_texts),
            "n_total": len(exp.total_texts),
            "ref": exp.reference_corpus_id,
            "target_set_path": exp.target_set_path,
        }

    overlap = lexical_texts & explicit_texts
    if overlap:
        raise LexicalControlError(
            f"Lexical/explicit text overlap ({len(overlap)} strings); "
            "variants must remain isolated"
        )

    return {
        "n_bundle_resolutions": 8,
        "lexical_bundles": lexical_bundles,
        "explicit_bundles_for_isolation": explicit_bundles,
        "lexical_reference_id": EXPECTED_LEXICAL_REF_ID,
        "explicit_reference_id": EXPECTED_EXPLICIT_REF_ID,
        "lexical_unique_texts": len(lexical_texts),
        "explicit_unique_texts": len(explicit_texts),
        "cross_variant_text_overlap": 0,
        "isolation_ok": True,
    }


def _annotate_lexical_engram(
    extracted: ExtractedEngram,
    *,
    model_state: str,
    checkpoint_path: str | None,
) -> ExtractedEngram:
    meta = dict(extracted.metadata or {})
    meta["extraction_variant"] = LEXICAL_VARIANT
    meta["model_state"] = model_state
    meta["engram_source"] = f"{model_state}_lexical_control_extraction"
    meta["not_reused_from_other_variant"] = True
    meta["not_reused_from_base_if_ft"] = model_state != "base"
    if checkpoint_path:
        meta["checkpoint_path"] = checkpoint_path
    extracted.metadata = meta
    return extracted


def collect_lexical_engrams_once(
    loaded: LoadedModel,
    *,
    model_state: str,
    config_path: Path | str | None = None,
    fact_ids: Sequence[str] = FACT_IDS,
) -> dict[str, ExtractedEngram]:
    """Collect Engrams once with lexical_control texts (no explicit mix)."""
    if loaded.model_id != EXPECTED_MODEL_ID:
        raise LexicalControlError(
            f"Expected {EXPECTED_MODEL_ID!r}, got {loaded.model_id!r}"
        )
    out: dict[str, ExtractedEngram] = {}
    for fid in fact_ids:
        extracted = extract_engram(
            loaded.model,
            loaded.tokenizer,
            fid,
            LEXICAL_VARIANT,  # type: ignore[arg-type]
            model_id=loaded.model_id,
            config_path=config_path,
        )
        if extracted.variant != LEXICAL_VARIANT:
            raise LexicalControlError(
                f"{fid}: Engram variant {extracted.variant!r} != lexical_control"
            )
        if extracted.reference_corpus_id != EXPECTED_LEXICAL_REF_ID:
            raise LexicalControlError(
                f"{fid}: Engram ref {extracted.reference_corpus_id!r}"
            )
        out[fid] = _annotate_lexical_engram(
            extracted,
            model_state=model_state,
            checkpoint_path=loaded.source_path,
        )
    return out


def compute_lexical_matrix(
    loaded: LoadedModel,
    *,
    model_stage: str,
    model_state: str,
    alphas_path: Path | str | None = None,
    config_path: Path | str | None = None,
    allow_override: bool = False,
    override_statuses: tuple[str, ...] = (),
    seed: int | None = None,
) -> LexicalMatrixResult:
    """Intervention–response matrix using lexical_control extraction + BASE alphas."""
    require_ai_engram_version(EXPECTED_ENGRAM_VERSION)
    assert_lexical_isolation(config_path)

    cfg = load_experiment_config(config_path)
    seeds = cfg.get("seeds") or {}
    if seed is None:
        seed = int(seeds.get("global") if seeds.get("global") is not None else 42)
    torch.manual_seed(seed)

    alpha_path = Path(alphas_path) if alphas_path else DEFAULT_ALPHAS_PATH
    try:
        alpha_specs = load_calibrated_base_alphas(alpha_path)
        row_facts = eligible_row_facts(
            alpha_specs,
            allow_override=allow_override,
            override_statuses=override_statuses,
        )
    except BaseMatrixError as exc:
        raise LexicalControlError(str(exc)) from exc

    if not row_facts:
        raise LexicalControlError(
            "No eligible selective rows with BASE-calibrated alphas. "
            "Do not invent alphas; complete explicit BASE calibration first."
        )

    # Collect lexical Engrams once from THIS model state (BASE or FT)
    engrams = collect_lexical_engrams_once(
        loaded, model_state=model_state, config_path=config_path
    )
    baselines: dict[str, TraceVector] = {
        fid: engram_to_trace(engrams[fid]) for fid in FACT_IDS
    }

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
            engrams[i_fact].engram,
            alpha_i,
            inplace=False,
        )
        for j_fact in FACT_IDS:
            j = index[j_fact]
            if i_fact == j_fact:
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

            post = extract_engram(
                edited,
                loaded.tokenizer,
                j_fact,
                LEXICAL_VARIANT,  # type: ignore[arg-type]
                model_id=loaded.model_id,
                config_path=config_path,
            )
            post = _annotate_lexical_engram(
                post,
                model_state=f"{model_state}|intervene={i_fact}",
                checkpoint_path=loaded.source_path,
            )
            post_trace = engram_to_trace(post)
            stats = primary_response(baselines[j_fact], post_trace)
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
            del post, post_trace
        del edited
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    repro = reproducibility_record(loaded)
    notes = [
        "Secondary lexical-control extraction experiment.",
        "Uses frozen lexical_control targets + shared_reference_lexical_control.",
        "Alphas are BASE explicit-calibrated values — NOT recalibrated on lexical text.",
        "Same cosine-distance metric and eligibility as primary matrices.",
        "Engrams collected from this model state only (no cross-model Engram reuse).",
    ]
    return LexicalMatrixResult(
        model_id=loaded.model_id,
        model_stage=model_stage,
        model_state=model_state,
        extraction_variant=LEXICAL_VARIANT,
        reference_corpus_id=EXPECTED_LEXICAL_REF_ID,
        ai_engram_version=EXPECTED_ENGRAM_VERSION,
        fact_ids=list(FACT_IDS),
        matrix=dist_mat,
        cosine_similarity_matrix=sim_mat,
        row_alphas=row_alphas,
        row_eligibility=row_eligibility,
        baseline_trace_ids={
            fid: (
                f"lexical_trace:{fid}:{LEXICAL_VARIANT}:{model_state}:"
                f"dim={baselines[fid].vector.numel()}"
            )
            for fid in FACT_IDS
        },
        diagonal_convention=(
            "null (NaN): fact not compared to itself for primary interaction analysis"
        ),
        primary_metric=(
            "cosine_distance = 1 - cosine_similarity("
            "baseline_lexical_j, post_lexical_j | intervene_i)"
        ),
        alpha_source=str(alpha_path),
        source_checkpoint_path=loaded.source_path,
        notes=notes,
        cells=cells,
        runtime={
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "device": repro.get("device"),
            "dtype": repro.get("dtype"),
            "software": repro.get("software"),
        },
    )


def descriptive_delta_comparison(
    delta_lexical: list[list[float | None]],
    delta_explicit: list[list[float | None]] | None,
    fact_ids: Sequence[str],
) -> dict[str, Any]:
    """Simple descriptive comparison — not a second inferential framework."""
    m_in_l, m_c_l, t_l = primary_contrast(
        delta_lexical,
        fact_ids,
        OBSERVED_IN_DOMAIN_PAIRS,
        OBSERVED_CONTROL_PAIRS,
    )
    out: dict[str, Any] = {
        "lexical": {
            "mean_in_domain_delta": m_in_l,
            "mean_control_delta": m_c_l,
            "contrast_T": t_l,
        },
        "explicit_primary": None,
        "comparison": None,
        "notes": [
            "Descriptive only; does not replace the primary exact permutation test.",
            "Positive T means in-domain Delta-M exceeds in-domain→control Delta-M.",
        ],
    }
    if delta_explicit is None:
        out["comparison"] = {
            "status": "primary_explicit_delta_unavailable",
            "message": (
                "Primary explicit matrices not present; lexical Delta-M computed "
                "but cross-variant comparison deferred."
            ),
        }
        return out

    m_in_e, m_c_e, t_e = primary_contrast(
        delta_explicit,
        fact_ids,
        OBSERVED_IN_DOMAIN_PAIRS,
        OBSERVED_CONTROL_PAIRS,
    )
    out["explicit_primary"] = {
        "mean_in_domain_delta": m_in_e,
        "mean_control_delta": m_c_e,
        "contrast_T": t_e,
    }
    same_sign = (t_l >= 0 and t_e >= 0) or (t_l < 0 and t_e < 0)
    out["comparison"] = {
        "tennis_vs_control_direction_same_sign": same_sign,
        "T_lexical": t_l,
        "T_explicit": t_e,
        "T_difference_lexical_minus_explicit": float(t_l - t_e),
        "mean_in_domain_difference_lexical_minus_explicit": float(m_in_l - m_in_e),
        "mean_control_difference_lexical_minus_explicit": float(m_c_l - m_c_e),
        "magnitude_note": (
            "Inspect |T| change to judge sensitivity to explicit lexical cues; "
            "no automatic significance claim."
        ),
    }
    return out


def _try_load_primary_explicit_delta() -> tuple[
    list[list[float | None]] | None, dict[str, Any]
]:
    meta: dict[str, Any] = {
        "base_matrix_path": str(DEFAULT_BASE_MATRIX_PATH),
        "ft_matrix_path": str(DEFAULT_FT_MATRIX_PATH),
        "available": False,
    }
    if not DEFAULT_BASE_MATRIX_PATH.is_file() or not DEFAULT_FT_MATRIX_PATH.is_file():
        meta["reason"] = "primary matrix JSON files missing"
        return None, meta
    try:
        base = load_matrix_json(DEFAULT_BASE_MATRIX_PATH)
        ft = load_matrix_json(DEFAULT_FT_MATRIX_PATH)
        if base.get("extraction_variant") != EXPLICIT_VARIANT:
            raise DeltaAnalysisError("primary base matrix is not explicit")
        if ft.get("extraction_variant") != EXPLICIT_VARIANT:
            raise DeltaAnalysisError("primary FT matrix is not explicit")
        from .delta_analysis import _distance_matrix, validate_matrix_pair

        validate_matrix_pair(
            base,
            ft,
            base_path=DEFAULT_BASE_MATRIX_PATH,
            ft_path=DEFAULT_FT_MATRIX_PATH,
        )
        delta = compute_delta_m(
            _distance_matrix(base),
            _distance_matrix(ft),
            list(FACT_IDS),
        )
        meta["available"] = True
        return delta, meta
    except (DeltaAnalysisError, OSError, json.JSONDecodeError, KeyError) as exc:
        meta["reason"] = str(exc)
        return None, meta


def run_lexical_control_experiment(
    *,
    config_path: Path | str | None = None,
    alphas_path: Path | str | None = None,
    device: Optional[str] = None,
    allow_override: bool = False,
    override_statuses: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Full lexical-control BASE + FT matrices + Delta-M + descriptive comparison."""
    from .ft_matrix import load_primary_finetuned_model

    isolation = assert_lexical_isolation(config_path)
    base_loaded = load_model(config_path, device=device, local_files_only=True)
    base_mat = compute_lexical_matrix(
        base_loaded,
        model_stage="M_base",
        model_state="base",
        alphas_path=alphas_path,
        config_path=config_path,
        allow_override=allow_override,
        override_statuses=override_statuses,
    )

    ft_loaded = load_primary_finetuned_model(config_path, device=device)
    ft_mat = compute_lexical_matrix(
        ft_loaded,
        model_stage="M_FT",
        model_state="primary_finetuned",
        alphas_path=alphas_path,
        config_path=config_path,
        allow_override=allow_override,
        override_statuses=override_statuses,
    )

    delta = compute_delta_m(base_mat.matrix, ft_mat.matrix, list(FACT_IDS))
    explicit_delta, explicit_meta = _try_load_primary_explicit_delta()
    comparison = descriptive_delta_comparison(delta, explicit_delta, list(FACT_IDS))

    return {
        "status": "complete",
        "experiment": "lexical_control_alternative_hypothesis",
        "role": "secondary_sensitivity_analysis",
        "model_id": EXPECTED_MODEL_ID,
        "ai_engram_version": EXPECTED_ENGRAM_VERSION,
        "extraction_variant": LEXICAL_VARIANT,
        "reference_corpus_id": EXPECTED_LEXICAL_REF_ID,
        "fact_ordering": list(FACT_IDS),
        "alpha_policy": (
            "Reuse BASE explicit-calibrated alphas; do not recalibrate on lexical_control."
        ),
        "fixed_alphas_used": base_mat.row_alphas,
        "isolation": isolation,
        "base_lexical_matrix": base_mat.to_dict(),
        "ft_lexical_matrix": ft_mat.to_dict(),
        "delta_m_lexical": delta,
        "primary_explicit_delta_meta": explicit_meta,
        "descriptive_comparison": comparison,
        "software": _software(),
        "runtime": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "notes": [
            "Does not modify primary Delta-M statistical test.",
            "Descriptive comparison only when primary explicit matrices exist.",
            "FT lexical Engrams extracted separately from BASE lexical Engrams.",
        ],
        "scientific_interpretation": "none — report matrices/comparison; no auto claims",
    }


def write_lexical_control_outputs(
    payload: dict[str, Any],
    *,
    json_path: Path | None = None,
    md_path: Path | None = None,
) -> dict[str, Path]:
    json_path = json_path or DEFAULT_OUTPUT_JSON
    md_path = md_path or DEFAULT_OUTPUT_MD
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fact_ids = payload.get("fact_ordering") or list(FACT_IDS)
    delta = payload.get("delta_m_lexical") or []
    base_m = (payload.get("base_lexical_matrix") or {}).get(
        "matrix_cosine_distance"
    ) or []
    ft_m = (payload.get("ft_lexical_matrix") or {}).get("matrix_cosine_distance") or []
    comp = payload.get("descriptive_comparison") or {}
    alphas = payload.get("fixed_alphas_used") or {}

    def _fmt_mat(mat: list, title: str) -> list[str]:
        lines = [f"## {title}", "", "| i \\ j | " + " | ".join(fact_ids) + " |"]
        lines.append("|-------|" + "|".join(["------"] * len(fact_ids)) + "|")
        for i, i_fact in enumerate(fact_ids):
            cells = []
            for j in range(len(fact_ids)):
                v = mat[i][j] if i < len(mat) else None
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    cells.append("—")
                else:
                    cells.append(f"{float(v):.6f}")
            lines.append(f"| {i_fact} | " + " | ".join(cells) + " |")
        lines.append("")
        return lines

    lines = [
        "# Lexical-control alternative hypothesis",
        "",
        f"- status: `{payload.get('status')}`",
        f"- model_id: `{payload.get('model_id')}`",
        f"- extraction_variant: `{payload.get('extraction_variant')}`",
        f"- reference: `{payload.get('reference_corpus_id')}`",
        f"- AI-Engram: `{payload.get('ai_engram_version')}`",
        f"- alpha_policy: {payload.get('alpha_policy')}",
        f"- fixed alphas: `{alphas}`",
        "",
        "Secondary analysis: same models/alphas/metrics; only extraction/reference "
        "text is lexical_control. Does **not** replace the primary explicit Delta-M test.",
        "",
    ]
    lines.extend(_fmt_mat(base_m, "Base lexical-control matrix (`M_base_lexical`)"))
    lines.extend(_fmt_mat(ft_m, "FT lexical-control matrix (`M_FT_lexical`)"))
    lines.extend(_fmt_mat(delta, "Lexical Delta-M (`M_FT_lexical − M_base_lexical`)"))

    lex = comp.get("lexical") or {}
    exp = comp.get("explicit_primary")
    cmp_ = comp.get("comparison") or {}
    lines.extend(
        [
            "## Descriptive comparison to primary explicit Delta-M",
            "",
            f"- lexical mean in-domain ΔM: `{lex.get('mean_in_domain_delta')}`",
            f"- lexical mean control ΔM: `{lex.get('mean_control_delta')}`",
            f"- lexical contrast T: `{lex.get('contrast_T')}`",
            "",
        ]
    )
    if exp:
        lines.extend(
            [
                f"- explicit mean in-domain ΔM: `{exp.get('mean_in_domain_delta')}`",
                f"- explicit mean control ΔM: `{exp.get('mean_control_delta')}`",
                f"- explicit contrast T: `{exp.get('contrast_T')}`",
                f"- same-sign tennis-vs-control pattern: "
                f"`{cmp_.get('tennis_vs_control_direction_same_sign')}`",
                f"- T_lexical − T_explicit: "
                f"`{cmp_.get('T_difference_lexical_minus_explicit')}`",
                "",
            ]
        )
    else:
        lines.append(
            f"- primary explicit comparison: `{cmp_.get('status')}` — "
            f"{cmp_.get('message', '')}"
        )
        lines.append("")

    lines.extend(
        [
            "No automatic significance claim. Research write-up should interpret "
            "whether the tennis-vs-control pattern remains and how magnitude shifts.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"lexical_control_json": json_path, "lexical_control_md": md_path}


def result_schema() -> dict[str, Any]:
    return {
        "files": {
            "results/raw/alternative_hypotheses/lexical_control.json": (
                "machine-readable lexical matrices + Delta-M + comparison"
            ),
            "results/tables/lexical_control.md": "human-readable summary",
        },
        "extraction_variant": LEXICAL_VARIANT,
        "reference_corpus_id": EXPECTED_LEXICAL_REF_ID,
        "alpha_policy": "BASE explicit-calibrated alphas; no lexical recalibration",
        "Delta_M_lexical": "M_FT_lexical - M_base_lexical",
        "fact_order": list(FACT_IDS),
        "comparison": "descriptive vs primary explicit Delta-M (not a second exact test)",
        "secondary": True,
    }


def validate_lexical_control_without_model(
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Config/data dry-validate — no Engram, weights, FT, or fabricated matrices."""
    require_ai_engram_version()
    isolation = assert_lexical_isolation(config_path)

    alphas_path = DEFAULT_ALPHAS_PATH
    alphas_status = "present" if alphas_path.is_file() else "missing"
    alpha_probe: Any = None
    if alphas_path.is_file():
        try:
            specs = load_calibrated_base_alphas(alphas_path)
            rows = eligible_row_facts(specs)
            alpha_probe = {
                "selective_rows": rows,
                "alphas": {
                    k: {"status": v.status, "alpha": v.alpha}
                    for k, v in specs.items()
                },
            }
        except BaseMatrixError as exc:
            alpha_probe = {"error": str(exc)}

    # Metric self-check (reuse primary metric)
    from .metrics import cosine_distance_vectors, cosine_similarity_vectors

    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([1.0, 0.0, 0.0])
    c = torch.tensor([0.0, 1.0, 0.0])
    assert abs(cosine_distance_vectors(a, b) - 0.0) < 1e-6
    assert abs(cosine_similarity_vectors(a, c) - 0.0) < 1e-6

    weights = find_local_snapshot(EXPECTED_MODEL_ID) is not None
    from .ft_matrix import primary_ft_adapter_available

    adapter_ok, adapter_path, adapter_reason = primary_ft_adapter_available(config_path)
    explicit_delta, explicit_meta = _try_load_primary_explicit_delta()

    pending: list[str] = []
    if not weights:
        pending.append("base_model_weights_unavailable")
    if not adapter_ok:
        pending.append("primary_ft_checkpoint_unavailable")
    if alphas_status == "missing":
        pending.append("base_alphas_missing")

    return {
        "status": "pending_prerequisites" if pending else "ready_when_executed",
        "experiment": "lexical_control_alternative_hypothesis",
        "secondary": True,
        "model_id": EXPECTED_MODEL_ID,
        "extraction_variant": LEXICAL_VARIANT,
        "reference_corpus_id": EXPECTED_LEXICAL_REF_ID,
        "fact_order": list(FACT_IDS),
        "alpha_policy": (
            "Reuse BASE explicit-calibrated alphas; do not recalibrate on lexical_control."
        ),
        "calibrated_alphas_file": str(alphas_path),
        "calibrated_alphas_status": alphas_status,
        "alpha_probe": alpha_probe,
        "isolation": isolation,
        "metric_self_check_ok": True,
        "base_weights_available_locally": weights,
        "primary_ft_adapter_available": adapter_ok,
        "primary_ft_adapter_path": str(adapter_path),
        "primary_ft_adapter_reason": adapter_reason,
        "primary_explicit_delta_available": explicit_delta is not None,
        "primary_explicit_delta_meta": explicit_meta,
        "result_schema": result_schema(),
        "pending_reasons": pending,
        "experiment_executed": False,
        "engram_executed": False,
        "matrix_fabricated": False,
        "software": _software(),
    }


def write_pending_status(report: dict[str, Any] | None = None) -> Path:
    path = DEFAULT_STATUS
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "status": "pending_weights_checkpoint_and_or_base_alphas",
        "experiment": "lexical_control_alternative_hypothesis",
        "secondary": True,
        "message": (
            "Lexical-control pipeline is implemented. Execution awaits local "
            "Qwen3-1.7B weights, primary LoRA checkpoint, and base_alphas.yaml. "
            "No lexical matrices or Delta-M values have been fabricated."
        ),
        "extraction_variant": LEXICAL_VARIANT,
        "reference_corpus_id": EXPECTED_LEXICAL_REF_ID,
        "alpha_policy": (
            "BASE explicit-calibrated alphas reused; no lexical recalibration"
        ),
        "schema": result_schema(),
        "experiment_executed": False,
    }
    if report is not None:
        payload["last_dry_validate"] = {
            k: report[k]
            for k in (
                "pending_reasons",
                "calibrated_alphas_status",
                "base_weights_available_locally",
                "primary_ft_adapter_available",
                "primary_explicit_delta_available",
                "experiment_executed",
            )
            if k in report
        }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path
