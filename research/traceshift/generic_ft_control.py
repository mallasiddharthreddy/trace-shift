"""Generic fine-tuning drift control (alternative hypothesis A).

Uses the already-trained Curie/Armstrong LoRA adapter. Does NOT retrain.
Does NOT recalibrate alphas. Does NOT recompute primary M_base / M_FT / Delta-M.
Does NOT run a second exact permutation test.

Scientific measurement matches the primary explicit FT matrix procedure:
same facts, explicit extraction, BASE alphas, AI-Engram 0.9.0, cosine distance.
Engrams are freshly extracted from the control-fine-tuned model.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from .base_matrix import (
    DEFAULT_ALPHAS_PATH,
    FACT_IDS,
    _fact_labels,
    load_calibrated_base_alphas,
)
from .delta_analysis import (
    DEFAULT_BASE_MATRIX_PATH,
    TENNIS_WITHIN_PAIRS,
    UNRELATED_WITHIN_PAIRS,
    compute_delta_m,
    load_matrix_json,
    primary_within_contrast,
    _distance_matrix,
)
from .engram import EXPECTED_ENGRAM_VERSION, EXPECTED_MODEL_ID
from .ft_matrix import (
    FinetunedCheckpointUnavailableError,
    FtMatrixError,
    FtMatrixResult,
    compute_ft_matrix,
    control_ft_adapter_available,
    load_control_finetuned_model,
)
from .model_loader import LoadedModel, ModelWeightsUnavailableError
from .paths import research_root

CONTROL_MODEL_STATE = "control_finetuned"
CONTROL_MODEL_STAGE = "M_control_FT"

DEFAULT_CONTROL_MATRIX_JSON = (
    research_root()
    / "results"
    / "raw"
    / "alternative_hypotheses"
    / "generic_ft_control_matrix.json"
)
DEFAULT_CONTROL_DELTA_JSON = (
    research_root()
    / "results"
    / "raw"
    / "alternative_hypotheses"
    / "generic_ft_control_delta_analysis.json"
)
DEFAULT_CONTROL_MATRIX_MD = (
    research_root() / "results" / "tables" / "generic_ft_control_matrix.md"
)
DEFAULT_CONTROL_DELTA_MD = (
    research_root() / "results" / "tables" / "generic_ft_control_delta_analysis.md"
)
DEFAULT_PRIMARY_DELTA_JSON = (
    research_root() / "results" / "raw" / "delta_analysis" / "delta_analysis.json"
)
DEFAULT_CONTROL_METADATA = (
    research_root()
    / "results"
    / "raw"
    / "finetuning"
    / "qwen3-1.7b-lora-control"
    / "training_metadata.json"
)
DEFAULT_CONTROL_VALIDITY = (
    research_root()
    / "results"
    / "raw"
    / "finetuning"
    / "qwen3-1.7b-lora-control"
    / "execution_validity.json"
)


class GenericFtControlError(RuntimeError):
    """Generic FT control analysis failure."""


def verify_control_training_artifacts(
    *,
    metadata_path: Path | None = None,
    validity_path: Path | None = None,
) -> dict[str, Any]:
    """Confirm the existing control adapter training record is valid (no retrain)."""
    metadata_path = metadata_path or DEFAULT_CONTROL_METADATA
    validity_path = validity_path or DEFAULT_CONTROL_VALIDITY
    errors: list[str] = []

    if not metadata_path.is_file():
        raise GenericFtControlError(f"Missing control training metadata: {metadata_path}")
    if not validity_path.is_file():
        raise GenericFtControlError(f"Missing control execution_validity: {validity_path}")

    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    validity = json.loads(validity_path.read_text(encoding="utf-8"))

    n_examples = meta.get("n_examples")
    epochs = (meta.get("hyperparameters") or {}).get("num_epochs")
    if n_examples != 40:
        errors.append(f"n_examples expected 40, got {n_examples!r}")
    if epochs != 3:
        errors.append(f"num_epochs expected 3, got {epochs!r}")
    if not validity.get("ok"):
        errors.append(f"execution_validity.ok is not true: {validity}")
    if validity.get("global_step") != 30:
        errors.append(f"global_step expected 30, got {validity.get('global_step')!r}")

    adapter = Path(meta.get("final_adapter_path") or "")
    if not (adapter / "adapter_config.json").is_file():
        errors.append(f"adapter_config.json missing under {adapter}")
    if not (adapter / "adapter_model.safetensors").is_file():
        errors.append(f"adapter_model.safetensors missing under {adapter}")

    # Phase-05 validate_ft historically checked nonzero LoRA norms in-memory but
    # did not persist lora_norm_max into execution_validity.json. If missing,
    # compute deterministically from the already-saved adapter weights (no retrain)
    # and persist with provenance. Requirement (must be > 0) is unchanged.
    lora_norm = validity.get("lora_norm_max")
    lora_norm_provenance = validity.get("lora_norm_max_provenance")
    if (not isinstance(lora_norm, (int, float)) or float(lora_norm) <= 0) and (
        adapter / "adapter_model.safetensors"
    ).is_file():
        try:
            from safetensors.torch import load_file

            tensors = load_file(str(adapter / "adapter_model.safetensors"))
            norms = [
                float(t.detach().float().norm().item())
                for name, t in tensors.items()
                if "lora_" in name
            ]
            if norms and max(norms) > 0:
                lora_norm = float(max(norms))
                lora_norm_provenance = {
                    "source": "final_adapter/adapter_model.safetensors",
                    "method": "max_frobenius_norm_over_lora_named_tensors",
                    "n_lora_tensors": len(norms),
                    "n_nonzero_lora_tensors": sum(1 for n in norms if n > 0),
                    "note": (
                        "Backfilled because Phase-05 execution_validity omitted "
                        "lora_norm_max after an in-memory nonzero-norm check."
                    ),
                }
                validity = dict(validity)
                validity["lora_norm_max"] = lora_norm
                validity["lora_norm_max_provenance"] = lora_norm_provenance
                validity_path.write_text(
                    json.dumps(validity, indent=2) + "\n", encoding="utf-8"
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(
                f"failed to backfill lora_norm_max from adapter safetensors: "
                f"{type(exc).__name__}: {exc}"
            )

    if not isinstance(lora_norm, (int, float)) or float(lora_norm) <= 0:
        errors.append(f"lora_norm_max must be > 0, got {lora_norm!r}")

    if errors:
        raise GenericFtControlError(
            "Control training validity checks failed:\n- " + "\n- ".join(errors)
        )

    return {
        "n_examples": n_examples,
        "num_epochs": epochs,
        "global_step": validity.get("global_step"),
        "lora_norm_max": lora_norm,
        "lora_norm_max_provenance": lora_norm_provenance,
        "execution_validity_ok": True,
        "final_adapter_path": str(adapter),
        "dataset_id": meta.get("dataset_id"),
        "unrelated_finetuning_control": bool(meta.get("unrelated_finetuning_control")),
    }


def verify_control_adapter_active(loaded: LoadedModel) -> dict[str, Any]:
    """Confirm PEFT adapter is loaded, active, and executing on CUDA."""
    model = loaded.model
    device = loaded.device
    checks: dict[str, Any] = {
        "device": str(device),
        "dtype": str(loaded.dtype),
        "source_path": loaded.source_path,
        "is_peft_model": type(model).__name__,
    }

    if device.type != "cuda":
        raise GenericFtControlError(
            f"Control model must execute on CUDA L4; got device={device}"
        )
    if not torch.cuda.is_available():
        raise GenericFtControlError("CUDA not available for control matrix")

    # PEFT active adapters
    active = None
    if hasattr(model, "active_adapters"):
        active = model.active_adapters
    elif hasattr(model, "active_adapter"):
        active = model.active_adapter
    checks["active_adapters"] = active
    if not active:
        raise GenericFtControlError("No active PEFT adapter on control model")

    # Nonzero LoRA parameter presence (eval mode may freeze requires_grad)
    lora_all = [(n, p) for n, p in model.named_parameters() if "lora_" in n.lower()]
    if not lora_all:
        raise GenericFtControlError("No LoRA parameters found on loaded control model")
    norms = [float(p.detach().float().norm().item()) for _, p in lora_all]
    max_norm = max(norms) if norms else 0.0
    checks["n_lora_tensors"] = len(lora_all)
    checks["lora_param_norm_max"] = max_norm
    if max_norm <= 0:
        raise GenericFtControlError("Loaded control LoRA parameter norms are all zero")

    # Confirm a parameter lives on CUDA
    sample = next(model.parameters())
    checks["sample_param_device"] = str(sample.device)
    if sample.device.type != "cuda":
        raise GenericFtControlError(
            f"Model parameters not on CUDA (got {sample.device})"
        )

    return checks


def _fmt_mat(matrix: list[list[float | None]], title: str) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "Rows = intervened fact *i*; columns = measured fact *j*. `—` = diagonal.",
        "",
        "| i \\ j | " + " | ".join(FACT_IDS) + " |",
        "|-------|" + "|".join(["------"] * len(FACT_IDS)) + "|",
    ]
    for i, i_fact in enumerate(FACT_IDS):
        cells_s = []
        for j in range(len(FACT_IDS)):
            v = matrix[i][j]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cells_s.append("—")
            else:
                cells_s.append(f"{float(v):.6f}")
        lines.append(f"| {i_fact} | " + " | ".join(cells_s) + " |")
    lines.append("")
    return lines


def validate_control_matrix_result(
    result: FtMatrixResult,
    *,
    alphas_path: Path | str | None = None,
) -> list[str]:
    """Hard validity checks; raises on failure.

    Alphas must match ``base_alphas.yaml`` for all six selective facts with
    non-null alphas present in ``result.row_alphas`` (no hardcoded alpha pins).
    """
    errors: list[str] = []
    if result.model_state != CONTROL_MODEL_STATE:
        errors.append(f"model_state expected {CONTROL_MODEL_STATE!r}, got {result.model_state!r}")
    if result.extraction_variant != "explicit":
        errors.append(f"extraction_variant must be explicit, got {result.extraction_variant!r}")
    if result.ai_engram_version != EXPECTED_ENGRAM_VERSION:
        errors.append(
            f"ai_engram_version expected {EXPECTED_ENGRAM_VERSION}, "
            f"got {result.ai_engram_version}"
        )
    if "lora-control" not in result.source_checkpoint_path.replace("\\", "/"):
        errors.append(
            f"source_checkpoint_path does not look like control adapter: "
            f"{result.source_checkpoint_path}"
        )

    try:
        alpha_specs = load_calibrated_base_alphas(alphas_path or DEFAULT_ALPHAS_PATH)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"cannot load base_alphas.yaml: {exc}")
        alpha_specs = {}

    expected_alphas: dict[str, float] = {}
    for fid in FACT_IDS:
        spec = alpha_specs.get(fid)
        if spec is None:
            errors.append(f"{fid}: missing from base_alphas.yaml")
            continue
        if spec.status != "selective":
            errors.append(
                f"{fid}: expected status='selective' in base_alphas.yaml, "
                f"got {spec.status!r}"
            )
            continue
        if spec.alpha is None:
            errors.append(f"{fid}: selective but alpha is null in base_alphas.yaml")
            continue
        expected_alphas[fid] = float(spec.alpha)

    if set(expected_alphas) != set(FACT_IDS):
        errors.append(
            f"Need selective non-null alphas for all 6 facts in base_alphas.yaml; "
            f"got {sorted(expected_alphas)}"
        )
    elif set(result.row_alphas) != set(FACT_IDS):
        errors.append(
            f"row_alphas must cover all 6 facts; got {sorted(result.row_alphas)}"
        )
    else:
        for fid in FACT_IDS:
            got = result.row_alphas.get(fid)
            want = expected_alphas[fid]
            if got is None:
                errors.append(f"row_alphas[{fid}] is null; expected {want}")
            elif abs(float(got) - want) > 1e-12:
                errors.append(
                    f"row_alphas[{fid}]={got!r} != base_alphas.yaml {want!r}"
                )

    if set(result.fact_ids) != set(FACT_IDS) or result.fact_ids != list(FACT_IDS):
        errors.append(f"fact_ids order mismatch: {result.fact_ids}")

    n = len(FACT_IDS)
    if len(result.matrix) != n:
        errors.append(f"matrix must be {n}×{n}, got row count {len(result.matrix)}")
    for i in range(n):
        if i >= len(result.matrix) or len(result.matrix[i]) != n:
            errors.append(f"matrix row {i} not length {n}")
            continue
        for j in range(n):
            v = result.matrix[i][j]
            if i == j:
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    errors.append(f"diagonal [{i},{j}] must be null, got {v!r}")
            else:
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    errors.append(f"off-diagonal [{FACT_IDS[i]}→{FACT_IDS[j]}] missing")
                elif isinstance(v, float) and (math.isinf(v)):
                    errors.append(f"off-diagonal [{FACT_IDS[i]}→{FACT_IDS[j]}] is Inf")
                elif not isinstance(v, (int, float)):
                    errors.append(f"off-diagonal [{FACT_IDS[i]}→{FACT_IDS[j]}] not numeric")

    device = (result.runtime or {}).get("device")
    if device not in ("cuda", "cuda:0") and not str(device).startswith("cuda"):
        errors.append(f"runtime.device must be cuda, got {device!r}")

    if errors:
        raise GenericFtControlError(
            "Control matrix validity failed:\n- " + "\n- ".join(errors)
        )
    return errors


def analyze_control_delta(
    control_matrix: FtMatrixResult,
    *,
    base_path: Path | None = None,
    primary_delta_path: Path | None = None,
) -> dict[str, Any]:
    """Descriptive Delta-M_control vs saved M_base; compare T_generic_within to T_tennis.

    Uses TENNIS_WITHIN vs UNRELATED_WITHIN (same primary contrast as Delta-M).
    Reads primary ``observed_contrast`` dynamically — does not pin a fixed T value.
    No permutation test / no new p-value.
    """
    base_path = base_path or DEFAULT_BASE_MATRIX_PATH
    primary_delta_path = primary_delta_path or DEFAULT_PRIMARY_DELTA_JSON

    base = load_matrix_json(base_path)
    if base.get("extraction_variant") != "explicit":
        raise GenericFtControlError("M_base must be explicit extraction")
    if base.get("ai_engram_version") != EXPECTED_ENGRAM_VERSION:
        raise GenericFtControlError("M_base AI-Engram version mismatch")
    if list(base.get("fact_ids") or []) != list(FACT_IDS):
        raise GenericFtControlError(
            f"M_base fact_ids must be {list(FACT_IDS)}, got {base.get('fact_ids')}"
        )

    base_mat = _distance_matrix(base)
    ft_mat = control_matrix.matrix
    if len(ft_mat) != len(FACT_IDS) or any(len(r) != len(FACT_IDS) for r in ft_mat):
        raise GenericFtControlError(
            f"Control matrix must be {len(FACT_IDS)}×{len(FACT_IDS)}"
        )
    delta = compute_delta_m(base_mat, ft_mat, FACT_IDS)
    m_ten, m_unrel, t_generic = primary_within_contrast(delta, FACT_IDS)

    if not primary_delta_path.is_file():
        raise GenericFtControlError(
            f"Primary delta_analysis.json missing at {primary_delta_path}"
        )
    primary = json.loads(primary_delta_path.read_text(encoding="utf-8"))
    t_tennis = float(primary["observed_contrast"])
    # Prefer explicit T_within key when present (6-fact rewrite).
    if "T_within" in primary and primary["T_within"] is not None:
        t_tennis = float(primary["T_within"])

    cell_values = {
        f"{a}->{b}": float(delta[list(FACT_IDS).index(a)][list(FACT_IDS).index(b)])
        for a, b in list(TENNIS_WITHIN_PAIRS) + list(UNRELATED_WITHIN_PAIRS)
    }

    return {
        "status": "complete",
        "experiment": "generic_fine_tuning_drift_control_alternative_hypothesis_A",
        "design": "balanced_6fact_3domain",
        "model_id": EXPECTED_MODEL_ID,
        "ai_engram_version": EXPECTED_ENGRAM_VERSION,
        "extraction_variant": "explicit",
        "base_matrix_path": str(base_path),
        "control_ft_matrix_model_state": control_matrix.model_state,
        "control_source_checkpoint": control_matrix.source_checkpoint_path,
        "alpha_source": control_matrix.alpha_source,
        "row_alphas": control_matrix.row_alphas,
        "matrix_shape": [len(FACT_IDS), len(FACT_IDS)],
        "definition": {
            "Delta_M_control[i,j]": (
                "M_control_FT[i,j] - M_base[i,j] (off-diagonal; diagonal null)"
            ),
            "T_generic_within": (
                "mean(Delta_M_control[TENNIS_WITHIN]) - "
                "mean(Delta_M_control[UNRELATED_WITHIN]); "
                "same formula as primary T_within / T_obs"
            ),
            "T_control": "alias of T_generic_within",
            "role": (
                "DESCRIPTIVE alternative-hypothesis comparison only. "
                "No second exact permutation test. Primary C(6,2)=15 cell test unchanged."
            ),
        },
        "tennis_within_pairs": [list(p) for p in TENNIS_WITHIN_PAIRS],
        "unrelated_within_pairs": [list(p) for p in UNRELATED_WITHIN_PAIRS],
        "in_domain_pair_definition": [list(p) for p in TENNIS_WITHIN_PAIRS],
        "control_pair_definition": [list(p) for p in UNRELATED_WITHIN_PAIRS],
        "delta_m_control_matrix": delta,
        "primary_cells": cell_values,
        "mean_tennis_within_delta_m_control": m_ten,
        "mean_unrelated_within_delta_m_control": m_unrel,
        "mean_in_domain_delta_m_control": m_ten,
        "mean_control_delta_m_control": m_unrel,
        "T_generic_within": t_generic,
        "T_control": t_generic,
        "T_tennis": t_tennis,
        "T_tennis_minus_T_generic": float(t_tennis - t_generic),
        "T_tennis_minus_T_control": float(t_tennis - t_generic),
        "comparison": {
            "T_tennis": t_tennis,
            "T_generic_within": t_generic,
            "T_control": t_generic,
            "T_tennis_minus_T_generic": float(t_tennis - t_generic),
            "T_tennis_minus_T_control": float(t_tennis - t_generic),
            "mean_tennis_within_tennis_ft": primary.get("mean_tennis_within_delta")
            or primary.get("mean_in_domain_delta_m"),
            "mean_unrelated_within_tennis_ft": primary.get(
                "mean_unrelated_within_delta"
            )
            or primary.get("mean_control_delta_m"),
            "mean_in_domain_tennis": primary.get("mean_tennis_within_delta")
            or primary.get("mean_in_domain_delta_m"),
            "mean_control_tennis": primary.get("mean_unrelated_within_delta")
            or primary.get("mean_control_delta_m"),
            "mean_tennis_within_control_ft": m_ten,
            "mean_unrelated_within_control_ft": m_unrel,
            "mean_in_domain_control_ft": m_ten,
            "mean_control_control_ft": m_unrel,
            "same_sign_T": (t_tennis > 0 and t_generic > 0)
            or (t_tennis < 0 and t_generic < 0)
            or (t_tennis == 0 and t_generic == 0),
            "abs_T_control_vs_abs_T_tennis": {
                "abs_T_generic": abs(t_generic),
                "abs_T_control": abs(t_generic),
                "abs_T_tennis": abs(t_tennis),
                "abs_T_control_smaller": abs(t_generic) < abs(t_tennis),
            },
        },
        "inferential": {
            "permutation_test_performed": False,
            "p_value": None,
            "note": (
                "Primary exact one-sided p remains from the tennis-FT C(6,2)=15 "
                "within-cell assignment test only. This control contrast is descriptive."
            ),
        },
        "claim_boundaries": {
            "supports": [
                "Quantitative descriptive comparison of structural Delta-M contrast "
                "after unrelated Curie/Armstrong FT vs after tennis FT, under the "
                "same 6-fact / 3-domain measurement procedure.",
            ],
            "does_not_support": [
                "Ruling out generic fine-tuning drift as a formal causal claim.",
                "A second significance test or revised primary p-value.",
                "Population-level generalization beyond the six frozen facts.",
                "That the tennis effect is definitively domain-specific.",
            ],
        },
        "runtime": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "control_matrix_runtime": control_matrix.runtime,
        },
        "notes": [
            "Uses already-trained control adapter; no retraining.",
            "Uses existing primary M_base; M_base not recomputed.",
            "Fresh Engrams from control FT model; not reused from base or tennis FT.",
            "BASE alphas unchanged; not recalibrated on control model.",
            "No lexical-control re-run.",
            "No second permutation test.",
            "Balanced 6-fact design: T_generic_within uses TENNIS_WITHIN vs "
            "UNRELATED_WITHIN (swimming+acting).",
        ],
    }


def write_generic_ft_control_outputs(
    matrix_result: FtMatrixResult,
    delta_payload: dict[str, Any],
    *,
    validity: dict[str, Any],
    adapter_checks: dict[str, Any],
    training_checks: dict[str, Any],
) -> dict[str, Path]:
    raw_dir = research_root() / "results" / "raw" / "alternative_hypotheses"
    tables_dir = research_root() / "results" / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    matrix_json = raw_dir / "generic_ft_control_matrix.json"
    delta_json = raw_dir / "generic_ft_control_delta_analysis.json"
    matrix_md = tables_dir / "generic_ft_control_matrix.md"
    delta_md = tables_dir / "generic_ft_control_delta_analysis.md"

    matrix_payload = matrix_result.to_dict()
    matrix_payload["fact_labels"] = _fact_labels()
    matrix_payload["experiment"] = (
        "generic_fine_tuning_drift_control_alternative_hypothesis_A"
    )
    matrix_payload["validity_checks"] = {
        "training": training_checks,
        "adapter_active": adapter_checks,
        "matrix": validity,
    }
    matrix_json.write_text(json.dumps(matrix_payload, indent=2), encoding="utf-8")
    delta_json.write_text(json.dumps(delta_payload, indent=2), encoding="utf-8")

    labels = _fact_labels()
    m_lines = [
        "# Generic FT control intervention–response matrix (`M_control_FT`)",
        "",
        "Alternative hypothesis A: unrelated Curie/Armstrong fine-tuning drift.",
        "Same measurement procedure as primary explicit FT matrix; different model state.",
        "",
        f"- model_id: `{matrix_result.model_id}`",
        f"- model_state: `{matrix_result.model_state}`",
        f"- AI-Engram: `{matrix_result.ai_engram_version}`",
        f"- extraction_variant: `{matrix_result.extraction_variant}`",
        f"- primary_metric: `{matrix_result.primary_metric}`",
        f"- diagonal: {matrix_result.diagonal_convention}",
        f"- source_checkpoint: `{matrix_result.source_checkpoint_path}`",
        f"- alpha_source: `{matrix_result.alpha_source}` (BASE-calibrated; not recalibrated)",
        f"- device: `{(matrix_result.runtime or {}).get('device')}`",
        "",
        "## Fact labels",
        "",
        "| fact_id | label |",
        "|---------|-------|",
    ]
    for fid in matrix_result.fact_ids:
        m_lines.append(f"| {fid} | {labels.get(fid, '')} |")
    m_lines.extend(
        [
            "",
            "## Intervention alphas (rows; BASE-calibrated)",
            "",
            "| row fact | status | alpha |",
            "|----------|--------|-------|",
        ]
    )
    for fid in matrix_result.fact_ids:
        m_lines.append(
            f"| {fid} | {matrix_result.row_eligibility.get(fid, '')} | "
            f"{matrix_result.row_alphas.get(fid)} |"
        )
    m_lines.append("")
    m_lines.extend(_fmt_mat(matrix_result.matrix, "Matrix values (cosine distance)"))
    m_lines.extend(
        [
            "No significance claims. Does not replace primary tennis `M_FT`.",
            "",
        ]
    )
    matrix_md.write_text("\n".join(m_lines), encoding="utf-8")

    d_lines = [
        "# Generic FT control Delta-M (descriptive)",
        "",
        "ΔM_control = M_control_FT − M_base (existing primary base matrix).",
        "T_control is **descriptive only** — no second permutation test.",
        "",
    ]
    d_lines.extend(
        _fmt_mat(delta_payload["delta_m_control_matrix"], "Delta-M_control matrix")
    )
    cmp_ = delta_payload["comparison"]
    d_lines.extend(
        [
            "## Primary cell group means",
            "",
            "Contrast: `T_generic_within = mean(tennis within) − "
            "mean(unrelated within)` on the 6×6 Delta-M_control matrix.",
            "",
            f"- mean tennis-within ΔM_control: "
            f"`{delta_payload.get('mean_tennis_within_delta_m_control')}`",
            f"- mean unrelated-within ΔM_control: "
            f"`{delta_payload.get('mean_unrelated_within_delta_m_control')}`",
            f"- **T_generic_within**: `{delta_payload.get('T_generic_within')}`",
            "",
            "## Comparison to tennis fine-tuning (descriptive)",
            "",
            f"- T_tennis (primary observed_contrast / T_within): `{cmp_['T_tennis']}`",
            f"- T_generic_within: `{cmp_['T_generic_within']}`",
            f"- T_tennis − T_generic: `{cmp_['T_tennis_minus_T_generic']}`",
            f"- same-sign T: `{cmp_['same_sign_T']}`",
            f"- |T_generic| < |T_tennis|: "
            f"`{cmp_['abs_T_control_vs_abs_T_tennis']['abs_T_control_smaller']}`",
            "",
            "## Claim boundaries",
            "",
            "Supports: quantitative descriptive structural comparison under identical "
            "6-fact / 3-domain measurement procedure.",
            "",
            "Does **not** support: ruling out generic FT drift as a causal claim; "
            "revising the primary C(6,2)=15 p-value; population generalization; "
            "definitively tennis-specific causation.",
            "",
            f"Primary T_tennis read dynamically from delta_analysis.json: "
            f"`{cmp_['T_tennis']}`.",
            "",
        ]
    )
    delta_md.write_text("\n".join(d_lines), encoding="utf-8")

    return {
        "generic_ft_control_matrix_json": matrix_json,
        "generic_ft_control_delta_json": delta_json,
        "generic_ft_control_matrix_md": matrix_md,
        "generic_ft_control_delta_md": delta_md,
    }


def run_generic_ft_control_experiment(
    *,
    config_path: Path | str | None = None,
    alphas_path: Path | str | None = None,
    device: str | None = None,
    allow_override: bool = False,
    override_statuses: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Full control matrix + descriptive Delta-M using existing control adapter."""
    training_checks = verify_control_training_artifacts()
    available, adapter_path, reason = control_ft_adapter_available(config_path)
    if not available:
        raise FinetunedCheckpointUnavailableError(reason)

    loaded = load_control_finetuned_model(config_path, device=device)
    adapter_checks = verify_control_adapter_active(loaded)

    try:
        matrix_result = compute_ft_matrix(
            loaded,
            alphas_path=alphas_path or DEFAULT_ALPHAS_PATH,
            config_path=config_path,
            allow_override=allow_override,
            override_statuses=override_statuses,
            model_state=CONTROL_MODEL_STATE,
            model_stage=CONTROL_MODEL_STAGE,
        )
    finally:
        # Aggressively release model state after matrix compute.
        del loaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    validate_control_matrix_result(
        matrix_result, alphas_path=alphas_path or DEFAULT_ALPHAS_PATH
    )
    delta_payload = analyze_control_delta(matrix_result)
    paths = write_generic_ft_control_outputs(
        matrix_result,
        delta_payload,
        validity={"ok": True},
        adapter_checks=adapter_checks,
        training_checks=training_checks,
    )

    return {
        "matrix": matrix_result.to_dict(),
        "delta": delta_payload,
        "paths": {k: str(v) for k, v in paths.items()},
        "training_checks": training_checks,
        "adapter_checks": adapter_checks,
        "adapter_path": str(adapter_path),
    }


def validate_generic_ft_control_without_model(
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Dry validate wiring without loading weights / extracting Engrams."""
    training_ok = False
    training: dict[str, Any] | None = None
    try:
        training = verify_control_training_artifacts()
        training_ok = True
    except GenericFtControlError as exc:
        training = {"error": str(exc)}

    available, path, reason = control_ft_adapter_available(config_path)
    primary_delta_ok = DEFAULT_PRIMARY_DELTA_JSON.is_file()
    base_ok = DEFAULT_BASE_MATRIX_PATH.is_file()
    return {
        "experiment": "generic_fine_tuning_drift_control_alternative_hypothesis_A",
        "control_adapter_available": available,
        "control_adapter_path": str(path),
        "control_adapter_reason": reason,
        "training_artifacts_ok": training_ok,
        "training_checks": training,
        "base_matrix_present": base_ok,
        "primary_delta_present": primary_delta_ok,
        "matrix_executed": False,
        "retraining": False,
        "permutation_test": False,
    }


def write_pending_status(report: dict[str, Any] | None = None) -> Path:
    path = (
        research_root()
        / "results"
        / "raw"
        / "alternative_hypotheses"
        / "generic_ft_control_STATUS.yaml"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "status": "pending_or_incomplete",
        "experiment": "generic_fine_tuning_drift_control_alternative_hypothesis_A",
        "message": (
            "Generic FT control structural analysis not yet complete. "
            "Requires existing control adapter + M_base. No values fabricated."
        ),
        "retraining": False,
        "permutation_test": False,
    }
    if report is not None:
        payload["last_dry_validate"] = report
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path
