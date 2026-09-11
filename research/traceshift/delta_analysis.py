"""Primary Delta-M analysis and exact permutation test for TraceShift.

BALANCED 6-FACT design (frozen; do not reinterpret):

    Delta_M[i, j] = M_FT[i, j] - M_base[i, j]   (off-diagonal; diagonal null)

Primary WITHIN-DOMAIN cells (6 directed):
    TENNIS:   F01 → F02,  F02 → F01
    SWIMMING: F03 → F04,  F04 → F03
    ACTING:   F05 → F06,  F06 → F05

Primary contrast:
    T_within / T_obs =
        mean(tennis within-domain Delta-M)
        - mean(unrelated within-domain Delta-M)
    where tennis = F01↔F02 (2 cells), unrelated = swimming+acting (4 cells).

PRIMARY exact permutation test (cell assignment, not fact-label swap):
    Hold the six within-domain directed Delta-M cell VALUES fixed.
    Enumerate every choice of 2 of these 6 CELLS as the "tennis" group: C(6,2)=15.
    For each assignment, T = mean(chosen 2) - mean(remaining 4).
    Include the observed assignment (the two tennis cells).
    One-sided exact p = (# with T >= T_obs) / 15.
    This is NOT the old C(4,2) fact-label swap of in-domain vs control facts.

SECONDARY descriptive (compute and store; not primary inference):
    CROSS_DOMAIN_PAIRS = F01/F02 → each of F03,F04,F05,F06 (8 cells)
    T_cross = mean(tennis within) - mean(cross-domain)
"""

from __future__ import annotations

import importlib.metadata
import itertools
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from .engram import EXPECTED_ENGRAM_VERSION, EXPECTED_MODEL_ID
from .paths import research_root

Pair = tuple[str, str]

# Balanced 6-fact design (frozen). Defined here so Delta-M does not depend on
# an unfinished base_matrix FACT_IDS migration; must match F01–F06 matrices.
FACT_IDS: tuple[str, ...] = ("F01", "F02", "F03", "F04", "F05", "F06")

# Domain within-domain directed cells (primary universe for the exact test).
TENNIS_WITHIN_PAIRS: tuple[Pair, ...] = (("F01", "F02"), ("F02", "F01"))
SWIMMING_WITHIN_PAIRS: tuple[Pair, ...] = (("F03", "F04"), ("F04", "F03"))
ACTING_WITHIN_PAIRS: tuple[Pair, ...] = (("F05", "F06"), ("F06", "F05"))

WITHIN_DOMAIN_PAIRS: tuple[Pair, ...] = (
    TENNIS_WITHIN_PAIRS + SWIMMING_WITHIN_PAIRS + ACTING_WITHIN_PAIRS
)
UNRELATED_WITHIN_PAIRS: tuple[Pair, ...] = (
    SWIMMING_WITHIN_PAIRS + ACTING_WITHIN_PAIRS
)

# Secondary descriptive only (not primary inference).
CROSS_DOMAIN_PAIRS: tuple[Pair, ...] = (
    ("F01", "F03"),
    ("F01", "F04"),
    ("F01", "F05"),
    ("F01", "F06"),
    ("F02", "F03"),
    ("F02", "F04"),
    ("F02", "F05"),
    ("F02", "F06"),
)

# Back-compat aliases used by sibling control modules (tennis = primary within).
OBSERVED_IN_DOMAIN_FACTS: tuple[str, ...] = ("F01", "F02")
OBSERVED_CONTROL_FACTS: tuple[str, ...] = ("F03", "F04", "F05", "F06")
OBSERVED_IN_DOMAIN_PAIRS: tuple[Pair, ...] = TENNIS_WITHIN_PAIRS
OBSERVED_CONTROL_PAIRS: tuple[Pair, ...] = UNRELATED_WITHIN_PAIRS

DEFAULT_BASE_MATRIX_PATH = (
    research_root() / "results" / "raw" / "base_matrix" / "base_matrix.json"
)
DEFAULT_FT_MATRIX_PATH = (
    research_root() / "results" / "raw" / "ft_matrix" / "ft_matrix.json"
)
DEFAULT_OUTPUT_DIR = research_root() / "results" / "raw" / "delta_analysis"
DEFAULT_TABLE_PATH = research_root() / "results" / "tables" / "delta_analysis.md"

EXACT_PERMUTATION_COUNT = 15  # C(6,2)


class DeltaAnalysisError(RuntimeError):
    """Incompatible or missing matrix inputs for Delta-M analysis."""


@dataclass
class DeltaAnalysisResult:
    status: str
    model_id: str
    base_matrix_path: str
    ft_matrix_path: str
    fact_ids: list[str]
    tennis_within_pairs: list[list[str]]
    unrelated_within_pairs: list[list[str]]
    cross_domain_pairs: list[list[str]]
    within_domain_pairs: list[list[str]]
    within_domain_cell_values: dict[str, float]
    cross_domain_cell_values: dict[str, float]
    delta_m: list[list[float | None]]
    mean_tennis_within_delta: float
    mean_unrelated_within_delta: float
    mean_cross_domain_delta: float
    observed_contrast: float  # T_obs / T_within
    t_cross: float
    permutation_count: int
    null_distribution: list[dict[str, Any]]
    exact_one_sided_p_value: float
    effect_size: dict[str, Any]
    software: dict[str, str]
    runtime: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    matrix_metadata: dict[str, Any] = field(default_factory=dict)

    # Legacy field names kept for callers that still expect them.
    @property
    def in_domain_fact_ids(self) -> list[str]:
        return list(OBSERVED_IN_DOMAIN_FACTS)

    @property
    def control_fact_ids(self) -> list[str]:
        return list(OBSERVED_CONTROL_FACTS)

    @property
    def in_domain_pairs(self) -> list[list[str]]:
        return self.tennis_within_pairs

    @property
    def control_pairs(self) -> list[list[str]]:
        return self.unrelated_within_pairs

    @property
    def mean_in_domain_delta(self) -> float:
        return self.mean_tennis_within_delta

    @property
    def mean_control_delta(self) -> float:
        return self.mean_unrelated_within_delta

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model_id": self.model_id,
            "base_matrix_path": self.base_matrix_path,
            "ft_matrix_path": self.ft_matrix_path,
            "design": "balanced_6fact_3domain",
            "fact_ordering": self.fact_ids,
            "tennis_within_pairs": self.tennis_within_pairs,
            "unrelated_within_pairs": self.unrelated_within_pairs,
            "cross_domain_pairs": self.cross_domain_pairs,
            "within_domain_pairs": self.within_domain_pairs,
            "within_domain_cell_values": self.within_domain_cell_values,
            "cross_domain_cell_values": self.cross_domain_cell_values,
            # Legacy keys (tennis / unrelated within-domain under 6-fact design).
            "in_domain_fact_ids": self.in_domain_fact_ids,
            "control_fact_ids": self.control_fact_ids,
            "in_domain_pair_definition": self.tennis_within_pairs,
            "control_pair_definition": self.unrelated_within_pairs,
            "definition": {
                "Delta_M[i,j]": "M_FT[i,j] - M_base[i,j] (off-diagonal; diagonal null)",
                "T_within / T_observed": (
                    "mean(tennis within-domain Delta-M) - "
                    "mean(unrelated within-domain Delta-M); "
                    "tennis = F01↔F02 (2 cells); unrelated = swimming+acting (4 cells)"
                ),
                "T_cross": (
                    "mean(tennis within) - mean(cross-domain); "
                    "cross-domain = F01/F02 → F03,F04,F05,F06 (8 cells); "
                    "descriptive only, not primary inference"
                ),
                "permutation": (
                    "Exact cell permutation: hold the 6 within-domain directed "
                    "Delta-M values fixed; enumerate all C(6,2)=15 choices of 2 "
                    "cells as the tennis group; T = mean(chosen 2) - mean(remaining 4). "
                    "NOT the old C(4,2) fact-label swap."
                ),
                "interpretation_note": (
                    "Positive T_observed means tennis within-domain interactions "
                    "changed more under FT than unrelated within-domain interactions "
                    "(intervention-response sense only; not a broader causal claim)."
                ),
            },
            "delta_m_matrix": self.delta_m,
            "mean_tennis_within_delta_m": self.mean_tennis_within_delta,
            "mean_unrelated_within_delta_m": self.mean_unrelated_within_delta,
            "mean_cross_domain_delta_m": self.mean_cross_domain_delta,
            "mean_in_domain_delta_m": self.mean_tennis_within_delta,
            "mean_control_delta_m": self.mean_unrelated_within_delta,
            "observed_contrast": self.observed_contrast,
            "T_obs": self.observed_contrast,
            "T_within": self.observed_contrast,
            "T_cross": self.t_cross,
            "permutation_count": self.permutation_count,
            "null_distribution": self.null_distribution,
            "exact_one_sided_p_value": self.exact_one_sided_p_value,
            "effect_size": self.effect_size,
            "software": self.software,
            "runtime": self.runtime,
            "matrix_metadata": self.matrix_metadata,
            "notes": self.notes,
            "significance_language": (
                "Report the exact p-value only; do not auto-claim "
                "'statistically significant' from an arbitrary threshold."
            ),
            "design_note": (
                "Exact cell-permutation test conditional on the frozen balanced "
                "6-fact / 3-domain design; not a population-level generalization claim."
            ),
        }


def _software_versions() -> dict[str, str]:
    out: dict[str, str] = {
        "python": f"{__import__('sys').version_info.major}."
        f"{__import__('sys').version_info.minor}."
        f"{__import__('sys').version_info.micro}",
    }
    for pkg in ("numpy", "torch", "pyyaml"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "not-installed"
    return out


def default_matrix_paths() -> tuple[Path, Path]:
    return DEFAULT_BASE_MATRIX_PATH, DEFAULT_FT_MATRIX_PATH


def load_matrix_json(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise DeltaAnalysisError(f"Matrix file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise DeltaAnalysisError(f"Matrix JSON must be an object: {p}")
    return data


def _require_keys(data: dict[str, Any], keys: Iterable[str], *, label: str) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise DeltaAnalysisError(f"{label} missing keys: {missing}")


def _as_fact_list(data: dict[str, Any]) -> list[str]:
    facts = data.get("fact_ids") or data.get("fact_ordering")
    if not isinstance(facts, list) or not facts:
        raise DeltaAnalysisError("Matrix JSON missing fact_ids / fact_ordering")
    return [str(x) for x in facts]


def _distance_matrix(data: dict[str, Any]) -> list[list[Any]]:
    mat = data.get("matrix_cosine_distance")
    if mat is None:
        mat = data.get("matrix")
    if not isinstance(mat, list):
        raise DeltaAnalysisError("Matrix JSON missing matrix_cosine_distance")
    return mat


def _pair_key(pair: Pair) -> str:
    return f"{pair[0]}->{pair[1]}"


def validate_matrix_pair(
    base: dict[str, Any],
    ft: dict[str, Any],
    *,
    base_path: Path | str,
    ft_path: Path | str,
) -> dict[str, Any]:
    """Confirm BASE and FT matrices are compatible for Delta-M (6×6 complete)."""
    for label, data in (("base", base), ("ft", ft)):
        _require_keys(
            data,
            (
                "model_id",
                "ai_engram_version",
                "extraction_variant",
                "diagonal_convention",
            ),
            label=label,
        )

    base_facts = _as_fact_list(base)
    ft_facts = _as_fact_list(ft)
    if base_facts != ft_facts:
        raise DeltaAnalysisError(
            f"Fact ordering mismatch: base={base_facts} ft={ft_facts}"
        )
    if base_facts != list(FACT_IDS):
        raise DeltaAnalysisError(
            f"Fact ordering must be {list(FACT_IDS)}, got {base_facts}"
        )

    if base.get("model_id") != ft.get("model_id"):
        raise DeltaAnalysisError(
            f"model_id mismatch: base={base.get('model_id')!r} "
            f"ft={ft.get('model_id')!r}"
        )
    if base.get("model_id") != EXPECTED_MODEL_ID:
        raise DeltaAnalysisError(
            f"Unexpected model_id {base.get('model_id')!r}; "
            f"expected {EXPECTED_MODEL_ID!r}"
        )

    if base.get("extraction_variant") != "explicit" or ft.get("extraction_variant") != "explicit":
        raise DeltaAnalysisError(
            "Both matrices must use extraction_variant='explicit' "
            f"(base={base.get('extraction_variant')!r}, "
            f"ft={ft.get('extraction_variant')!r})"
        )

    if base.get("ai_engram_version") != ft.get("ai_engram_version"):
        raise DeltaAnalysisError(
            f"ai_engram_version mismatch: base={base.get('ai_engram_version')!r} "
            f"ft={ft.get('ai_engram_version')!r}"
        )
    if base.get("ai_engram_version") != EXPECTED_ENGRAM_VERSION:
        raise DeltaAnalysisError(
            f"Unexpected AI-Engram version {base.get('ai_engram_version')!r}; "
            f"expected {EXPECTED_ENGRAM_VERSION!r}"
        )

    bdiag = str(base.get("diagonal_convention") or "").lower()
    fdiag = str(ft.get("diagonal_convention") or "").lower()
    if ("null" not in bdiag and "nan" not in bdiag) or (
        "null" not in fdiag and "nan" not in fdiag
    ):
        raise DeltaAnalysisError(
            "Diagonal conventions must indicate null/NaN for both matrices"
        )

    base_mat = _distance_matrix(base)
    ft_mat = _distance_matrix(ft)
    n = len(base_facts)
    if n != 6:
        raise DeltaAnalysisError(
            f"Balanced 6-fact design requires exactly 6 facts, got {n}"
        )
    if len(base_mat) != n or len(ft_mat) != n:
        raise DeltaAnalysisError("Matrix row count does not match fact ordering")
    for i in range(n):
        if len(base_mat[i]) != n or len(ft_mat[i]) != n:
            raise DeltaAnalysisError("Matrix is not square / fact-aligned")
        for j in range(n):
            if i == j:
                for label, v in (("base", base_mat[i][j]), ("ft", ft_mat[i][j])):
                    if v is not None and not (
                        isinstance(v, float) and math.isnan(v)
                    ):
                        raise DeltaAnalysisError(
                            f"{label} diagonal cell [{base_facts[i]},{base_facts[j]}] "
                            f"must be null/NaN, got {v!r}"
                        )
            else:
                for label, v in (("base", base_mat[i][j]), ("ft", ft_mat[i][j])):
                    if v is None or (
                        isinstance(v, float) and math.isnan(v)
                    ):
                        raise DeltaAnalysisError(
                            f"{label} off-diagonal cell "
                            f"[{base_facts[i]}→{base_facts[j]}] missing/non-numeric. "
                            "The frozen exact cell-permutation test requires a complete "
                            "6×6 off-diagonal ΔM, which in turn requires all six "
                            "facts to have been selective matrix rows (BASE + FT). "
                            "Do not invent cell values; complete selective alpha "
                            "calibration for F01–F06 or stop before Delta-M."
                        )
                    if not isinstance(v, (int, float)):
                        raise DeltaAnalysisError(
                            f"{label} off-diagonal cell "
                            f"[{base_facts[i]}→{base_facts[j]}] not numeric: {v!r}"
                        )

    return {
        "base_path": str(base_path),
        "ft_path": str(ft_path),
        "model_id": base["model_id"],
        "ai_engram_version": base["ai_engram_version"],
        "extraction_variant": "explicit",
        "fact_ids": base_facts,
        "design": "balanced_6fact_3domain",
        "base_diagonal_convention": base.get("diagonal_convention"),
        "ft_diagonal_convention": ft.get("diagonal_convention"),
        "base_model_stage": base.get("model_stage"),
        "ft_model_stage": ft.get("model_stage"),
        "ft_model_state": ft.get("model_state"),
        "compatible": True,
    }


def compute_delta_m(
    base_mat: Sequence[Sequence[Any]],
    ft_mat: Sequence[Sequence[Any]],
    fact_ids: Sequence[str],
) -> list[list[float | None]]:
    """Element-wise Delta-M = M_FT - M_base; diagonal stays null."""
    n = len(fact_ids)
    out: list[list[float | None]] = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                out[i][j] = None
                continue
            out[i][j] = float(ft_mat[i][j]) - float(base_mat[i][j])
    return out


def cell_value(
    delta_m: Sequence[Sequence[float | None]],
    fact_ids: Sequence[str],
    pair: Pair,
) -> float:
    i = list(fact_ids).index(pair[0])
    j = list(fact_ids).index(pair[1])
    v = delta_m[i][j]
    if v is None or (isinstance(v, float) and math.isnan(v)):
        raise DeltaAnalysisError(f"Required Delta-M cell {pair[0]}→{pair[1]} is null")
    return float(v)


def mean_over_pairs(
    delta_m: Sequence[Sequence[float | None]],
    fact_ids: Sequence[str],
    pairs: Sequence[Pair],
) -> float:
    if not pairs:
        raise DeltaAnalysisError("Empty pair list")
    vals = [cell_value(delta_m, fact_ids, p) for p in pairs]
    return float(sum(vals) / len(vals))


def within_domain_cell_values(
    delta_m: Sequence[Sequence[float | None]],
    fact_ids: Sequence[str],
    pairs: Sequence[Pair] = WITHIN_DOMAIN_PAIRS,
) -> dict[str, float]:
    return {_pair_key(p): cell_value(delta_m, fact_ids, p) for p in pairs}


def primary_within_contrast(
    delta_m: Sequence[Sequence[float | None]],
    fact_ids: Sequence[str],
    *,
    tennis_pairs: Sequence[Pair] = TENNIS_WITHIN_PAIRS,
    unrelated_pairs: Sequence[Pair] = UNRELATED_WITHIN_PAIRS,
) -> tuple[float, float, float]:
    """Return (mean_tennis_within, mean_unrelated_within, T_within)."""
    m_tennis = mean_over_pairs(delta_m, fact_ids, tennis_pairs)
    m_unrel = mean_over_pairs(delta_m, fact_ids, unrelated_pairs)
    return m_tennis, m_unrel, float(m_tennis - m_unrel)


def cross_domain_contrast(
    delta_m: Sequence[Sequence[float | None]],
    fact_ids: Sequence[str],
    *,
    tennis_pairs: Sequence[Pair] = TENNIS_WITHIN_PAIRS,
    cross_pairs: Sequence[Pair] = CROSS_DOMAIN_PAIRS,
) -> tuple[float, float, float]:
    """Secondary descriptive: (mean_tennis, mean_cross, T_cross)."""
    m_tennis = mean_over_pairs(delta_m, fact_ids, tennis_pairs)
    m_cross = mean_over_pairs(delta_m, fact_ids, cross_pairs)
    return m_tennis, m_cross, float(m_tennis - m_cross)


def pairs_for_assignment(
    in_domain: Sequence[str],
    controls: Sequence[str],
) -> tuple[list[Pair], list[Pair]]:
    """Legacy helper: both directed edges within in_domain; in→each control.

    Retained for sibling modules. Primary inference uses
    ``enumerate_within_cell_assignments`` / ``exact_permutation_test`` instead.
    """
    if len(in_domain) != 2:
        raise DeltaAnalysisError(
            f"Expected 2 in-domain facts, got in_domain={list(in_domain)}"
        )
    a, b = in_domain[0], in_domain[1]
    in_pairs: list[Pair] = [(a, b), (b, a)]
    ctrl_pairs: list[Pair] = [(a, c) for c in controls] + [(b, c) for c in controls]
    return in_pairs, ctrl_pairs


def primary_contrast(
    delta_m: Sequence[Sequence[float | None]],
    fact_ids: Sequence[str],
    in_domain_pairs: Sequence[Pair],
    control_pairs: Sequence[Pair],
) -> tuple[float, float, float]:
    """Legacy API: (mean_group_a, mean_group_b, mean_a - mean_b)."""
    m_in = mean_over_pairs(delta_m, fact_ids, in_domain_pairs)
    m_ctrl = mean_over_pairs(delta_m, fact_ids, control_pairs)
    return m_in, m_ctrl, float(m_in - m_ctrl)


def enumerate_within_cell_assignments(
    within_pairs: Sequence[Pair] = WITHIN_DOMAIN_PAIRS,
) -> list[tuple[tuple[Pair, ...], tuple[Pair, ...]]]:
    """All C(6,2) ways to choose 2 of the 6 within-domain cells as 'tennis'.

    Returns list of (chosen_tennis_cells, remaining_unrelated_cells).
    Order is deterministic: combinations in the order of ``within_pairs``.
    """
    cells = tuple(within_pairs)
    if len(cells) != 6:
        raise DeltaAnalysisError(
            f"Exact cell test expects exactly 6 within-domain cells, got {len(cells)}"
        )
    assignments: list[tuple[tuple[Pair, ...], tuple[Pair, ...]]] = []
    for chosen in itertools.combinations(range(6), 2):
        tennis = tuple(cells[i] for i in chosen)
        unrelated = tuple(cells[i] for i in range(6) if i not in chosen)
        assignments.append((tennis, unrelated))
    if len(assignments) != EXACT_PERMUTATION_COUNT:
        raise DeltaAnalysisError(
            f"Expected C(6,2)={EXACT_PERMUTATION_COUNT} assignments, "
            f"got {len(assignments)}"
        )
    return assignments


def enumerate_group_assignments(
    fact_ids: Sequence[str] = FACT_IDS,
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Deprecated: old C(4,2) fact-label assignments.

    Kept only so older callers do not crash on import; primary inference uses
    ``enumerate_within_cell_assignments``.
    """
    facts = list(fact_ids)
    if len(facts) < 2:
        raise DeltaAnalysisError(f"Need at least 2 facts, got {len(facts)}")
    assignments: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for combo in itertools.combinations(facts, 2):
        in_dom = tuple(sorted(combo, key=lambda f: facts.index(f)))
        ctrl = tuple(f for f in facts if f not in in_dom)
        assignments.append((in_dom, ctrl))
    assignments.sort(key=lambda x: (facts.index(x[0][0]), facts.index(x[0][1])))
    return assignments


def exact_permutation_test(
    delta_m: Sequence[Sequence[float | None]],
    fact_ids: Sequence[str],
    *,
    within_pairs: Sequence[Pair] = WITHIN_DOMAIN_PAIRS,
    observed_tennis_pairs: Sequence[Pair] = TENNIS_WITHIN_PAIRS,
) -> dict[str, Any]:
    """Exact one-sided cell-permutation test over all C(6,2)=15 assignments.

    Null: the six within-domain Delta-M cell values are fixed; which two cells
    are labeled 'tennis' is exchangeable. p = (# assignments with T >= T_obs) / 15,
    including the observed tennis-cell assignment (exact-test convention).
    """
    obs_tennis = tuple(observed_tennis_pairs)
    # Prefer documented ordered pairs when they match the frozen tennis cells.
    if set(obs_tennis) == set(TENNIS_WITHIN_PAIRS):
        tennis_obs = list(TENNIS_WITHIN_PAIRS)
        unrelated_obs = list(UNRELATED_WITHIN_PAIRS)
    else:
        tennis_obs = list(obs_tennis)
        unrelated_obs = [p for p in within_pairs if p not in set(obs_tennis)]

    m_tennis, m_unrel, t_obs = primary_within_contrast(
        delta_m, fact_ids, tennis_pairs=tennis_obs, unrelated_pairs=unrelated_obs
    )

    cell_vals = within_domain_cell_values(delta_m, fact_ids, within_pairs)
    obs_tennis_set = set(obs_tennis)

    null: list[dict[str, Any]] = []
    assignments = enumerate_within_cell_assignments(within_pairs)
    n_ge = 0
    for tennis_cells, unrelated_cells in assignments:
        tennis_vals = [cell_vals[_pair_key(p)] for p in tennis_cells]
        unrel_vals = [cell_vals[_pair_key(p)] for p in unrelated_cells]
        t = float(sum(tennis_vals) / len(tennis_vals) - sum(unrel_vals) / len(unrel_vals))
        is_observed = set(tennis_cells) == obs_tennis_set
        if t >= t_obs - 1e-15:
            n_ge += 1
        null.append(
            {
                "tennis_cells": [list(p) for p in tennis_cells],
                "unrelated_cells": [list(p) for p in unrelated_cells],
                # Legacy keys for older consumers of null_distribution rows.
                "in_domain_pairs": [list(p) for p in tennis_cells],
                "control_pairs": [list(p) for p in unrelated_cells],
                "contrast": t,
                "is_observed_assignment": is_observed,
            }
        )

    n = len(assignments)
    if n != EXACT_PERMUTATION_COUNT:
        raise DeltaAnalysisError(
            f"Empty or incomplete null distribution: expected "
            f"{EXACT_PERMUTATION_COUNT}, got {n}"
        )
    p_value = float(n_ge) / float(n)

    return {
        "mean_tennis_within_delta": m_tennis,
        "mean_unrelated_within_delta": m_unrel,
        "mean_in_domain_delta": m_tennis,
        "mean_control_delta": m_unrel,
        "observed_contrast": t_obs,
        "permutation_count": n,
        "n_as_extreme_or_more": n_ge,
        "exact_one_sided_p_value": p_value,
        "null_distribution": null,
        "tennis_within_pairs": [list(p) for p in tennis_obs],
        "unrelated_within_pairs": [list(p) for p in unrelated_obs],
        "in_domain_pairs": [list(p) for p in tennis_obs],
        "control_pairs": [list(p) for p in unrelated_obs],
        "within_domain_cell_values": cell_vals,
    }


def analyze_delta_m(
    *,
    base_path: Path | str | None = None,
    ft_path: Path | str | None = None,
) -> DeltaAnalysisResult:
    """Load matrices, compute Delta-M, and run the exact primary cell permutation."""
    bpath = Path(base_path) if base_path else DEFAULT_BASE_MATRIX_PATH
    fpath = Path(ft_path) if ft_path else DEFAULT_FT_MATRIX_PATH

    base = load_matrix_json(bpath)
    ft = load_matrix_json(fpath)
    meta = validate_matrix_pair(base, ft, base_path=bpath, ft_path=fpath)

    fact_ids = meta["fact_ids"]
    delta = compute_delta_m(_distance_matrix(base), _distance_matrix(ft), fact_ids)
    perm = exact_permutation_test(delta, fact_ids)

    _, mean_cross, t_cross = cross_domain_contrast(delta, fact_ids)
    cross_vals = {
        _pair_key(p): cell_value(delta, fact_ids, p) for p in CROSS_DOMAIN_PAIRS
    }

    ratio: float | None = None
    m_unrel = perm["mean_unrelated_within_delta"]
    m_tennis = perm["mean_tennis_within_delta"]
    if abs(m_unrel) > 1e-12:
        ratio = float(m_tennis / m_unrel)

    notes = [
        "Primary inferential test is on within-domain Delta-M contrast only "
        "(not separate tests on M_base or M_FT).",
        "Exact enumeration of C(6,2)=15 choices of 2 within-domain cells as "
        "tennis; remaining 4 are unrelated. Not the old C(4,2) fact-label swap.",
        "One-sided p-value includes the observed assignment (exact-test convention).",
        "T_cross (tennis within vs cross-domain) is secondary descriptive only.",
        "Do not auto-claim statistical significance from an arbitrary α threshold.",
        "Inference is conditional on the frozen balanced 6-fact / 3-domain design.",
    ]

    return DeltaAnalysisResult(
        status="complete",
        model_id=meta["model_id"],
        base_matrix_path=str(bpath),
        ft_matrix_path=str(fpath),
        fact_ids=list(fact_ids),
        tennis_within_pairs=perm["tennis_within_pairs"],
        unrelated_within_pairs=perm["unrelated_within_pairs"],
        cross_domain_pairs=[list(p) for p in CROSS_DOMAIN_PAIRS],
        within_domain_pairs=[list(p) for p in WITHIN_DOMAIN_PAIRS],
        within_domain_cell_values=dict(perm["within_domain_cell_values"]),
        cross_domain_cell_values=cross_vals,
        delta_m=delta,
        mean_tennis_within_delta=perm["mean_tennis_within_delta"],
        mean_unrelated_within_delta=perm["mean_unrelated_within_delta"],
        mean_cross_domain_delta=mean_cross,
        observed_contrast=perm["observed_contrast"],
        t_cross=t_cross,
        permutation_count=perm["permutation_count"],
        null_distribution=perm["null_distribution"],
        exact_one_sided_p_value=perm["exact_one_sided_p_value"],
        effect_size={
            "mean_tennis_within_delta_m": perm["mean_tennis_within_delta"],
            "mean_unrelated_within_delta_m": perm["mean_unrelated_within_delta"],
            "mean_cross_domain_delta_m": mean_cross,
            "T_obs": perm["observed_contrast"],
            "T_within": perm["observed_contrast"],
            "T_cross": t_cross,
            "observed_contrast": perm["observed_contrast"],
            "ratio_tennis_over_unrelated": ratio,
            "ratio_note": (
                "Optional; primary effect-size summary is T_within, not the ratio."
                if ratio is not None
                else "Ratio omitted (unrelated mean near zero / unstable)."
            ),
        },
        software=_software_versions(),
        runtime={
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "random_seed": None,
            "enumeration": "exact_within_cell_combinations_C(6,2)",
            "n_as_extreme_or_more": perm["n_as_extreme_or_more"],
        },
        notes=notes,
        matrix_metadata=meta,
    )


def write_delta_analysis_outputs(
    result: DeltaAnalysisResult,
    *,
    raw_dir: Path | None = None,
    table_path: Path | None = None,
) -> dict[str, Path]:
    raw_dir = raw_dir or DEFAULT_OUTPUT_DIR
    table_path = table_path or DEFAULT_TABLE_PATH
    raw_dir.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = raw_dir / "delta_analysis.json"
    json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    lines = [
        "# Delta-M analysis (balanced 6-fact / 3-domain)",
        "",
        f"- status: `{result.status}`",
        f"- model_id: `{result.model_id}`",
        f"- design: `balanced_6fact_3domain`",
        f"- base matrix: `{result.base_matrix_path}`",
        f"- FT matrix: `{result.ft_matrix_path}`",
        f"- AI-Engram / variant: see matrix metadata (must both be explicit / matching)",
        "",
        "## Definitions",
        "",
        "- `Delta_M[i,j] = M_FT[i,j] - M_base[i,j]` (diagonal null; full 6×6)",
        "- Tennis within-domain (2 cells): "
        + ", ".join(f"{a}→{b}" for a, b in result.tennis_within_pairs),
        "- Unrelated within-domain (4 cells; swimming+acting): "
        + ", ".join(f"{a}→{b}" for a, b in result.unrelated_within_pairs),
        "- Cross-domain (8 cells; secondary only): "
        + ", ".join(f"{a}→{b}" for a, b in result.cross_domain_pairs),
        "- Primary: `T_within = mean(tennis within) − mean(unrelated within)`",
        "- Secondary: `T_cross = mean(tennis within) − mean(cross-domain)`",
        "",
        "## Within-domain cell values",
        "",
    ]
    for key, val in result.within_domain_cell_values.items():
        lines.append(f"- `{key}`: `{val:.6f}`")

    lines.extend(
        [
            "",
            "## Delta-M matrix (cosine-distance change)",
            "",
            "Rows = intervened fact *i*; columns = measured fact *j*. `—` = diagonal.",
            "",
        ]
    )
    header = "| i \\ j | " + " | ".join(result.fact_ids) + " |"
    sep = "|-------|" + "|".join(["------"] * len(result.fact_ids)) + "|"
    lines.append(header)
    lines.append(sep)
    for i, i_fact in enumerate(result.fact_ids):
        cells = []
        for j in range(len(result.fact_ids)):
            v = result.delta_m[i][j]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append("—")
            else:
                cells.append(f"{v:.6f}")
        lines.append(f"| {i_fact} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Primary group means and contrast",
            "",
            f"- mean tennis within Delta-M: `{result.mean_tennis_within_delta:.6f}`",
            f"- mean unrelated within Delta-M: "
            f"`{result.mean_unrelated_within_delta:.6f}`",
            f"- observed contrast T_within / T_obs: "
            f"`{result.observed_contrast:.6f}`",
            "",
            "## Secondary descriptive (not primary inference)",
            "",
            f"- mean cross-domain Delta-M: `{result.mean_cross_domain_delta:.6f}`",
            f"- T_cross: `{result.t_cross:.6f}`",
            "",
            "## Exact cell-permutation test",
            "",
            f"- assignments in null distribution: `{result.permutation_count}` "
            "(all C(6,2)=15 choices of 2 of the 6 within-domain cells as tennis; "
            "remaining 4 = unrelated)",
            f"- exact one-sided p-value (T ≥ T_obs, including observed): "
            f"`{result.exact_one_sided_p_value:.6f}`",
            "",
            "This is **not** the old C(4,2) fact-label swap of in-domain vs "
            "control facts.",
            "",
            "### Null distribution (15 contrasts)",
            "",
            "| tennis cells | unrelated cells | contrast | observed? |",
            "|--------------|-----------------|----------|-----------|",
        ]
    )
    for row in result.null_distribution:
        tennis_s = ", ".join(f"{a}→{b}" for a, b in row["tennis_cells"])
        unrel_s = ", ".join(f"{a}→{b}" for a, b in row["unrelated_cells"])
        lines.append(
            f"| {tennis_s} | {unrel_s} | {row['contrast']:.6f} | "
            + ("yes" if row["is_observed_assignment"] else "no")
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation placeholder",
            "",
            "Positive T_within means tennis within-domain interactions changed more "
            "under tennis fine-tuning than unrelated within-domain interactions "
            "(intervention–response sense). Research write-up should interpret the "
            "exact p-value; this table does **not** auto-claim statistical significance.",
            "",
            "## Design note",
            "",
            "This is an exact cell-permutation test **conditional on the frozen "
            "balanced 6-fact / 3-domain design**. It does not claim population-level "
            "generalization from six facts.",
            "",
        ]
    )
    table_path.write_text("\n".join(lines), encoding="utf-8")
    return {"delta_analysis_json": json_path, "delta_analysis_md": table_path}


def result_schema() -> dict[str, Any]:
    return {
        "files": {
            "results/raw/delta_analysis/delta_analysis.json": "machine-readable analysis",
            "results/tables/delta_analysis.md": "human-readable summary",
        },
        "design": "balanced_6fact_3domain",
        "Delta_M": "M_FT - M_base (off-diagonal; full 6×6)",
        "tennis_within_pairs": [list(p) for p in TENNIS_WITHIN_PAIRS],
        "unrelated_within_pairs": [list(p) for p in UNRELATED_WITHIN_PAIRS],
        "cross_domain_pairs": [list(p) for p in CROSS_DOMAIN_PAIRS],
        "within_domain_pairs": [list(p) for p in WITHIN_DOMAIN_PAIRS],
        "T_within / T_observed": (
            "mean(tennis within) - mean(unrelated within)"
        ),
        "T_cross": "mean(tennis within) - mean(cross-domain) [descriptive only]",
        "permutation": (
            "exact enumeration of C(6,2)=15 choices of 2 of 6 within-domain "
            "cells as tennis (cell permutation; not fact-label swap)"
        ),
        "p_value": "one-sided fraction with T >= T_obs including observed",
        "fact_order": list(FACT_IDS),
    }


def validate_delta_analysis_without_matrices() -> dict[str, Any]:
    """Schema / algorithm dry-validate without fabricating experiment results."""
    base_path, ft_path = default_matrix_paths()
    base_exists = base_path.is_file()
    ft_exists = ft_path.is_file()

    facts = list(FACT_IDS)
    n = len(facts)
    synth: list[list[float | None]] = [[None] * n for _ in range(n)]
    idx = {f: i for i, f in enumerate(facts)}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            synth[i][j] = 0.05

    # Tennis within large; unrelated within smaller; cross-domain intermediate.
    synth[idx["F01"]][idx["F02"]] = 0.50
    synth[idx["F02"]][idx["F01"]] = 0.40
    synth[idx["F03"]][idx["F04"]] = 0.10
    synth[idx["F04"]][idx["F03"]] = 0.10
    synth[idx["F05"]][idx["F06"]] = 0.10
    synth[idx["F06"]][idx["F05"]] = 0.10
    for a, b in CROSS_DOMAIN_PAIRS:
        synth[idx[a]][idx[b]] = 0.20

    perm = exact_permutation_test(synth, facts)
    assert perm["permutation_count"] == EXACT_PERMUTATION_COUNT == 15
    assert abs(perm["mean_tennis_within_delta"] - 0.45) < 1e-12
    assert abs(perm["mean_unrelated_within_delta"] - 0.10) < 1e-12
    assert abs(perm["observed_contrast"] - 0.35) < 1e-12
    assert 0.0 < perm["exact_one_sided_p_value"] <= 1.0
    assert sum(1 for r in perm["null_distribution"] if r["is_observed_assignment"]) == 1
    assert len(perm["null_distribution"]) == 15

    assignments = enumerate_within_cell_assignments()
    assert len(assignments) == 15
    # Observed tennis assignment must appear exactly once among the 15.
    obs = set(TENNIS_WITHIN_PAIRS)
    assert sum(1 for t, _ in assignments if set(t) == obs) == 1

    _, mean_cross, t_cross = cross_domain_contrast(synth, facts)
    assert abs(mean_cross - 0.20) < 1e-12
    assert abs(t_cross - 0.25) < 1e-12

    pending_reasons: list[str] = []
    if not base_exists:
        pending_reasons.append("base_matrix_json_missing")
    if not ft_exists:
        pending_reasons.append("ft_matrix_json_missing")

    return {
        "status": "pending_matrix_inputs" if pending_reasons else "inputs_present",
        "model_id": EXPECTED_MODEL_ID,
        "ai_engram_version": EXPECTED_ENGRAM_VERSION,
        "design": "balanced_6fact_3domain",
        "base_matrix_path": str(base_path),
        "ft_matrix_path": str(ft_path),
        "base_matrix_exists": base_exists,
        "ft_matrix_exists": ft_exists,
        "fact_order": facts,
        "tennis_within_pairs": [list(p) for p in TENNIS_WITHIN_PAIRS],
        "unrelated_within_pairs": [list(p) for p in UNRELATED_WITHIN_PAIRS],
        "cross_domain_pairs": [list(p) for p in CROSS_DOMAIN_PAIRS],
        "within_domain_pairs": [list(p) for p in WITHIN_DOMAIN_PAIRS],
        "result_schema": result_schema(),
        "algorithm_self_check": {
            "permutation_count": perm["permutation_count"],
            "expected_permutation_count": 15,
            "enumeration": "C(6,2)_within_domain_cells",
            "synthetic_T_within": perm["observed_contrast"],
            "synthetic_T_cross": t_cross,
            "synthetic_p_in_unit_interval": True,
            "observed_assignment_counted_once": True,
            "note": (
                "Synthetic Delta-M used only to verify C(6,2)=15 cell-permutation "
                "and contrast math; not an experimental result and not written to "
                "delta_analysis.json."
            ),
        },
        "pending_reasons": pending_reasons,
        "analysis_executed": False,
        "matrix_fabricated": False,
        "operational_precondition_for_exact_permutation": (
            "Both base_matrix.json and ft_matrix.json must have numeric values in "
            "ALL off-diagonal cells (all six facts selective rows). Incomplete "
            "selective-only matrices are refused — not patched with invented values."
        ),
        "software": _software_versions(),
    }


def write_pending_status(report: dict[str, Any] | None = None) -> Path:
    raw_dir = DEFAULT_OUTPUT_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "STATUS.yaml"
    payload: dict[str, Any] = {
        "status": "pending_base_and_or_ft_matrix_json",
        "model_id": EXPECTED_MODEL_ID,
        "design": "balanced_6fact_3domain",
        "message": (
            "Delta-M analysis pipeline is implemented for the balanced 6-fact "
            "design. Execution awaits base_matrix.json and ft_matrix.json from "
            "completed matrix runs. No Delta-M values, p-values, or significance "
            "claims have been fabricated."
        ),
        "definition": (
            "Delta_M = M_FT - M_base; "
            "T_within = mean(tennis within) - mean(unrelated within); "
            "T_cross = mean(tennis within) - mean(cross-domain) [descriptive]"
        ),
        "tennis_within_pairs": [list(p) for p in TENNIS_WITHIN_PAIRS],
        "unrelated_within_pairs": [list(p) for p in UNRELATED_WITHIN_PAIRS],
        "cross_domain_pairs": [list(p) for p in CROSS_DOMAIN_PAIRS],
        "permutation": "exact C(6,2)=15 within-domain cell assignments",
        "schema": result_schema(),
        "analysis_executed": False,
    }
    if report is not None:
        payload["last_dry_validate"] = {
            k: report[k]
            for k in (
                "base_matrix_exists",
                "ft_matrix_exists",
                "pending_reasons",
                "analysis_executed",
            )
            if k in report
        }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path
