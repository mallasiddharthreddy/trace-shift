"""Primary Delta-M analysis and exact permutation test for TraceShift.

Scientific definitions (do not reinterpret):

    Delta_M[i, j] = M_FT[i, j] - M_base[i, j]   (off-diagonal; diagonal null)

Primary in-domain cells:
    F01 → F02,  F02 → F01

Primary in-domain→control cells:
    F01 → F03, F01 → F04, F02 → F03, F02 → F04

Primary contrast:
    T = mean(Delta_M[in_domain]) - mean(Delta_M[in_domain_to_control])

Inference is an exact randomization test on the frozen four-fact design:
enumerate all C(4,2)=6 assignments of two in-domain vs two control facts,
recompute T under each assignment with the observed Delta-M held fixed,
and report a one-sided exact p-value including the observed assignment.
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

from .base_matrix import FACT_IDS
from .engram import EXPECTED_ENGRAM_VERSION, EXPECTED_MODEL_ID
from .paths import research_root

Pair = tuple[str, str]

# Frozen primary contrast definitions (documented; not alternative experiments).
OBSERVED_IN_DOMAIN_FACTS: tuple[str, ...] = ("F01", "F02")
OBSERVED_CONTROL_FACTS: tuple[str, ...] = ("F03", "F04")
OBSERVED_IN_DOMAIN_PAIRS: tuple[Pair, ...] = (("F01", "F02"), ("F02", "F01"))
OBSERVED_CONTROL_PAIRS: tuple[Pair, ...] = (
    ("F01", "F03"),
    ("F01", "F04"),
    ("F02", "F03"),
    ("F02", "F04"),
)

DEFAULT_BASE_MATRIX_PATH = (
    research_root() / "results" / "raw" / "base_matrix" / "base_matrix.json"
)
DEFAULT_FT_MATRIX_PATH = (
    research_root() / "results" / "raw" / "ft_matrix" / "ft_matrix.json"
)
DEFAULT_OUTPUT_DIR = research_root() / "results" / "raw" / "delta_analysis"
DEFAULT_TABLE_PATH = research_root() / "results" / "tables" / "delta_analysis.md"


class DeltaAnalysisError(RuntimeError):
    """Incompatible or missing matrix inputs for Delta-M analysis."""


@dataclass
class DeltaAnalysisResult:
    status: str
    model_id: str
    base_matrix_path: str
    ft_matrix_path: str
    fact_ids: list[str]
    in_domain_fact_ids: list[str]
    control_fact_ids: list[str]
    in_domain_pairs: list[list[str]]
    control_pairs: list[list[str]]
    delta_m: list[list[float | None]]
    mean_in_domain_delta: float
    mean_control_delta: float
    observed_contrast: float
    permutation_count: int
    null_distribution: list[dict[str, Any]]
    exact_one_sided_p_value: float
    effect_size: dict[str, Any]
    software: dict[str, str]
    runtime: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    matrix_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model_id": self.model_id,
            "base_matrix_path": self.base_matrix_path,
            "ft_matrix_path": self.ft_matrix_path,
            "fact_ordering": self.fact_ids,
            "in_domain_fact_ids": self.in_domain_fact_ids,
            "control_fact_ids": self.control_fact_ids,
            "in_domain_pair_definition": self.in_domain_pairs,
            "control_pair_definition": self.control_pairs,
            "definition": {
                "Delta_M[i,j]": "M_FT[i,j] - M_base[i,j] (off-diagonal; diagonal null)",
                "T_observed": (
                    "mean(Delta_M[in_domain_pairs]) - "
                    "mean(Delta_M[in_domain_to_control_pairs])"
                ),
                "interpretation_note": (
                    "Positive T_observed means FT changed the selected in-domain "
                    "interactions more than the selected control interactions "
                    "(intervention-response sense only; not a broader causal claim)."
                ),
            },
            "delta_m_matrix": self.delta_m,
            "mean_in_domain_delta_m": self.mean_in_domain_delta,
            "mean_control_delta_m": self.mean_control_delta,
            "observed_contrast": self.observed_contrast,
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
                "Exact randomization test conditional on the frozen four-fact design; "
                "not a population-level generalization claim."
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


def validate_matrix_pair(
    base: dict[str, Any],
    ft: dict[str, Any],
    *,
    base_path: Path | str,
    ft_path: Path | str,
) -> dict[str, Any]:
    """Confirm BASE and FT matrices are compatible for Delta-M."""
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

    # Diagonal convention: both should declare null/NaN semantics (string match loose)
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
    if len(base_mat) != n or len(ft_mat) != n:
        raise DeltaAnalysisError("Matrix row count does not match fact ordering")
    for i in range(n):
        if len(base_mat[i]) != n or len(ft_mat[i]) != n:
            raise DeltaAnalysisError("Matrix is not square / fact-aligned")
        for j in range(n):
            if i == j:
                # diagonal must be null-like
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
                            "The frozen exact permutation test requires a complete "
                            "4×4 off-diagonal ΔM, which in turn requires all four "
                            "facts to have been selective matrix rows (BASE + FT). "
                            "Do not invent cell values; complete selective alpha "
                            "calibration for F01–F04 or stop before Delta-M."
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


def pairs_for_assignment(
    in_domain: Sequence[str],
    controls: Sequence[str],
) -> tuple[list[Pair], list[Pair]]:
    """Build primary-style pair sets for a group assignment.

    In-domain: both directed edges between the two in-domain facts.
    Control: each in-domain fact → each control fact.
    """
    if len(in_domain) != 2 or len(controls) != 2:
        raise DeltaAnalysisError(
            f"Expected 2 in-domain and 2 control facts, got "
            f"in_domain={list(in_domain)} controls={list(controls)}"
        )
    a, b = in_domain[0], in_domain[1]
    in_pairs: list[Pair] = [(a, b), (b, a)]
    ctrl_pairs: list[Pair] = [
        (a, controls[0]),
        (a, controls[1]),
        (b, controls[0]),
        (b, controls[1]),
    ]
    return in_pairs, ctrl_pairs


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


def primary_contrast(
    delta_m: Sequence[Sequence[float | None]],
    fact_ids: Sequence[str],
    in_domain_pairs: Sequence[Pair],
    control_pairs: Sequence[Pair],
) -> tuple[float, float, float]:
    """Return (mean_in_domain, mean_control, T)."""
    m_in = mean_over_pairs(delta_m, fact_ids, in_domain_pairs)
    m_ctrl = mean_over_pairs(delta_m, fact_ids, control_pairs)
    return m_in, m_ctrl, float(m_in - m_ctrl)


def enumerate_group_assignments(
    fact_ids: Sequence[str] = FACT_IDS,
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """All ways to choose 2 in-domain facts (rest = controls). Order within groups sorted."""
    facts = list(fact_ids)
    if len(facts) != 4:
        raise DeltaAnalysisError(
            f"Exact test expects exactly 4 facts, got {len(facts)}"
        )
    assignments: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for combo in itertools.combinations(facts, 2):
        in_dom = tuple(sorted(combo, key=lambda f: facts.index(f)))
        ctrl = tuple(f for f in facts if f not in in_dom)
        assignments.append((in_dom, ctrl))
    # Deterministic order by in-domain tuple
    assignments.sort(key=lambda x: (facts.index(x[0][0]), facts.index(x[0][1])))
    return assignments


def exact_permutation_test(
    delta_m: Sequence[Sequence[float | None]],
    fact_ids: Sequence[str],
    *,
    observed_in_domain: Sequence[str] = OBSERVED_IN_DOMAIN_FACTS,
    observed_controls: Sequence[str] = OBSERVED_CONTROL_FACTS,
) -> dict[str, Any]:
    """Exact one-sided randomization test over all C(4,2) group assignments.

    Null: Delta-M is fixed; group labels (in-domain vs control) are exchangeable
    under size-(2,2) assignments. p = (# assignments with T >= T_obs) / N,
    including the observed assignment (exact-test convention).
    """
    obs_in = tuple(observed_in_domain)
    obs_ctrl = tuple(observed_controls)
    # Normalize observed group order to fact_ids order for membership compare
    obs_in_sorted = tuple(sorted(obs_in, key=lambda f: list(fact_ids).index(f)))
    obs_ctrl_sorted = tuple(f for f in fact_ids if f not in obs_in_sorted)

    in_pairs_obs, ctrl_pairs_obs = pairs_for_assignment(obs_in_sorted, obs_ctrl_sorted)
    # Prefer the documented ordered pairs when they match the frozen observed groups
    if set(obs_in_sorted) == set(OBSERVED_IN_DOMAIN_FACTS) and set(
        obs_ctrl_sorted
    ) == set(OBSERVED_CONTROL_FACTS):
        in_pairs_obs = list(OBSERVED_IN_DOMAIN_PAIRS)
        ctrl_pairs_obs = list(OBSERVED_CONTROL_PAIRS)

    m_in, m_ctrl, t_obs = primary_contrast(
        delta_m, fact_ids, in_pairs_obs, ctrl_pairs_obs
    )

    null: list[dict[str, Any]] = []
    assignments = enumerate_group_assignments(fact_ids)
    n_ge = 0
    for in_dom, ctrl in assignments:
        in_pairs, ctrl_pairs = pairs_for_assignment(in_dom, ctrl)
        _, _, t = primary_contrast(delta_m, fact_ids, in_pairs, ctrl_pairs)
        is_observed = set(in_dom) == set(obs_in_sorted)
        if t >= t_obs - 1e-15:  # inclusive; float-safe for equality
            n_ge += 1
        null.append(
            {
                "in_domain_facts": list(in_dom),
                "control_facts": list(ctrl),
                "in_domain_pairs": [list(p) for p in in_pairs],
                "control_pairs": [list(p) for p in ctrl_pairs],
                "contrast": t,
                "is_observed_assignment": is_observed,
            }
        )

    n = len(assignments)
    if n == 0:
        raise DeltaAnalysisError("Empty null distribution")
    p_value = float(n_ge) / float(n)

    return {
        "mean_in_domain_delta": m_in,
        "mean_control_delta": m_ctrl,
        "observed_contrast": t_obs,
        "permutation_count": n,
        "n_as_extreme_or_more": n_ge,
        "exact_one_sided_p_value": p_value,
        "null_distribution": null,
        "in_domain_pairs": [list(p) for p in in_pairs_obs],
        "control_pairs": [list(p) for p in ctrl_pairs_obs],
    }


def analyze_delta_m(
    *,
    base_path: Path | str | None = None,
    ft_path: Path | str | None = None,
) -> DeltaAnalysisResult:
    """Load matrices, compute Delta-M, and run the exact primary permutation test."""
    bpath = Path(base_path) if base_path else DEFAULT_BASE_MATRIX_PATH
    fpath = Path(ft_path) if ft_path else DEFAULT_FT_MATRIX_PATH

    base = load_matrix_json(bpath)
    ft = load_matrix_json(fpath)
    meta = validate_matrix_pair(base, ft, base_path=bpath, ft_path=fpath)

    fact_ids = meta["fact_ids"]
    delta = compute_delta_m(_distance_matrix(base), _distance_matrix(ft), fact_ids)
    perm = exact_permutation_test(delta, fact_ids)

    ratio: float | None = None
    m_ctrl = perm["mean_control_delta"]
    m_in = perm["mean_in_domain_delta"]
    if abs(m_ctrl) > 1e-12:
        ratio = float(m_in / m_ctrl)

    notes = [
        "Primary inferential test is on Delta-M contrast only "
        "(not separate tests on M_base or M_FT).",
        "Exact enumeration of C(4,2)=6 size-(2,2) group assignments; no Monte Carlo.",
        "One-sided p-value includes the observed assignment (exact-test convention).",
        "Do not auto-claim statistical significance from an arbitrary α threshold.",
        "Inference is conditional on the frozen four-fact design.",
    ]

    return DeltaAnalysisResult(
        status="complete",
        model_id=meta["model_id"],
        base_matrix_path=str(bpath),
        ft_matrix_path=str(fpath),
        fact_ids=list(fact_ids),
        in_domain_fact_ids=list(OBSERVED_IN_DOMAIN_FACTS),
        control_fact_ids=list(OBSERVED_CONTROL_FACTS),
        in_domain_pairs=perm["in_domain_pairs"],
        control_pairs=perm["control_pairs"],
        delta_m=delta,
        mean_in_domain_delta=perm["mean_in_domain_delta"],
        mean_control_delta=perm["mean_control_delta"],
        observed_contrast=perm["observed_contrast"],
        permutation_count=perm["permutation_count"],
        null_distribution=perm["null_distribution"],
        exact_one_sided_p_value=perm["exact_one_sided_p_value"],
        effect_size={
            "mean_in_domain_delta_m": perm["mean_in_domain_delta"],
            "mean_control_delta_m": perm["mean_control_delta"],
            "observed_contrast": perm["observed_contrast"],
            "ratio_in_domain_over_control": ratio,
            "ratio_note": (
                "Optional; primary effect-size summary is the contrast, not the ratio."
                if ratio is not None
                else "Ratio omitted (control mean near zero / unstable)."
            ),
        },
        software=_software_versions(),
        runtime={
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "random_seed": None,
            "enumeration": "exact_combinations_C(4,2)",
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
        "# Delta-M analysis (primary contrast)",
        "",
        f"- status: `{result.status}`",
        f"- model_id: `{result.model_id}`",
        f"- base matrix: `{result.base_matrix_path}`",
        f"- FT matrix: `{result.ft_matrix_path}`",
        f"- AI-Engram / variant: see matrix metadata (must both be explicit / matching)",
        "",
        "## Definitions",
        "",
        "- `Delta_M[i,j] = M_FT[i,j] - M_base[i,j]` (diagonal null)",
        "- In-domain facts: " + ", ".join(result.in_domain_fact_ids),
        "- Control facts: " + ", ".join(result.control_fact_ids),
        "- In-domain pairs: "
        + ", ".join(f"{a}→{b}" for a, b in result.in_domain_pairs),
        "- Control pairs (in-domain→control): "
        + ", ".join(f"{a}→{b}" for a, b in result.control_pairs),
        "- `T = mean(in-domain Delta-M) − mean(control Delta-M)`",
        "",
        "## Delta-M matrix (cosine-distance change)",
        "",
        "Rows = intervened fact *i*; columns = measured fact *j*. `—` = diagonal.",
        "",
    ]
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
            f"- mean in-domain Delta-M: `{result.mean_in_domain_delta:.6f}`",
            f"- mean control Delta-M: `{result.mean_control_delta:.6f}`",
            f"- observed contrast T: `{result.observed_contrast:.6f}`",
            "",
            "## Exact permutation test",
            "",
            f"- assignments in null distribution: `{result.permutation_count}` "
            "(all size-(2,2) labelings of the four facts)",
            f"- exact one-sided p-value (T ≥ T_obs, including observed): "
            f"`{result.exact_one_sided_p_value:.6f}`",
            "",
            "### Null distribution",
            "",
            "| in-domain | control | contrast | observed? |",
            "|-----------|---------|----------|-----------|",
        ]
    )
    for row in result.null_distribution:
        lines.append(
            "| "
            + ", ".join(row["in_domain_facts"])
            + " | "
            + ", ".join(row["control_facts"])
            + f" | {row['contrast']:.6f} | "
            + ("yes" if row["is_observed_assignment"] else "no")
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation placeholder",
            "",
            "Positive T means the selected in-domain interactions changed more under "
            "tennis fine-tuning than the selected in-domain→control interactions "
            "(intervention–response sense). Research write-up should interpret the "
            "exact p-value; this table does **not** auto-claim statistical significance.",
            "",
            "## Design note",
            "",
            "This is an exact randomization test **conditional on the frozen "
            "four-fact design**. It does not claim population-level generalization "
            "from four facts.",
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
        "Delta_M": "M_FT - M_base (off-diagonal)",
        "in_domain_pairs": [list(p) for p in OBSERVED_IN_DOMAIN_PAIRS],
        "control_pairs": [list(p) for p in OBSERVED_CONTROL_PAIRS],
        "T_observed": "mean(in_domain) - mean(control)",
        "permutation": "exact enumeration of C(4,2)=6 group assignments",
        "p_value": "one-sided fraction with T >= T_obs including observed",
        "fact_order": list(FACT_IDS),
    }


def validate_delta_analysis_without_matrices() -> dict[str, Any]:
    """Schema / algorithm dry-validate without fabricating experiment results."""
    base_path, ft_path = default_matrix_paths()
    base_exists = base_path.is_file()
    ft_exists = ft_path.is_file()

    # Deterministic synthetic Delta-M to exercise enumeration only (not written as results)
    facts = list(FACT_IDS)
    # Construct a tiny synthetic delta with known contrast properties
    synth = [[None] * 4 for _ in range(4)]
    # Make observed in-domain cells large, control smaller
    idx = {f: i for i, f in enumerate(facts)}
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            synth[i][j] = 0.1
    synth[idx["F01"]][idx["F02"]] = 0.5
    synth[idx["F02"]][idx["F01"]] = 0.4
    synth[idx["F01"]][idx["F03"]] = 0.1
    synth[idx["F01"]][idx["F04"]] = 0.1
    synth[idx["F02"]][idx["F03"]] = 0.1
    synth[idx["F02"]][idx["F04"]] = 0.1

    perm = exact_permutation_test(synth, facts)
    assert perm["permutation_count"] == 6
    assert abs(perm["mean_in_domain_delta"] - 0.45) < 1e-12
    assert abs(perm["mean_control_delta"] - 0.1) < 1e-12
    assert abs(perm["observed_contrast"] - 0.35) < 1e-12
    assert 0.0 < perm["exact_one_sided_p_value"] <= 1.0
    assert sum(1 for r in perm["null_distribution"] if r["is_observed_assignment"]) == 1

    assignments = enumerate_group_assignments(facts)
    assert len(assignments) == 6

    pending_reasons: list[str] = []
    if not base_exists:
        pending_reasons.append("base_matrix_json_missing")
    if not ft_exists:
        pending_reasons.append("ft_matrix_json_missing")

    return {
        "status": "pending_matrix_inputs" if pending_reasons else "inputs_present",
        "model_id": EXPECTED_MODEL_ID,
        "ai_engram_version": EXPECTED_ENGRAM_VERSION,
        "base_matrix_path": str(base_path),
        "ft_matrix_path": str(ft_path),
        "base_matrix_exists": base_exists,
        "ft_matrix_exists": ft_exists,
        "fact_order": facts,
        "in_domain_facts": list(OBSERVED_IN_DOMAIN_FACTS),
        "control_facts": list(OBSERVED_CONTROL_FACTS),
        "in_domain_pairs": [list(p) for p in OBSERVED_IN_DOMAIN_PAIRS],
        "control_pairs": [list(p) for p in OBSERVED_CONTROL_PAIRS],
        "result_schema": result_schema(),
        "algorithm_self_check": {
            "permutation_count": perm["permutation_count"],
            "synthetic_T": perm["observed_contrast"],
            "synthetic_p_in_unit_interval": True,
            "observed_assignment_counted_once": True,
            "note": (
                "Synthetic Delta-M used only to verify enumeration/contrast math; "
                "not an experimental result and not written to delta_analysis.json."
            ),
        },
        "pending_reasons": pending_reasons,
        "analysis_executed": False,
        "matrix_fabricated": False,
        "operational_precondition_for_exact_permutation": (
            "Both base_matrix.json and ft_matrix.json must have numeric values in "
            "ALL off-diagonal cells (all four facts selective rows). Incomplete "
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
        "message": (
            "Delta-M analysis pipeline is implemented. Execution awaits "
            "base_matrix.json and ft_matrix.json from completed matrix runs. "
            "No Delta-M values, p-values, or significance claims have been fabricated."
        ),
        "definition": "Delta_M = M_FT - M_base; T = mean(in_domain) - mean(control)",
        "in_domain_pairs": [list(p) for p in OBSERVED_IN_DOMAIN_PAIRS],
        "control_pairs": [list(p) for p in OBSERVED_CONTROL_PAIRS],
        "permutation": "exact C(4,2)=6 group assignments",
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
