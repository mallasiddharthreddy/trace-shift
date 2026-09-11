#!/usr/bin/env python3
"""Primary Delta-M analysis + exact cell-permutation test CLI for TraceShift.

Balanced 6-fact / 3-domain design:
  T_within = mean(tennis within) - mean(unrelated within)
  Exact test: C(6,2)=15 choices of 2 of 6 within-domain cells as tennis.

Usage:
  .venv/bin/python research/scripts/run_delta_analysis.py --dry-validate
  .venv/bin/python research/scripts/run_delta_analysis.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_pkg_path() -> None:
    research = Path(__file__).resolve().parent.parent
    if str(research) not in sys.path:
        sys.path.insert(0, str(research))


def main(argv: list[str] | None = None) -> int:
    _ensure_pkg_path()
    from traceshift.delta_analysis import (
        DeltaAnalysisError,
        analyze_delta_m,
        validate_delta_analysis_without_matrices,
        write_delta_analysis_outputs,
        write_pending_status,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Delta-M = M_FT - M_base (6×6); exact C(6,2)=15 cell-permutation "
            "test on tennis within-domain vs unrelated within-domain contrast "
            "(balanced 6-fact / 3-domain design)."
        )
    )
    parser.add_argument(
        "--base-matrix",
        type=Path,
        default=None,
        help="Path to base_matrix.json",
    )
    parser.add_argument(
        "--ft-matrix",
        type=Path,
        default=None,
        help="Path to ft_matrix.json",
    )
    parser.add_argument(
        "--dry-validate",
        action="store_true",
        help="Validate schema/algorithm without requiring matrix JSON results",
    )
    args = parser.parse_args(argv)

    print("=== TraceShift Delta-M analysis (balanced 6-fact) ===")
    print("Delta_M[i,j] = M_FT[i,j] - M_base[i,j]")
    print("T_within = mean(tennis within) - mean(unrelated within)")
    print("T_cross  = mean(tennis within) - mean(cross-domain)  [descriptive]")
    print("test: exact C(6,2)=15 within-domain cell permutation (one-sided)")
    print()

    if args.dry_validate:
        report = validate_delta_analysis_without_matrices()
        status_path = write_pending_status(report)
        print(json.dumps(report, indent=2))
        print()
        print(f"wrote pending status: {status_path}")
        print("DRY VALIDATE: OK (no fabricated Delta-M / p-values)")
        return 0

    try:
        result = analyze_delta_m(
            base_path=args.base_matrix,
            ft_path=args.ft_matrix,
        )
    except DeltaAnalysisError as exc:
        report = validate_delta_analysis_without_matrices()
        status_path = write_pending_status(report)
        print(f"DELTA ANALYSIS: PENDING/REFUSED — {exc}")
        print()
        print(json.dumps(report, indent=2))
        print()
        print(f"wrote pending status: {status_path}")
        print("No Delta-M values or p-values fabricated.")
        return 0

    paths = write_delta_analysis_outputs(result)
    print("DELTA ANALYSIS: complete")
    print(f"T_within / T_obs: {result.observed_contrast}")
    print(f"T_cross: {result.t_cross}")
    print(f"exact one-sided p: {result.exact_one_sided_p_value}")
    print(f"permutation_count: {result.permutation_count}")
    print("(No automatic significance claim — report exact p-value only.)")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
