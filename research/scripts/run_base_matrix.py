#!/usr/bin/env python3
"""BASE intervention-response matrix CLI for TraceShift.

Default without local weights: dry validation only (no fabricated matrix values).

Usage:
  .venv/bin/python research/scripts/run_base_matrix.py --dry-validate
  .venv/bin/python research/scripts/run_base_matrix.py
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
    from traceshift.base_matrix import (
        BaseMatrixError,
        compute_base_matrix,
        validate_base_matrix_pipeline_without_model,
        write_base_matrix_outputs,
        write_pending_status,
    )
    from traceshift.model_loader import ModelWeightsUnavailableError, load_model
    from traceshift.paths import default_experiment_config

    parser = argparse.ArgumentParser(
        description="BASE-model intervention-response matrix (explicit extraction)."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--alphas",
        type=Path,
        default=None,
        help="Path to base_alphas.yaml from alpha calibration",
    )
    parser.add_argument("--dry-validate", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--allow-override-statuses",
        type=str,
        default="",
        help="Comma-separated statuses allowed only with explicit override recording "
        "(e.g. nonselective). Default: selective only.",
    )
    args = parser.parse_args(argv)
    config_path = args.config or default_experiment_config()

    print("=== TraceShift BASE intervention-response matrix ===")
    print(f"config: {config_path}")
    print("definition: M_base[i,j] = response of j after intervene on i")
    print("primary metric: cosine_distance = 1 - cosine_similarity")
    print("variant: explicit only; diagonal null; selective rows by default")
    print()

    if args.dry_validate:
        report = validate_base_matrix_pipeline_without_model(config_path)
        status_path = write_pending_status()
        print(json.dumps(report, indent=2))
        print()
        print(f"wrote pending status: {status_path}")
        print("DRY VALIDATE: OK (no model / Engram / matrix execution)")
        return 0

    try:
        loaded = load_model(config_path, device=args.device, local_files_only=True)
    except ModelWeightsUnavailableError as exc:
        report = validate_base_matrix_pipeline_without_model(config_path)
        status_path = write_pending_status()
        print("MODEL LOAD: PENDING — weights unavailable locally")
        print(str(exc))
        print()
        print(json.dumps(report, indent=2))
        print()
        print(f"wrote pending status: {status_path}")
        print("No matrix values fabricated.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"MODEL LOAD: FAIL — {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    overrides = tuple(
        s.strip() for s in args.allow_override_statuses.split(",") if s.strip()
    )
    try:
        result = compute_base_matrix(
            loaded,
            alphas_path=args.alphas,
            config_path=config_path,
            allow_override=bool(overrides),
            override_statuses=overrides,
        )
    except BaseMatrixError as exc:
        print(f"BASE MATRIX: REFUSED — {exc}", file=sys.stderr)
        write_pending_status()
        return 1

    paths = write_base_matrix_outputs(result)
    print("BASE MATRIX: complete (no significance claims)")
    print(f"rows with alphas: {result.row_alphas}")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
