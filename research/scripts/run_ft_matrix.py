#!/usr/bin/env python3
"""FINE-TUNED intervention-response matrix CLI for TraceShift.

Uses BASE-calibrated alphas + Engrams extracted from the primary FT model.

Usage:
  .venv/bin/python research/scripts/run_ft_matrix.py --dry-validate
  .venv/bin/python research/scripts/run_ft_matrix.py
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
    from traceshift.ft_matrix import (
        FinetunedCheckpointUnavailableError,
        FtMatrixError,
        compute_ft_matrix,
        validate_ft_matrix_pipeline_without_model,
        write_ft_matrix_outputs,
        write_pending_status,
    )
    from traceshift.model_loader import ModelWeightsUnavailableError
    from traceshift.paths import default_experiment_config

    parser = argparse.ArgumentParser(
        description=(
            "Primary fine-tuned intervention-response matrix (explicit extraction; "
            "BASE-calibrated alphas)."
        )
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--alphas",
        type=Path,
        default=None,
        help="Path to base_alphas.yaml (BASE calibration; not FT-recalibrated)",
    )
    parser.add_argument("--dry-validate", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--allow-override-statuses",
        type=str,
        default="",
        help="Comma-separated statuses allowed only with explicit override recording",
    )
    args = parser.parse_args(argv)
    config_path = args.config or default_experiment_config()

    print("=== TraceShift FT intervention-response matrix ===")
    print(f"config: {config_path}")
    print("definition: M_FT[i,j] = response of j after intervene on i (primary FT model)")
    print("alphas: BASE-calibrated (fixed); Engrams: collected from FT model")
    print("variant: explicit only; diagonal null; selective rows by default")
    print()

    if args.dry_validate:
        report = validate_ft_matrix_pipeline_without_model(config_path)
        status_path = write_pending_status(report=report)
        print(json.dumps(report, indent=2))
        print()
        print(f"wrote pending status: {status_path}")
        print("DRY VALIDATE: OK (no model / Engram / matrix / training execution)")
        return 0

    try:
        from traceshift.ft_matrix import load_primary_finetuned_model

        loaded = load_primary_finetuned_model(config_path, device=args.device)
    except (ModelWeightsUnavailableError, FinetunedCheckpointUnavailableError) as exc:
        report = validate_ft_matrix_pipeline_without_model(config_path)
        status_path = write_pending_status(report=report)
        print("LOAD: PENDING — base weights and/or primary FT checkpoint unavailable")
        print(str(exc))
        print()
        print(json.dumps(report, indent=2))
        print()
        print(f"wrote pending status: {status_path}")
        print("No matrix values fabricated.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"LOAD: FAIL — {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    overrides = tuple(
        s.strip() for s in args.allow_override_statuses.split(",") if s.strip()
    )
    try:
        result = compute_ft_matrix(
            loaded,
            alphas_path=args.alphas,
            config_path=config_path,
            allow_override=bool(overrides),
            override_statuses=overrides,
        )
    except FtMatrixError as exc:
        print(f"FT MATRIX: REFUSED — {exc}", file=sys.stderr)
        write_pending_status()
        return 1

    paths = write_ft_matrix_outputs(result)
    print("FT MATRIX: complete (no significance claims)")
    print(f"model_state: {result.model_state}")
    print(f"rows with BASE alphas: {result.row_alphas}")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
