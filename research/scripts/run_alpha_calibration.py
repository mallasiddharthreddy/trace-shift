#!/usr/bin/env python3
"""BASE-model alpha calibration CLI for TraceShift.

Default: dry validation (no weight download). Full run only if local weights exist.

Usage:
  .venv/bin/python research/scripts/run_alpha_calibration.py --dry-validate
  .venv/bin/python research/scripts/run_alpha_calibration.py
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
    from traceshift.alpha_calibration import (
        calibrate_base_model,
        validate_calibration_pipeline_without_model,
        write_calibration_outputs,
        write_pending_status,
    )
    from traceshift.model_loader import ModelWeightsUnavailableError, load_model
    from traceshift.paths import default_experiment_config

    parser = argparse.ArgumentParser(
        description="BASE-model AI-Engram alpha calibration (dose selection only)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment.yaml",
    )
    parser.add_argument(
        "--dry-validate",
        action="store_true",
        help="Validate config/control-flow/schema without loading the model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device when running full calibration",
    )
    args = parser.parse_args(argv)
    config_path = args.config or default_experiment_config()

    print("=== TraceShift BASE alpha calibration ===")
    print(f"config: {config_path}")
    print("scope: baseline recall + Engram collect-once + alpha sweep (explicit only)")
    print("out of scope: matrices, FT, delta, stats, lexical-control, paid GPU")
    print()

    if args.dry_validate:
        report = validate_calibration_pipeline_without_model(config_path)
        status_path = write_pending_status()
        print(json.dumps(report, indent=2))
        print()
        print(f"wrote pending status: {status_path}")
        print("DRY VALIDATE: OK (no model / Engram / intervention executed)")
        return 0

    try:
        loaded = load_model(config_path, device=args.device, local_files_only=True)
    except ModelWeightsUnavailableError as exc:
        report = validate_calibration_pipeline_without_model(config_path)
        status_path = write_pending_status()
        print("MODEL LOAD: PENDING — weights unavailable locally")
        print(str(exc))
        print()
        print("Falling back to dry validation:")
        print(json.dumps(report, indent=2))
        print()
        print(f"wrote pending status: {status_path}")
        print("No calibration results fabricated.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"MODEL LOAD: FAIL — {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    result = calibrate_base_model(loaded, config_path=config_path)
    paths = write_calibration_outputs(result)
    print("CALIBRATION: complete (dose/eligibility only)")
    for f in result.facts:
        print(
            f"  {f.fact_id}: status={f.status} alpha={f.selected_alpha} "
            f"baseline={f.baseline.status}"
        )
    print("outputs:")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
