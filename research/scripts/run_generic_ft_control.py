#!/usr/bin/env python3
"""Generic fine-tuning drift control (alt hypothesis A) CLI for TraceShift.

Loads the already-trained Curie/Armstrong control adapter. Does NOT retrain.
Computes M_control_FT + descriptive Delta-M_control vs existing M_base.
Does NOT modify primary results or run a second permutation test.

Usage:
  .venv/bin/python research/scripts/run_generic_ft_control.py --dry-validate
  .venv/bin/python research/scripts/run_generic_ft_control.py
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
    from traceshift.ft_matrix import FinetunedCheckpointUnavailableError
    from traceshift.generic_ft_control import (
        GenericFtControlError,
        run_generic_ft_control_experiment,
        validate_generic_ft_control_without_model,
        write_pending_status,
    )
    from traceshift.model_loader import ModelWeightsUnavailableError
    from traceshift.paths import default_experiment_config

    parser = argparse.ArgumentParser(
        description=(
            "Generic FT drift control: M_control_FT + descriptive Delta-M_control "
            "using existing Curie/Armstrong adapter (no retrain)."
        )
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--alphas",
        type=Path,
        default=None,
        help="BASE explicit-calibrated alphas (not recalibrated on control model)",
    )
    parser.add_argument("--dry-validate", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--allow-override-statuses",
        type=str,
        default="",
        help="Comma-separated eligibility overrides (recorded; default selective only)",
    )
    args = parser.parse_args(argv)
    config_path = args.config or default_experiment_config()

    print("=== TraceShift generic fine-tuning drift control ===")
    print(f"config: {config_path}")
    print("model_state: control_finetuned (Curie/Armstrong LoRA)")
    print("alphas: BASE explicit-calibrated (NOT recalibrated)")
    print("role: descriptive alt-hypothesis A (does not replace primary p=1/6)")
    print("retraining: FORBIDDEN")
    print()

    if args.dry_validate:
        report = validate_generic_ft_control_without_model(config_path)
        status_path = write_pending_status(report)
        print(json.dumps(report, indent=2))
        print()
        print(f"wrote pending status: {status_path}")
        print("DRY VALIDATE: OK (no model / Engram / matrix execution)")
        return 0

    overrides = tuple(
        s.strip() for s in args.allow_override_statuses.split(",") if s.strip()
    )
    try:
        out = run_generic_ft_control_experiment(
            config_path=config_path,
            alphas_path=args.alphas,
            device=args.device,
            allow_override=bool(overrides),
            override_statuses=overrides,
        )
    except (ModelWeightsUnavailableError, FinetunedCheckpointUnavailableError) as exc:
        report = validate_generic_ft_control_without_model(config_path)
        status_path = write_pending_status(report)
        print(f"LOAD: PENDING/FAIL — {exc}")
        print(json.dumps(report, indent=2))
        print(f"wrote pending status: {status_path}")
        return 1
    except (GenericFtControlError, Exception) as exc:  # noqa: BLE001
        print(f"GENERIC FT CONTROL: FAIL — {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    delta = out["delta"]
    print("GENERIC FT CONTROL: complete (descriptive only)")
    print(f"adapter: {out['adapter_path']}")
    print(f"T_control: {delta['T_control']}")
    print(f"T_tennis: {delta['T_tennis']}")
    print(f"T_tennis - T_control: {delta['T_tennis_minus_T_control']}")
    print(f"mean in-domain ΔM_control: {delta['mean_in_domain_delta_m_control']}")
    print(f"mean control ΔM_control: {delta['mean_control_delta_m_control']}")
    for k, p in out["paths"].items():
        print(f"  {k}: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
