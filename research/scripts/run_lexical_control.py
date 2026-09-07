#!/usr/bin/env python3
"""Lexical-control alternative-hypothesis CLI for TraceShift.

Secondary analysis: same models/BASE alphas/metrics; lexical_control extraction only.

Usage:
  .venv/bin/python research/scripts/run_lexical_control.py --dry-validate
  .venv/bin/python research/scripts/run_lexical_control.py
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
    from traceshift.lexical_control import (
        LexicalControlError,
        run_lexical_control_experiment,
        validate_lexical_control_without_model,
        write_lexical_control_outputs,
        write_pending_status,
    )
    from traceshift.model_loader import ModelWeightsUnavailableError
    from traceshift.paths import default_experiment_config

    parser = argparse.ArgumentParser(
        description=(
            "Lexical-control secondary experiment (BASE+FT matrices, Delta-M_lexical, "
            "descriptive comparison to primary explicit Delta-M)."
        )
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--alphas",
        type=Path,
        default=None,
        help="BASE explicit-calibrated alphas (not lexical-recalibrated)",
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

    print("=== TraceShift lexical-control alternative hypothesis ===")
    print(f"config: {config_path}")
    print("variant: lexical_control + shared_reference_lexical_control")
    print("alphas: BASE explicit-calibrated (NOT recalibrated on lexical text)")
    print("role: secondary sensitivity analysis (does not replace primary Delta-M test)")
    print()

    if args.dry_validate:
        report = validate_lexical_control_without_model(config_path)
        status_path = write_pending_status(report)
        print(json.dumps(report, indent=2))
        print()
        print(f"wrote pending status: {status_path}")
        print("DRY VALIDATE: OK (no model / Engram / matrix / FT execution)")
        return 0

    overrides = tuple(
        s.strip() for s in args.allow_override_statuses.split(",") if s.strip()
    )
    try:
        payload = run_lexical_control_experiment(
            config_path=config_path,
            alphas_path=args.alphas,
            device=args.device,
            allow_override=bool(overrides),
            override_statuses=overrides,
        )
    except (ModelWeightsUnavailableError, FinetunedCheckpointUnavailableError) as exc:
        report = validate_lexical_control_without_model(config_path)
        status_path = write_pending_status(report)
        print("LOAD: PENDING — weights and/or primary FT checkpoint unavailable")
        print(str(exc))
        print()
        print(json.dumps(report, indent=2))
        print()
        print(f"wrote pending status: {status_path}")
        print("No lexical matrix values fabricated.")
        return 0
    except LexicalControlError as exc:
        print(f"LEXICAL CONTROL: REFUSED — {exc}", file=sys.stderr)
        write_pending_status()
        return 1

    paths = write_lexical_control_outputs(payload)
    print("LEXICAL CONTROL: complete (descriptive; no auto significance claim)")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
