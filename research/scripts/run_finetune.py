#!/usr/bin/env python3
"""TraceShift LoRA fine-tuning CLI (primary or control).

Dataset selection is explicit — no silent primary↔control fallback.

Usage:
  .venv/bin/python research/scripts/run_finetune.py --dataset primary --dry-validate
  .venv/bin/python research/scripts/run_finetune.py --dataset control --dry-validate
  .venv/bin/python research/scripts/run_finetune.py --dataset control   # when weights exist
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
    from traceshift.finetune import (
        FinetuneConfigError,
        run_finetune,
        validate_finetune_pipeline_without_model,
        write_control_status,
        write_pending_status,
    )
    from traceshift.finetune_data import FinetuneDataError
    from traceshift.model_loader import ModelWeightsUnavailableError
    from traceshift.paths import default_experiment_config

    parser = argparse.ArgumentParser(
        description=(
            "LoRA SFT for TraceShift. Requires explicit --dataset primary|control. "
            "control → frozen Curie/Armstrong corpus only (unrelated FT control)."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=("primary", "control"),
        help=(
            "Which frozen corpus to train on (required; no default fallback). "
            "'control' maps only to control_finetuning_dataset_spec.yaml."
        ),
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--dry-validate",
        action="store_true",
        help="Validate config/dataset/hparams without loading weights or training",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)
    config_path = args.config or default_experiment_config()
    kind = args.dataset  # type: ignore[assignment]

    print("=== TraceShift fine-tuning ===")
    print(f"config: {config_path}")
    print(f"dataset selection: {kind} (explicit)")
    if kind == "control":
        print("role: unrelated fine-tuning control (alt hypothesis A)")
        print("dataset file: control_finetuning_dataset_spec.yaml only")
    print("identical hparams for primary/control; corpus is the only intended difference")
    print()

    if args.dry_validate:
        try:
            report = validate_finetune_pipeline_without_model(
                kind, config_path=config_path
            )
        except (FinetuneConfigError, FinetuneDataError) as exc:
            print(f"DRY VALIDATE: FAIL — {exc}", file=sys.stderr)
            return 1

        # Always print comparability for either selection
        print(report.get("comparability_report_text", ""))
        print()
        print(json.dumps(report, indent=2))
        print()

        if kind == "control":
            status_path = write_control_status(report=report)
            print(f"wrote control status (primary STATUS untouched): {status_path}")
        else:
            status_path = write_pending_status(kind, report=report)
            print(f"wrote pending status: {status_path}")
        print("DRY VALIDATE: OK (no model load / training)")
        return 0

    # Non-dry path: attempt load; if weights missing, dry-validate + pending only.
    try:
        result = run_finetune(kind, config_path=config_path, device=args.device)
    except ModelWeightsUnavailableError as exc:
        report = validate_finetune_pipeline_without_model(kind, config_path=config_path)
        print("MODEL LOAD: PENDING — weights unavailable locally")
        print(str(exc))
        print()
        print(report.get("comparability_report_text", ""))
        print()
        print(json.dumps(report, indent=2))
        print()
        if kind == "control":
            status_path = write_control_status(report=report)
            print(f"wrote control status: {status_path}")
        else:
            status_path = write_pending_status(kind, report=report)
            print(f"wrote pending status: {status_path}")
        print("No training results fabricated.")
        return 0
    except (FinetuneConfigError, FinetuneDataError) as exc:
        print(f"FINETUNE: REFUSED — {exc}", file=sys.stderr)
        return 1

    print("FINETUNE: complete")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
