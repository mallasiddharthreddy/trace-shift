#!/usr/bin/env python3
"""CLI for TraceShift model load + frozen cloze scoring (engineering validation).

Does not run AI-Engram, fine-tuning, alpha calibration, or matrices.
Does not download model weights.

Usage (from repo root, with .venv active):
  python research/scripts/run_baseline_cloze.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_pkg_path() -> Path:
    research = Path(__file__).resolve().parent.parent
    if str(research) not in sys.path:
        sys.path.insert(0, str(research))
    return research


def main(argv: list[str] | None = None) -> int:
    _ensure_pkg_path()
    from traceshift.cloze import evaluate_all_cloze, format_evaluation_report
    from traceshift.model_loader import (
        ModelWeightsUnavailableError,
        load_model,
        reproducibility_record,
    )
    from traceshift.paths import default_cloze_probes, default_experiment_config

    parser = argparse.ArgumentParser(
        description="Load frozen Qwen config and score TraceShift cloze probes."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment.yaml (default: research/configs/experiment.yaml)",
    )
    parser.add_argument(
        "--cloze",
        type=Path,
        default=None,
        help="Path to cloze_probes.yaml (default: research/data/frozen/cloze_probes.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device: cuda | mps | cpu (default: auto)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write machine-readable results JSON",
    )
    args = parser.parse_args(argv)

    config_path = args.config or default_experiment_config()
    cloze_path = args.cloze or default_cloze_probes()

    print("=== TraceShift inference layer ===")
    print(f"config: {config_path}")
    print(f"cloze:  {cloze_path}")
    print("policy: no weight download; no AI-Engram / FT / alpha / matrix")
    print()

    try:
        loaded = load_model(config_path, device=args.device, local_files_only=True)
    except ModelWeightsUnavailableError as exc:
        print("MODEL LOAD: PENDING — weights unavailable locally")
        print(str(exc))
        print()
        print(
            "Implemented: reusable loader + deterministic cloze scorer. "
            "Execution remains pending until Qwen/Qwen3-1.7B is available locally."
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"MODEL LOAD: FAIL — {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    repro = reproducibility_record(loaded)
    print("MODEL LOAD: OK")
    print(f"  model_id:     {repro['model_id']}")
    print(f"  tokenizer_id: {repro['tokenizer_id']}")
    print(f"  device:       {repro['device']}")
    print(f"  dtype:        {repro['dtype']}")
    print(f"  source:       {repro['source_path']}")
    print(f"  software:     {repro['software']}")
    print()

    evaluation = evaluate_all_cloze(loaded, cloze_path)
    print(format_evaluation_report(evaluation, repro))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"reproducibility": repro, "evaluation": evaluation.to_dict()}
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
