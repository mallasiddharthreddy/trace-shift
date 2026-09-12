# TraceShift research tree

Completed **balanced 6-fact / 3-domain** experiment. Start at the [repository root README](../README.md).

## Layout

```
research/
├── configs/          # Frozen experiment + environment config
├── data/frozen/      # Facts, extraction, cloze, FT corpora, references
├── traceshift/       # Library (matrices, Engram, FT, ΔM, controls)
├── scripts/          # CLI entry points (validate / run stages)
├── results/          # Canonical JSON + markdown tables (see results/README.md)
├── reports/          # Methodology, evidence audit, full-matrix comparison
└── archive/          # Historical pre-GPU / overnight tooling (not primary cites)
```

## Canonical documents

| Doc | Role |
|-----|------|
| [reports/METHODOLOGY.md](reports/METHODOLOGY.md) | Design, metrics, statistics |
| [reports/final_evidence_audit.md](reports/final_evidence_audit.md) | Completion / validity checklist |
| [reports/tennis_vs_generic_delta_m_full_matrix_comparison.md](reports/tennis_vs_generic_delta_m_full_matrix_comparison.md) | Cell/group tennis vs generic ΔM |
| [results/README.md](results/README.md) | Index of result files |
| [archive/README.md](archive/README.md) | What is historical |

## Inspecting outputs (no GPU)

```bash
# Frozen-spec integrity (no model download)
python research/scripts/validate_frozen_experiment.py

# Human tables
ls research/results/tables/

# Primary numbers
python -c "import json; d=json.load(open('research/results/raw/delta_analysis/delta_analysis.json')); print(d['observed_contrast'], d.get('exact_one_sided_p_value') or d.get('p_exact'))"
```

Full matrix / Engram regeneration needs local `Qwen/Qwen3-1.7B`, CUDA, and LoRA adapters (adapters are not in git).

## Status

Scientific run **complete**. Do not treat `archive/` pre-GPU notes or Git-history 4-fact outputs as the final result.
