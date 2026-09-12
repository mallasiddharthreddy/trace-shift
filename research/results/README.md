# Canonical results (completed 6-fact experiment)

Do **not** regenerate these files for routine inspection. They are the frozen outputs of the completed run.

## Where to look

| Stage | Machine-readable | Human table |
|-------|------------------|-------------|
| Baseline cloze recall | `raw/baseline_recall/baseline_cloze.json` | — |
| Alpha calibration | `raw/alpha_calibration/base_alphas.yaml`, `base_alpha_calibration.json` | `tables/alpha_calibration_summary.md` |
| Base matrix `M_base` | `raw/base_matrix/base_matrix.json` | `tables/base_matrix.md` |
| Tennis LoRA training metadata | `raw/finetuning/qwen3-1.7b-lora-tennis-primary/` | — |
| Generic-control LoRA metadata | `raw/finetuning/qwen3-1.7b-lora-control/` | — |
| Tennis FT matrix `M_FT` | `raw/ft_matrix/ft_matrix.json` | `tables/ft_matrix.md` |
| Tennis ΔM + exact permutation | `raw/delta_analysis/delta_analysis.json` | `tables/delta_analysis.md` |
| Generic-control matrix | `raw/alternative_hypotheses/generic_ft_control_matrix.json` | `tables/generic_ft_control_matrix.md` |
| Generic-control ΔM | `raw/alternative_hypotheses/generic_ft_control_delta_analysis.json` | `tables/generic_ft_control_delta_analysis.md` |
| Lexical extraction control | `raw/alternative_hypotheses/lexical_control.json` | `tables/lexical_control.md` |
| Run provenance / phase markers | `raw/experiment_run/` | — |

Adapter weights (`final_adapter/`, checkpoints) are **not** in git (see root `.gitignore`). Training metadata and execution-validity JSON are.

## Design note

All matrices above are **6×6** (`balanced_6fact_3domain`). Older 4-fact outputs are not in this tree; see `research/archive/README.md` and Git history.

`figures/` is reserved for optional plots; this repository ships tables/JSON as the primary deliverable.
