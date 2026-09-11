# Generic FT control intervention–response matrix (`M_control_FT`)

Alternative hypothesis A: unrelated Curie/Armstrong fine-tuning drift.
Same measurement procedure as primary explicit FT matrix; different model state.

- model_id: `Qwen/Qwen3-1.7B`
- model_state: `control_finetuned`
- AI-Engram: `0.9.0`
- extraction_variant: `explicit`
- primary_metric: `cosine_distance = 1 - cosine_similarity(baseline_FT_j, post_FT_j | intervene_i)`
- diagonal: null (NaN): fact not compared to itself for primary interaction analysis
- source_checkpoint: `/home/siddharth/trace-shift/research/results/raw/finetuning/qwen3-1.7b-lora-control/final_adapter`
- alpha_source: `/home/siddharth/trace-shift/research/results/raw/alpha_calibration/base_alphas.yaml` (BASE-calibrated; not recalibrated)
- device: `cuda`

## Fact labels

| fact_id | label |
|---------|-------|
| F01 | Roger Federer — tennis |
| F02 | Rafael Nadal — tennis |
| F03 | Michael Phelps — swimming |
| F04 | Katie Ledecky — swimming |
| F05 | Amitabh Bachchan — acting |
| F06 | Shah Rukh Khan — acting |

## Intervention alphas (rows; BASE-calibrated)

| row fact | status | alpha |
|----------|--------|-------|
| F01 | selective | 0.4 |
| F02 | selective | 0.4 |
| F03 | selective | 0.8 |
| F04 | selective | 0.8 |
| F05 | selective | 0.4 |
| F06 | selective | 0.2 |

## Matrix values (cosine distance)

Rows = intervened fact *i*; columns = measured fact *j*. `—` = diagonal.

| i \ j | F01 | F02 | F03 | F04 | F05 | F06 |
|-------|------|------|------|------|------|------|
| F01 | — | 0.381594 | 0.037994 | 0.030618 | 0.018937 | 0.022941 |
| F02 | 0.344395 | — | 0.037580 | 0.035393 | 0.021059 | 0.012477 |
| F03 | 0.178446 | 0.193957 | — | 0.255656 | 0.071460 | 0.077403 |
| F04 | 0.179854 | 0.215229 | 0.269672 | — | 0.072500 | 0.068155 |
| F05 | 0.020372 | 0.015643 | -0.007687 | -0.013179 | — | 0.364771 |
| F06 | -0.002850 | -0.005589 | -0.025360 | -0.027349 | 0.245827 | — |

No significance claims. Does not replace primary tennis `M_FT`.
