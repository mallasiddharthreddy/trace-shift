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
| F04 | Amitabh Bachchan — acting |

## Intervention alphas (rows; BASE-calibrated)

| row fact | status | alpha |
|----------|--------|-------|
| F01 | selective | 0.6 |
| F02 | selective | 0.4 |
| F03 | selective | 0.4 |
| F04 | selective | 0.4 |

## Matrix values (cosine distance)

Rows = intervened fact *i*; columns = measured fact *j*. `—` = diagonal.

| i \ j | F01 | F02 | F03 | F04 |
|-------|------|------|------|------|
| F01 | — | 0.374385 | 0.078445 | 0.008225 |
| F02 | 0.398375 | — | 0.060899 | -0.006322 |
| F03 | 0.053895 | 0.042390 | — | -0.022012 |
| F04 | 0.006763 | -0.008687 | -0.014484 | — |

No significance claims. Does not replace primary tennis `M_FT`.
