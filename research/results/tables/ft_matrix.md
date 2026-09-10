# Fine-tuned intervention–response matrix (`M_FT`)

- model_id: `Qwen/Qwen3-1.7B`
- model_state: `primary_finetuned`
- AI-Engram: `0.9.0`
- extraction_variant: `explicit`
- primary_metric: `cosine_distance = 1 - cosine_similarity(baseline_FT_j, post_FT_j | intervene_i)`
- diagonal: null (NaN): fact not compared to itself for primary interaction analysis
- source_checkpoint: `/home/siddharth/trace-shift/research/results/raw/finetuning/qwen3-1.7b-lora-tennis-primary/final_adapter`
- alpha_source: `/home/siddharth/trace-shift/research/results/raw/alpha_calibration/base_alphas.yaml` (BASE-calibrated; not recalibrated)

Definition: `M_FT[i,j]` = response of fact **j** after intervening on fact **i** in the primary fine-tuned model.

## Fact labels

| fact_id | label |
|---------|-------|
| F01 | Roger Federer — tennis |
| F02 | Rafael Nadal — tennis |
| F03 | Michael Phelps — swimming |
| F04 | Amitabh Bachchan — acting |

## Intervention alphas (rows; BASE-calibrated)

| row fact | label | status | alpha |
|----------|-------|--------|-------|
| F01 | Roger Federer — tennis | selective | 0.6 |
| F02 | Rafael Nadal — tennis | selective | 0.4 |
| F03 | Michael Phelps — swimming | selective | 0.4 |
| F04 | Amitabh Bachchan — acting | selective | 0.4 |

## Matrix values (cosine distance)

Rows = intervened fact *i*; columns = measured fact *j*. `—` = diagonal or ineligible row.

| i \ j | F01 | F02 | F03 | F04 |
|-------|------|------|------|------|
| F01 | — | 0.391161 | 0.090571 | 0.015317 |
| F02 | 0.401242 | — | 0.040217 | -0.006151 |
| F03 | 0.062770 | 0.042740 | — | -0.015896 |
| F04 | 0.005212 | -0.001587 | -0.009444 | — |

No scientific significance claims or delta interpretation in this table.
