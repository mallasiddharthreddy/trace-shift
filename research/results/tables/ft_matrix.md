# Fine-tuned intervention–response matrix (`M_FT`)

- model_id: `Qwen/Qwen3-1.7B`
- model_state: `primary_finetuned`
- AI-Engram: `0.9.0`
- extraction_variant: `explicit`
- primary_metric: `cosine_distance = 1 - cosine_similarity(baseline_FT_j, post_FT_j | intervene_i)`
- diagonal: null (NaN): fact not compared to itself for primary interaction analysis
- source_checkpoint: `research/results/raw/finetuning/qwen3-1.7b-lora-tennis-primary/final_adapter`
- alpha_source: `research/results/raw/alpha_calibration/base_alphas.yaml` (BASE-calibrated; not recalibrated)

Definition: `M_FT[i,j]` = response of fact **j** after intervening on fact **i** in the primary fine-tuned model.

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

| row fact | label | status | alpha |
|----------|-------|--------|-------|
| F01 | Roger Federer — tennis | selective | 0.4 |
| F02 | Rafael Nadal — tennis | selective | 0.4 |
| F03 | Michael Phelps — swimming | selective | 0.8 |
| F04 | Katie Ledecky — swimming | selective | 0.8 |
| F05 | Amitabh Bachchan — acting | selective | 0.4 |
| F06 | Shah Rukh Khan — acting | selective | 0.2 |

## Matrix values (cosine distance)

Rows = intervened fact *i*; columns = measured fact *j*. `—` = diagonal or ineligible row.

| i \ j | F01 | F02 | F03 | F04 | F05 | F06 |
|-------|------|------|------|------|------|------|
| F01 | — | 0.325179 | 0.035246 | 0.030424 | 0.018793 | 0.008967 |
| F02 | 0.356506 | — | 0.033887 | 0.023108 | 0.023482 | 0.020417 |
| F03 | 0.173805 | 0.145769 | — | 0.250185 | 0.055427 | 0.051665 |
| F04 | 0.150034 | 0.151444 | 0.255362 | — | 0.063523 | 0.058817 |
| F05 | 0.017425 | 0.016777 | -0.007228 | -0.010621 | — | 0.353692 |
| F06 | -0.006768 | -0.005615 | -0.024871 | -0.028805 | 0.220393 | — |

No scientific significance claims or delta interpretation in this table.
