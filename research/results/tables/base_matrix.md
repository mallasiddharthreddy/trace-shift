# Base intervention–response matrix (`M_base`)

- model_id: `Qwen/Qwen3-1.7B`
- AI-Engram: `0.9.0`
- extraction_variant: `explicit`
- primary_metric: `cosine_distance = 1 - cosine_similarity(baseline_j, post_j | intervene_i)`
- diagonal: null (NaN): fact not compared to itself for primary interaction analysis

Definition: `M_base[i,j]` = response of fact **j** after intervening on fact **i**.

## Fact labels

| fact_id | label |
|---------|-------|
| F01 | Roger Federer — tennis |
| F02 | Rafael Nadal — tennis |
| F03 | Michael Phelps — swimming |
| F04 | Katie Ledecky — swimming |
| F05 | Amitabh Bachchan — acting |
| F06 | Shah Rukh Khan — acting |

## Intervention alphas (rows)

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
| F01 | — | 0.326948 | 0.033843 | 0.028348 | 0.015854 | 0.009606 |
| F02 | 0.335785 | — | 0.033515 | 0.032522 | 0.025615 | 0.014705 |
| F03 | 0.184481 | 0.171207 | — | 0.239425 | 0.073165 | 0.066560 |
| F04 | 0.170510 | 0.207571 | 0.258230 | — | 0.078965 | 0.075350 |
| F05 | 0.022336 | 0.014566 | -0.008496 | -0.015388 | — | 0.370535 |
| F06 | -0.003060 | -0.009064 | -0.028516 | -0.030228 | 0.263332 | — |

No scientific significance claims or tennis-vs-control interpretation in this table.
