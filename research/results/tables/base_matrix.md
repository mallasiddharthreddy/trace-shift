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
| F04 | Amitabh Bachchan — acting |

## Intervention alphas (rows)

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
| F01 | — | 0.358328 | 0.076776 | 0.016992 |
| F02 | 0.383712 | — | 0.049951 | 0.000490 |
| F03 | 0.062983 | 0.044152 | — | -0.017841 |
| F04 | 0.012014 | -0.003225 | -0.012890 | — |

No scientific significance claims or tennis-vs-control interpretation in this table.
