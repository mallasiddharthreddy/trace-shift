# Delta-M analysis (balanced 6-fact / 3-domain)

- status: `complete`
- model_id: `Qwen/Qwen3-1.7B`
- design: `balanced_6fact_3domain`
- base matrix: `research/results/raw/base_matrix/base_matrix.json`
- FT matrix: `research/results/raw/ft_matrix/ft_matrix.json`
- AI-Engram / variant: see matrix metadata (must both be explicit / matching)

## Definitions

- `Delta_M[i,j] = M_FT[i,j] - M_base[i,j]` (diagonal null; full 6×6)
- Tennis within-domain (2 cells): F01→F02, F02→F01
- Unrelated within-domain (4 cells; swimming+acting): F03→F04, F04→F03, F05→F06, F06→F05
- Cross-domain (8 cells; secondary only): F01→F03, F01→F04, F01→F05, F01→F06, F02→F03, F02→F04, F02→F05, F02→F06
- Primary: `T_within = mean(tennis within) − mean(unrelated within)`
- Secondary: `T_cross = mean(tennis within) − mean(cross-domain)`

## Within-domain cell values

- `F01->F02`: `-0.001769`
- `F02->F01`: `0.020721`
- `F03->F04`: `0.010760`
- `F04->F03`: `-0.002868`
- `F05->F06`: `-0.016843`
- `F06->F05`: `-0.042939`

## Delta-M matrix (cosine-distance change)

Rows = intervened fact *i*; columns = measured fact *j*. `—` = diagonal.

| i \ j | F01 | F02 | F03 | F04 | F05 | F06 |
|-------|------|------|------|------|------|------|
| F01 | — | -0.001769 | 0.001404 | 0.002076 | 0.002939 | -0.000640 |
| F02 | 0.020721 | — | 0.000373 | -0.009413 | -0.002133 | 0.005711 |
| F03 | -0.010675 | -0.025438 | — | 0.010760 | -0.017737 | -0.014895 |
| F04 | -0.020476 | -0.056127 | -0.002868 | — | -0.015442 | -0.016532 |
| F05 | -0.004911 | 0.002211 | 0.001269 | 0.004767 | — | -0.016843 |
| F06 | -0.003709 | 0.003449 | 0.003645 | 0.001422 | -0.042939 | — |

## Primary group means and contrast

- mean tennis within Delta-M: `0.009476`
- mean unrelated within Delta-M: `-0.012972`
- observed contrast T_within / T_obs: `0.022448`

## Secondary descriptive (not primary inference)

- mean cross-domain Delta-M: `0.000040`
- T_cross: `0.009437`

## Exact cell-permutation test

- assignments in null distribution: `15` (all C(6,2)=15 choices of 2 of the 6 within-domain cells as tennis; remaining 4 = unrelated)
- exact one-sided p-value (T ≥ T_obs, including observed): `0.133333`

This is **not** the old C(4,2) fact-label swap of in-domain vs control facts.

### Null distribution (15 contrasts)

| tennis cells | unrelated cells | contrast | observed? |
|--------------|-----------------|----------|-----------|
| F01→F02, F02→F01 | F03→F04, F04→F03, F05→F06, F06→F05 | 0.022448 | yes |
| F01→F02, F03→F04 | F02→F01, F04→F03, F05→F06, F06→F05 | 0.014978 | no |
| F01→F02, F04→F03 | F02→F01, F03→F04, F05→F06, F06→F05 | 0.004757 | no |
| F01→F02, F05→F06 | F02→F01, F03→F04, F04→F03, F06→F05 | -0.005725 | no |
| F01→F02, F06→F05 | F02→F01, F03→F04, F04→F03, F05→F06 | -0.025297 | no |
| F02→F01, F03→F04 | F01→F02, F04→F03, F05→F06, F06→F05 | 0.031846 | no |
| F02→F01, F04→F03 | F01→F02, F03→F04, F05→F06, F06→F05 | 0.021625 | no |
| F02→F01, F05→F06 | F01→F02, F03→F04, F04→F03, F06→F05 | 0.011143 | no |
| F02→F01, F06→F05 | F01→F02, F03→F04, F04→F03, F05→F06 | -0.008429 | no |
| F03→F04, F04→F03 | F01→F02, F02→F01, F05→F06, F06→F05 | 0.014154 | no |
| F03→F04, F05→F06 | F01→F02, F02→F01, F04→F03, F06→F05 | 0.003672 | no |
| F03→F04, F06→F05 | F01→F02, F02→F01, F04→F03, F05→F06 | -0.015900 | no |
| F04→F03, F05→F06 | F01→F02, F02→F01, F03→F04, F06→F05 | -0.006549 | no |
| F04→F03, F06→F05 | F01→F02, F02→F01, F03→F04, F05→F06 | -0.026121 | no |
| F05→F06, F06→F05 | F01→F02, F02→F01, F03→F04, F04→F03 | -0.036602 | no |

## Interpretation placeholder

Positive T_within means tennis within-domain interactions changed more under tennis fine-tuning than unrelated within-domain interactions (intervention–response sense). Research write-up should interpret the exact p-value; this table does **not** auto-claim statistical significance.

## Design note

This is an exact cell-permutation test **conditional on the frozen balanced 6-fact / 3-domain design**. It does not claim population-level generalization from six facts.
