# Delta-M analysis (primary contrast)

- status: `complete`
- model_id: `Qwen/Qwen3-1.7B`
- base matrix: `/home/siddharth/trace-shift/research/results/raw/base_matrix/base_matrix.json`
- FT matrix: `/home/siddharth/trace-shift/research/results/raw/ft_matrix/ft_matrix.json`
- AI-Engram / variant: see matrix metadata (must both be explicit / matching)

## Definitions

- `Delta_M[i,j] = M_FT[i,j] - M_base[i,j]` (diagonal null)
- In-domain facts: F01, F02
- Control facts: F03, F04
- In-domain pairs: F01→F02, F02→F01
- Control pairs (in-domain→control): F01→F03, F01→F04, F02→F03, F02→F04
- `T = mean(in-domain Delta-M) − mean(control Delta-M)`

## Delta-M matrix (cosine-distance change)

Rows = intervened fact *i*; columns = measured fact *j*. `—` = diagonal.

| i \ j | F01 | F02 | F03 | F04 |
|-------|------|------|------|------|
| F01 | — | 0.032833 | 0.013795 | -0.001675 |
| F02 | 0.017529 | — | -0.009734 | -0.006641 |
| F03 | -0.000213 | -0.001412 | — | 0.001945 |
| F04 | -0.006802 | 0.001638 | 0.003445 | — |

## Primary group means and contrast

- mean in-domain Delta-M: `0.025181`
- mean control Delta-M: `-0.001064`
- observed contrast T: `0.026245`

## Exact permutation test

- assignments in null distribution: `6` (all size-(2,2) labelings of the four facts)
- exact one-sided p-value (T ≥ T_obs, including observed): `0.166667`

### Null distribution

| in-domain | control | contrast | observed? |
|-----------|---------|----------|-----------|
| F01, F02 | F03, F04 | 0.026245 | yes |
| F01, F03 | F02, F04 | -0.001132 | no |
| F01, F04 | F02, F03 | -0.017166 | no |
| F02, F03 | F01, F04 | -0.008728 | no |
| F02, F04 | F01, F03 | -0.003611 | no |
| F03, F04 | F01, F02 | 0.004392 | no |

## Interpretation placeholder

Positive T means the selected in-domain interactions changed more under tennis fine-tuning than the selected in-domain→control interactions (intervention–response sense). Research write-up should interpret the exact p-value; this table does **not** auto-claim statistical significance.

## Design note

This is an exact randomization test **conditional on the frozen four-fact design**. It does not claim population-level generalization from four facts.
