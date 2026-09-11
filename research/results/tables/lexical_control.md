# Lexical-control alternative hypothesis

- status: `complete`
- model_id: `Qwen/Qwen3-1.7B`
- extraction_variant: `lexical_control`
- reference: `shared_reference_lexical_control`
- AI-Engram: `0.9.0`
- alpha_policy: Reuse BASE explicit-calibrated alphas; do not recalibrate on lexical_control.
- fixed alphas: `{'F01': 0.4, 'F02': 0.4, 'F03': 0.8, 'F04': 0.8, 'F05': 0.4, 'F06': 0.2}`

Secondary analysis: same models/alphas/metrics; only extraction/reference text is lexical_control. Does **not** replace the primary explicit Delta-M test.

## Base lexical-control matrix (`M_base_lexical`)

| i \ j | F01 | F02 | F03 | F04 | F05 | F06 |
|-------|------|------|------|------|------|------|
| F01 | — | 0.298442 | 0.017314 | 0.007554 | 0.010365 | 0.003708 |
| F02 | 0.293698 | — | 0.019534 | 0.011946 | 0.022370 | 0.016082 |
| F03 | 0.105600 | 0.093650 | — | 0.301799 | 0.064429 | 0.060122 |
| F04 | 0.084697 | 0.097254 | 0.295792 | — | 0.071290 | 0.071742 |
| F05 | 0.004813 | 0.001959 | -0.006438 | -0.014354 | — | 0.275535 |
| F06 | -0.020067 | -0.023727 | -0.030466 | -0.035612 | 0.191230 | — |

## FT lexical-control matrix (`M_FT_lexical`)

| i \ j | F01 | F02 | F03 | F04 | F05 | F06 |
|-------|------|------|------|------|------|------|
| F01 | — | 0.263432 | 0.018418 | 0.009194 | 0.005945 | -0.000860 |
| F02 | 0.283201 | — | 0.025118 | 0.014927 | 0.013069 | 0.016251 |
| F03 | 0.102513 | 0.086433 | — | 0.302791 | 0.056292 | 0.051988 |
| F04 | 0.086824 | 0.087367 | 0.288737 | — | 0.065277 | 0.064582 |
| F05 | 0.000864 | 0.007260 | -0.006176 | -0.009265 | — | 0.261226 |
| F06 | -0.026214 | -0.002397 | -0.034129 | -0.035299 | 0.152899 | — |

## Lexical Delta-M (`M_FT_lexical − M_base_lexical`)

| i \ j | F01 | F02 | F03 | F04 | F05 | F06 |
|-------|------|------|------|------|------|------|
| F01 | — | -0.035010 | 0.001104 | 0.001640 | -0.004419 | -0.004568 |
| F02 | -0.010497 | — | 0.005583 | 0.002982 | -0.009301 | 0.000169 |
| F03 | -0.003088 | -0.007217 | — | 0.000991 | -0.008137 | -0.008134 |
| F04 | 0.002126 | -0.009887 | -0.007055 | — | -0.006013 | -0.007160 |
| F05 | -0.003949 | 0.005301 | 0.000262 | 0.005090 | — | -0.014308 |
| F06 | -0.006146 | 0.021330 | -0.003663 | 0.000314 | -0.038331 | — |

## Descriptive comparison to primary explicit Delta-M

Primary contrast (balanced 6-fact / 3-domain): `T_within = mean(tennis within) − mean(unrelated within)` (swimming+acting). Matrices are 6×6.

- lexical mean tennis-within ΔM: `-0.022753626108169556`
- lexical mean unrelated-within ΔM: `-0.01467578113079071`
- **T_lexical_within**: `-0.008077844977378845`
- T_lexical_cross (secondary): `-0.021902307868003845`

- explicit mean tennis-within ΔM: `0.00947609543800354`
- explicit mean unrelated-within ΔM: `-0.012972384691238403`
- **T_explicit_within**: `0.022448480129241943`
- T_explicit_cross (secondary): `0.009436555206775665`
- same-sign T_within: `False`
- T_lexical_within − T_explicit_within: `-0.03052632510662079`

No automatic significance claim. Research write-up should interpret whether the tennis-vs-unrelated-within pattern remains and how magnitude shifts under lexical_control extraction.
