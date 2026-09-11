# Generic FT control Delta-M (descriptive)

ΔM_control = M_control_FT − M_base (existing primary base matrix).
T_control is **descriptive only** — no second permutation test.

## Delta-M_control matrix

Rows = intervened fact *i*; columns = measured fact *j*. `—` = diagonal.

| i \ j | F01 | F02 | F03 | F04 | F05 | F06 |
|-------|------|------|------|------|------|------|
| F01 | — | 0.054645 | 0.004151 | 0.002270 | 0.003083 | 0.013335 |
| F02 | 0.008611 | — | 0.004065 | 0.002872 | -0.004556 | -0.002229 |
| F03 | -0.006034 | 0.022750 | — | 0.016231 | -0.001704 | 0.010843 |
| F04 | 0.009344 | 0.007658 | 0.011442 | — | -0.006465 | -0.007194 |
| F05 | -0.001964 | 0.001077 | 0.000809 | 0.002208 | — | -0.005764 |
| F06 | 0.000210 | 0.003475 | 0.003156 | 0.002878 | -0.017505 | — |

## Primary cell group means

Contrast: `T_generic_within = mean(tennis within) − mean(unrelated within)` on the 6×6 Delta-M_control matrix.

- mean tennis-within ΔM_control: `0.03162813186645508`
- mean unrelated-within ΔM_control: `0.0011012554168701172`
- **T_generic_within**: `0.03052687644958496`

## Comparison to tennis fine-tuning (descriptive)

- T_tennis (primary observed_contrast / T_within): `0.022448480129241943`
- T_generic_within: `0.03052687644958496`
- T_tennis − T_generic: `-0.008078396320343018`
- same-sign T: `True`
- |T_generic| < |T_tennis|: `False`

## Claim boundaries

Supports: quantitative descriptive structural comparison under identical 6-fact / 3-domain measurement procedure.

Does **not** support: ruling out generic FT drift as a causal claim; revising the primary C(6,2)=15 p-value; population generalization; definitively tennis-specific causation.

Primary T_tennis read dynamically from delta_analysis.json: `0.022448480129241943`.
