# Generic FT control Delta-M (descriptive)

ΔM_control = M_control_FT − M_base (existing primary base matrix).
T_control is **descriptive only** — no second permutation test.

## Delta-M_control matrix

Rows = intervened fact *i*; columns = measured fact *j*. `—` = diagonal.

| i \ j | F01 | F02 | F03 | F04 |
|-------|------|------|------|------|
| F01 | — | 0.016057 | 0.001669 | -0.008767 |
| F02 | 0.014663 | — | 0.010948 | -0.006812 |
| F03 | -0.009088 | -0.001762 | — | -0.004171 |
| F04 | -0.005251 | -0.005462 | -0.001595 | — |

## Primary cell group means

- mean in-domain ΔM_control: `0.015360027551651001`
- mean control ΔM_control: `-0.0007405579090118408`
- **T_control**: `0.016100585460662842`

## Comparison to tennis fine-tuning (descriptive)

- T_tennis (primary, unchanged): `0.026245146989822388`
- T_control: `0.016100585460662842`
- T_tennis − T_control: `0.010144561529159546`
- same-sign T: `True`
- |T_control| < |T_tennis|: `True`

## Claim boundaries

Supports: quantitative descriptive structural comparison under identical measurement procedure.

Does **not** support: ruling out generic FT drift as a causal claim; revising the primary p=1/6; population generalization; definitively tennis-specific causation.

Primary result remains: T_explicit = 0.026245146989822388, p = 0.166667.
