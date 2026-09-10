# Lexical-control alternative hypothesis

- status: `complete`
- model_id: `Qwen/Qwen3-1.7B`
- extraction_variant: `lexical_control`
- reference: `shared_reference_lexical_control`
- AI-Engram: `0.9.0`
- alpha_policy: Reuse BASE explicit-calibrated alphas; do not recalibrate on lexical_control.
- fixed alphas: `{'F01': 0.6, 'F02': 0.4, 'F03': 0.4, 'F04': 0.4}`

Secondary analysis: same models/alphas/metrics; only extraction/reference text is lexical_control. Does **not** replace the primary explicit Delta-M test.

## Base lexical-control matrix (`M_base_lexical`)

| i \ j | F01 | F02 | F03 | F04 |
|-------|------|------|------|------|
| F01 | — | 0.347177 | 0.045661 | 0.018219 |
| F02 | 0.339347 | — | 0.028978 | 0.013358 |
| F03 | 0.019039 | 0.012096 | — | -0.006543 |
| F04 | 0.010137 | 0.011494 | 0.010161 | — |

## FT lexical-control matrix (`M_FT_lexical`)

| i \ j | F01 | F02 | F03 | F04 |
|-------|------|------|------|------|
| F01 | — | 0.325130 | 0.046933 | 0.016992 |
| F02 | 0.329380 | — | 0.026447 | 0.006935 |
| F03 | 0.018723 | 0.009426 | — | -0.007086 |
| F04 | 0.003758 | 0.011938 | 0.006629 | — |

## Lexical Delta-M (`M_FT_lexical − M_base_lexical`)

| i \ j | F01 | F02 | F03 | F04 |
|-------|------|------|------|------|
| F01 | — | -0.022047 | 0.001272 | -0.001227 |
| F02 | -0.009968 | — | -0.002532 | -0.006422 |
| F03 | -0.000316 | -0.002671 | — | -0.000542 |
| F04 | -0.006379 | 0.000443 | -0.003532 | — |

## Descriptive comparison to primary explicit Delta-M

- lexical mean in-domain ΔM: `-0.01600739359855652`
- lexical mean control ΔM: `-0.0022272318601608276`
- lexical contrast T: `-0.013780161738395691`

- explicit mean in-domain ΔM: `0.02518138289451599`
- explicit mean control ΔM: `-0.0010637640953063965`
- explicit contrast T: `0.026245146989822388`
- same-sign tennis-vs-control pattern: `False`
- T_lexical − T_explicit: `-0.04002530872821808`

No automatic significance claim. Research write-up should interpret whether the tennis-vs-control pattern remains and how magnitude shifts.
