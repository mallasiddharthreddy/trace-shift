# Methodology (canonical)

Authoritative design for the completed TraceShift experiment. Frozen inputs live under `research/data/frozen/` and `research/configs/experiment.yaml`.

## Question

Does narrow LoRA fine-tuning on tennis biographies selectively change how related factual traces interact under AI-Engram interventions, relative to unrelated within-domain pairs?

## Model and instrument

- Base model: `Qwen/Qwen3-1.7B`
- Instrument: AI-Engram **v0.9.0** (`get_engram` / `apply_engram`)
- Target modules: all 28 `model.layers.{0–27}.mlp.down_proj`
- Primary metric: cosine distance `1 − cos` between fact *j*’s baseline trace and its post-intervention trace after intervening on fact *i*

## Facts (balanced 6-fact / 3-domain)

| ID | Entity | Domain | Role |
|----|--------|--------|------|
| F01 | Roger Federer | tennis | FT target |
| F02 | Rafael Nadal | tennis | FT target |
| F03 | Michael Phelps | swimming | within-domain control |
| F04 | Katie Ledecky | swimming | within-domain control |
| F05 | Amitabh Bachchan | acting | within-domain control |
| F06 | Shah Rukh Khan | acting | within-domain control |

## Extraction and references

- Per fact: **5** held-out extraction sentences (variants: `explicit`, `lexical_control`)
- Shared reference corpus per variant: **30** sentences (concatenation of the six target sets)
- Extraction text is held out from fine-tuning

## Cloze / alpha calibration

- 3 cloze probes per fact; **≥2 of 3** correct ⇒ recall PASS
- Alphas calibrated **once on `M_base`** (smallest selective α on a fixed grid) and **held fixed** for all matrix comparisons
- Primary matrices use **explicit** extraction; lexical condition reuses the same alphas

## Matrices and ΔM

- `M_base[i,j]`, `M_FT[i,j]`: response of fact *j* after intervening on fact *i* (diagonal null)
- `ΔM = M_FT − M_base` (cell-wise)
- Tennis FT and generic-control FT use the **same** `M_base`

## Fine-tuning

- LoRA: r=16, α=32, dropout 0.05, targets `q/k/v/o_proj`
- 3 epochs, LR `2e-4`, seed `42`, identical procedure for both runs
- Primary corpus: 40 Federer/Nadal biography sentences
- Generic control: 40 Curie/Armstrong biography sentences (only corpus differs)

## Primary statistic

\[
T = \overline{\Delta M}_{\text{tennis within}} - \overline{\Delta M}_{\text{unrelated within}}
\]

- Tennis within (2 cells): F01→F02, F02→F01
- Unrelated within (4 cells): F03↔F04, F05↔F06
- Exact one-sided permutation: all \(\binom{6}{2}=15\) ways to assign which 2 of the 6 within-domain cells are “tennis”; p = fraction with \(T \ge T_{\mathrm{obs}}\)

`T_generic` and `T_lexical` use the **same formula** but are **descriptive only** (no second permutation test). Lexical = tennis FT vs base under lexical extraction wording, not a third fine-tune.

## What this does / does not claim

Supports: descriptive and exact-permutation evidence **conditional on these six frozen facts** that fine-tuning changes measured intervention–response structure, and that a tennis-specific semantic story is **not** uniquely supported once generic FT and lexical controls are inspected.

Does not support: population generalization; causal proof that tennis content alone drove `T_obs`; significance of the control contrasts.
