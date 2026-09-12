# Full-matrix comparison: tennis FT vs generic FT Delta-M

Canonical descriptive comparison of **existing** completed 6×6 matrices only.
No new experiments, p-values, or primary-statistic changes.
Sources:
- tennis ΔM: `research/results/raw/delta_analysis/delta_analysis.json`
- generic ΔM: `research/results/raw/alternative_hypotheses/generic_ft_control_delta_analysis.json`
- lexical ΔM (context only): `research/results/raw/alternative_hypotheses/lexical_control.json`
- shared base: `research/results/raw/base_matrix/base_matrix.json`
- tennis M_FT / generic M_FT: `ft_matrix.json` / `generic_ft_control_matrix.json`

Definition: `diff[i,j] = ΔM_tennis[i,j] − ΔM_generic[i,j]`.

## Reference matrices (existing)

### Tennis ΔM
| i\\j | F01 | F02 | F03 | F04 | F05 | F06 |
|-------|------|------|------|------|------|------|
| F01 | — | -0.001769 | 0.001404 | 0.002076 | 0.002939 | -0.000640 |
| F02 | 0.020721 | — | 0.000373 | -0.009413 | -0.002133 | 0.005711 |
| F03 | -0.010675 | -0.025438 | — | 0.010760 | -0.017737 | -0.014895 |
| F04 | -0.020476 | -0.056127 | -0.002868 | — | -0.015442 | -0.016532 |
| F05 | -0.004911 | 0.002211 | 0.001269 | 0.004767 | — | -0.016843 |
| F06 | -0.003709 | 0.003449 | 0.003645 | 0.001422 | -0.042939 | — |

### Generic ΔM
| i\\j | F01 | F02 | F03 | F04 | F05 | F06 |
|-------|------|------|------|------|------|------|
| F01 | — | 0.054645 | 0.004151 | 0.002270 | 0.003083 | 0.013335 |
| F02 | 0.008611 | — | 0.004065 | 0.002872 | -0.004556 | -0.002229 |
| F03 | -0.006034 | 0.022750 | — | 0.016231 | -0.001704 | 0.010843 |
| F04 | 0.009344 | 0.007658 | 0.011442 | — | -0.006465 | -0.007194 |
| F05 | -0.001964 | 0.001077 | 0.000809 | 0.002208 | — | -0.005764 |
| F06 | 0.000210 | 0.003475 | 0.003156 | 0.002878 | -0.017505 | — |

### Difference (tennis − generic)
| i\\j | F01 | F02 | F03 | F04 | F05 | F06 |
|-------|------|------|------|------|------|------|
| F01 | — | -0.056415 | -0.002747 | -0.000194 | -0.000144 | -0.013975 |
| F02 | 0.012111 | — | -0.003692 | -0.012285 | 0.002423 | 0.007940 |
| F03 | -0.004641 | -0.048188 | — | -0.005471 | -0.016033 | -0.025738 |
| F04 | -0.029820 | -0.063785 | -0.014310 | — | -0.008977 | -0.009338 |
| F05 | -0.002947 | 0.001134 | 0.000459 | 0.002559 | — | -0.011079 |
| F06 | -0.003919 | -0.000026 | 0.000489 | -0.001456 | -0.025434 | — |

## Grouped cell-by-cell comparison

### A. Tennis within-domain

| cell | ΔM_tennis | ΔM_generic | tennis − generic |
|------|-----------|------------|------------------|
| F01→F02 | -0.001769 | 0.054645 | -0.056415 |
| F02→F01 | 0.020721 | 0.008611 | 0.012111 |

- **group mean tennis**: `0.009476`
- **group mean generic**: `0.031628`
- **group mean (tennis − generic)**: `-0.022152`
- mean |cell| tennis / generic: `0.011245` / `0.031628`

### B. Swimming within-domain

| cell | ΔM_tennis | ΔM_generic | tennis − generic |
|------|-----------|------------|------------------|
| F03→F04 | 0.010760 | 0.016231 | -0.005471 |
| F04→F03 | -0.002868 | 0.011442 | -0.014310 |

- **group mean tennis**: `0.003946`
- **group mean generic**: `0.013837`
- **group mean (tennis − generic)**: `-0.009890`
- mean |cell| tennis / generic: `0.006814` / `0.013837`

### C. Acting within-domain

| cell | ΔM_tennis | ΔM_generic | tennis − generic |
|------|-----------|------------|------------------|
| F05→F06 | -0.016843 | -0.005764 | -0.011079 |
| F06→F05 | -0.042939 | -0.017505 | -0.025434 |

- **group mean tennis**: `-0.029891`
- **group mean generic**: `-0.011634`
- **group mean (tennis − generic)**: `-0.018257`
- mean |cell| tennis / generic: `0.029891` / `0.011634`

### D. Tennis -> other-domain

| cell | ΔM_tennis | ΔM_generic | tennis − generic |
|------|-----------|------------|------------------|
| F01→F03 | 0.001404 | 0.004151 | -0.002747 |
| F01→F04 | 0.002076 | 0.002270 | -0.000194 |
| F01→F05 | 0.002939 | 0.003083 | -0.000144 |
| F01→F06 | -0.000640 | 0.013335 | -0.013975 |
| F02→F03 | 0.000373 | 0.004065 | -0.003692 |
| F02→F04 | -0.009413 | 0.002872 | -0.012285 |
| F02→F05 | -0.002133 | -0.004556 | 0.002423 |
| F02→F06 | 0.005711 | -0.002229 | 0.007940 |

- **group mean tennis**: `0.000040`
- **group mean generic**: `0.002874`
- **group mean (tennis − generic)**: `-0.002834`
- mean |cell| tennis / generic: `0.003086` / `0.004570`

### E. Other-domain -> tennis

| cell | ΔM_tennis | ΔM_generic | tennis − generic |
|------|-----------|------------|------------------|
| F03→F01 | -0.010675 | -0.006034 | -0.004641 |
| F03→F02 | -0.025438 | 0.022750 | -0.048188 |
| F04→F01 | -0.020476 | 0.009344 | -0.029820 |
| F04→F02 | -0.056127 | 0.007658 | -0.063785 |
| F05→F01 | -0.004911 | -0.001964 | -0.002947 |
| F05→F02 | 0.002211 | 0.001077 | 0.001134 |
| F06→F01 | -0.003709 | 0.000210 | -0.003919 |
| F06→F02 | 0.003449 | 0.003475 | -0.000026 |

- **group mean tennis**: `-0.014460`
- **group mean generic**: `0.004564`
- **group mean (tennis − generic)**: `-0.019024`
- mean |cell| tennis / generic: `0.015875` / `0.006564`

### F. Other cross-domain (swim <-> act)

| cell | ΔM_tennis | ΔM_generic | tennis − generic |
|------|-----------|------------|------------------|
| F03→F05 | -0.017737 | -0.001704 | -0.016033 |
| F03→F06 | -0.014895 | 0.010843 | -0.025738 |
| F04→F05 | -0.015442 | -0.006465 | -0.008977 |
| F04→F06 | -0.016532 | -0.007194 | -0.009338 |
| F05→F03 | 0.001269 | 0.000809 | 0.000459 |
| F05→F04 | 0.004767 | 0.002208 | 0.002559 |
| F06→F03 | 0.003645 | 0.003156 | 0.000489 |
| F06→F04 | 0.001422 | 0.002878 | -0.001456 |

- **group mean tennis**: `-0.006688`
- **group mean generic**: `0.000566`
- **group mean (tennis − generic)**: `-0.007254`
- mean |cell| tennis / generic: `0.009464` / `0.004407`

## Lexical-control ΔM (context; not the main contrast)
| cell | ΔM_lexical |
|------|------------|
| F01→F02 | -0.035010 |
| F02→F01 | -0.010497 |
| F03→F04 | 0.000991 |
| F04→F03 | -0.007055 |
| F05→F06 | -0.014308 |
| F06→F05 | -0.038331 |

- lexical mean tennis-within: `-0.022754`
- lexical mean unrelated-within: `-0.014676`
- lexical T_within: `-0.008078`

## Cross-domain drift breadth (descriptive)
- tennis mean |ΔM| all off-diagonal: `0.010777`
  - D. Tennis -> other-domain: mean `0.000040`, mean|·| `0.003086`
  - E. Other-domain -> tennis: mean `-0.014460`, mean|·| `0.015875`
  - F. Other cross-domain (swim <-> act): mean `-0.006688`, mean|·| `0.009464`
- generic mean |ΔM| all off-diagonal: `0.007951`
  - D. Tennis -> other-domain: mean `0.002874`, mean|·| `0.004570`
  - E. Other-domain -> tennis: mean `0.004564`, mean|·| `0.006564`
  - F. Other cross-domain (swim <-> act): mean `0.000566`, mean|·| `0.004407`

## Descriptive answers

### 1. Does tennis FT selectively increase the tennis pair while leaving other within-domain pairs near zero?
- Tennis within mean ΔM_tennis = `0.009476` (cells: F01→F02 negative small, F02→F01 positive; asymmetric).
- Swimming within mean = `0.003946`; acting within mean = `-0.029891`; unrelated-within mean = `-0.012972`.
- Partially: tennis within is the only within-domain block with a **positive** mean under tennis FT, but it is not uniformly elevated (F01→F02 ≈ 0), and unrelated within are not all near zero — acting within is distinctly **negative** (especially F06→F05), swimming is mixed/near-zero positive mean.
- So selectivity vs unrelated is mainly tennis-positive vs acting-negative, not a clean “tennis up / others ≈ 0” pattern.

### 2. Does generic FT increase the swimming/acting within-domain pairs too?
- Swimming within mean ΔM_generic = `0.013837` (both directed cells positive).
- Acting within mean ΔM_generic = `-0.011634` (both negative).
- Unrelated-within mean = `0.001101`.
- Generic FT **does raise swimming within-domain** links, but **does not raise acting within-domain** (acting shrinks). So not a uniform “all within-domain pairs increase” effect.

### 3. Does generic FT produce broader cross-domain drift?
- mean |ΔM| all off-diagonal: tennis `0.010777` vs generic `0.007951`.
- mean |ΔM| all cross-domain blocks (D+E+F): tennis `0.009475` vs generic `0.005180`.
- Tennis→other mean: tennis `0.000040` vs generic `0.002874`; other→tennis: tennis `-0.014460` vs generic `0.004564`.
- **No** in the sense of larger average |cross-domain| drift: tennis FT has **larger** mean absolute cross-domain change, driven especially by **other→tennis** shrinkages (large negative cells under tennis FT, e.g. F04→F02).
- Generic cross-domain means are closer to zero / smaller |·| on average; tennis FT shows more structured (often negative) other→tennis movement.

### 4. Which model produces the more domain-selective pattern?
- Relative within contrast T_within = mean(tennis within) − mean(unrelated within): tennis FT `0.022448`; generic FT `0.030527` (matches existing reports).
- Absolute tennis-within elevation is **larger under generic FT** (`0.031628` vs `0.009476`), and generic’s T_within is also slightly larger.
- Domain-selectivity as “tennis up, other within near zero, cross near zero”: **neither is clean**. Tennis FT: tennis modest+/acting down; generic FT: tennis strongly+/swim up/acting down.
- If selectivity means *preferential tennis-within vs other within under a tennis-trained model relative to a generic-trained control*, the difference matrix shows tennis−generic **negative** on tennis-within (generic raised tennis more) and mixed elsewhere — so the **generic control does not make tennis FT look uniquely selective**; if anything generic produces a larger tennis-within ΔM.

### 5. Does full-matrix evidence strengthen or weaken the claim that narrow tennis FT causes a tennis-specific structural change?
- **Weakens** the strong causal claim of tennis-specific structural change, as a descriptive control comparison:
  1. Generic (Curie/Armstrong) FT also elevates tennis within-domain ΔM — and more so than tennis FT.
  2. Tennis FT’s positive T_within is partly driven by **acting within-domain decreases**, not only tennis increases.
  3. Tennis FT shows substantial **other→tennis** ΔM (often large negative), i.e. non-local structure change beyond the tennis pair.
  4. Lexical-control extraction flips the sign of T_within (context), so the explicit-matrix tennis elevation is extraction-sensitive.
- This does **not** overturn the primary frozen statistic or its exact p-value; it is a descriptive alternative-hypothesis comparison that reduces confidence in interpreting the primary contrast as uniquely tennis-FT-caused domain-specific restructuring.
