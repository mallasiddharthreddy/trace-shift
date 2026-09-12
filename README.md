# TraceShift

Does narrow fine-tuning selectively rewire related factual memories?

## Question

After LoRA fine-tuning on a small tennis-biography corpus, do Federer/Nadal factual traces interact more strongly under targeted AI-Engram interventions than unrelated swimming and acting pairs do?

## Why this is interesting

Narrow fine-tuning leaves readable activation / weight-space traces. A natural follow-up: is that a generic “you’ve been trained recently” drift, or a restructuring of how specific facts relate to each other? TraceShift measures **intervention–response** structure with AI-Engram (parameter-space traces + edits), compares base vs fine-tuned 6×6 matrices, and checks the result against an unrelated fine-tune and a lexical extraction control.

## Main finding

Fine-tuning **does** change the measured interaction matrix, but these experiments **do not** establish a clean tennis-content-specific rewiring effect.

| Analysis | Result |
|----------|--------|
| Primary tennis FT contrast \(T\) | **0.02245** |
| Exact one-sided permutation \(p\) | **2/15 ≈ 0.133** |
| Generic FT control \(T_{\mathrm{generic}}\) | **0.03053** (same sign, larger) |
| Lexical extraction control \(T_{\mathrm{lexical}}\) | **−0.00808** (sign flips) |

\(T\) is **not** “the Federer↔Nadal cell average alone.” It is:

\[
T = \overline{\Delta M}_{\text{tennis within}} - \overline{\Delta M}_{\text{unrelated within}}
\]

with tennis within = F01↔F02 (2 cells) and unrelated within = swimming + acting pairs (4 cells).

In plain language: tennis fine-tuning yields a positive contrast, but an identical LoRA run on Curie/Armstrong biographies produces a **larger** nominal tennis contrast, and stripping domain words from extraction probes flips the sign. That is evidence **against** attributing the primary number mainly to tennis-specific semantic content under this measurement.

Full write-ups: [`research/reports/METHODOLOGY.md`](research/reports/METHODOLOGY.md), [`research/reports/final_evidence_audit.md`](research/reports/final_evidence_audit.md), [`research/reports/tennis_vs_generic_delta_m_full_matrix_comparison.md`](research/reports/tennis_vs_generic_delta_m_full_matrix_comparison.md).

## Experimental design

- **Model:** `Qwen/Qwen3-1.7B`
- **Facts:** six entities, three domains (two each) — tennis Federer/Nadal; swimming Phelps/Ledecky; acting Bachchan/Shah Rukh Khan
- **Instrument:** AI-Engram v0.9.0 over all 28 `mlp.down_proj` layers
- **Extraction:** 5 held-out sentences/fact; shared 30-sentence reference corpus (explicit + lexical variants)
- **Alphas:** calibrated on the base model only; held fixed across comparisons
- **Fine-tuning:** LoRA (r=16, α=32, dropout 0.05, `q/k/v/o_proj`), 3 epochs, LR 2e-4, seed 42
- **Controls:** (1) identical LoRA on Curie/Armstrong biographies; (2) lexical extraction wording on the tennis FT path

## Results at a glance

| Analysis | Result |
|----------|--------|
| Primary \(T_{\mathrm{obs}}\) (tennis FT) | 0.02245; \(p = 0.133\) (exact, 15 assignments) |
| Generic-control \(T\) | 0.03053 (descriptive; no second \(p\)) |
| Lexical-control \(T\) | −0.00808 (descriptive; tennis FT + lexical probes) |
| Full-matrix pattern | Not “tennis up / others flat”; acting within-domain ΔM falls under both FTs; tennis FT shows larger other→tennis \|ΔM\| than generic |

Canonical JSON/tables: [`research/results/README.md`](research/results/README.md).

## Repository structure

```
.
├── README.md                 # This overview
├── requirements.txt          # Direct Python deps (Python 3.12)
├── .gitignore
└── research/
    ├── configs/              # experiment.yaml, environment.yaml
    ├── data/frozen/          # Locked facts, probes, corpora
    ├── traceshift/           # Pipeline library
    ├── scripts/              # CLIs (validate / stage runners)
    ├── results/              # Canonical outputs (raw JSON + tables)
    ├── reports/              # Methodology + audits + matrix comparison
    └── archive/              # Pre-GPU notes, overnight GPU scripts, smoke logs
```

## Reproducibility / inspection

**Inspect without GPU:** open `research/results/tables/` and the JSON files listed in [`research/results/README.md`](research/results/README.md). Validate frozen specs with:

```bash
python research/scripts/validate_frozen_experiment.py
```

**Full regeneration** (matrices, Engrams, LoRA) requires local `Qwen/Qwen3-1.7B` weights, a CUDA environment matching the original run, and LoRA adapters (adapters are **not** committed). This repo does **not** claim one-command end-to-end reproduction.

Historical overnight orchestrators (hardcoded VM paths) live under [`research/archive/scripts/`](research/archive/scripts/).

## Status

This repository contains the **completed** balanced 6-fact TraceShift experiment and analysis. Primary inference is conditional on the frozen design; treat generic/lexical contrasts as descriptive checks, not independently significant tests.
