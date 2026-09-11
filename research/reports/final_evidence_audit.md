# TraceShift final evidence / validity audit (balanced 6-fact)

**Generated (UTC):** 2026-09-11T19:55:14.264255+00:00  
**Scientific start:** 2026-09-10T22:16:42Z  
**Total scientific seconds:** 77910  
**Overnight exit:** 0

## FINAL EVIDENCE AUDIT: PASS

### Completed checklist

| Item | Present |
|------|---------|
| Baseline 6 facts | True |
| Alphas | True |
| M_base 6×6 | True |
| Tennis FT validity | True |
| Generic FT validity | True |
| M_FT tennis | True |
| M_FT generic | True |
| Delta-M tennis + p | True |
| Lexical | True |
| Consolidation | True |

### Primary result

- T_obs = `0.022448480129241943`
- p_exact = `0.13333333333333333`
- T_cross (descriptive) = `None`
- T_generic_within (descriptive) = `0.03052687644958496`
- T_tennis − T_generic = `-0.008078396320343018`
- T_lexical_within (descriptive) = `None`

### Claim boundaries

**Can claim (conservative):**
- Measured baseline recall and selective/nonselective alphas under frozen 6-fact design.
- Descriptive and exact-permutation results for tennis FT within-domain vs unrelated within-domain cells (p over 15 assignments).
- Descriptive comparison to generic FT and lexical extraction.

**Cannot claim:**
- Population generalization beyond six frozen facts.
- That descriptive generic/lexical contrasts are statistically significant (no second test).
- Causality of tennis fine-tuning beyond the intervention–response operationalization.

### Engineering notes

- Prior Phase-2 interruption (first attempt) was session/tmux teardown after weight load, not an OOM at that timestamp.
- Resume skipped completed baseline; continued from alpha calibration inside detached tmux `traceshift-run-6fact`.
- GPU-first load; Engram CPU storage / empty_cache memory hygiene as previously engineering-fixed.
- Do not cite stale 4-fact outputs; those live only in Git history.
