# Alpha calibration summary (BASE model)

- model_id: `Qwen/Qwen3-1.7B`
- AI-Engram: `0.9.0`
- extraction_variant: `explicit`
- alpha_grid: `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]`
- device/dtype: `cuda` / `bfloat16`
- seed: `42`

Dose/eligibility only — not a scientific effect-size result.

| fact_id | baseline | status | selected_alpha | reason |
|---------|----------|--------|----------------|--------|
| F01 | PASS | selective | 0.4 | smallest alpha with target gate FAIL and zero PASS→FAIL collateral |
| F02 | PASS | selective | 0.4 | smallest alpha with target gate FAIL and zero PASS→FAIL collateral |
| F03 | PASS | selective | 0.8 | smallest alpha with target gate FAIL and zero PASS→FAIL collateral |
| F04 | PASS | selective | 0.8 | smallest alpha with target gate FAIL and zero PASS→FAIL collateral |
| F05 | PASS | selective | 0.4 | smallest alpha with target gate FAIL and zero PASS→FAIL collateral |
| F06 | PASS | selective | 0.2 | smallest alpha with target gate FAIL and zero PASS→FAIL collateral |
