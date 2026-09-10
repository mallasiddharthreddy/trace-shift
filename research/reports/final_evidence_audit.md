# TraceShift final evidence / validity audit

**Audit date (UTC):** 2026-09-10 (post-run inspection)  
**Addendum (UTC):** 2026-09-10 — generic FT control structural measurement completed (no retrain; primary outputs unchanged)  
**Audited run summary:** `research/reports/experiment_status_summary.md`

---

## FINAL EVIDENCE AUDIT: PASS WITH WARNINGS

---

## 1. Control fine-tuning question (most important)

### A. Was the Curie/Armstrong control model trained successfully?

**YES** — training executed and passed the execution-validity gate.

| Check | Evidence | Result |
|-------|----------|--------|
| Dataset size | `…/qwen3-1.7b-lora-control/training_metadata.json` → `n_examples: 40`, `dataset_id: control_general_ft` | 40 |
| Epochs | metadata `num_epochs: 3`; `checkpoint-30/trainer_state.json` → `epoch: 3.0` | 3 |
| Optimizer steps | `global_step: 30` (expected \(40/(1\times4)\times3=30\)) | 30 |
| Loss trajectory | 30 finite losses; first≈3.266 → last≈1.280; min≈1.096; max≈3.641 | finite + changed |
| Nonzero LoRA gradients | `grad_norm` values in trainer log all > 0 (e.g. 3.98, 4.02, …) | yes |
| Nonzero LoRA update | `execution_validity.json` → `lora_norm_max: 2.3635…`; adapter weights present | yes |
| Checkpoint/reload | `final_adapter/adapter_model.safetensors` (25,720,120 bytes); validity `ok: true` | yes |

**Stale-file warning:** `research/results/raw/finetuning/control_status.yaml` still says `status: pending_not_executed` / `training_executed: false`. That file is **outdated dry-validate status**, not the scientific truth. Prefer `training_metadata.*` + `execution_validity.json` + on-disk adapter.

### BEFORE vs AFTER (structural control)

| Stage | What existed |
|-------|----------------|
| **BEFORE** (primary run freeze) | Control FT trained successfully; **no** control matrix / control ΔM / tennis-vs-unrelated-FT comparison |
| **AFTER** (this add-on) | Same adapter reused (SHA unchanged; no retrain); `M_control_FT` + descriptive `ΔM_control` + `T_control` vs `T_tennis` measured |

### B. Was a CONTROL intervention-response matrix computed for the Curie/Armstrong model?

**YES (after add-on).**

- Artifact: `research/results/raw/alternative_hypotheses/generic_ft_control_matrix.json`
- `model_state: control_finetuned`
- `source_checkpoint_path: …/qwen3-1.7b-lora-control/final_adapter`
- Explicit extraction; BASE alphas `{F01:0.6, F02:0.4, F03:0.4, F04:0.4}`; device `cuda`; AI-Engram `0.9.0`
- All four rows complete; diagonal null

### C. Was any control Delta-M computed (Curie/Armstrong FT vs base)?

**YES (descriptive).**

- Artifact: `research/results/raw/alternative_hypotheses/generic_ft_control_delta_analysis.json`
- Definition: `ΔM_control = M_control_FT − M_base` (existing primary `M_base`; not recomputed)
- **No** second exact permutation test / **no** new p-value

### D. Was any quantitative tennis-FT vs unrelated-FT comparison performed?

**YES (descriptive only).**

### E. Metrics / values / files

| Metric | Value | File |
|--------|-------|------|
| mean in-domain ΔM_control | `0.015360027551651001` | `generic_ft_control_delta_analysis.json` |
| mean control ΔM_control | `-0.0007405579090118408` | same |
| T_control | `0.016100585460662842` | same |
| T_tennis (primary, unchanged) | `0.026245146989822388` | `delta_analysis.json` + comparison block |
| T_tennis − T_control | `0.010144561529159546` | same |
| same-sign T | true | same |

Supports: under identical measurement procedure, unrelated Curie/Armstrong FT produces a **same-sign but smaller** descriptive tennis-vs-fact-control contrast than tennis FT.

Does **not** support: ruling out generic FT drift; causal tennis-specificity; a revised primary p-value; population generalization.

### F. Updated conclusion

> Control FT was trained successfully **and** its downstream intervention–response structure was measured. A quantitative descriptive comparison to tennis FT now exists. Generic-fine-tuning drift is **partially weakened as a sole/complete explanation of the tennis magnitude** (T_control < T_tennis) but **remains unresolved** because T_control is same-sign and non-negligible (~61% of T_tennis).
---

## 2. Primary result consistency (raw files)

### Baseline (`baseline_cloze.json`)

| Fact | Status | Wins |
|------|--------|------|
| F01 | PASS | 2/3 |
| F02 | PASS | 2/3 |
| F03 | PASS | 3/3 |
| F04 | PASS | 3/3 |

Matches summary.

### Alpha (`base_alphas.yaml` + calibration JSON)

| Fact | Status | Alpha | Selective row at α |
|------|--------|-------|--------------------|
| F01 | selective | 0.6 | target FAIL, collateral 0 |
| F02 | selective | 0.4 | target FAIL, collateral 0 |
| F03 | selective | 0.4 | target FAIL, collateral 0 |
| F04 | selective | 0.4 | target FAIL, collateral 0 |

Frozen grid used: `{0,0.2,0.4,0.6,0.8,1.0,1.5,2.0}`. No smaller selective α exists for any fact. Matches summary.

### Base matrix (`base_matrix.json`) — cosine distance

All four rows present; diagonal null.

```
F01: [—, 0.358328, 0.076776, 0.016992]
F02: [0.383712, —, 0.049951, 0.000490]
F03: [0.062983, 0.044152, —, -0.017841]
F04: [0.012014, -0.003225, -0.012890, —]
```

`row_alphas` = `{F01:0.6,F02:0.4,F03:0.4,F04:0.4}`; all `selective`.  
`runtime.device=cuda`, `dtype=bfloat16`, `ai_engram_version=0.9.0`, `extraction_variant=explicit`.

### FT matrix (`ft_matrix.json`) — cosine distance

```
F01: [—, 0.391161, 0.090571, 0.015317]
F02: [0.401242, —, 0.040217, -0.006151]
F03: [0.062770, 0.042740, —, -0.015896]
F04: [0.005212, -0.001587, -0.009444, —]
```

`source_checkpoint_path` = tennis primary adapter only.  
Same BASE alphas. Notes: Engrams from FT model; **no FT alpha recalibration**.

### Delta-M primary cells

| Cell | Value |
|------|-------|
| F01→F02 | 0.0328332781791687 |
| F02→F01 | 0.01752948760986328 |
| F01→F03 | 0.013795137405395508 |
| F01→F04 | −0.001675248146057129 |
| F02→F03 | −0.009734153747558594 |
| F02→F04 | −0.006640791893005371 |

- mean in-domain ΔM = **0.02518138289451599**
- mean control ΔM = **−0.0010637640953063965**
- **T_obs = 0.026245146989822388**

Tables (`delta_analysis.md`) match to displayed rounding.

### Permutation

Six assignments present; observed (F01,F02)/(F03,F04) included once.  
Contrasts: `[0.026245, -0.001132, -0.017166, -0.008728, -0.003611, 0.004392]`  
**p = 1/6 = 0.1666…** Matches summary/table.

### Lexical control

- mean in-domain ΔM_lex = **−0.01600739359855652**
- mean control ΔM_lex = **−0.0022272318601608276**
- **T_lex = −0.013780161738395691**
- Sign vs explicit T: **different** (`tennis_vs_control_direction_same_sign: false`)
- Uses lexical extraction + shared lexical reference; BASE explicit alphas reused; FT = tennis primary (not Curie/Armstrong).

### Summary ↔ raw mismatches

| Item | Issue |
|------|-------|
| `control_status.yaml` / possibly primary `STATUS.yaml` | Still dry-validate / “not executed” language despite completed training |
| `phase_times.jsonl` | Records tennis/control FT as `FAIL` on first attempt; later recovered; summary correctly says PASS via validity JSON |
| Cosine-distance cells | Several **negative** distances in M_base/M_FT (implies cos>1 numerically) — present in raw; not highlighted in status summary |

No mismatch on headline PASS/alphas/T_obs/p/lexical T between `experiment_status_summary.md` and primary raw JSON.

---

## 3. Alpha / intervention validity

| Requirement | Verdict | Evidence |
|-------------|---------|----------|
| Alphas from frozen grid | PASS | calibration `alpha_grid` = config grid |
| Target-fail + zero collateral | PASS | per-fact sweep rows at selected α |
| Smallest such α | PASS | no smaller `selective_ok` rows |
| Same BASE alphas on FT matrix | PASS | identical `row_alphas`; `alpha_source` → `base_alphas.yaml` |
| No FT recalibration | PASS | FT notes + lexical `alpha_policy` |
| AI-Engram 0.9.0 | PASS | base/ft/delta/lexical/calibration fields |
| Base Engrams not reused on FT | PASS | FT notes: collected from primary FT model |
| FT Engrams separate | PASS | same |
| Primary = explicit | PASS | `extraction_variant: explicit` |
| Lexical = lexical_control | PASS | variant + isolation_ok; 0 text overlap |
| Diagonal null | PASS | all diag `null` |
| Metric = 1 − cos | PASS | `primary_metric` strings in matrices |

---

## 4. Fine-tuning validity (both runs)

Frozen settings confirmed in both metadata files:

`seed=42`, `epochs=3`, `LR=2e-4`, `AdamW`, `cosine`, `batch=1`, `accum=4`, `max_seq=256`, `wd=0`, `warmup=5`, `LoRA r=16 α=32 dropout=0.05`, targets `q/k/v/o_proj`.

| | Tennis | Control |
|--|--------|---------|
| Examples | 40 | 40 |
| Steps | 30/30 | 30/30 |
| Loss finite + changed | yes | yes |
| grad_norm > 0 | yes | yes |
| LoRA norm after reload | 2.368 | 2.364 |
| Adapter reload/exec | ok | ok |
| Downstream model used | **yes** (FT + lexical FT) | **no** |

Tennis metadata note: rewritten after TorchVersion YAML failure; **training not re-run**.

---

## 5. Engineering deviation audit

| Deviation | Operational change | Sci measurement risk | Spec changed? | Class |
|-----------|--------------------|----------------------|---------------|-------|
| `device_map="cuda"` in `model_loader.py` | Avoid CPU-resident full load | Low if weights identical on GPU | No | **A** |
| Engram `storage_device="cpu"` + CPU Engram tensors (`engram.py`) | Covariances/projections off GPU | Mild numerical-path difference vs GPU accum possible; math API unchanged | No | **A** (borderline **B** for float path) |
| `empty_cache` after matrix rows | Memory hygiene | None | No | **A** |
| First base-matrix CUDA OOM then resume with above | Same science path after fix | None if final matrix is post-fix | No | **A** |
| Tennis metadata TorchVersion str fix | Serialization only | None | No | **A** |
| Control validity script path bug fix | Gate script only | None | No | **A** |
| `validate_runtime_stack` skipped | Engineering gate unused | None for science (other gates used) | No | **A** |
| No swap / no VM change | As reported | N/A | No | **A** |
| Control FT trained but unused in matrices (BEFORE) | Coverage gap closed by post-hoc structural add-on | See AFTER row | No | **B** historically |
| Control structural add-on (AFTER): `M_control_FT` + descriptive ΔM | Measurement of alt A; no primary rewrite; no new p | Descriptive only; same-sign residual drift | No (primary unchanged) | **A** (planned alt analysis) |
| Negative cosine-distance cells | Float numerics in 1−cos | Small cell-level noise; affects tiny cells | No | **B** (numerics) |
| Stale `control_status.yaml` | Misleading status file | Documentation hazard for writers | No | **A** (docs) |

**No category C (genuine scientific protocol rewrite) found** in frozen inputs/config digests or primary analysis definitions.

---

## 6. Memory / device audit

- Scientific CLIs invoked with `--device cuda` (8× in `master.log`).
- Saved matrix runtimes: `device: cuda`, `dtype: bfloat16` (base, FT, lexical base).
- Calibration runtime: `device: cuda`.
- No evidence scientific phases ran entirely on CPU.
- No swap workaround in scientific run reports.
- Memory fixes did not change model id, dtype policy (bf16), alphas, texts, or test definition.
- **Incomplete evidence:** per-kernel device of Engram covariance ops under `storage_device=cpu` is CPU by design; model weights/activations for forward passes remain GPU-placed per loader/runtime fields.

---

## 7. Claim boundary audit (conservative)

### Can safely claim

**A. Baseline structure (descriptive):** On M_base, tennis–tennis cells are large (~0.36–0.38) vs tennis→control / control cells (mostly ≪0.1).  
**B. Change after tennis FT (descriptive):** Primary ΔM shows positive mean in-domain change and near-zero/negative mean tennis→control change; T_obs≈0.026.  
**C. Statistical support:** Exact 4-fact permutation **p=1/6**. Under conventional α=0.05, **not significant**. Strongest honest statement: “most extreme of 6 assignments; not distinguishable from chance at 0.05.”  
**D. Lexical control:** Explicit T and lexical T **differ in sign**; effect is **sensitive to extraction wording** under this frozen design (descriptive only).  
**E. Generic FT drift:** Control model trained validly **and** measured: T_control≈0.0161 vs T_tennis≈0.0262 (same sign; control smaller). Partially weakens “identical generic drift explains all of tennis T”; **does not** rule out a substantial generic-FT contribution.  
**F. Population generalization:** **Cannot** claim; exact test is conditional on four frozen facts.

### Cannot claim

- That tennis FT statistically significantly reshapes in-domain interactions (p=1/6).
- That generic fine-tuning is ruled out.
- That the effect is definitively tennis-specific / causal.
- That the effect is robust to lexical/extraction form (lexical T flips sign).
- That results generalize beyond F01–F04.
- That the control contrast has its own inferential p-value (descriptive only).

---

## 8. MATS-writing evidence table

| Question | Evidence file | Exact result | Supports | Does NOT support |
|----------|---------------|--------------|----------|------------------|
| Baseline recall | `raw/baseline_recall/baseline_cloze.json` | All PASS (2/3,2/3,3/3,3/3) | Eligibility for calibration | Scientific effect size |
| Alpha calibration | `raw/alpha_calibration/base_alphas.yaml`, `base_alpha_calibration.json` | 0.6/0.4/0.4/0.4 selective | Dose selection under frozen rule | That α is “optimal” scientifically |
| Base matrix | `raw/base_matrix/base_matrix.json` | Full 4×4 off-diag; tennis–tennis large | Pre-FT interaction structure | Causation / FT effect |
| Tennis FT validity | `…/tennis-primary/training_metadata.json`, `execution_validity.json`, `trainer_state.json` | 40/3/30; loss↓; LoRA≠0; reload OK | Training executed as specified | That FT improved “true” tennis knowledge |
| Control FT validity | `…/lora-control/training_metadata.json`, `execution_validity.json` | Same technical PASS | Control **training** succeeded | Alone, not a structural comparison |
| Generic FT control (measured) | `raw/alternative_hypotheses/generic_ft_control_matrix.json`, `generic_ft_control_delta_analysis.json` | T_control=0.016100…; T_tennis−T_control=0.010145…; same-sign | Descriptive structural comparison under identical procedure | Ruling out generic drift; tennis-specificity; new p-value |
| Primary ΔM | `raw/delta_analysis/delta_analysis.json` | T_obs=0.026245… | Descriptive tennis-FT change contrast | Significance at 0.05 |
| Exact permutation | same | p=1/6; 6 contrasts; observed included | Exact 4-fact test executed | Population inference; “significant finding” |
| Lexical control | `raw/alternative_hypotheses/lexical_control.json` | T_lex=−0.01378…; opposite sign | Sensitivity to extraction wording | Robustness of primary T |
| Engineering deviations | `traceshift/{model_loader,engram,base_matrix,ft_matrix,lexical_control,finetune}.py`; `experiment_run/master.log` | GPU map; CPU Engram storage; OOM resume; metadata fixes | Run feasibility on 15 GiB/L4 | Spec rewrite |
| Frozen integrity | `experiment_run/consolidation_summary.json` digests | All 6 frozen SHA256 still match | Inputs unchanged post-run | — |

---

## 9. Repository integrity

- Frozen facts/extraction/reference/cloze/`experiment.yaml` digests **unchanged** vs consolidation record.
- Final raw outputs + tables + reports present.
- `git ls-files`: **no** tracked `.safetensors` / weights / credentials.
- Adapters remain local + gitignored (correct).
- Leftover untracked live-runtime helpers (`smoke.json`, etc.) are non-secret; optional cleanup later.
- **Preserve before VM shutdown (critical):**
  - both `final_adapter/` directories (gitignored weights)
  - checkpoints if desired for audit (gitignored)
  - `research/results/raw/**` JSON/YAML already in git
  - `research/results/raw/experiment_run/master.log` (local; gitignored `*.log`) — useful for OOM/device audit

---

## 10. Decisions for shutdown / writing

1. **Generic-fine-tuning-control answer:** Control FT **trained successfully** and, in a post-primary add-on, **was used** for a full explicit intervention matrix + descriptive ΔM. Quantitative comparison: T_control≈0.0161 vs T_tennis≈0.0262 (same sign; tennis larger by ≈0.0101). Generic-drift alternative is **partially weakened for magnitude-equivalence** but **not ruled out**.

2. **Scientific validity concerns:**
   - Primary inferential result remains **weak** (p=1/6; unchanged).
   - Lexical control **undermines robustness** of the explicit T direction.
   - Generic FT control is **descriptive only** (no second exact test).
   - Negative cosine-distance cells → minor numerical warning.
   - Stale status YAMLs → do not cite them for training completion.

3. **Must preserve:** tennis + control `final_adapter/` on disk (or offline archive) before VM delete; committed raw/tables/reports; also new `generic_ft_control_*.json/md` and `generic_ft_control.log`.

4. **GPU safe to shut down?** **YES**, provided adapters (and preferably logs) are copied/backed up first.

5. **Ready to freeze and move to writing?** **YES, with warnings** — freeze the measured package including the generic-FT control add-on; write within claim boundaries; do **not** overclaim significance, robustness, or “generic drift ruled out.”

### Claim boundaries for the control result (explicit)

**A. Supports:** descriptive evidence that unrelated Curie/Armstrong FT under the same recipe produces a same-sign but smaller T than tennis FT.

**B. Does not support:** ruling out generic FT drift; proving tennis-specific causality; establishing significance; population generalization.

**C. Generic-fine-tuning-drift alternative:** **partially weakened** (magnitude not identical) / **remains unresolved** (same-sign residual ≈61% of tennis T).

**D. Methodological limitations remaining:** n=1 control corpus; descriptive only; four-fact design; primary p still 1/6; lexical sign flip.
