# TraceShift scientific experiment status summary

**Run host:** `traceshift-l4` (remote GCP NVIDIA L4)  
**Execution:** persistent tmux session `traceshift-run` (does not depend on Mac/Cursor connectivity)  
**Scientific clock start (UTC):** 2026-09-10T13:35:20Z  
**Scientific clock end (UTC):** 2026-09-10T17:15:29Z  
**Approximate total scientific wall time:** **13209 s (~3.67 h)**  
**Pre-science runtime verification commit:** `df51386805bab550fff305291b02c5360058d27a`

This document records measured outcomes and engineering notes. It is **not** a polished narrative essay or MATS application.

---

## Hardware / software (execution)

| Item | Value |
|------|-------|
| GPU | NVIDIA L4 |
| NVIDIA driver | 580.178.04 |
| CUDA (driver / torch) | 13.0 / 13.0 |
| Python | 3.12.13 (`.venv`) |
| torch | 2.14.0+cu130 |
| transformers | 5.16.1 |
| peft | 0.20.0 |
| ai-engram | 0.9.0 |

Frozen model id remained `Qwen/Qwen3-1.7B`. Frozen facts/extraction/cloze/alpha grid/FT hparams were not edited for outcome.

---

## Phase outcomes

### 1. Baseline recall (M_base)
All four facts **PASS** (2-of-3 gate):

| Fact | Status |
|------|--------|
| F01 Federer | PASS |
| F02 Nadal | PASS |
| F03 Phelps | PASS |
| F04 Bachchan | PASS |

Artifact: `results/raw/baseline_recall/baseline_cloze.json`

### 2. Alpha calibration (BASE, explicit)
All four **selective**:

| Fact | Status | Alpha |
|------|--------|-------|
| F01 | selective | 0.6 |
| F02 | selective | 0.4 |
| F03 | selective | 0.4 |
| F04 | selective | 0.4 |

Artifacts: `results/raw/alpha_calibration/base_alphas.yaml`, `base_alpha_calibration.json`, tables summary.

### 3. Base matrix `M_base`
**Complete** (all four selective rows). Primary metric cosine-distance.  
Artifact: `results/raw/base_matrix/base_matrix.json`, `results/tables/base_matrix.md`

### 4. Tennis LoRA fine-tuning
**Completed** under frozen hparams (seed 42, 3 epochs, LR 2e-4, LoRA r16/α32, q/k/v/o).  
Execution-validity gate: **PASS** (40 examples, 30 optimizer steps, finite loss trajectory, nonzero LoRA norms, adapter reload+generate OK).  
Note: first metadata write failed on `TorchVersion` YAML serialization after training already finished; metadata rewritten without re-training; `str(torch.__version__)` fix applied.

### 5. Control LoRA fine-tuning (Curie/Armstrong)
**Completed** with identical hparams, different corpus only.  
Execution-validity gate: **PASS**.

Adapters themselves are gitignored (local on VM under `final_adapter/`).

### 6. FT matrix `M_FT`
**Complete** using tennis FT model + **fixed BASE alphas** + freshly extracted FT Engrams.  
Artifact: `results/raw/ft_matrix/ft_matrix.json`

### 7. Delta-M + exact permutation
**Complete.**

- mean in-domain ΔM = `0.02518138289451599`
- mean control ΔM = `-0.0010637640953063965`
- **T_obs = `0.026245146989822388`**
- Exact one-sided p (C(4,2)=6, observed included) = **`1/6 ≈ 0.16666666666666666`**
- Six permutation contrasts:  
  `[0.026245146989822388, -0.0011315643787384033, -0.017166420817375183, -0.008728355169296265, -0.003611132502555847, 0.0043923258781433105]`

Limitation (frozen design): exact test over four factual traces only; not a population-level generalization claim.

### 8. Lexical-control (descriptive)
**Complete.** Same models/BASE alphas/metric; lexical extraction/reference only.

Descriptive contrasts (not a second primary inferential test):

| Condition | T |
|-----------|---|
| Explicit primary | +0.026245146989822388 |
| Lexical | −0.013780161738395691 |
| Same sign? | **false** |

Artifact: `results/raw/alternative_hypotheses/lexical_control.json`, `results/tables/lexical_control.md`

### 9. Consolidation
All required scientific outputs present. Frozen-input SHA-256 digests recorded in `results/raw/experiment_run/consolidation_summary.json`.

---

## Technical deviations (engineering only; methodology unchanged)

1. **GPU-direct model load** (`device_map="cuda"`) in `model_loader` to avoid CPU-resident full copies on ~15 GiB host RAM.
2. **AI-Engram `EditorConfig(storage_device="cpu")`** for covariance accumulation + move Engram tensors to CPU after extract — same 0.9.0 math; prevents CUDA OOM when holding model + intervention copy on 24 GiB L4.
3. **`torch.cuda.empty_cache()`** after each matrix intervention row.
4. First base-matrix attempt OOM’d before (2); resumed after placement fix; **no science change**.
5. Tennis FT metadata YAML `TorchVersion` serialization bug fixed; training not re-run.
6. Control FT validity gate script bug (path shadowed by generate tensor) fixed; training not re-run.
7. **`validate_runtime_stack.py` not used** as a gate (CPU model load incompatible with host RAM).
8. No swap added; no VM resize; no hyperparameter/alpha/fact/text changes after seeing results.

---

## Integrity statement

No scientific methodology was changed after observing results. Frozen facts, extraction texts, cloze probes, alpha grid, model id, AI-Engram version, LoRA targets/hparams, and primary permutation procedure remained as frozen. Negative/ambiguous inferential outcomes (exact p = 1/6; lexical T opposite sign) are reported as measured.
