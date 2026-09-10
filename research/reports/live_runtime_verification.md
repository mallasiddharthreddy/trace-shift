# TraceShift live runtime verification (GPU-first)

**Verification timestamp (UTC):** 2026-09-10T12:37:44Z  
**Mode:** GPU-first only (no CPU-resident full model copy)  
**Machine:** GCP L4 VM (`g2-standard-4` class; ~15 GiB host RAM)  
**Hostname context:** `traceshift-l4`  
**Overall decision:** **READY**  
**20-hour research clock:** **NOT STARTED**

This is an **engineering** verification that the frozen TraceShift stack can execute on the NVIDIA L4. It is **not** a scientific result. No baseline recall, alpha calibration, fine-tuning, matrices, Delta-M, permutation tests, or lexical-control experiments were run.

Machine-readable smoke record: `results/raw/live_runtime_verification/gpu_first_smoke.json`.

---

## Status summary

| Check | Verdict |
|-------|---------|
| GPU / driver | **PASS** |
| PyTorch GPU | **PASS** |
| Qwen GPU load (bf16, direct CUDA) | **PASS** |
| Architecture (28× `mlp.down_proj`) | **PASS** |
| LoRA targets `q/k/v/o_proj` | **PASS** |
| AI-Engram smoke (`get_engram` + `apply_engram`) | **PASS** |
| Trace + cosine-distance smoke | **PASS** |
| PEFT/LoRA trainer initialization (no train) | **PASS** |
| `validate_runtime_stack.py` (CPU model load path) | **WARNING / NOT APPLICABLE** on this low-RAM VM |

---

## 1. VM / GPU / driver

| Item | Value |
|------|-------|
| GPU | NVIDIA L4 (1×) |
| VRAM | 23034 MiB |
| NVIDIA driver | 580.178.04 |
| CUDA (driver-reported) | 13.0 |
| Host RAM | ~15 GiB |
| Disk (`/`) | ~146 GiB total; ample free |
| Swap added for this verification | **No** (explicitly refused) |

`nvidia-smi` succeeds and reports the L4.

---

## 2. Python environment

| Item | Value |
|------|-------|
| Interpreter | `.venv` Python **3.12.13** |
| System Python (not used for experiment) | 3.10.12 |
| Torch wheel | `torch==2.14.0+cu130` (CUDA 13.0 index; matches driver CUDA 13.0) |

### Exact installed direct package versions

| Package | Version |
|---------|---------|
| torch | 2.14.0+cu130 |
| transformers | 5.16.1 |
| peft | 0.20.0 |
| ai-engram | 0.9.0 |
| tqdm | 4.70.0 |
| PyYAML | 6.0.3 |

Frozen scientific pins were **not** upgraded/downgraded. Frozen experiment YAML/data were **not** modified.

---

## 3. PyTorch CUDA — PASS

- `torch.cuda.is_available()` → `True`
- `torch.cuda.device_count()` → `1`
- `torch.cuda.get_device_name(0)` → `NVIDIA L4`
- `torch.version.cuda` → `13.0`
- Tiny GPU matmul succeeded

---

## 4. Qwen/Qwen3-1.7B GPU load — PASS

Loaded from local HF snapshot with **direct GPU placement**:

- `device_map="cuda"`
- frozen dtype `bfloat16` (`ai_engram.precision.model_weights`)
- `local_files_only=True` (weights already cached; no second CPU-resident full copy)

| Observation | Value |
|-------------|-------|
| Model id | `Qwen/Qwen3-1.7B` |
| Parameter device | `cuda:0` |
| Parameter dtype | `torch.bfloat16` |
| GPU memory allocated | ~3.45 GB |
| Process RSS after load | ~4.9 GB |
| Harmless inference | OK (`"The capital of France is Paris..."`) |

Snapshot path:  
`~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`

---

## 5. Architecture — PASS

- Transformer layers: **28**
- All frozen Engram targets present:
  - `model.layers.0.mlp.down_proj` … `model.layers.27.mlp.down_proj` (28/28)
- LoRA attention targets present:
  - `q_proj`, `k_proj`, `v_proj`, `o_proj`

---

## 6. AI-Engram 0.9.0 smoke — PASS

Engineering-only smoke (not calibration, not a scientific measurement):

- Import `engram` / `get_engram` / `apply_engram`: OK
- Version: **0.9.0**
- One construction: fact **F01**, variant **explicit**, frozen target modules/layers
- `get_engram` → Engram with **28** layer entries
- `apply_engram` on a fresh copy (`inplace=False`) at smoke α=`0.1` only

**Not performed:** alpha grid sweep, selective criterion, FT recalibration.

---

## 7. Trace pipeline smoke — PASS

- Minimal baseline trace from F01 Engram (float32 concatenated projections)
- Minimal post-intervention re-extraction after smoke apply
- Shapes matched: `[352321536]`
- Cosine-distance metric executed (`1 - cos`)

Smoke distances are **not** scientific outcomes and must not be interpreted as experiment results. **No** base/FT matrix was written.

---

## 8. PEFT / LoRA initialization — PASS

- Frozen LoRA config initialized (`r=16`, `α=32`, dropout `0.05`)
- Targets verified: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `Trainer` constructed on the GPU-loaded PEFT-wrapped model
- **`trainer.train()` was NOT called**
- Neither tennis nor control model was trained

---

## 9. Validator results

| Script | Result |
|--------|--------|
| `validate_frozen_experiment.py` | Previously PASS (exit 0; frozen YAML integrity) |
| Dry validators (`--dry-validate` / finetune dry) | Previously PASS (exit 0) in earlier session |
| `validate_runtime_stack.py` | **Not invoked** in this GPU-first pass |

### `validate_runtime_stack.py` — WARNING / NOT APPLICABLE (low-RAM VM)

**Do not treat stock `validate_runtime_stack.py` as a gate on this g2-standard-4 host** while it still uses CPU-side model materialization.

**Incompatible check:** `check_model_path()` in `research/scripts/validate_runtime_stack.py` (approx. lines 475–481):

```python
model = transformers.AutoModelForCausalLM.from_pretrained(
    str(snap),
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
```

This call has **no** `device_map="cuda"` / `.to("cuda")`, so it builds a **full CPU-resident** bf16 Qwen3-1.7B. On ~15 GiB host RAM with the CUDA torch stack, that path is **OOM-killed (exit 137)**.

**Follow-on blocked by the same load:** `check_minimal_engram()` depends on the model returned by `check_model_path()`.

**Not changed:** validator left unmodified (would be a device-placement engineering fix only; scientific freeze untouched). GPU-first smoke above already covers environment, CUDA, local weights, 28× `down_proj`, and a minimal `get_engram`/`apply_engram` path on the L4.

---

## 10. Warnings / blockers

| Item | Severity | Notes |
|------|----------|-------|
| CPU-side `validate_runtime_stack` model load | **WARNING / N/A** | Insufficient host RAM; use GPU-first path instead |
| Transformers deprecation notice `torch_dtype` → `dtype` | Warning only | Cosmetic; load still succeeds |
| Scientific experiment | Not started | Intentional |

**Blockers for live GPU engineering readiness:** none.

---

## 11. Explicit non-execution confirmation

This verification did **not**:

- start the 20-hour research clock
- add swap
- retry CPU-side full model load as a requirement
- modify frozen configs/data/hyperparameters
- change frozen package pins
- run baseline cloze recall
- calibrate alphas
- fine-tune tennis or control adapters
- compute base/FT matrices, Delta-M, or permutation tests
- run lexical-control experiments

---

## 12. Final decision

**LIVE RUNTIME VERIFICATION: PASS (READY)**  
**RESEARCH CLOCK: NOT STARTED**

The frozen TraceShift stack is confirmed operable on this NVIDIA L4 via GPU-direct placement. Scientific measurement may begin only after an explicit operator decision to start the research clock.
