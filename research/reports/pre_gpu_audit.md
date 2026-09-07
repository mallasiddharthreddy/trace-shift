# TraceShift pre-GPU integration audit

**Audit timestamp (UTC):** 2026-09-04T11:06:05Z  
**Overall status:** `READY_WITH_WARNINGS`  
**20-hour research clock may begin:** **NO** (await GCP CUDA env + local Qwen3-1.7B weights + live module verification)  
**Safe to move engineering work to paid GPU prep:** **YES, with warnings** (code/data wiring passes dry validation; execution prerequisites outstanding)

This is an **engineering audit**, not a scientific result. No model weights were downloaded, no GPU was used, and no Engram / matrix / Delta-M / FT experiment was executed.

---

## 1. Overall status

| Verdict | Meaning |
|---------|---------|
| **READY_WITH_WARNINGS** | Frozen design + pipeline dry validations PASS. Path/schema wiring is consistent. AI-Engram 0.9.0 API signatures match installed package. Live Qwen3-1.7B module checks and CUDA training remain pending. Operational precondition: all four facts must be `selective` before the exact Delta-M permutation can run. |

Not `READY` because weights are absent and live architecture checks are unverified.  
Not `BLOCKED` because no hard code incompatibility was found that would prevent a correctly provisioned GCP run when preconditions are met.

---

## 2. Component status table

| Component | Status | Evidence | Action required |
|-----------|--------|----------|-----------------|
| Frozen facts / extraction / reference / cloze / FT corpora | **PASS** | `validate_frozen_experiment.py` → 75 checks PASS | None |
| Environment deps (torch/transformers/peft/ai-engram) | **PASS** | Local `.venv` 3.12.14; torch 2.14.0; transformers 5.16.1; peft 0.20.0; ai-engram 0.9.0 | On GCP: CUDA-matched torch ≥2.0 |
| AI-Engram API (`get_engram` / `apply_engram`) | **VERIFIED FROM SOURCE/API** | Installed signatures match `engram.extract_engram` / `apply_intervention` kwargs; `inplace=False` default | Live Engram call still **REQUIRES LIVE MODEL** |
| Model loader | **PASS (pending weights)** | `local_files_only`; refuses `allow_download`; explicit `device='cuda'` now refuses silent CPU fallback | Download `Qwen/Qwen3-1.7B`; always pass `--device cuda` on GCP |
| Cloze scorer config | **PASS** | 3 probes/fact; threshold 2; candidates `{tennis,swimming,acting,cricket}`; TF sum-logprob | Live scoring **REQUIRES LIVE MODEL** |
| Alpha calibration pipeline | **PASS (dry)** | Frozen grid enforced; collect-once; selective criterion; writes `base_alphas.yaml` | Run after weights |
| Base matrix | **PASS (dry)** | Explicit only; BASE alphas; `1−cos`; diagonal null; F01–F04 | Run after alphas |
| Fine-tuning (primary) | **PASS (dry)** | Federer/Nadal 40; LoRA r16/α32; out `qwen3-1.7b-lora-tennis-primary/` | Train on GCP CUDA |
| Fine-tuning (control) | **PASS (dry)** | Curie/Armstrong 40; identical hparams; distinct path; `control_status.yaml` | Optional secondary train |
| FT matrix | **PASS (dry)** | Loads `final_adapter`; FT Engrams separate; BASE alphas; same metric | Run after primary FT |
| Delta-M + exact permutation | **PASS (dry) + WARNING** | C(4,2)=6; one-sided p; refuses incomplete matrices with clear message | Requires **all four** facts `selective` |
| Lexical-control alt | **PASS (dry)** | Isolation OK (8 bundles); BASE alphas reused; descriptive compare only | Secondary after primary matrices |
| Qwen 28×`down_proj` / LoRA qkvo | **REQUIRES LIVE MODEL** | Declared in `experiment.yaml` + runtime `_module_exists` / `verify_lora_target_modules` | Verify on first GCP load |
| Device selection (auto) | **WARNING** | Auto path still CUDA→MPS→CPU if `--device` omitted | Always `--device cuda` on GCP |
| `tokenizer_identifier: null` | **WARNING** | Falls back to model id at load time | Pin revision later if needed; not inventing now |
| Precision note | **WARNING** | Config lists `response_trace_storage: float16` but traces built as float32 in code (safer for metrics) | Accept as engineering choice; do not change science |

---

## 3. End-to-end dependency graph

```
baseline cloze (M_base)
    → alpha calibration (explicit) → base_alphas.yaml
        → base matrix (explicit) → base_matrix.json
            → primary LoRA FT (tennis) → …/qwen3-1.7b-lora-tennis-primary/final_adapter/
                → FT matrix (explicit, BASE alphas) → ft_matrix.json
                    → Delta-M + exact permutation → delta_analysis.json

Secondary:
  control LoRA FT (Curie/Armstrong) → …/qwen3-1.7b-lora-control/final_adapter/
  lexical-control BASE+FT matrices → ΔM_lexical → descriptive vs explicit ΔM
```

**Schema checks (source):** writers/readers agree on `base_alphas.yaml`, `base_matrix.json`, `ft_matrix.json`, `final_adapter/`, delta/lexical output paths. No circular imports found.

**Operational precondition (not a silent bug):** matrix pipelines default to `selective` rows only; the frozen exact permutation needs a full off-diagonal 4×4 ΔM. Incomplete matrices are **refused** (clear error), not filled with invented values. Experiment plan: obtain selective alphas for F01–F04 before claiming Delta-M.

---

## 4. Dry-validation results (this audit)

| Check | Exit | Result |
|-------|------|--------|
| `validate_frozen_experiment.py` | 0 | PASS (75 checks) |
| `validate_runtime_stack.py` | 0 | OK; pending weights/modules (33 pending, 0 failed) |
| Cloze config load | 0 | probes/threshold/candidates OK |
| `run_alpha_calibration.py --dry-validate` | 0 | OK |
| `run_base_matrix.py --dry-validate` | 0 | OK |
| `run_finetune.py --dataset primary --dry-validate` | 0 | OK + comparability |
| `run_finetune.py --dataset control --dry-validate` | 0 | OK + control_status |
| `run_ft_matrix.py --dry-validate` | 0 | OK |
| `run_delta_analysis.py --dry-validate` | 0 | OK (+ completeness precondition documented) |
| `run_lexical_control.py --dry-validate` | 0 | OK; isolation_ok |

No dry validation failed due to a package API mismatch.

---

## 5. Engineering fixes applied during audit

Scientific decisions / frozen YAML data were **not** changed.

1. **`delta_analysis.validate_matrix_pair`** — clearer refusal when off-diagonal cells are null, documenting the all-four-selective precondition for the exact test.
2. **`base_matrix.load_calibrated_base_alphas`** — reads default path from `intervention.calibrated_alphas_path` in `experiment.yaml` (same string as before; removes latent hard-code drift).
3. **`model_loader.resolve_device`** — explicit `device='cuda'` raises if CUDA unavailable (no silent CPU when CUDA was requested).

---

## 6. GCP external prerequisites

| Item | Requirement |
|------|-------------|
| Python | **3.12.x** dedicated env (not system 3.14) |
| CUDA / PyTorch | CUDA-matched **torch≥2.0** (A100-compatible wheel from pytorch.org index) |
| Packages | `pip install -r requirements.txt` after torch; pin **`ai-engram==0.9.0`**; transformers≥4.51; peft≥0.13; tqdm; PyYAML |
| Model weights | **`Qwen/Qwen3-1.7B`** in HF cache (`models--Qwen--Qwen3-1.7B`) |
| Auth | Public model; HF token only if hub rate-limits require it |
| Storage (approx.) | ~4–8 GB model (bf16) + env (~few GB) + checkpoints/adapters (LoRA small) + Engram scratch; budget **≥40 GB** free on instance disk for comfort |
| Device policy | Always pass **`--device cuda`** to experiment CLIs |

---

## 7. Commands that SHOULD eventually run on GCP

```bash
# 1) Environment
python3.12 -m venv .venv && source .venv/bin/activate
# install CUDA-matched torch first (example index — pick cuXXX for the image)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 2) Dependency / API verification
python -c "import torch,transformers,peft,engram,yaml,importlib.metadata as m; assert torch.cuda.is_available(); assert m.version('ai-engram')=='0.9.0'"
python research/scripts/validate_runtime_stack.py

# 3) Model availability (download only in explicit GPU-prep step)
# huggingface-cli download Qwen/Qwen3-1.7B   # or equivalent; not run in this audit
python -c "from pathlib import Path; import os; ..."  # confirm snapshot exists

# 4) Final pre-experiment dry validators (still no research clock)
python research/scripts/validate_frozen_experiment.py
python research/scripts/run_alpha_calibration.py --dry-validate
python research/scripts/run_base_matrix.py --dry-validate
python research/scripts/run_finetune.py --dataset primary --dry-validate
python research/scripts/run_finetune.py --dataset control --dry-validate
python research/scripts/run_ft_matrix.py --dry-validate
python research/scripts/run_delta_analysis.py --dry-validate
python research/scripts/run_lexical_control.py --dry-validate

# 5) Only after live module check succeeds and research plan is signed off:
# python research/scripts/run_baseline_cloze.py --device cuda
# ... then alpha → base matrix → FT → FT matrix → delta ...
```

**Do not** start the 20-hour research clock until: CUDA torch verified, weights local, `validate_runtime_stack.py` shows model modules verified (not pending), and operators accept the all-four-selective Delta precondition.

---

## 8. Confirmation of non-execution

This audit did **not**:

- download Qwen3-1.7B weights;
- use a GPU / paid instance;
- fine-tune;
- run AI-Engram extraction or intervention;
- compute real alphas, matrices, Delta-M, or p-values;
- run alternative-hypothesis experiments;
- begin the 20-hour research clock.
