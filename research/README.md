# MATS 12.0 — Trace-Shift Research Experiment

20-hour research experiment workspace. Structure only; pipeline and results are intentionally not implemented here.

## 1. Research question

Does narrow fine-tuning on a small tennis-biography corpus change intervention–response relationships among related factual traces more than among unrelated controls?

We test whether fine-tuning on tennis biographies selectively reshapes how intervening on one tennis-related factual association affects other tennis-related associations (vs. swimming/acting controls), relative to a fixed base model.

## 2. Frozen factual associations

Canonical spec: `data/frozen/facts.yaml` (IDs `F01`–`F04`).

| ID  | Entity           | Domain   | Group       | Tennis FT |
|-----|------------------|----------|-------------|-----------|
| F01 | Roger Federer    | tennis   | in_domain   | yes       |
| F02 | Rafael Nadal     | tennis   | in_domain   | yes       |
| F03 | Michael Phelps   | swimming | control     | no        |
| F04 | Amitabh Bachchan | acting   | control     | no        |

**Why these four:** Minimal contrast for the research question — two facts in the tennis fine-tuning domain and two facts outside it — without claiming an external semantic hierarchy, DBpedia labels, or ontology levels. Federer/Nadal share the FT domain; Phelps (swimming) and Bachchan (acting) are unrelated comparison facts.

**Stability rule:** Do not add facts during the experiment unless the current design fails a pre-specified sanity check and the change is recorded in `reports/research_log/`.

Not decided yet: intervention-response metric name details and GPU runtime validation. Numerical recall/alpha/FT outcomes are unknown until models are run.

## 2b. Frozen extraction sets and reference corpora

Canonical specs:

- `data/frozen/extraction_sets_explicit.yaml`
- `data/frozen/extraction_sets_lexical.yaml`
- `data/frozen/reference_corpus.yaml`

**Design (locked):**

- Each fact `F01`–`F04` has a **5-sentence** held-out extraction target set per variant.
- Variants are separate frozen conditions: **`explicit`** (domain/attribute stated openly) and **`lexical_control`** (same semantic target; avoid unnecessary domain-label repetition).
- For each variant, there is **one shared reference corpus** = concatenation of that variant’s four target sets (**20 sentences**, order F01→F04).
- Target sentences are **members of** the corresponding shared reference corpus; no extra reference-only factual statements were added in this freeze.
- The same reference corpus is shared across all four facts within a variant.
- Extraction sentences must remain **held out** from fine-tuning data (not copied or trivially paraphrased into FT examples).

## 2c. Frozen cloze probes and alpha-calibration protocol

Canonical specs:

- `data/frozen/cloze_probes.yaml`
- `configs/experiment.yaml` → `alpha_calibration` (+ `intervention.alpha_grid`)

**Why fixed cloze probes:** Same short, closed-form prefixes are used for baseline recall on `M_base`, post-FT recall on `M_FT`, and base-model α dose selection — so pre/post comparisons are not confounded by changing question wording or open-ended decoding.

**Retrieval gate:** Each fact has **3** cloze probes (entity named; one blank for the attribute). Rank the expected answer against a shared candidate set `{tennis, swimming, acting, cricket}` via **teacher-forced token log-probability**. Fact-level recall **PASS** if the correct answer wins on **≥2 of 3** probes; otherwise **FAIL**.

**Base selective-α criterion:** After baseline PASS, sweep α ∈ `{0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0}` on `M_base` only. Choose the **smallest** α such that (a) the target fails its gate and (b) no other fact undergoes pass→fail. Mark `selective` if both hold; else fall back to smallest target-erasing α as `nonselective` (not primary unless research-logged). Do not auto-extend the grid.

**Why base alphas are held fixed:** Primary pre/post-FT comparison applies the **same** base-calibrated α values to both `M_base` and `M_FT` intervention–response matrices. FT re-calibration is secondary sensitivity only.

**Role distinctions:** baseline recall = unintervened retrieval; target erasure = calibration dose-finding; intervention–response measurement = the scientific matrix analysis. Alpha calibration is **behavioral dose selection**, not the scientific result. **All numerical results remain unknown.**

## 2d. Frozen fine-tuning datasets

Canonical specs:

- `data/frozen/finetuning_dataset_spec.yaml` — primary tennis biographies (`tennis_biography_ft`)
- `data/frozen/control_finetuning_dataset_spec.yaml` — matched unrelated control (`control_general_ft`)

**Sizes:** primary = **40** examples (**20** Federer / **20** Nadal); control = **40** examples (**20** Marie Curie / **20** Neil Armstrong).

**Why Federer/Nadal:** They are the two in-domain facts (`F01`/`F02`). Narrow FT on their biographies is the treatment that should, under the research hypothesis, reshape intervention–response links among related tennis traces more than among unrelated ones.

**Why F03/F04 are excluded from primary FT:** Phelps and Bachchan are evaluation controls. Including them in training would contaminate the unrelated-trace comparison.

**Why training is disjoint from extraction/evaluation:** Extraction sentences and cloze probes estimate and gate the same associations; using them (or trivial paraphrases) as FT text would confound trace measurement with train/test leakage. FT examples are separate biographical statements.

**Control purpose:** Same size/format, unrelated subjects, no Federer/Nadal — tests whether matrix changes are generic fine-tuning effects (alternative hypothesis A) rather than tennis-domain-specific.

Training hyperparameters (LR, epochs, LoRA, etc.) are **not** chosen in this freeze; see §2e for the frozen base model.

## 2e. Frozen base model

**`Qwen/Qwen3-1.7B`** is the frozen base model (`M_base`). Config: `configs/experiment.yaml` → `model.model_id` / `model.base_model_id`.

It was selected primarily to reuse the already validated AI-Engram/Qwen implementation from the previous TraceShift/NeurIPS experiment and reduce engineering risk for a 20-hour MATS run. The primary fine-tuned model will be produced by fine-tuning this same base checkpoint on the frozen tennis corpus.

No model execution or compatibility verification is claimed yet (later pre-GPU step). GPU runtime configuration remains unfrozen; see §2f for frozen fine-tuning hyperparameters.

## 2f. Frozen fine-tuning configuration

Canonical spec: `configs/experiment.yaml` → `fine_tuning`.

**Method:** LoRA / PEFT (not full-model FT) on `Qwen/Qwen3-1.7B`.

**Major hyperparameters:** seed `42`; `3` epochs; LR `2e-4`; AdamW + cosine; batch `1` × grad-accum `4` (effective `4`); max seq len `256`; weight decay `0`; warmup `5` steps; LoRA `r=16`, `α=32`, dropout `0.05`, targets `q/k/v/o_proj`.

**Same-procedure requirement:** Primary (tennis) and control (Curie/Armstrong) runs use **identical** hyperparameters and training procedure; **only the training corpus differs**.

**Why simple:** No validated Qwen3-1.7B SFT recipe was present in this repo; LoRA minimizes compute/engineering risk for a 40-example narrow-domain model-diff. Configuration is conservative by design.

**Not claimed:** Actual training performance, loss curves, or runtime compatibility have **not** been measured (later pre-GPU engineering task).

## 2g. Frozen AI-Engram configuration

Canonical spec: `configs/experiment.yaml` → `ai_engram`.

**Version:** `ai-engram==0.9.0` (import `engram`). Preferred over previously used **0.8.0** as the pinned official closed-form release for this experiment with the faster deterministic implementation.

**API:** `get_engram(model, tokenizer, forget=..., total=...)` then `apply_engram(model, engram, alpha=...)`. Collect once per model/fact/reference config; reuse the Engram for alpha sweeps.

**Target modules:** all Qwen3-1.7B hidden-layer `mlp.down_proj` modules `model.layers.0`–`27`.

**Precision:** weights `bfloat16`; Engram projection/intervention arithmetic `float32`; response-trace storage `float16`; metric computation and baseline traces `float32`.

**Alpha rule:** calibrate on `M_base` with the frozen behavioral criterion; hold those alphas fixed for the primary base-vs-FT comparison; FT recalibration is secondary only.

**Caveats:** Engram = fact-conditioned parameter projection (not a literal memory slot; not raw base−FT weight diff). Do not inject an FT Engram into the base model. Compare separate measurements in each model on the same extraction/reference inputs.

**No execution yet** — install/runtime verification is a later pre-GPU engineering task.

## 2h. Frozen-spec validator

Entry point: `scripts/validate_frozen_experiment.py`

```bash
python scripts/validate_frozen_experiment.py
```

This is a **pre-experiment integrity check** over frozen YAML/config only (facts, extraction sets, reference corpora, cloze probes, FT corpora, and `configs/experiment.yaml`). Run it before any model or pipeline execution.

It does **not** validate model loading, AI-Engram runtime/API compatibility, training, or GPU behavior.

## 2i. Pre-GPU runtime validation

Entry point: `scripts/validate_runtime_stack.py`

```bash
python scripts/validate_runtime_stack.py
```

Engineering validation of the **Qwen/Qwen3-1.7B + AI-Engram 0.9.0** software path against the frozen config — before paid GPU time.

**What it verifies when prerequisites exist:** environment/package versions, CUDA probe, local model load (`local_files_only`, no download), all 28 `mlp.down_proj` module paths, frozen YAML/config wiring, and optionally one minimal `get_engram` construction (no `apply_engram`, no alpha/matrix/FT).

**What remains unverified until prerequisites are available:** any step skipped when torch/transformers/peft/`ai-engram==0.9.0` or local `Qwen3-1.7B` weights are missing. The script does **not** install packages or download weights.

This is **engineering validation only** and does **not** count as a research result.

## 2j. Environment setup (Python 3.12)

Specs:

- `configs/environment.yaml` — environment policy and dependency notes
- repo-root `requirements.txt` — pip-installable direct dependencies

**Python:** **3.12** in a dedicated venv/conda env for the research experiment. Do **not** use the host system Python 3.14. 3.12 is within official ranges for `ai-engram==0.9.0` (≥3.9) and current Transformers/PEFT (≥3.10), and is a stable research default.

**Direct dependencies:** `ai-engram==0.9.0` (exact), `torch>=2.0`, `transformers>=4.51.0`, `peft>=0.13.0`, `tqdm>=4.60`, `PyYAML>=6.0`.

**CUDA:** The CUDA-specific PyTorch wheel/index is **not** hard-coded here; resolve it during GCP environment setup once the GPU/runtime is known (still satisfying `torch>=2.0`).

**Install outline (GCP / dedicated env only):** create Python 3.12 env → install CUDA-matched `torch` if on GPU → `pip install -r requirements.txt` → run `scripts/validate_runtime_stack.py` before paid experiment time.

**Local `.venv` status:** A dedicated repo-root `.venv` was created successfully with Python **3.12.14**. Direct dependency versions are recorded under `configs/environment.yaml` → `local_venv.installed_versions`. Package/API checks (`import engram`, `get_engram` / `apply_engram`) succeeded. **This environment verification does not constitute model or experiment execution** (no Qwen load, no Engram extraction, no FT/alpha/matrix/GPU runs).

## 2k. Inference layer

Package: `research/traceshift/` · CLI: `scripts/run_baseline_cloze.py`

```bash
.venv/bin/python research/scripts/run_baseline_cloze.py
```

Reusable **model-loading + deterministic cloze-scoring** layer for later pipeline stages.

- **Loader** (`traceshift.model_loader`): reads `model_id` / dtype from frozen `configs/experiment.yaml` (not hard-coded); loads tokenizer + model with `local_files_only`; auto device CUDA→MPS→CPU; clear error if weights are missing (no download).
- **Cloze scoring** (`traceshift.cloze`): teacher-forced **sum** of candidate token log-probabilities over the frozen candidate set; per-probe ranking; frozen **3-probe / 2-of-3** PASS rule (ties vs expected = loss).
- **Local execution** is engineering validation only. **No research experiment has started** — this layer does not run AI-Engram, alpha calibration, interventions, matrices, or fine-tuning.

## 2l. AI-Engram extraction/intervention layer

Package module: `research/traceshift/engram.py`

Frozen instrument: **AI-Engram v0.9.0** (`get_engram` / `apply_engram`).

- **Target/reference:** for each fact + variant, `forget` = that fact’s 5-sentence extraction set; `total` = the matching shared reference corpus (20 sentences).
- **Variant isolation:** `explicit` and `lexical_control` are passed explicitly and must not be mixed (mismatch raises).
- **Collect once / reuse for alpha:** extract with `get_engram` once; later alpha sweeps call `apply_engram` on the stored Engram (no recollect). Default intervention uses `inplace=False` so the canonical model is not permanently mutated.
- **Trace:** concatenated float32 flattened layer projections across configured `mlp.down_proj` modules — an AI-Engram-derived / fact-conditioned parameter projection, not a literal memory slot.
- **Status:** API + frozen data resolution validated without weights. **No actual model Engram extraction has been performed yet** (Qwen3-1.7B not local).

## 2m. Alpha calibration pipeline

Package: `research/traceshift/alpha_calibration.py` · CLI: `scripts/run_alpha_calibration.py`

```bash
.venv/bin/python research/scripts/run_alpha_calibration.py --dry-validate
.venv/bin/python research/scripts/run_alpha_calibration.py   # when local weights exist
```

BASE-model **behavioral dose selection** only (not a scientific effect-size result).

- **Baseline recall prerequisite:** all four facts scored with frozen cloze probes; FAIL → `baseline_recall_failed` (no alpha).
- **Frozen alpha grid:** `{0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0}` — never auto-extended.
- **Selective criterion:** smallest α with (a) target 2-of-3 FAIL and (b) zero PASS→FAIL collateral; else smallest target-erasure α as `nonselective`.
- **Base-model only:** FT recalibration is out of scope here.
- **Collect once / reuse Engram:** primary path uses **explicit** extraction + shared explicit reference; Engram collected once per eligible fact, then reused across the alpha sweep on fresh model copies.
- **Outputs:** `results/raw/alpha_calibration/` + `results/tables/alpha_calibration_summary.*` — base alphas become the **fixed primary** doses for later base and FT matrices.
- **Status:** pipeline implemented; **no calibration results yet** because Qwen3-1.7B is not available locally (dry-validate only).

## 2n. Base intervention-response matrix

Package: `research/traceshift/base_matrix.py` (+ `metrics.py`) · CLI: `scripts/run_base_matrix.py`

```bash
.venv/bin/python research/scripts/run_base_matrix.py --dry-validate
.venv/bin/python research/scripts/run_base_matrix.py   # when weights + base_alphas.yaml exist
```

**Definition:** `M_base[i,j]` = response of fact **j** after intervention on fact **i**.

- **Primary metric:** `cosine_distance = 1 - cosine_similarity`, where similarity matches the prior TraceShift/AI-Engram global float32 flattened-projection convention.
- **Baseline traces:** Engrams for F01–F04 collected once (explicit); TraceVectors derived and reused for every row.
- **One row = one independent intervention** on a fresh model copy (`apply_engram(..., inplace=False)`); canonical BASE is not mutated; the same collected Engram is applied at the **fixed calibrated BASE alpha** (no recollect).
- **Eligibility:** default rows are `selective` only; missing alpha refused; `baseline_recall_failed` / `no_erasure_in_grid` / `nonselective` excluded unless an explicit override is recorded (no invented alphas).
- **Variant:** primary matrix is **explicit** only (lexical-control later).
- **Diagonal:** null/`—` (fact not compared to itself for primary interaction analysis).
- **Outputs:** `results/raw/base_matrix/base_matrix.json`, `results/tables/base_matrix.md` (pending: `STATUS.yaml` only until weights + alphas exist).
- **Status:** pipeline implemented; **no matrix results yet** (weights / calibrated alphas pending).

## 2o. Fine-tuning pipeline

Package: `research/traceshift/finetune.py` (+ `finetune_data.py`) · CLI: `scripts/run_finetune.py`

```bash
.venv/bin/python research/scripts/run_finetune.py --dataset primary --dry-validate
.venv/bin/python research/scripts/run_finetune.py --dataset control --dry-validate
.venv/bin/python research/scripts/run_finetune.py --dataset primary   # when local weights exist
```

**Primary vs control:** `--dataset primary` → frozen Federer/Nadal corpus (`tennis_biography_ft`); `--dataset control` → frozen Curie/Armstrong corpus (`control_general_ft`). Selection is required and explicit — no silent cross-fallback.

**Identical procedure:** same LoRA/SFT code path, same seed, same hyperparameters from `configs/experiment.yaml` → `fine_tuning`. The **only intended difference** between runs is the training corpus.

**Frozen LoRA / train config (consumed from config, not hard-coded in the trainer):** seed `42`; `3` epochs; LR `2e-4`; AdamW + cosine; batch `1` × accum `4`; max seq `256`; weight decay `0`; warmup `5`; LoRA `r=16`, `α=32`, dropout `0.05`, targets `q/k/v/o_proj`.

**Outputs:**
- primary → `results/raw/finetuning/qwen3-1.7b-lora-tennis-primary/`
- control → `results/raw/finetuning/qwen3-1.7b-lora-control/`
- each run writes `final_adapter/` + `training_metadata.{yaml,json}` when training actually executes

**Dry-validate vs training:** `--dry-validate` checks dataset identity/counts, frozen hparams, output paths, and preprocessing config without loading weights. Full training requires local `Qwen/Qwen3-1.7B` and verifies LoRA target modules exist before `Trainer.train()`.

**Reproducibility:** Python/NumPy/Torch seeds are set; CUDA/cuDNN may still be nondeterministic — bitwise-identical reruns are not claimed.

**Status:** pipeline implemented; **no training results yet** (weights pending).

## 2o-bis. Unrelated fine-tuning control

Same CLI / trainer as §2o — **no second training stack**.

```bash
.venv/bin/python research/scripts/run_finetune.py --dataset control --dry-validate
```

**Purpose:** Test the alternative explanation that observed intervention–response / Delta-M changes after tennis FT are merely **generic fine-tuning effects**, rather than domain-local to Federer/Nadal biographies.

**Procedure:** Exact same base model, LoRA config, optimizer, LR, epochs, batch/accum, seq length, seed, preprocessing, and metadata conventions as the primary tennis run. **Only the training corpus differs** — frozen `control_general_ft` (Marie Curie + Neil Armstrong, 40 examples) at `data/frozen/control_finetuning_dataset_spec.yaml`.

**Output (when executed later):** `results/raw/finetuning/qwen3-1.7b-lora-control/` with `final_adapter/` + training metadata marked `unrelated_finetuning_control: true`.

**Status:** control path dry-validated; **control training has not been executed** (`results/raw/finetuning/control_status.yaml`).

## 2p. Fine-tuned intervention-response matrix

Package: `research/traceshift/ft_matrix.py` · CLI: `scripts/run_ft_matrix.py`

```bash
.venv/bin/python research/scripts/run_ft_matrix.py --dry-validate
.venv/bin/python research/scripts/run_ft_matrix.py   # when base weights + primary LoRA + base_alphas exist
```

**Definition:** `M_FT[i,j]` = response of fact **j** after independently intervening on fact **i** in the **primary fine-tuned** model (same cosine-distance measurement as `M_base`).

- **FT Engrams:** collected from the primary tennis LoRA model (not reused from BASE). Metadata records `model_state=primary_finetuned` and the checkpoint path.
- **Fixed alphas:** BASE-calibrated `alpha_i` from `base_alphas.yaml` — **no FT recalibration**, no new/random alphas.
- **Same explicit corpus:** frozen 5-sentence targets + shared explicit reference (lexical-control excluded).
- **One row = one independent intervention** on a fresh FT-model copy (`inplace=False`); canonical FT model untouched; baseline FT traces computed once.
- **Eligibility:** same as base matrix (`selective` by default).
- **Outputs:** `results/raw/ft_matrix/ft_matrix.json`, `results/tables/ft_matrix.md` (pending: `STATUS.yaml` until weights + primary checkpoint + alphas exist).
- **Status:** pipeline implemented; **no matrix results yet**.

## 2q. Delta-M and statistical analysis

Package: `research/traceshift/delta_analysis.py` · CLI: `scripts/run_delta_analysis.py`

```bash
.venv/bin/python research/scripts/run_delta_analysis.py --dry-validate
.venv/bin/python research/scripts/run_delta_analysis.py   # when both matrix JSON files exist
```

**Delta-M:** `Delta_M[i,j] = M_FT[i,j] − M_base[i,j]` (same fact order; diagonal null).

**Primary cells:**
- In-domain: `F01→F02`, `F02→F01`
- Control (in-domain→control): `F01→F03`, `F01→F04`, `F02→F03`, `F02→F04`

**Primary contrast:** `T = mean(in-domain Delta-M) − mean(control Delta-M)`.

**Exact permutation test:** hold observed Delta-M fixed; enumerate all `C(4,2)=6` assignments of two in-domain vs two control facts; recompute `T` under each assignment; one-sided exact p-value = fraction with `T ≥ T_obs` (observed assignment included). Not a population generalization from four facts.

**Why Delta-M:** the primary scientific question is whether tennis FT **changes** in-domain interactions more than the selected controls; inference is on that pre/post contrast, not separate tests on `M_base` or `M_FT`.

**Outputs:** `results/raw/delta_analysis/delta_analysis.json`, `results/tables/delta_analysis.md` (pending: `STATUS.yaml` until both matrices exist).

**Status:** pipeline implemented; **no numerical analysis result yet**.

## 2r. Lexical-control alternative hypothesis

Package: `research/traceshift/lexical_control.py` · CLI: `scripts/run_lexical_control.py`

```bash
.venv/bin/python research/scripts/run_lexical_control.py --dry-validate
.venv/bin/python research/scripts/run_lexical_control.py   # when weights + primary LoRA + base_alphas exist
```

**What is tested:** Whether the primary tennis-vs-control pre/post-FT interaction pattern depends mainly on **explicit lexical/domain cues** in extraction text (alternative explanation), rather than the intended factual associations.

**Why lexical-control:** Same four facts, same BASE and primary FT models, same BASE-calibrated alphas, same intervention–response metric and fact order — but frozen `lexical_control` target sets + `shared_reference_lexical_control` instead of explicit. Fine-tuning corpus is unchanged. Alphas are **not** recalibrated on lexical text.

**Procedure:** `M_base_lexical` → `M_FT_lexical` (FT Engrams extracted separately) → `Delta_M_lexical = M_FT_lexical − M_base_lexical` → descriptive comparison to primary explicit Delta-M (direction/magnitude). Does **not** replace the primary exact permutation test.

**Outputs:** `results/raw/alternative_hypotheses/lexical_control.json`, `results/tables/lexical_control.md` (pending status until prerequisites exist).

**Status:** secondary pipeline implemented; **no results yet**.

## 2s. Pre-GPU integration audit

Engineering audit (not a research result): [`reports/pre_gpu_audit.md`](reports/pre_gpu_audit.md) · status [`results/raw/pre_gpu_audit/STATUS.yaml`](results/raw/pre_gpu_audit/STATUS.yaml).

- **Overall:** `READY_WITH_WARNINGS` — dry validations PASS; wiring/API consistent with `ai-engram==0.9.0`.
- **Not begun:** paid GPU execution, weight download in this audit, or the **20-hour research clock** (`research_clock_may_begin: false`).
- **Before clock start on GCP:** CUDA torch, local `Qwen/Qwen3-1.7B`, live module verification, and `--device cuda` on all experiment CLIs.

## 2t. Repository preflight (pre–first commit)

Hygiene note only (not scientific): see [`reports/repository_preflight.md`](reports/repository_preflight.md).

- Root `.gitignore` excludes `.venv/`, caches, secrets, weights, and training checkpoints; frozen data/configs/source/docs/pending STATUS files remain versionable.
- **`~/Projects/trace-shift` is not yet its own Git repository** (no local `.git/`). It currently appears only as an untracked path under a parent home-directory Git work tree — do not first-commit TraceShift into that parent repo.
- No commit/push was performed in the preflight.

## 3. Frozen experimental architecture

Models:

- `M_base` — base model (identifier set in `configs/experiment.yaml`)
- `M_FT` — same architecture after tennis-biography fine-tuning

Shared frozen assets (under `data/frozen/`):

- Facts: associations frozen (`facts.yaml`)
- Extraction sets + shared reference corpora: frozen (`extraction_sets_*.yaml`, `reference_corpus.yaml`)
- Cloze probes + alpha-calibration protocol: frozen (`cloze_probes.yaml`, `configs/experiment.yaml`)
- Fine-tuning corpora: frozen (`finetuning_dataset_spec.yaml`, `control_finetuning_dataset_spec.yaml`)
- Fine-tuning method/hyperparameters: frozen (`configs/experiment.yaml` → `fine_tuning`)
- AI-Engram 0.9.0 config: frozen (`configs/experiment.yaml` → `ai_engram`)

Planned experimental stages (directories under `experiments/`; not implemented):

1. **Baseline cloze recall** — behavioral recall on frozen probes for `M_base` (and later `M_FT`)
2. **Alpha calibration** — per-fact AI-Engram α calibration on `M_base` only
3. **Base intervention–response matrix** — intervene on each fact, measure responses across facts (calibrated α)
4. **Fine-tuning** — produce `M_FT` from the frozen tennis FT dataset spec
5. **FT matrix** — same interventions/alphas on `M_FT`
6. **Delta analysis** — Δ = matrix(`M_FT`) − matrix(`M_base`)
7. **Statistics** — significance / randomization tests on Δ structure
8. **Alternative hypotheses**
   - **A.** Unrelated / general fine-tuning control
   - **B.** Lexical-control extraction set

Outputs land in `results/`; write-ups in `reports/`.

## 4. Directory structure

```
research/
├── README.md                 # This file
├── configs/                  # experiment.yaml, environment.yaml
├── data/
│   ├── raw/                  # Incoming source material (unprocessed)
│   ├── processed/            # Intermediate cleaned artifacts
│   ├── frozen/               # Locked associations, extraction, cloze, FT corpora, reference
│   └── metadata/             # Dataset / run metadata
├── experiments/
│   ├── baseline_recall/
│   ├── alpha_calibration/
│   ├── base_matrix/
│   ├── finetuning/
│   ├── ft_matrix/
│   ├── delta_analysis/
│   ├── statistics/
│   └── alternative_hypotheses/
├── results/
│   ├── tables/
│   ├── figures/
│   └── raw/
├── reports/
│   ├── research_log/         # Time log for the 20-hour window
│   ├── draft/                # Paper / note drafts
│   └── application/          # MATS application materials
└── scripts/                  # validators, cloze, alpha, matrices, finetune, delta, lexical
├── traceshift/               # … + lexical_control (secondary alt hypothesis)
```

## 5. Intentionally unimplemented

Still not implemented:

- Experimental pipeline code (extraction, intervention, matrix construction, recall/calibration runners)
- AI-Engram extraction runs
- Model fine-tuning / training runs
- AI-Engram install/runtime verification and any Engram extraction/apply runs
- Runtime compatibility verification of the frozen LoRA config
- Any numerical recall, alpha, matrix, training, or statistical results
- GPU usage or environment / package changes

FT corpora, base model ID, FT hyperparameters, and AI-Engram 0.9.0 config are frozen; no execution has occurred.

## 6. Gate rule — no GPU until local pipeline is validated

**No GPU experiment begins until the full local pipeline is validated.**

That means: end-to-end dry runs on CPU (or otherwise non-GPU) with frozen schemas, config wiring, and I/O paths must succeed and be logged before any GPU fine-tuning, intervention sweep, or matrix collection starts. Document validation in `reports/research_log/` before requesting GPU time.
