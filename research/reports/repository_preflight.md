# TraceShift repository preflight (pre–first commit)

**Date:** 2026-09-04  
**Scope:** local hygiene only — no `git init`, no commit, no push, no model/GPU work.

---

## 1. Git repository status

| Question | Answer |
|----------|--------|
| Does `~/Projects/trace-shift/.git/` exist? | **No** |
| Is TraceShift its own Git repository? | **No** |
| What does `git` see from inside the project? | Parent work tree rooted at **`/Users/siddharth`** (home-directory Git repo on branch `main`) |
| Is `Projects/trace-shift/` tracked there? | **No** — currently shows as **untracked** (`?? Projects/trace-shift/`) under the home repo |

**Critical risk before any commit:** do **not** `git add` TraceShift into the home-directory repository. The first TraceShift commit should happen only after a **dedicated** `git init` inside `~/Projects/trace-shift` (or a new remote-backed clone), in a later task.

This preflight task did **not** run `git init`.

---

## 2. Intended files to commit (after dedicated init)

Approximate **safe-to-commit set** with the new root `.gitignore` applied: **~63 files** (source + frozen data + configs + docs + pending STATUS YAML + `.gitignore` + this report).

Categories:

- Root: `requirements.txt`, `.gitignore`
- `research/configs/` — `experiment.yaml`, `environment.yaml`
- `research/data/frozen/` — all frozen YAML scientific assets (+ empty-dir `.gitkeep`s under data/)
- `research/traceshift/` — Python package (`.py` only; `__pycache__` ignored)
- `research/scripts/` — validators and run CLIs
- `research/reports/` — `pre_gpu_audit.md`, `repository_preflight.md`, research log CSV, `.gitkeep`s
- `research/results/raw/**/STATUS*.yaml` and `control_status.yaml` — pending/engineering status only
- `research/README.md`
- Placeholder `.gitkeep` trees under `experiments/`, `results/tables|figures`, etc.

---

## 3. Ignored files / categories

Created root **`.gitignore`** covering:

| Category | Examples |
|----------|----------|
| Env / caches | `.venv/`, `__pycache__/`, `*.py[cod]`, pytest/mypy/ruff caches |
| OS / editor | `.DS_Store`, `.idea/`, `.vscode/` |
| Secrets | `.env*`, `*.pem`, `*.key`, service-account / ADC JSON names, HF token files |
| Model weights / HF caches | `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, `.cache/`, hub dirs |
| Training artifacts | `final_adapter/`, `checkpoint-*/`, wandb/tensorboard local dirs |
| Logs / temp | `*.log`, `tmp/` |

**Explicitly not ignored:** frozen YAML, configs, source, small STATUS YAML/JSON schemas, research reports.

Present locally but ignored: **`.venv/` (~910 MB)** — must never be committed.

---

## 4. Suspected secrets

| Finding | Detail |
|---------|--------|
| In-repo secret pattern scan | **No hits** under `trace-shift` (excluding `.venv`) |
| Suspicious credential filenames | **None** found |
| Outside-repo note | Host SSH `known_hosts` was updated by earlier GCP SSH attempts — **outside this project**; not a TraceShift commit concern |

No secret values are reproduced here.

---

## 5. Large files

| Path | Size | Commit? |
|------|------|---------|
| `.venv/` (tree) | ~910 MB | **No** — ignored |
| Non-venv files | all **&lt; 1 MB** | Yes (content files) |
| `*.safetensors` / `*.pt` / checkpoints in project | **None present** | N/A |

No &gt;50 MB scientific/result files inside the project tree outside `.venv`.

---

## 6. Accidental cloud / runtime artifacts

| Item | Classification |
|------|----------------|
| In-repo GCP scripts, credentials, cloud-init, SSH configs | **None found** |
| In-repo nvidia / VM copy / setup logs | **None found** |
| Local `.venv` from Mac CPU engineering | **Ignore** (`.gitignore`) |
| Host `~/.ssh/known_hosts` entries from GCP SSH | **Outside repo** — leave alone; do not copy into project |
| GCP project still may have VM `traceshift-live-verify` (L4) from a prior session | **Cloud resource**, not a local file — stop/delete via `gcloud` if still running (ops action, not a git file) |

---

## 7. Pending / fabricated-result check

Inspected all `research/results/raw/**/STATUS*.yaml` and `control_status.yaml`:

- Statuses are **pending** / engineering (`pending_*`, `READY_WITH_WARNINGS`, `training_executed: false`, `matrix_executed: false`, etc.).
- No fabricated alpha tables, matrix numeric grids, Delta-M values, p-values, or training metrics files (`base_alphas.yaml`, `base_matrix.json`, `ft_matrix.json`, `delta_analysis.json`, adapters) are present.
- Mentions of “fabricated” are **negations** in status messages only.

**Verdict:** safe to version these pending status artifacts as current engineering state.

---

## 8. Exact actions required before the first Git commit

Do **not** perform destructive deletes of scientific data. Recommended sequence for a **later** commit task:

1. **Do not** `git add Projects/trace-shift` from `/Users/siddharth` (home repo).
2. Optionally add `Projects/` or `Projects/trace-shift/` to the **home** repo’s `.gitignore` so the project never gets swept into that repo accidentally.
3. In a follow-up task: `cd ~/Projects/trace-shift && git init` (dedicated repo only).
4. Confirm `.gitignore` is present (done in this preflight).
5. `git status` and stage only the safe set (no `.venv`, no weights, no secrets).
6. Create GitHub remote / first commit only when explicitly requested.
7. If GCP VM `traceshift-live-verify` is still billed/running, stop or delete it via GCP console/`gcloud` (infrastructure hygiene, separate from git).

---

## Confirmation

This preflight did **not**: initialize a TraceShift Git repo, create a GitHub repository, commit, push, install packages, download weights, use a GPU, run Qwen, or start the research clock.
