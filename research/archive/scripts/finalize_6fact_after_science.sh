# ARCHIVED — moved from research/scripts/. Historical overnight/GPU tooling; paths may point at the original VM layout.
# See research/archive/README.md. Not part of the canonical CLI surface.

#!/usr/bin/env bash
# Wait for 6-fact scientific run to finish, then write reports + commit/push.
# Runs in its own detached tmux so Mac/Cursor disconnect is safe.
set -uo pipefail

cd "${HOME}/trace-shift" || exit 1
REPO="${HOME}/trace-shift"
LOGDIR="${REPO}/research/results/raw/experiment_run"
MASTER="${LOGDIR}/master.log"
EXIT_FILE="${LOGDIR}/overnight_exit.txt"
CONS_DONE="${LOGDIR}/phase_markers/10_consolidation.done"
PY="${REPO}/.venv/bin/python"
FINAL_LOG="${LOGDIR}/finalize.log"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$FINAL_LOG" "$MASTER"; }

log "finalize_watcher_start pid=$$ TMUX=${TMUX:-unset}"

MARK="${LOGDIR}/phase_markers"
REQUIRED_MARKERS=(
  01_baseline_recall
  02_alpha_calibration
  03_base_matrix
  04_tennis_finetune
  05_control_finetune
  06_ft_matrix
  07_delta_analysis
  08_generic_ft_control
  09_lexical_control
  10_consolidation
)
CONS_JSON="${LOGDIR}/consolidation_summary.json"

science_complete() {
  local p
  for p in "${REQUIRED_MARKERS[@]}"; do
    [[ -f "${MARK}/${p}.done" ]] || return 1
  done
  [[ -f "$CONS_JSON" ]] || return 1
  "$PY" - "$CONS_JSON" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    d = json.loads(p.read_text())
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if d.get("all_required_present") else 1)
PY
}

# When invoked from resume wrapper after science exit=0, do NOT poll forever.
# Refuse to finalize unless all markers + consolidation are present.
if ! science_complete; then
  log "ABORT: refused to finalize — markers/consolidation incomplete (overnight_exit alone is never enough)"
  exit 2
fi

log "science_complete markers+consolidation ok — proceeding to reports/git"
sleep 2

log "generating reports via python"
"$PY" - <<'PY' | tee -a "$FINAL_LOG"
import json, math
from pathlib import Path
from datetime import datetime, timezone

root = Path("/home/siddharth/trace-shift/research")
raw = root / "results" / "raw"
tables = root / "results" / "tables"
reports = root / "reports"
reports.mkdir(parents=True, exist_ok=True)

def load_json(p):
    p = Path(p)
    if not p.is_file():
        return None
    return json.loads(p.read_text())

baseline = load_json(raw / "baseline_recall" / "baseline_cloze.json")
alphas_y = raw / "alpha_calibration" / "base_alphas.yaml"
alpha_json = load_json(raw / "alpha_calibration" / "base_alpha_calibration.json")
base_m = load_json(raw / "base_matrix" / "base_matrix.json")
ft_m = load_json(raw / "ft_matrix" / "ft_matrix.json")
delta = load_json(raw / "delta_analysis" / "delta_analysis.json")
generic = load_json(raw / "alternative_hypotheses" / "generic_ft_control_delta_analysis.json")
generic_m = load_json(raw / "alternative_hypotheses" / "generic_ft_control_matrix.json")
lexical = load_json(raw / "alternative_hypotheses" / "lexical_control.json")
cons = load_json(raw / "experiment_run" / "consolidation_summary.json")
phase_times = []
pt = raw / "experiment_run" / "phase_times.jsonl"
if pt.is_file():
    for line in pt.read_text().splitlines():
        line=line.strip()
        if line:
            phase_times.append(json.loads(line))

start_utc = (raw / "experiment_run" / "experiment_start_utc.txt").read_text().strip() if (raw / "experiment_run" / "experiment_start_utc.txt").is_file() else "?"
total_s = (raw / "experiment_run" / "total_scientific_seconds.txt").read_text().strip() if (raw / "experiment_run" / "total_scientific_seconds.txt").is_file() else "?"
exit_rc = (raw / "experiment_run" / "overnight_exit.txt").read_text().strip() if (raw / "experiment_run" / "overnight_exit.txt").is_file() else "?"

# Baseline table
bl_lines = []
if baseline:
    facts = baseline.get("evaluation", {}).get("facts") or []
    for f in facts:
        bl_lines.append(f"| {f.get('fact_id')} | {f.get('entity','')} | {f.get('status')} | {f.get('wins')}/{f.get('n_probes')} |")

# Alphas
import yaml
alpha_rows = []
if alphas_y.is_file():
    ay = yaml.safe_load(alphas_y.read_text())
    # flexible schema
    specs = ay.get("alphas") or ay.get("facts") or ay
    if isinstance(specs, dict):
        for fid in ["F01","F02","F03","F04","F05","F06"]:
            v = specs.get(fid)
            if isinstance(v, dict):
                alpha_rows.append(f"| {fid} | {v.get('status')} | {v.get('alpha')} |")
            elif v is not None:
                alpha_rows.append(f"| {fid} | ? | {v} |")

# Delta primary
t_obs = delta.get("observed_contrast") if delta else None
p_val = delta.get("exact_one_sided_p_value") if delta else None
null = delta.get("null_distribution") if delta else []
t_cross = delta.get("t_cross") if delta else None
mean_tennis = delta.get("mean_tennis_within_delta") if delta else None
mean_unrel = delta.get("mean_unrelated_within_delta") if delta else None
within_cells = delta.get("within_domain_cell_values") if delta else {}

# Generic
t_gen = None
t_diff = None
if generic:
    t_gen = generic.get("T_control") or generic.get("T_generic_within") or generic.get("observed_contrast")
    cmp_ = generic.get("comparison") or {}
    t_diff = cmp_.get("T_tennis_minus_T_control") or cmp_.get("T_tennis_minus_T_generic")
    if t_diff is None and t_obs is not None and t_gen is not None:
        t_diff = float(t_obs) - float(t_gen)
    if generic.get("T_generic_within") is not None:
        t_gen = generic.get("T_generic_within")
    # also try nested
    for k in ("T_generic_within", "mean_in_domain_delta_m_control"):
        pass

# Lexical
t_lex = None
if lexical:
    t_lex = (
        lexical.get("T_lexical_within")
        or (lexical.get("lexical_contrast") or {}).get("T")
        or (lexical.get("descriptive") or {}).get("T_lexical_within")
    )
    # common schema from updated lexical_control
    if t_lex is None:
        dlex = lexical.get("delta_lexical") or lexical.get("lexical") or {}
        t_lex = dlex.get("observed_contrast") or dlex.get("T_lexical_within")
    if t_lex is None and "comparison" in lexical:
        t_lex = lexical["comparison"].get("T_lexical") or lexical["comparison"].get("T_lexical_within")

# FT validity
def ft_valid(kind):
    p = raw / "finetuning" / ("qwen3-1.7b-lora-tennis-primary" if kind=="tennis" else "qwen3-1.7b-lora-control") / "execution_validity.json"
    j = load_json(p)
    return j

tv = ft_valid("tennis")
gv = ft_valid("generic")

status = "COMPLETE" if str(exit_rc) in ("0",) or (cons and cons.get("all_required_present")) else "INCOMPLETE_OR_FAIL"

summary = f"""# TraceShift balanced 6-fact experiment status summary

**Design:** balanced_6fact_3domain (tennis / swimming / acting; two facts each)  
**Run host:** `traceshift-l4` (remote GCP NVIDIA L4)  
**Execution:** persistent tmux session `traceshift-run-6fact`  
**Scientific clock start (UTC):** {start_utc}  
**Approx total scientific wall time:** {total_s} s  
**Overnight exit code:** {exit_rc}  
**Status:** {status}

This document records measured outcomes and engineering notes. It is **not** a polished MATS narrative.

---

## Fact universe (frozen)

| Domain | Facts |
|--------|-------|
| Tennis | F01 Roger Federer, F02 Rafael Nadal |
| Swimming | F03 Michael Phelps, F04 Katie Ledecky |
| Acting | F05 Amitabh Bachchan, F06 Shah Rukh Khan |

Within-domain pairs: F01↔F02, F03↔F04, F05↔F06.

---

## 1. Baseline recall

| Fact | Entity | Status | Wins |
|------|--------|--------|------|
{chr(10).join(bl_lines) if bl_lines else "| (missing) | | | |"}

Artifact: `results/raw/baseline_recall/baseline_cloze.json`

---

## 2. Alpha calibration (BASE, explicit)

| Fact | Status | Alpha |
|------|--------|-------|
{chr(10).join(alpha_rows) if alpha_rows else "| (missing) | | |"}

Artifacts: `results/raw/alpha_calibration/base_alphas.yaml`

---

## 3. Matrices

- M_base 6×6: {"present" if base_m else "MISSING"} (`model_stage={base_m.get("model_stage") if base_m else None}`)
- M_FT tennis 6×6: {"present" if ft_m else "MISSING"} (`model_state={ft_m.get("model_state") if ft_m else None}`)
- M_FT generic 6×6: {"present" if generic_m else "MISSING"} (`model_state={generic_m.get("model_state") if generic_m else None}`)

---

## 4. Fine-tuning validity

- Tennis: {tv}
- Generic control: {gv}

---

## 5. Primary Delta-M (tennis FT) — inferential

- mean tennis within-domain ΔM: `{mean_tennis}`
- mean unrelated within-domain ΔM: `{mean_unrel}`
- **T_obs (T_within):** `{t_obs}`
- Exact one-sided p (C(6,2)=15 cell permutation): `{p_val}`
- Secondary descriptive T_cross: `{t_cross}`
- Within-domain cells: `{json.dumps(within_cells)}`

Null distribution (15 contrasts):
```
{json.dumps(null, indent=2) if null else "missing"}
```

Limitation: exact test over six directed within-domain cells; not population-level inference.

---

## 6. Generic FT control (descriptive)

- T_generic_within: `{t_gen}`
- T_tennis − T_generic: `{t_diff}`
- No second p-value.

Artifacts: `results/raw/alternative_hypotheses/generic_ft_control_*.json`

---

## 7. Lexical-control (descriptive)

- T_lexical_within: `{t_lex}`
- Artifact: `results/raw/alternative_hypotheses/lexical_control.json`

---

## 8. Phase timings

```
{json.dumps(phase_times, indent=2)}
```

---

## 9. Consolidation

```
{json.dumps(cons, indent=2) if cons else "missing"}
```

---

## Integrity

Frozen 6-fact inputs were not rewritten after measurements began for outcome shopping.
Primary inferential result is the tennis within-domain vs unrelated within-domain contrast with exact p over 15 cell assignments.
Generic FT and lexical results are descriptive only.
"""

(reports / "experiment_status_summary.md").write_text(summary)

# Evidence audit (compact)
audit = f"""# TraceShift final evidence / validity audit (balanced 6-fact)

**Generated (UTC):** {datetime.now(timezone.utc).isoformat()}  
**Scientific start:** {start_utc}  
**Total scientific seconds:** {total_s}  
**Overnight exit:** {exit_rc}

## FINAL EVIDENCE AUDIT: {"PASS" if status=="COMPLETE" else "PASS WITH WARNINGS / FAIL"}

### Completed checklist

| Item | Present |
|------|---------|
| Baseline 6 facts | {baseline is not None} |
| Alphas | {alphas_y.is_file()} |
| M_base 6×6 | {base_m is not None} |
| Tennis FT validity | {bool(tv and tv.get("ok"))} |
| Generic FT validity | {bool(gv and gv.get("ok"))} |
| M_FT tennis | {ft_m is not None} |
| M_FT generic | {generic_m is not None} |
| Delta-M tennis + p | {delta is not None} |
| Lexical | {lexical is not None} |
| Consolidation | {bool(cons and cons.get("all_required_present"))} |

### Primary result

- T_obs = `{t_obs}`
- p_exact = `{p_val}`
- T_cross (descriptive) = `{t_cross}`
- T_generic_within (descriptive) = `{t_gen}`
- T_tennis − T_generic = `{t_diff}`
- T_lexical_within (descriptive) = `{t_lex}`

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
"""
(reports / "final_evidence_audit.md").write_text(audit)
print("WROTE", reports / "experiment_status_summary.md")
print("WROTE", reports / "final_evidence_audit.md")
print("STATUS", status)
print("T_obs", t_obs, "p", p_val, "T_generic", t_gen, "T_lex", t_lex)
PY

if ! science_complete; then
  log "ABORT: science incomplete after report generation — not committing"
  exit 2
fi

log "git commit/push (science markers+consolidation verified complete)"
cd "$REPO"
# Stage intended scientific artifacts; exclude weights/adapters/venv/cache
git add \
  research/data/frozen \
  research/configs/experiment.yaml \
  research/traceshift \
  research/scripts \
  research/results/raw/baseline_recall \
  research/results/raw/alpha_calibration \
  research/results/raw/base_matrix \
  research/results/raw/ft_matrix \
  research/results/raw/delta_analysis \
  research/results/raw/alternative_hypotheses \
  research/results/raw/finetuning/*.yaml \
  research/results/raw/finetuning/*.json \
  research/results/raw/finetuning/*/execution_validity.json \
  research/results/raw/finetuning/*/training_metadata.json \
  research/results/raw/finetuning/*/training_metadata.yaml \
  research/results/raw/experiment_run/consolidation_summary.json \
  research/results/raw/experiment_run/phase_times.jsonl \
  research/results/raw/experiment_run/pre_run_frozen_digests.json \
  research/results/raw/experiment_run/experiment_start_utc.txt \
  research/results/raw/experiment_run/experiment_start_epoch.txt \
  research/results/raw/experiment_run/total_scientific_seconds.txt \
  research/results/raw/experiment_run/overnight_exit.txt \
  research/results/raw/experiment_run/phase_markers \
  research/results/tables \
  research/reports/experiment_status_summary.md \
  research/reports/final_evidence_audit.md \
  2>/dev/null || true

# Do not force-add logs if gitignored; try anyway for key non-ignored artifacts
git status -sb | tee -a "$FINAL_LOG"

if git diff --cached --quiet; then
  log "WARN: nothing staged — attempting broader add of results (respecting gitignore)"
  git add research/results/raw research/results/tables research/reports research/data/frozen research/configs research/traceshift research/scripts || true
fi

git commit -m "$(cat <<'EOF'
Balanced 6-fact TraceShift experiment results

EOF
)" 2>&1 | tee -a "$FINAL_LOG"

git push origin main 2>&1 | tee -a "$FINAL_LOG"
git rev-parse HEAD | tee "${LOGDIR}/final_commit_hash.txt" | tee -a "$FINAL_LOG"
git status -sb | tee -a "$FINAL_LOG"
log "finalize_watcher_end"
exit 0
