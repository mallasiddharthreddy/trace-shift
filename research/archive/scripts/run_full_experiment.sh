# ARCHIVED — moved from research/scripts/. Historical overnight/GPU tooling; paths may point at the original VM layout.
# See research/archive/README.md. Not part of the canonical CLI surface.

#!/usr/bin/env bash
# TraceShift end-to-end scientific experiment orchestrator (GPU / tmux).
# Resume-safe: skips phases whose completion markers already exist and look valid.
# Does NOT modify frozen scientific inputs. Does NOT start from Mac-side processes.
set -uo pipefail

REPO="${HOME}/trace-shift"
cd "$REPO" || exit 1
PY="${REPO}/.venv/bin/python"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RAW="${REPO}/research/results/raw"
LOGDIR="${RAW}/experiment_run"
MARK="${LOGDIR}/phase_markers"
mkdir -p "$LOGDIR" "$MARK" \
  "${RAW}/baseline_recall" \
  "${REPO}/research/reports/research_log"

MASTER_LOG="${LOGDIR}/master.log"
TIME_LOG="${REPO}/research/reports/research_log/time_log.csv"
PHASE_TIMES="${LOGDIR}/phase_times.jsonl"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER_LOG"; }

# Append lightweight research time-log row (approximate wall times only).
append_time() {
  local elapsed="$1" task="$2" result="$3" notes="$4"
  # Ensure header
  if [[ ! -s "$TIME_LOG" ]]; then
    echo "timestamp,elapsed_project_time,task,result,notes_decision" > "$TIME_LOG"
  elif ! head -1 "$TIME_LOG" | grep -q "timestamp,elapsed_project_time"; then
    echo "timestamp,elapsed_project_time,task,result,notes_decision" > "$TIME_LOG"
  fi
  printf '%s,%s,"%s","%s","%s"\n' "$(ts)" "$elapsed" "$task" "$result" "$notes" >> "$TIME_LOG"
}

record_phase() {
  local phase="$1" start_epoch="$2" result="$3" notes="$4"
  local end_epoch now_elapsed
  end_epoch=$(date +%s)
  now_elapsed=$(( end_epoch - start_epoch ))
  local total=$(( end_epoch - EXPERIMENT_START_EPOCH ))
  printf '{"phase":"%s","result":"%s","elapsed_s":%s,"total_scientific_s":%s,"notes":"%s","utc":"%s"}\n' \
    "$phase" "$result" "$now_elapsed" "$total" "$notes" "$(ts)" >> "$PHASE_TIMES"
  append_time "${total}s" "$phase" "$result" "phase_wall=${now_elapsed}s; ${notes}"
  if [[ "$phase" == "${CURRENT_PHASE:-}" ]]; then CURRENT_PHASE=""; CURRENT_PHASE_T0=""; fi
}

gc_cuda() {
  "$PY" - <<'PY'
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"cuda_mem_after_gc allocated_gb={alloc:.3f} reserved_gb={reserved:.3f}")
print("gc_ok")
PY
}

mark_done() {
  local phase="$1"
  echo "$(ts)" > "${MARK}/${phase}.done"
}

is_done() {
  local phase="$1"
  [[ -f "${MARK}/${phase}.done" ]]
}

run_cmd() {
  local phase="$1"
  shift
  local logfile="${LOGDIR}/${phase}.log"
  log "BEGIN ${phase}: $*"
  # Do NOT enable set -e here — that would leak into the caller and turn
  # later benign failures (and mid-file-edit oddities) into hard aborts.
  set +e
  "$@" > >(tee -a "$logfile" "$MASTER_LOG") 2>&1
  local rc=$?
  set +e
  log "END ${phase} exit=${rc}"
  return "$rc"
}

# -------------------- FT execution-validity gate --------------------
validate_ft() {
  local kind="$1"  # primary|control
  "$PY" - "$kind" <<'PY'
import json, math, sys
from pathlib import Path

kind = sys.argv[1]
root = Path("/home/siddharth/trace-shift/research/results/raw/finetuning")
if kind == "primary":
    run_dir = root / "qwen3-1.7b-lora-tennis-primary"
else:
    run_dir = root / "qwen3-1.7b-lora-control"
adapter = run_dir / "final_adapter"
meta_y = run_dir / "training_metadata.yaml"
meta_j = run_dir / "training_metadata.json"
trainer_state = None
# HF may write trainer_state under output_dir or checkpoint-*.
# MUST pick the state with maximum global_step (epoch saves create
# checkpoint-10/20/30; unsorted glob()[0] can falsely read mid-run state).
cands = list(run_dir.glob("**/trainer_state.json"))
errors = []
ok = True
trainer_state_used = None
steps = None

if not adapter.is_dir():
    errors.append(f"missing adapter dir {adapter}")
    ok = False
else:
    has_w = any(adapter.glob("adapter_model.*")) or any(adapter.glob("*.safetensors"))
    if not has_w:
        # peft may use adapter_model.safetensors or pytorch_model.bin
        files = list(adapter.iterdir())
        if not files:
            errors.append("adapter dir empty")
            ok = False

meta = None
if meta_j.is_file():
    meta = json.loads(meta_j.read_text())
elif meta_y.is_file():
    import yaml
    meta = yaml.safe_load(meta_y.read_text())
else:
    errors.append("missing training_metadata")
    ok = False

if meta:
    n = int(meta.get("n_examples") or 0)
    if n != 40:
        errors.append(f"n_examples={n} expected 40")
        ok = False
    hp = meta.get("hyperparameters") or {}
    epochs = int(hp.get("num_epochs") or 0)
    if epochs != 3:
        errors.append(f"num_epochs={epochs} expected 3")
        ok = False
    # Expected optimizer steps: 40 / (batch1 * accum4) = 10 per epoch * 3 = 30
    expected_steps = 30
    if cands:
        parsed = []
        for p in cands:
            try:
                st = json.loads(p.read_text())
                gs = st.get("global_step")
                if gs is None:
                    continue
                parsed.append((int(gs), p, st))
            except Exception as e:  # noqa: BLE001
                errors.append(f"unreadable trainer_state {p}: {e}")
                ok = False
        if not parsed:
            errors.append("no readable trainer_state.json with global_step")
            ok = False
        else:
            parsed.sort(key=lambda t: t[0])
            steps, trainer_state_used, st = parsed[-1]  # max global_step
            log_hist = st.get("log_history") or []
            losses = [h.get("loss") for h in log_hist if "loss" in h]
            if not losses:
                errors.append("no loss values in trainer_state log_history")
                ok = False
            else:
                if any((x is None) or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) for x in losses):
                    errors.append("NaN/Inf in training loss")
                    ok = False
                if len(losses) >= 2 and max(losses) == min(losses) and losses[0] == 0.0:
                    errors.append("loss trajectory flat at zero — optimization suspicious")
                    ok = False
            if int(steps) != expected_steps:
                errors.append(
                    f"global_step={steps} expected {expected_steps} "
                    f"(from {trainer_state_used})"
                )
                ok = False
    else:
        errors.append("no trainer_state.json found under run dir")
        ok = False

# Reload adapter onto fresh base
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

lora_norm_max = None
lora_norm_source = None
snap_roots = list(Path.home().joinpath(".cache/huggingface/hub").glob("models--Qwen--Qwen3-1.7B/snapshots/*"))
if not snap_roots:
    errors.append("base weights snapshot missing for reload check")
    ok = False
else:
    snap = sorted(snap_roots)[0]
    try:
        tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            str(snap), local_files_only=True, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="cuda",
        )
        model = PeftModel.from_pretrained(base, str(adapter))
        model.eval()
        # nonzero LoRA params check — also persist lora_norm_max for Phase-08 bookkeeping
        lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
        if not lora_params:
            errors.append("no lora_ parameters after reload")
            ok = False
        else:
            norms = [float(p.detach().float().norm().cpu()) for p in lora_params]
            lora_norm_max = float(max(norms)) if norms else None
            lora_norm_source = "peft_reload_named_parameters"
            if all(n == 0.0 for n in norms):
                errors.append("all LoRA parameter norms are zero after reload")
                ok = False
        ids = tok("Roger Federer is a", return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=4, do_sample=False)
        _ = tok.decode(out[0], skip_special_tokens=True)
        del model, base, tok
        torch.cuda.empty_cache()
    except Exception as e:
        errors.append(f"reload/exec failed: {type(e).__name__}: {e}")
        ok = False

# Fallback: Frobenius max from saved adapter (no base model / no retrain)
if lora_norm_max is None:
    try:
        from safetensors.torch import load_file
        w = adapter / "adapter_model.safetensors"
        if w.is_file():
            tensors = load_file(str(w))
            norms = [float(t.detach().float().norm().item()) for n, t in tensors.items() if "lora_" in n]
            if norms:
                lora_norm_max = float(max(norms))
                lora_norm_source = "adapter_model.safetensors"
    except Exception:
        pass

report = {
    "kind": kind,
    "ok": ok,
    "errors": errors,
    "adapter": str(adapter),
    "expected_optimizer_steps": 30,
    "n_examples_expected": 40,
    "epochs_expected": 3,
    "global_step": steps,
    "trainer_state_path": str(trainer_state_used) if trainer_state_used else None,
    "lora_norm_max": lora_norm_max,
    "lora_norm_max_provenance": {
        "source": lora_norm_source,
        "method": "max_frobenius_norm_over_lora_named_tensors",
    } if lora_norm_max is not None else None,
}
out_path = run_dir / "execution_validity.json"
out_path.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
sys.exit(0 if ok else 1)
PY
}

# -------------------- Start scientific clock at first measurement --------------------
if [[ -f "${LOGDIR}/experiment_start_epoch.txt" ]]; then
  EXPERIMENT_START_EPOCH=$(cat "${LOGDIR}/experiment_start_epoch.txt")
  log "Resuming; scientific start epoch already recorded: $EXPERIMENT_START_EPOCH"
else
  EXPERIMENT_START_EPOCH=$(date +%s)
  echo "$EXPERIMENT_START_EPOCH" > "${LOGDIR}/experiment_start_epoch.txt"
  echo "$(ts)" > "${LOGDIR}/experiment_start_utc.txt"
  log "SCIENTIFIC CLOCK START (first measurement about to begin): $(cat ${LOGDIR}/experiment_start_utc.txt)"
  append_time "0s" "scientific_clock_start" "STARTED" "First scientific measurement phase beginning on remote L4 in tmux traceshift-run-6fact (balanced 6-fact / 3-domain)"
fi

log "Host=$(hostname) repo=$REPO commit=$(git rev-parse HEAD)"
log "DESIGN=balanced_6fact_3domain FACTS=F01..F06"
log "Inside tmux? TMUX=${TMUX:-unset}"
nvidia-smi | tee -a "$MASTER_LOG" || true
gc_cuda | tee -a "$MASTER_LOG"

FAIL=0
CURRENT_PHASE=""
CURRENT_PHASE_T0=""

on_unexpected_exit() {
  local rc=$?
  # Disarm first so any re-exit cannot recurse, and so trap body status
  # cannot override the real orchestrator exit code (e.g. failed [[ → 1).
  trap - EXIT
  if [[ -n "${CURRENT_PHASE:-}" && -n "${CURRENT_PHASE_T0:-}" ]]; then
    if [[ ! -f "${MARK}/${CURRENT_PHASE}.done" ]]; then
      record_phase "$CURRENT_PHASE" "$CURRENT_PHASE_T0" "INTERRUPTED" "orchestrator exited rc=${rc} before phase marker" || true
      log "INTERRUPTED ${CURRENT_PHASE} rc=${rc}" || true
    fi
  fi
  exit "$rc"
}
trap on_unexpected_exit EXIT


# ========== PHASE 1: Baseline recall ==========
PHASE=01_baseline_recall
if is_done "$PHASE" && [[ -f "${RAW}/baseline_recall/baseline_cloze.json" ]]; then
  log "SKIP $PHASE (already done)"
else
  t0=$(date +%s)
  CURRENT_PHASE="$PHASE"
  CURRENT_PHASE_T0="$t0"
  if run_cmd "$PHASE" "$PY" research/scripts/run_baseline_cloze.py --device cuda \
      --json-out "${RAW}/baseline_recall/baseline_cloze.json"; then
    mark_done "$PHASE"; CURRENT_PHASE=""; CURRENT_PHASE_T0=""
    record_phase "$PHASE" "$t0" "PASS" "baseline cloze on M_base"
  else
    record_phase "$PHASE" "$t0" "FAIL" "baseline cloze failed"
    FAIL=1
  fi
  gc_cuda | tee -a "$MASTER_LOG"
fi
[[ $FAIL -eq 0 ]] || { log "STOP after phase 1"; exit 1; }

# ========== PHASE 2: Alpha calibration ==========
PHASE=02_alpha_calibration
if is_done "$PHASE" && [[ -f "${RAW}/alpha_calibration/base_alphas.yaml" ]]; then
  log "SKIP $PHASE (already done)"
else
  t0=$(date +%s)
  CURRENT_PHASE="$PHASE"
  CURRENT_PHASE_T0="$t0"
  if run_cmd "$PHASE" "$PY" research/scripts/run_alpha_calibration.py --device cuda; then
    if [[ -f "${RAW}/alpha_calibration/base_alphas.yaml" ]]; then
      mark_done "$PHASE"; CURRENT_PHASE=""; CURRENT_PHASE_T0=""
      record_phase "$PHASE" "$t0" "PASS" "base alpha calibration"
    else
      record_phase "$PHASE" "$t0" "FAIL" "missing base_alphas.yaml"
      FAIL=1
    fi
  else
    record_phase "$PHASE" "$t0" "FAIL" "alpha calibration exit nonzero"
    FAIL=1
  fi
  gc_cuda | tee -a "$MASTER_LOG"
fi
[[ $FAIL -eq 0 ]] || { log "STOP after phase 2"; exit 1; }

# ========== PHASE 3: Base matrix ==========
PHASE=03_base_matrix
if is_done "$PHASE" && [[ -f "${RAW}/base_matrix/base_matrix.json" ]]; then
  log "SKIP $PHASE (already done)"
else
  t0=$(date +%s)
  CURRENT_PHASE="$PHASE"
  CURRENT_PHASE_T0="$t0"
  if run_cmd "$PHASE" "$PY" research/scripts/run_base_matrix.py --device cuda; then
    if [[ -f "${RAW}/base_matrix/base_matrix.json" ]]; then
      mark_done "$PHASE"; CURRENT_PHASE=""; CURRENT_PHASE_T0=""
      record_phase "$PHASE" "$t0" "PASS" "M_base computed"
    else
      # May legitimately refuse incomplete selective set — capture status
      record_phase "$PHASE" "$t0" "INCOMPLETE_OR_FAIL" "no base_matrix.json; see status"
      # If STATUS says incomplete due to eligibility, continue only if documented
      FAIL=1
    fi
  else
    record_phase "$PHASE" "$t0" "FAIL" "base matrix exit nonzero"
    FAIL=1
  fi
  gc_cuda | tee -a "$MASTER_LOG"
fi
[[ $FAIL -eq 0 ]] || { log "STOP after phase 3 — inspect alpha selectivity / matrix status"; exit 1; }

# ========== PHASE 4: Tennis LoRA FT ==========
# If a final_adapter already exists, re-validate ONLY (do not retrain on a
# bookkeeping false-negative). Retrain only if adapter missing or truly invalid.
PHASE=04_tennis_finetune
TENNIS_ADAPTER="${RAW}/finetuning/qwen3-1.7b-lora-tennis-primary/final_adapter"
if is_done "$PHASE" && [[ -d "$TENNIS_ADAPTER" ]]; then
  log "SKIP $PHASE (already done)"
else
  t0=$(date +%s)
  CURRENT_PHASE="$PHASE"
  CURRENT_PHASE_T0="$t0"
  if [[ -d "$TENNIS_ADAPTER" ]]; then
    log "Phase 4: existing tennis adapter found — re-validating without retrain"
    if validate_ft primary > "${LOGDIR}/04_tennis_validity.log" 2>&1; then
      mark_done "$PHASE"; CURRENT_PHASE=""; CURRENT_PHASE_T0=""
      record_phase "$PHASE" "$t0" "PASS" "tennis LoRA FT revalidated (no retrain)"
    else
      cat "${LOGDIR}/04_tennis_validity.log" | tee -a "$MASTER_LOG"
      log "Phase 4: existing adapter failed validity — retraining ONCE under frozen settings"
      if run_cmd "$PHASE" "$PY" research/scripts/run_finetune.py --dataset primary --device cuda; then
        if validate_ft primary > "${LOGDIR}/04_tennis_validity.log" 2>&1; then
          mark_done "$PHASE"; CURRENT_PHASE=""; CURRENT_PHASE_T0=""
          record_phase "$PHASE" "$t0" "PASS" "tennis LoRA FT retrained + validity gate"
        else
          cat "${LOGDIR}/04_tennis_validity.log" | tee -a "$MASTER_LOG"
          record_phase "$PHASE" "$t0" "FAIL" "tennis FT validity gate failed after retrain"
          FAIL=1
        fi
      else
        record_phase "$PHASE" "$t0" "FAIL" "tennis FT retrain exit nonzero"
        FAIL=1
      fi
    fi
  else
    if run_cmd "$PHASE" "$PY" research/scripts/run_finetune.py --dataset primary --device cuda; then
      if validate_ft primary > "${LOGDIR}/04_tennis_validity.log" 2>&1; then
        mark_done "$PHASE"; CURRENT_PHASE=""; CURRENT_PHASE_T0=""
        record_phase "$PHASE" "$t0" "PASS" "tennis LoRA FT + validity gate"
      else
        cat "${LOGDIR}/04_tennis_validity.log" | tee -a "$MASTER_LOG"
        record_phase "$PHASE" "$t0" "FAIL" "tennis FT validity gate failed"
        FAIL=1
      fi
    else
      record_phase "$PHASE" "$t0" "FAIL" "tennis FT exit nonzero"
      FAIL=1
    fi
  fi
  gc_cuda | tee -a "$MASTER_LOG"
fi
[[ $FAIL -eq 0 ]] || { log "STOP after phase 4"; exit 1; }

# ========== PHASE 5: Control LoRA FT ==========
PHASE=05_control_finetune
CONTROL_ADAPTER="${RAW}/finetuning/qwen3-1.7b-lora-control/final_adapter"
if is_done "$PHASE" && [[ -d "$CONTROL_ADAPTER" ]]; then
  log "SKIP $PHASE (already done)"
else
  t0=$(date +%s)
  CURRENT_PHASE="$PHASE"
  CURRENT_PHASE_T0="$t0"
  if [[ -d "$CONTROL_ADAPTER" ]]; then
    log "Phase 5: existing control adapter found — re-validating without retrain"
    if validate_ft control > "${LOGDIR}/05_control_validity.log" 2>&1; then
      mark_done "$PHASE"; CURRENT_PHASE=""; CURRENT_PHASE_T0=""
      record_phase "$PHASE" "$t0" "PASS" "control LoRA FT revalidated (no retrain)"
    else
      cat "${LOGDIR}/05_control_validity.log" | tee -a "$MASTER_LOG"
      log "Phase 5: existing control adapter failed validity — retraining ONCE"
      if run_cmd "$PHASE" "$PY" research/scripts/run_finetune.py --dataset control --device cuda; then
        if validate_ft control > "${LOGDIR}/05_control_validity.log" 2>&1; then
          mark_done "$PHASE"; CURRENT_PHASE=""; CURRENT_PHASE_T0=""
          record_phase "$PHASE" "$t0" "PASS" "control LoRA FT retrained + validity gate"
        else
          cat "${LOGDIR}/05_control_validity.log" | tee -a "$MASTER_LOG"
          record_phase "$PHASE" "$t0" "FAIL" "control FT validity gate failed after retrain"
          FAIL=1
        fi
      else
        record_phase "$PHASE" "$t0" "FAIL" "control FT retrain exit nonzero"
        FAIL=1
      fi
    fi
  else
    if run_cmd "$PHASE" "$PY" research/scripts/run_finetune.py --dataset control --device cuda; then
      if validate_ft control > "${LOGDIR}/05_control_validity.log" 2>&1; then
        mark_done "$PHASE"; CURRENT_PHASE=""; CURRENT_PHASE_T0=""
        record_phase "$PHASE" "$t0" "PASS" "control LoRA FT + validity gate"
      else
        cat "${LOGDIR}/05_control_validity.log" | tee -a "$MASTER_LOG"
        record_phase "$PHASE" "$t0" "FAIL" "control FT validity gate failed"
        FAIL=1
      fi
    else
      record_phase "$PHASE" "$t0" "FAIL" "control FT exit nonzero"
      FAIL=1
    fi
  fi
  gc_cuda | tee -a "$MASTER_LOG"
fi
[[ $FAIL -eq 0 ]] || { log "STOP after phase 5"; exit 1; }

# ========== PHASE 6: Tennis FT matrix ==========
PHASE=06_ft_matrix
if is_done "$PHASE" && [[ -f "${RAW}/ft_matrix/ft_matrix.json" ]]; then
  log "SKIP $PHASE (already done)"
else
  t0=$(date +%s)
  CURRENT_PHASE="$PHASE"
  CURRENT_PHASE_T0="$t0"
  if run_cmd "$PHASE" "$PY" research/scripts/run_ft_matrix.py --device cuda; then
    if [[ -f "${RAW}/ft_matrix/ft_matrix.json" ]]; then
      mark_done "$PHASE"; CURRENT_PHASE=""; CURRENT_PHASE_T0=""
      record_phase "$PHASE" "$t0" "PASS" "M_FT_tennis 6x6 computed"
    else
      record_phase "$PHASE" "$t0" "FAIL" "missing ft_matrix.json"
      FAIL=1
    fi
  else
    record_phase "$PHASE" "$t0" "FAIL" "ft matrix exit nonzero"
    FAIL=1
  fi
  gc_cuda | tee -a "$MASTER_LOG"
fi
[[ $FAIL -eq 0 ]] || { log "STOP after phase 6"; exit 1; }

# ========== PHASE 7: Delta-M + 15-cell permutation (tennis primary) ==========
PHASE=07_delta_analysis
if is_done "$PHASE" && [[ -f "${RAW}/delta_analysis/delta_analysis.json" ]]; then
  log "SKIP $PHASE (already done)"
else
  t0=$(date +%s)
  CURRENT_PHASE="$PHASE"
  CURRENT_PHASE_T0="$t0"
  if run_cmd "$PHASE" "$PY" research/scripts/run_delta_analysis.py; then
    if [[ -f "${RAW}/delta_analysis/delta_analysis.json" ]]; then
      mark_done "$PHASE"; CURRENT_PHASE=""; CURRENT_PHASE_T0=""
      record_phase "$PHASE" "$t0" "PASS" "Delta-M tennis + C(6,2)=15 exact permutation"
    else
      record_phase "$PHASE" "$t0" "FAIL" "missing delta_analysis.json"
      FAIL=1
    fi
  else
    record_phase "$PHASE" "$t0" "FAIL" "delta analysis exit nonzero"
    FAIL=1
  fi
fi
[[ $FAIL -eq 0 ]] || { log "STOP after phase 7"; exit 1; }

# ========== PHASE 8: Generic FT control matrix (Curie/Armstrong) ==========
# After primary T_obs so descriptive T_tennis - T_generic comparison can load it.
PHASE=08_generic_ft_control
if is_done "$PHASE" && [[ -f "${RAW}/alternative_hypotheses/generic_ft_control_matrix.json" ]] \
   && [[ -f "${RAW}/alternative_hypotheses/generic_ft_control_delta_analysis.json" ]]; then
  log "SKIP $PHASE (already done)"
else
  t0=$(date +%s)
  CURRENT_PHASE="$PHASE"
  CURRENT_PHASE_T0="$t0"
  if run_cmd "$PHASE" "$PY" research/scripts/run_generic_ft_control.py --device cuda; then
    if [[ -f "${RAW}/alternative_hypotheses/generic_ft_control_matrix.json" ]] \
       && [[ -f "${RAW}/alternative_hypotheses/generic_ft_control_delta_analysis.json" ]]; then
      mark_done "$PHASE"; CURRENT_PHASE=""; CURRENT_PHASE_T0=""
      record_phase "$PHASE" "$t0" "PASS" "M_FT_generic 6x6 + descriptive T_generic_within"
    else
      record_phase "$PHASE" "$t0" "FAIL" "missing generic_ft_control outputs"
      FAIL=1
    fi
  else
    record_phase "$PHASE" "$t0" "FAIL" "generic FT control exit nonzero"
    FAIL=1
  fi
  gc_cuda | tee -a "$MASTER_LOG"
fi
[[ $FAIL -eq 0 ]] || { log "STOP after phase 8"; exit 1; }

# ========== PHASE 9: Lexical control ==========
PHASE=09_lexical_control
if is_done "$PHASE" && [[ -f "${RAW}/alternative_hypotheses/lexical_control.json" ]]; then
  log "SKIP $PHASE (already done)"
else
  t0=$(date +%s)
  CURRENT_PHASE="$PHASE"
  CURRENT_PHASE_T0="$t0"
  if run_cmd "$PHASE" "$PY" research/scripts/run_lexical_control.py --device cuda; then
    if [[ -f "${RAW}/alternative_hypotheses/lexical_control.json" ]]; then
      mark_done "$PHASE"; CURRENT_PHASE=""; CURRENT_PHASE_T0=""
      record_phase "$PHASE" "$t0" "PASS" "lexical-control 6x6 matrices + T_lexical_within"
    else
      record_phase "$PHASE" "$t0" "FAIL" "missing lexical_control.json"
      FAIL=1
    fi
  else
    record_phase "$PHASE" "$t0" "FAIL" "lexical control exit nonzero"
    FAIL=1
  fi
  gc_cuda | tee -a "$MASTER_LOG"
fi
[[ $FAIL -eq 0 ]] || { log "STOP after phase 9"; exit 1; }

# ========== PHASE 10: Consolidation ==========
PHASE=10_consolidation
t0=$(date +%s)
  CURRENT_PHASE="$PHASE"
  CURRENT_PHASE_T0="$t0"
"$PY" - <<'PY' | tee -a "$MASTER_LOG"
import hashlib, json
from pathlib import Path
import importlib.metadata as m
import torch

root = Path("/home/siddharth/trace-shift/research")
checks = {
  "facts": root/"data/frozen/facts.yaml",
  "explicit": root/"data/frozen/extraction_sets_explicit.yaml",
  "lexical": root/"data/frozen/extraction_sets_lexical.yaml",
  "reference": root/"data/frozen/reference_corpus.yaml",
  "cloze": root/"data/frozen/cloze_probes.yaml",
  "experiment": root/"configs/experiment.yaml",
}
digests = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k,p in checks.items()}
required = [
  root/"results/raw/baseline_recall/baseline_cloze.json",
  root/"results/raw/alpha_calibration/base_alphas.yaml",
  root/"results/raw/base_matrix/base_matrix.json",
  root/"results/raw/finetuning/qwen3-1.7b-lora-tennis-primary/final_adapter",
  root/"results/raw/finetuning/qwen3-1.7b-lora-control/final_adapter",
  root/"results/raw/ft_matrix/ft_matrix.json",
  root/"results/raw/delta_analysis/delta_analysis.json",
  root/"results/raw/alternative_hypotheses/generic_ft_control_matrix.json",
  root/"results/raw/alternative_hypotheses/generic_ft_control_delta_analysis.json",
  root/"results/raw/alternative_hypotheses/lexical_control.json",
]
missing = [str(p) for p in required if not p.exists()]
# Quick structural checks
issues = []
try:
    base = json.loads((root/"results/raw/base_matrix/base_matrix.json").read_text())
    if base.get("fact_ids") != ["F01","F02","F03","F04","F05","F06"]:
        issues.append(f"base fact_ids={base.get('fact_ids')}")
    if len(base.get("matrix_cosine_distance") or []) != 6:
        issues.append("base matrix not 6x6")
except Exception as e:
    issues.append(f"base matrix check: {e}")
summary = {
  "design": "balanced_6fact_3domain",
  "frozen_input_sha256": digests,
  "required_outputs_missing": missing,
  "structural_issues": issues,
  "hardware": {
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
  },
  "software": {
    "python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
    "transformers": m.version("transformers"),
    "peft": m.version("peft"),
    "ai-engram": m.version("ai-engram"),
    "tqdm": m.version("tqdm"),
    "PyYAML": m.version("PyYAML"),
  },
  "all_required_present": len(missing)==0 and len(issues)==0,
}
out = root/"results/raw/experiment_run/consolidation_summary.json"
out.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
raise SystemExit(0 if summary["all_required_present"] else 1)
PY
cons_rc=$?
if [[ $cons_rc -eq 0 ]]; then
  mark_done "$PHASE"; CURRENT_PHASE=""; CURRENT_PHASE_T0=""
  record_phase "$PHASE" "$t0" "PASS" "consolidation checks (6-fact)"
else
  record_phase "$PHASE" "$t0" "FAIL" "consolidation missing outputs"
  FAIL=1
fi

TOTAL=$(( $(date +%s) - EXPERIMENT_START_EPOCH ))
echo "$TOTAL" > "${LOGDIR}/total_scientific_seconds.txt"
append_time "${TOTAL}s" "scientific_clock_end" "$([[ $FAIL -eq 0 ]] && echo COMPLETE || echo INCOMPLETE)" "total_scientific_wall=${TOTAL}s on remote L4 tmux traceshift-run-6fact"
log "SCIENTIFIC RUN FINISHED fail=$FAIL total_s=$TOTAL"
exit $FAIL
