# ARCHIVED — moved from research/scripts/. Historical overnight/GPU tooling; paths may point at the original VM layout.
# See research/archive/README.md. Not part of the canonical CLI surface.

#!/usr/bin/env bash
# Resilient overnight wrapper for TraceShift 6-fact experiment.
# Intended to run inside detached tmux so Mac/Cursor disconnect is safe.
set -uo pipefail

REPO="${HOME}/trace-shift"
cd "$REPO" || exit 1

LOGDIR="${REPO}/research/results/raw/experiment_run"
MASTER_LOG="${LOGDIR}/master.log"
GPU_LOG="${LOGDIR}/gpu_monitor.log"
EXIT_FILE="${LOGDIR}/overnight_exit.txt"
PY="${REPO}/.venv/bin/python"
DATE=/usr/bin/date
TEE=/usr/bin/tee
NVIDIA_SMI=/usr/bin/nvidia-smi

mkdir -p "$LOGDIR"

ts() { "$DATE" -u +"%Y-%m-%dT%H:%M:%SZ"; }

{
  echo "[$(ts)] overnight_wrapper_start host=$(hostname) pid=$$ TMUX=${TMUX:-unset}"
  echo "[$(ts)] python=${PY}"
  "$NVIDIA_SMI" --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
} | "$TEE" -a "$MASTER_LOG"

# Background GPU monitor while experiment runs (CSV + occasional pmon)
gpu_monitor() {
  echo "--- gpu_monitor start $(ts) ---" >> "$GPU_LOG"
  while true; do
    "$NVIDIA_SMI" --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw \
      --format=csv,noheader 2>/dev/null >> "$GPU_LOG" || true
    "$NVIDIA_SMI" pmon -c 1 2>/dev/null >> "$GPU_LOG" || true
    sleep 120
  done
}

gpu_monitor &
GPU_MON_PID=$!
echo "[$(ts)] gpu_monitor_pid=${GPU_MON_PID}" | "$TEE" -a "$MASTER_LOG"

cleanup() {
  # Never alter the wrapper's pending exit status.
  if [[ -n "${GPU_MON_PID:-}" ]] && kill -0 "$GPU_MON_PID" 2>/dev/null; then
    kill "$GPU_MON_PID" 2>/dev/null || true
    wait "$GPU_MON_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

set +e
bash "${REPO}/research/scripts/run_full_experiment.sh"
RC=$?
set +e

# Record science exit only; nonzero means incomplete/failed — do not finalize.
echo "$RC" > "$EXIT_FILE"
echo "[$(ts)] overnight_wrapper_end exit=${RC}" | "$TEE" -a "$MASTER_LOG"

trap - EXIT
cleanup

# Final reports + git ONLY after successful full science (exit 0) AND markers/consolidation.
if [[ "$RC" -eq 0 ]]; then
  echo "[$(ts)] invoking finalize_6fact_after_science.sh (science exit=0)" | "$TEE" -a "$MASTER_LOG"
  set +e
  bash "${REPO}/research/scripts/finalize_6fact_after_science.sh"
  FRC=$?
  set +e
  echo "[$(ts)] finalize_exit=${FRC}" | "$TEE" -a "$MASTER_LOG"
  exit "$FRC"
fi

exit "$RC"
