#!/bin/bash
# Keep leftover ablation+robustness resume alive. Do not re-run finished OOD
# (Gigaspeech included). Skip-if-done; diagnose last FAIL on restart.
set -u
cd /workspace/SpeechPTQ
LOG=logs/watch_ablation_resume.log
RESUME_LOG=logs/resume_ablation_then_robust.log
mkdir -p logs
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

finished() { grep -q '\[resume\] ALL DONE' "$RESUME_LOG" 2>/dev/null; }
running() { pgrep -f 'scripts/resume_ablation_then_robust.sh' >/dev/null; }
eval_running() {
  pgrep -f 'src/run_eval.py' >/dev/null && return 0
  pgrep -f 'src/run_eval_qwen3.py' >/dev/null && return 0
  pgrep -f 'src/simulate_robustness.py' >/dev/null && return 0
  return 1
}

diagnose() {
  echo "[watch] diagnose $(date -Is)" | tee -a "$LOG"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tee -a "$LOG"
  grep -E '=== FAIL|RuntimeError|CUDA out of memory|can.t start new thread|Killed|Traceback' \
    logs/scope_log_ablation.log logs/log_placement_ablation.log "$RESUME_LOG" 2>/dev/null \
    | tail -n 20 | tee -a "$LOG"
}

while true; do
  if finished; then
    echo "[watch] resume ALL DONE $(date -Is)" | tee -a "$LOG"
    exit 0
  fi
  if ! running; then
    diagnose
    echo "[watch] restart resume_ablation_then_robust.sh $(date -Is)" | tee -a "$LOG"
    nohup bash scripts/resume_ablation_then_robust.sh >/dev/null 2>&1 &
    sleep 90
    continue
  fi
  # Parent alive but child eval/sim died and script stuck (e.g. wait loop).
  if running && ! eval_running; then
    # run_scope may be between jobs for a few seconds; wait one more cycle.
    sleep 45
    if running && ! eval_running && ! finished; then
      echo "[watch] parent idle without eval; leave 60s then check again $(date -Is)" | tee -a "$LOG"
    fi
  fi
  sleep 60
done
