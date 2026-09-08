#!/bin/bash
# Restart skip-if-done W4A8 compare if the runner dies before finishing.
# After the in-flight pass prints ALL JOBS FINISHED, run one extra pass so
# jobs added later (GPTQ W4A16 / GPTQ+A8 leftovers) still execute.
set -u
cd /workspace/SpeechPTQ
LOG=logs/w4a8_method_compare.log
mkdir -p logs
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/SpeechPTQ/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/venv/main/bin/python

finished() { grep -q 'ALL JOBS FINISHED' "$LOG" 2>/dev/null; }
running() { pgrep -f 'src/run_w4a8_method_compare.py --run' >/dev/null; }
extra_needed() {
  [ ! -f results/compare/w4a8/whisper_awq_w4a8_test-clean.json ] && return 0
  [ ! -f results/compare/w4a8/whisper_awq_w4a8_test-other.json ] && return 0
  [ ! -f results/compare/w4a8/qwen_enc_awq_w4a8_test-clean.json ] && return 0
  [ ! -f results/compare/w4a8/qwen_enc_awq_w4a8_test-other.json ] && return 0
  # Encoder GPTQ: config.json alone is not a ckpt. Keep the extra pass
  # alive until weights exist and both W4A8 evals are written.
  if ! ls models/qwen3-asr-1.7b-W4A16-G128-llm-encgptq/*.safetensors >/dev/null 2>&1; then
    return 0
  fi
  [ ! -f results/compare/w4a8/qwen_enc_gptq_w4a8_test-clean.json ] && return 0
  [ ! -f results/compare/w4a8/qwen_enc_gptq_w4a8_test-other.json ] && return 0
  return 1
}

while true; do
  if finished && extra_needed && ! running; then
    echo "[watch] extra pass W4A8 leftovers  $(date -Is)" | tee -a "$LOG"
    nohup $PY src/run_w4a8_method_compare.py --run >> "$LOG" 2>&1 &
    sleep 60
    continue
  fi
  if finished && ! extra_needed; then
    echo "[watch] done $(date -Is)"
    exit 0
  fi
  if ! running; then
    echo "[watch] restart $(date -Is)" | tee -a "$LOG"
    nohup $PY src/run_w4a8_method_compare.py --run >> "$LOG" 2>&1 &
  fi
  sleep 60
done
