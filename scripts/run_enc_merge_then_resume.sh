#!/bin/bash
# After the in-flight Whisper RTN-other eval exits, run encoder-merge once,
# then resume the W4A8 method compare (skip-if-done).
set -u
cd /workspace/SpeechPTQ
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/SpeechPTQ/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/venv/main/bin/python
LOG=logs/enc_merge_whisper.log
mkdir -p logs results/diagnostics

echo "[queue] waiting for PID 51461 (whisper RTN other) $(date -Is)" | tee -a "$LOG"
while kill -0 51461 2>/dev/null; do sleep 15; done
echo "[queue] GPU free $(date -Is)" | tee -a "$LOG"

$PY src/run_enc_merge_whisper.py >> "$LOG" 2>&1
ec=$?
echo "[queue] enc-merge exit=$ec $(date -Is)" | tee -a "$LOG"

echo "[queue] resume W4A8 compare $(date -Is)" | tee -a logs/w4a8_method_compare.log
nohup $PY src/run_w4a8_method_compare.py --run >> logs/w4a8_method_compare.log 2>&1 &
nohup bash scripts/watch_w4a8_compare.sh >> logs/w4a8_watch.log 2>&1 &
echo "[queue] compare restarted pid=$! $(date -Is)" | tee -a "$LOG"
