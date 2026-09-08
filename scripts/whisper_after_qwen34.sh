#!/bin/bash
# After Qwen 3/4 finishes: Whisper SQ+RTN baseline (same recipe), then proposed.
set -x
source /venv/main/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/workspace/SpeechPTQ/src
cd /workspace/SpeechPTQ

QJSON=results/compare/qwen_baseline_sq_encw4a8_decw4a16.json
QPID="${QWEN34_PID:-470127}"

echo "waiting for Qwen 3/4 -> $QJSON (pid=$QPID)"
while [ ! -f "$QJSON" ]; do
  if [ -n "$QPID" ] && ! kill -0 "$QPID" 2>/dev/null; then
    echo "Qwen 3/4 process exited without JSON"
    exit 1
  fi
  sleep 20
done
while [ -n "$QPID" ] && kill -0 "$QPID" 2>/dev/null; do
  sleep 5
done
sleep 3
echo "Qwen 3/4 done, starting Whisper pair"

echo "=== Whisper Edge-ASR-style baseline: RTN W4A8, weight group=64 ==="
/venv/main/bin/python src/run_eval.py \
  --mode w4a4 \
  --w-mode uniform \
  --a-mode uniform_a8_dyn \
  --scope nolm \
  --group-size 64 \
  --split test-clean \
  --batch-size 16 \
  --backend fake \
  --out results/compare/whisper_baseline_edgeasr_rtn_w4a8_g64.json

echo "=== Whisper proposed: grouping-free log W4, enc FFN A4, enc rest A8, dec A16 ==="
/venv/main/bin/python src/run_eval.py \
  --mode w4a4 \
  --w-mode log \
  --a-mode mixed_log_encffn_enc_a8 \
  --scope nolm \
  --group-size 0 \
  --split test-clean \
  --batch-size 16 \
  --backend fake \
  --out results/compare/whisper_proposed_log_encffn_a4_enc_a8_dec_a16.json

echo "=== WHISPER PAIR DONE ==="
