#!/bin/bash
# Encoder FFN W4A16 ablation: uniform RTN vs log vs SmoothQuant vs GPTQ.
# Waits for any in-progress eval (Qwen3 proposed test-other) to release the GPU.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/SpeechPTQ/src PYTHONUNBUFFERED=1
source /venv/main/bin/activate
cd /workspace/SpeechPTQ

WAIT_PID="${1:-}"
if [[ -n "$WAIT_PID" ]]; then
  echo "[ablation] waiting for pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
  echo "[ablation] pid $WAIT_PID exited"
fi

echo "===== Whisper enc-FFN uniform RTN W4A16 ====="
python src/run_eval.py --mode w4a4 --w-mode uniform --a-mode none --scope enc_ffn --backend fake --split test-clean --batch-size 16

echo "===== Whisper enc-FFN log W4A16 ====="
python src/run_eval.py --mode w4a4 --w-mode log --a-mode none --scope enc_ffn --backend fake --split test-clean --batch-size 16

echo "===== Whisper enc-FFN SmoothQuant a=0.5 + uniform RTN W4A16 ====="
python src/run_eval.py --mode w4a4 --w-mode uniform --a-mode none --scope enc_ffn --backend fake --smoothquant-alpha 0.5 --split test-clean --batch-size 16 --calib-batches 4

echo "===== Whisper enc-FFN GPTQ W4A16 (oneshot) ====="
python src/quant_whisper_enc_ffn_gptq.py

echo "===== Whisper enc-FFN GPTQ W4A16 eval ====="
python src/run_eval.py --mode fp16 --model-id /workspace/SpeechPTQ/models/whisper-large-v3-encffn-W4A16-G128 --split test-clean --batch-size 16

echo "[ablation] Whisper encoder FFN W4A16 done (Qwen3 arms skipped)"
