#!/bin/bash
# Speech-frame packing: W1 (no 30s pad) → W2 (collapse+nopad) → Q1 (Qwen3 collapse).
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/SpeechPTQ/src PYTHONUNBUFFERED=1
source /venv/main/bin/activate
cd /workspace/SpeechPTQ

echo "===== W1 Whisper FP16 no-30s-pad test-clean ====="
python src/run_eval.py --mode fp16 --pack nopad --split test-clean --batch-size 16

echo "===== W2 Whisper FP16 RMS-collapse + no-pad test-clean ====="
python src/run_eval.py --mode fp16 --pack collapse --split test-clean --batch-size 16

echo "===== Q1 Qwen3 BF16 silence-collapse test-clean ====="
python src/run_eval_qwen3.py --mode silence_collapse --split test-clean --batch-size 8

echo "[packing] done"
