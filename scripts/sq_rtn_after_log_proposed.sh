#!/bin/bash
# After grouping-free log Proposed (current job): Proposed test-other, then
# grouped SQ+RTN W4A8 Baseline (clean + other). Do NOT re-run Proposed as SQ+RTN.
set -euo pipefail
set -x
source /venv/main/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/SpeechPTQ/src PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/SpeechPTQ

WAIT_PID="${WAIT_PID:-471582}"
echo "waiting for pid $WAIT_PID (grouping-free log proposed test-clean)"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done
echo "proposed test-clean finished"

PY=/venv/main/bin/python

echo "===== Proposed test-other Whisper (grouping-free log, enc FFN A4, rest A8) ====="
$PY src/run_eval.py --mode w4a4 --w-mode log --a-mode mixed_log_encffn_a8 \
  --scope nolm --group-size 0 --split test-other --batch-size 16 --backend fake \
  --out results/compare/whisper_log_w4_encffn_a4_resta8_test-other.json

echo "===== Proposed test-other Qwen3 (grouping-free log, AuT A8/FFN A4, LLM A16) ====="
$PY src/run_eval_qwen3.py --mode w4a4 --w-mode log --a-mode mixed_log_encffn_enc_a8 \
  --scope nolm --group-size 0 --split test-other --batch-size 8 --backend fake \
  --out results/compare/qwen_log_w4_encffn_a4_aut_a8_llm_a16_test-other.json

echo "===== Whisper baseline: SQ+RTN W4A8 g=64 test-clean ====="
$PY src/run_eval.py --mode w4a4 --w-mode uniform --a-mode uniform_a8_dyn \
  --scope nolm --smoothquant-alpha 0.5 --group-size 64 --split test-clean \
  --batch-size 16 --backend fake \
  --out results/compare/whisper_baseline_sq_rtn_w4a8_g64.json

echo "===== Qwen3 baseline: SQ+RTN W4A8 g=64, LLM A16 test-clean ====="
$PY src/run_eval_qwen3.py --mode w4a4 --w-mode uniform --a-mode encoder_a8 \
  --scope nolm --smoothquant-alpha 0.5 --group-size 64 --split test-clean \
  --batch-size 8 --backend fake \
  --out results/compare/qwen_baseline_sq_rtn_encw4a8_llma16_g64.json

echo "===== Whisper baseline test-other ====="
$PY src/run_eval.py --mode w4a4 --w-mode uniform --a-mode uniform_a8_dyn \
  --scope nolm --smoothquant-alpha 0.5 --group-size 64 --split test-other \
  --batch-size 16 --backend fake \
  --out results/compare/whisper_baseline_sq_rtn_w4a8_g64_test-other.json

echo "===== Qwen3 baseline test-other ====="
$PY src/run_eval_qwen3.py --mode w4a4 --w-mode uniform --a-mode encoder_a8 \
  --scope nolm --smoothquant-alpha 0.5 --group-size 64 --split test-other \
  --batch-size 8 --backend fake \
  --out results/compare/qwen_baseline_sq_rtn_encw4a8_llma16_g64_test-other.json

echo "[sq-rtn-table] WER jobs done"
