#!/bin/bash
# After Whisper proposed test-other: run Qwen GPTQ-LLM + encoder log W4A8 / FFN A4,
# then grouped SQ+RTN baselines. Skip doomed Qwen grouping-free-log proposed other.
set -euo pipefail
set -x
source /venv/main/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/SpeechPTQ/src PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/SpeechPTQ

WJSON=results/compare/whisper_log_w4_encffn_a4_resta8_test-other.json
echo "waiting for $WJSON"
while [ ! -f "$WJSON" ]; do sleep 15; done
sleep 5

# Stop the old waiter so it does not start Qwen grouping-free-log test-other.
if [ -f logs/sq_rtn_after_log_proposed.pid ]; then
  OLD=$(cat logs/sq_rtn_after_log_proposed.pid)
  kill "$OLD" 2>/dev/null || true
fi
pkill -f 'run_eval_qwen3.py --mode w4a4 --w-mode log --a-mode mixed_log_encffn_enc_a8 --scope nolm' || true
sleep 3

PY=/venv/main/bin/python
GPTQ=/workspace/SpeechPTQ/models/qwen3-asr-1.7b-W4A16-G128-llm

echo "===== Qwen Proposed: GPTQ LLM + encoder log W4A8, FFN log W4A4 test-clean ====="
$PY src/run_eval_qwen3.py --mode w4a4 --model-id "$GPTQ" \
  --w-mode log --a-mode mixed_log_encffn_enc_a8 --scope encoder \
  --group-size 0 --split test-clean --batch-size 8 --backend fake \
  --out results/compare/qwen_gptq_llm_enc_log_ffn_a4_resta8.json

echo "===== Qwen Proposed test-other (same recipe) ====="
$PY src/run_eval_qwen3.py --mode w4a4 --model-id "$GPTQ" \
  --w-mode log --a-mode mixed_log_encffn_enc_a8 --scope encoder \
  --group-size 0 --split test-other --batch-size 8 --backend fake \
  --out results/compare/qwen_gptq_llm_enc_log_ffn_a4_resta8_test-other.json

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
