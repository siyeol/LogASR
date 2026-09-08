#!/bin/bash
# Qwen: encoder uniform W g=64 + log activations (group-free);
#        decoder = GPTQ W4A16 G128 (checkpoint).
set -u
source /venv/main/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/SpeechPTQ/src
cd /workspace/SpeechPTQ

OUT=results/ablation_w4a4
mkdir -p "$OUT" logs
LOG=logs/qwen_enc_uni_g64_alog.log
exec > >(tee -a "$LOG") 2>&1

PY=/venv/main/bin/python
MODEL=/workspace/SpeechPTQ/models/qwen3-asr-1.7b-W4A16-G128-llm
G=64
SQ=0.5

done_ok() {
  local out="$1"
  [ -f "$out" ] || return 1
  $PY -c "
import json,sys
d=json.load(open(sys.argv[1]))
sys.exit(0 if d.get('wer') is not None else 1)
" "$out"
}

run_or_skip() {
  local out="$1"
  shift
  if done_ok "$out"; then
    echo "=== SKIP existing $out ==="
    return 0
  fi
  echo "=== RUN $* ==="
  echo "    -> $out"
  if "$@"; then
    echo "=== DONE $out ==="
  else
    echo "=== FAIL $out exit=$? ==="
  fi
}

echo "===== Qwen enc W-uni g=$G A-log g_a=0 + GPTQ LLM  $(date -Is) ====="

for split in test-clean test-other; do
  out="$OUT/qwen_gptq_enc_wuni_g${G}_alog_a0_${split}.json"
  run_or_skip "$out" \
    $PY src/run_eval_qwen3.py --mode w4a4 --split "$split" \
      --model-id "$MODEL" \
      --w-mode uniform --a-mode log_token --scope encoder \
      --group-size "$G" --act-group-size 0 --smoothquant-alpha "$SQ" \
      --backend fake --batch-size 8 --max-new-tokens 256 \
      --out "$out"
done

echo "ALL JOBS FINISHED $(date -Is)"
