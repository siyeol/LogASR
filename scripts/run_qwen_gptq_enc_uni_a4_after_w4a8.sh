#!/bin/bash
# After proposed W4A8 (Whisper+Qwen) finishes: Qwen baseline matched to proposed
# encoder scope. GPTQ LLM W4A16 + encoder uniform W g=64, uniform A grouping-free.
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
LOG=logs/qwen_gptq_enc_uni_a4_baseline.log
exec > >(tee -a "$LOG") 2>&1

PY=/venv/main/bin/python
GPTQ=/workspace/SpeechPTQ/models/qwen3-asr-1.7b-W4A16-G128-llm
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

echo "===== wait for proposed W4A8  $(date -Is) ====="
need=(
  "$OUT/whisper_unifW${G}_logA8_a0_test-clean.json"
  "$OUT/whisper_unifW${G}_logA8_a0_test-other.json"
  "$OUT/qwen_gptq_enc_wuni_g${G}_alogA8_a0_test-clean.json"
  "$OUT/qwen_gptq_enc_wuni_g${G}_alogA8_a0_test-other.json"
)
while true; do
  all=1
  for f in "${need[@]}"; do
    done_ok "$f" || all=0
  done
  [ "$all" -eq 1 ] && break
  echo "  waiting W4A8 JSONs  $(date -Is)"
  sleep 30
done
while pgrep -f 'scripts/run_proposed_w4a8_after_qwen_w4a4.sh' >/dev/null; do
  echo "  waiting W4A8 script exit  $(date -Is)"
  sleep 10
done
sleep 15
echo "===== proposed W4A8 complete  $(date -Is) ====="

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

echo "===== Qwen baseline GPTQ LLM + enc W-uni g=$G A-uni g_a=0 sq=$SQ  $(date -Is) ====="

for split in test-clean test-other; do
  out="$OUT/qwen_gptq_enc_wuni_g${G}_auni_a0_${split}.json"
  run_or_skip "$out" \
    $PY src/run_eval_qwen3.py --mode w4a4 --split "$split" \
      --model-id "$GPTQ" \
      --w-mode uniform --a-mode uniform_dyn --scope encoder \
      --group-size "$G" --act-group-size 0 --smoothquant-alpha "$SQ" \
      --backend fake --batch-size 8 --max-new-tokens 256 \
      --out "$out"
done

echo "ALL JOBS FINISHED $(date -Is)"
