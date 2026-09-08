#!/bin/bash
# After the in-flight Qwen W4A4 proposed job, run proposed W4A8:
#   Weight: uniform + grouping g=64
#   Activation: log, group-free, 8-bit (log_token_a8)
# Whisper: scope=nolm. Qwen: GPTQ LLM W4A16 + encoder this recipe.
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
LOG=logs/proposed_w4a8_unifW_logA.log
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

echo "===== wait for Qwen proposed W4A4  $(date -Is) ====="
Q1=$OUT/qwen_gptq_enc_wuni_g${G}_alog_a0_test-clean.json
Q2=$OUT/qwen_gptq_enc_wuni_g${G}_alog_a0_test-other.json
while ! done_ok "$Q1" || ! done_ok "$Q2"; do
  echo "  waiting Qwen W4A4 JSONs  $(date -Is)"
  sleep 30
done
while pgrep -f 'scripts/run_qwen_enc_uni_g64_alog.sh' >/dev/null; do
  echo "  waiting Qwen W4A4 script exit  $(date -Is)"
  sleep 10
done
# Let CUDA / HF threads unwind before the next model load.
sleep 15
echo "===== Qwen W4A4 proposed complete  $(date -Is) ====="
$PY -c "
import json
for p in [
  'results/ablation_w4a4/qwen_gptq_enc_wuni_g64_alog_a0_test-clean.json',
  'results/ablation_w4a4/qwen_gptq_enc_wuni_g64_alog_a0_test-other.json',
]:
    d=json.load(open(p))
    print(f\"  {p}: WER={d.get('wer')}\")
"

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

echo "===== proposed W4A8  W-uni g=$G  A-log8 g_a=0  sq=$SQ  $(date -Is) ====="

for split in test-clean test-other; do
  out="$OUT/whisper_unifW${G}_logA8_a0_${split}.json"
  run_or_skip "$out" \
    $PY src/run_eval.py --mode w4a4 --split "$split" \
      --w-mode uniform --a-mode log_token_a8 --scope nolm \
      --group-size "$G" --act-group-size 0 --smoothquant-alpha "$SQ" \
      --backend fake --batch-size 16 --max-new-tokens 224 --pack none \
      --out "$out"

  out="$OUT/qwen_gptq_enc_wuni_g${G}_alogA8_a0_${split}.json"
  run_or_skip "$out" \
    $PY src/run_eval_qwen3.py --mode w4a4 --split "$split" \
      --model-id "$GPTQ" \
      --w-mode uniform --a-mode log_token_a8 --scope encoder \
      --group-size "$G" --act-group-size 0 --smoothquant-alpha "$SQ" \
      --backend fake --batch-size 8 --max-new-tokens 256 \
      --out "$out"
done

echo "ALL JOBS FINISHED $(date -Is)"
