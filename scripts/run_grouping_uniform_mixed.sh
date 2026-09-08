#!/bin/bash
# Grouping baseline (uniform RTN + SmoothQuant 0.5 + g=64):
#   1) full W4A4
#   2) encoder-attn W4A4, rest W4A8
#   3) encoder-FFN W4A4, rest W4A8
#   4) encoder W4A4, rest W4A8
# Weights are grouped uniform W4 everywhere (scope=nolm). Qwen included.
set -u
source /venv/main/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/SpeechPTQ/src
cd /workspace/SpeechPTQ

OUT=results/ablation_w4a4
mkdir -p "$OUT" logs
LOG=logs/grouping_uniform_mixed.log
exec > >(tee -a "$LOG") 2>&1

PY=/venv/main/bin/python
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
    return 0
  fi
}

echo "===== uniform grouping mixed A4/A8 g=$G sq=$SQ  $(date -Is) ====="

# name  a_mode
for split in test-clean test-other; do
  for spec in \
    "full_a4 uniform_dyn" \
    "enc_attn_a4_resta8 mixed_uni_encattn_a4_a8" \
    "enc_ffn_a4_resta8 mixed_uni_encffn_a4_a8" \
    "encoder_a4_resta8 mixed_uni_encoder_a4_a8"
  do
    set -- $spec
    name=$1 amode=$2

    run_or_skip "$OUT/whisper_${name}_uniform_g${G}_${split}.json" \
      $PY src/run_eval.py --mode w4a4 --split "$split" \
        --w-mode uniform --a-mode "$amode" --scope nolm \
        --group-size "$G" --smoothquant-alpha "$SQ" \
        --backend fake --batch-size 16 --max-new-tokens 224 --pack none \
        --out "$OUT/whisper_${name}_uniform_g${G}_${split}.json"

    run_or_skip "$OUT/qwen_${name}_uniform_g${G}_${split}.json" \
      $PY src/run_eval_qwen3.py --mode w4a4 --split "$split" \
        --model-id Qwen/Qwen3-ASR-1.7B-hf \
        --w-mode uniform --a-mode "$amode" --scope nolm \
        --group-size "$G" --smoothquant-alpha "$SQ" \
        --backend fake --batch-size 8 --max-new-tokens 256 \
        --out "$OUT/qwen_${name}_uniform_g${G}_${split}.json"
  done
done

echo "[grouping-mixed] ALL JOBS FINISHED $(date -Is)"
