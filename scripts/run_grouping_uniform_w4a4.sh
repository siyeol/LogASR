#!/bin/bash
# Grouping *baseline* for the W4A4 ablation table: grouped uniform W4A4 (not log).
# Same stack as W4A8 (grouping): uniform RTN + SmoothQuant 0.5 + g=64, but activations INT4.
set -u
source /venv/main/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/SpeechPTQ/src
cd /workspace/SpeechPTQ

OUT=results/ablation_w4a4
mkdir -p "$OUT" logs
LOG=logs/grouping_uniform_w4a4.log
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

echo "===== uniform W4A4 grouping g=$G sq=$SQ  $(date -Is) ====="

for split in test-clean test-other; do
  for spec in \
    "enc_attn enc_attn" \
    "enc_ffn enc_ffn" \
    "encoder encoder" \
    "decoder decoder" \
    "nolm nolm"
  do
    set -- $spec
    name=$1 scope=$2

    run_or_skip "$OUT/whisper_${name}_uniform_g${G}_${split}.json" \
      $PY src/run_eval.py --mode w4a4 --split "$split" \
        --w-mode uniform --a-mode uniform_dyn --scope "$scope" \
        --group-size "$G" --smoothquant-alpha "$SQ" \
        --backend fake --batch-size 16 --max-new-tokens 224 --pack none \
        --out "$OUT/whisper_${name}_uniform_g${G}_${split}.json"

    run_or_skip "$OUT/qwen_${name}_uniform_g${G}_${split}.json" \
      $PY src/run_eval_qwen3.py --mode w4a4 --split "$split" \
        --model-id Qwen/Qwen3-ASR-1.7B-hf \
        --w-mode uniform --a-mode uniform_dyn --scope "$scope" \
        --group-size "$G" --smoothquant-alpha "$SQ" \
        --backend fake --batch-size 8 --max-new-tokens 256 \
        --out "$OUT/qwen_${name}_uniform_g${G}_${split}.json"
  done
done

echo "[grouping-uniform] ALL JOBS FINISHED $(date -Is)"
