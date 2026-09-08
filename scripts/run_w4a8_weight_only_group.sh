#!/bin/bash
# Production W4A8 proxy: grouped WEIGHTS, per-tensor A8 (no act grouping).
# Whisper: all Linear A8. Qwen: encoder A8, LLM activations FP (encoder_a8).
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
LOG=logs/w4a8_weight_only_group.log
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

run_bg() {
  local out="$1"
  shift
  if done_ok "$out"; then
    echo "=== SKIP existing $out ==="
    return 0
  fi
  echo "=== RUN $* ==="
  echo "    -> $out"
  "$@"
  local ec=$?
  if [ "$ec" -eq 0 ]; then
    echo "=== DONE $out ==="
  else
    echo "=== FAIL $out exit=$ec ==="
  fi
  return 0
}

echo "===== W4A8 weight-only grouping g_w=$G g_a=0 sq=$SQ  $(date -Is) ====="

for split in test-clean test-other; do
  echo "----- Whisper || Qwen W4A8  $split -----"
  pids=()

  out="$OUT/whisper_w${G}_a0_w4a8_${split}.json"
  run_bg "$out" \
    $PY src/run_eval.py --mode w4a4 --split "$split" \
      --w-mode uniform --a-mode uniform_a8_dyn --scope nolm \
      --group-size "$G" --act-group-size 0 --smoothquant-alpha "$SQ" \
      --backend fake --batch-size 16 --max-new-tokens 224 --pack none \
      --out "$out" &
  pids+=($!)
  sleep 45

  out="$OUT/qwen_w${G}_a0_w4a8_${split}.json"
  run_bg "$out" \
    $PY src/run_eval_qwen3.py --mode w4a4 --split "$split" \
      --model-id Qwen/Qwen3-ASR-1.7B-hf \
      --w-mode uniform --a-mode encoder_a8 --scope nolm \
      --group-size "$G" --act-group-size 0 --smoothquant-alpha "$SQ" \
      --backend fake --batch-size 8 --max-new-tokens 256 \
      --out "$out" &
  pids+=($!)

  for pid in "${pids[@]}"; do
    wait "$pid" || true
  done
done

echo "ALL JOBS FINISHED $(date -Is)"
