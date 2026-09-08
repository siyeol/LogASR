#!/bin/bash
# Encoder-attention log A4, remaining Linears uniform A8. Grouping-free log W4.
# Whisper: decoder also A8. Qwen: GPTQ LLM stays A16; encoder rest A8.
set -u
source /venv/main/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/SpeechPTQ/src
cd /workspace/SpeechPTQ

OUT=results/ablation_w4a4
mkdir -p "$OUT" logs
LOG=logs/attn_a4_rest_a8.log
exec > >(tee -a "$LOG") 2>&1

PY=/venv/main/bin/python
QWEN_GPTQ=/workspace/SpeechPTQ/models/qwen3-asr-1.7b-W4A16-G128-llm

# Let the in-flight W4A8 Whisper test-other finish first.
W8=/workspace/SpeechPTQ/results/ablation_w4a4/whisper_w64_a0_w4a8_test-other.json
echo "waiting for $W8 $(date -Is)"
while [ ! -f "$W8" ]; do sleep 20; done
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('wer') is not None else 1)" "$W8"
echo "W4A8 other done, starting attn-A4 rest-A8 $(date -Is)"
sleep 5

done_ok() {
  local out="$1"
  [ -f "$out" ] || return 1
  $PY -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('wer') is not None else 1)" "$out"
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
  if "$@"; then
    echo "=== DONE $out ==="
  else
    echo "=== FAIL $out exit=$? ==="
  fi
  return 0
}

for split in test-clean test-other; do
  echo "----- Whisper || Qwen  attn-A4 rest-A8  $split -----"
  pids=()
  out="$OUT/whisper_log_encattn_a4_resta8_${split}.json"
  run_bg "$out" \
    $PY src/run_eval.py --mode w4a4 --split "$split" \
      --w-mode log --a-mode mixed_log_encattn_a8 --scope nolm \
      --group-size 0 --backend fake --batch-size 16 \
      --out "$out" &
  pids+=($!)
  sleep 40
  out="$OUT/qwen_gptq_log_encattn_a4_resta8_${split}.json"
  run_bg "$out" \
    $PY src/run_eval_qwen3.py --mode w4a4 --split "$split" \
      --model-id "$QWEN_GPTQ" \
      --w-mode log --a-mode mixed_log_encattn_enc_a8 --scope encoder \
      --group-size 0 --backend fake --batch-size 8 \
      --out "$out" &
  pids+=($!)
  for pid in "${pids[@]}"; do
    wait "$pid" || true
  done
done

echo "ALL JOBS FINISHED $(date -Is)"
$PY - <<'PY'
import json
from pathlib import Path
root = Path("/workspace/SpeechPTQ/results/ablation_w4a4")
for name in [
    "whisper_log_encattn_a4_resta8_test-clean.json",
    "whisper_log_encattn_a4_resta8_test-other.json",
    "qwen_gptq_log_encattn_a4_resta8_test-clean.json",
    "qwen_gptq_log_encattn_a4_resta8_test-other.json",
]:
    p = root / name
    if not p.exists():
        print(f"{name}: MISSING")
        continue
    d = json.loads(p.read_text())
    print(f"{name}: wer={d['wer']:.2f} n={d['n_utts']} a={d.get('a_mode')}")
PY
