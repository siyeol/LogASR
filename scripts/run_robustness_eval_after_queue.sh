#!/bin/bash
# After ablation + proposed OOD + robustness simulation:
# Whisper/Qwen FP16 and proposed WER on test-clean-reverb and test-clean-dns.
set -u
source /venv/main/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/SpeechPTQ/src
cd /workspace/SpeechPTQ

PY=/venv/main/bin/python
QWEN_FP=Qwen/Qwen3-ASR-1.7B-hf
QWEN_GPTQ=/workspace/SpeechPTQ/models/qwen3-asr-1.7b-W4A16-G128-llm
OUT=results/robustness
mkdir -p "$OUT" logs
LOG=logs/robustness_eval.log
exec > >(tee -a "$LOG") 2>&1

queue_busy() {
  pgrep -f 'scripts/run_scope_log_ablation.sh' >/dev/null && return 0
  pgrep -f 'scripts/run_proposed_ood_after_ablation.sh' >/dev/null && return 0
  pgrep -f 'src/simulate_robustness.py' >/dev/null && return 0
  pgrep -f 'src/run_eval.py' >/dev/null && return 0
  pgrep -f 'src/run_eval_qwen3.py' >/dev/null && return 0
  return 1
}

flac_n() {
  local d="$1"
  find "$d" -name '*.flac' 2>/dev/null | wc -l
}

echo "===== wait for queued GPU/CPU jobs  $(date -Is) ====="
while queue_busy; do
  sleep 60
done
echo "===== wait for 2620 flacs on both robustness sets  $(date -Is) ====="
while [ "$(flac_n data/LibriSpeech/test-clean-reverb)" -lt 2620 ] || \
      [ "$(flac_n data/LibriSpeech/test-clean-dns)" -lt 2620 ]; do
  echo "  reverb=$(flac_n data/LibriSpeech/test-clean-reverb) dns=$(flac_n data/LibriSpeech/test-clean-dns)"
  sleep 60
done
echo "===== start robustness eval  $(date -Is) ====="

done_ok() {
  local out="$1"
  [ -f "$out" ] || return 1
  $PY -c "
import json,sys
d=json.load(open(sys.argv[1]))
sys.exit(0 if d.get('wer') is not None and int(d.get('n_utts') or 0)>=2000 else 1)
" "$out"
}

run_or_skip() {
  local out="$1"
  shift
  if done_ok "$out"; then
    echo "=== SKIP existing $out ==="
    return 0
  fi
  echo "=== RUN $(date -Is) $* ==="
  echo "    -> $out"
  if "$@"; then
    echo "=== DONE $(date -Is) $out ==="
  else
    echo "=== FAIL $out exit=$? $(date -Is) ==="
    return 0
  fi
}

PROP_W=(--mode w4a4 --w-mode uniform --a-mode log_token --scope nolm
        --group-size 64 --act-group-size 0 --smoothquant-alpha 0.5 --backend fake)
PROP_Q=(--mode w4a4 --w-mode uniform --a-mode log_token --scope encoder
        --group-size 64 --act-group-size 0 --smoothquant-alpha 0.5 --backend fake
        --model-id "$QWEN_GPTQ")

for split in test-clean-reverb test-clean-dns; do
  run_or_skip "$OUT/whisper_fp16_${split}.json" \
    $PY src/run_eval.py --mode fp16 --dataset librispeech --split "$split" \
      --batch-size 16 --max-new-tokens 224 --out "$OUT/whisper_fp16_${split}.json"

  run_or_skip "$OUT/whisper_proposed_${split}.json" \
    $PY src/run_eval.py "${PROP_W[@]}" --dataset librispeech --split "$split" \
      --batch-size 16 --max-new-tokens 224 --out "$OUT/whisper_proposed_${split}.json"

  run_or_skip "$OUT/qwen_fp16_${split}.json" \
    $PY src/run_eval_qwen3.py --mode fp16 --dataset librispeech --split "$split" \
      --model-id "$QWEN_FP" --batch-size 8 --max-new-tokens 256 \
      --out "$OUT/qwen_fp16_${split}.json"

  run_or_skip "$OUT/qwen_proposed_${split}.json" \
    $PY src/run_eval_qwen3.py "${PROP_Q[@]}" --dataset librispeech --split "$split" \
      --batch-size 8 --max-new-tokens 256 --out "$OUT/qwen_proposed_${split}.json"
done

echo "===== ROBUSTNESS EVAL FINISHED $(date -Is) ====="
$PY - <<'PY'
import json
from pathlib import Path
root = Path("/workspace/SpeechPTQ/results/robustness")
print(f"{'set':<22} {'W-fp16':>8} {'W-prop':>8} {'Q-fp16':>8} {'Q-prop':>8}")
for split in ("test-clean-reverb", "test-clean-dns"):
    def w(stem):
        p = root / f"{stem}_{split}.json"
        if not p.exists():
            return "MISSING"
        d = json.loads(p.read_text())
        wer = d.get("wer")
        return "FAIL" if wer is None else f"{wer:.2f}"
    print(f"{split:<22} {w('whisper_fp16'):>8} {w('whisper_proposed'):>8} {w('qwen_fp16'):>8} {w('qwen_proposed'):>8}")
PY
