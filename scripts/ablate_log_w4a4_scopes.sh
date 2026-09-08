#!/bin/bash
# Grouping-free log W4A4 by scope: enc_attn / encoder / decoder / nolm
# on LibriSpeech test-clean and test-other for Whisper and Qwen3-ASR.
set -u
source /venv/main/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/SpeechPTQ/src
cd /workspace/SpeechPTQ

OUTDIR=results/ablation_w4a4
mkdir -p "$OUTDIR" logs
LOG=logs/ablate_log_w4a4_scopes.log
exec > >(tee -a "$LOG") 2>&1

run_or_skip() {
  local out="$1"
  shift
  if [ -f "$out" ]; then
    local n
    n=$(python -c "import json; print(json.load(open('$out')).get('n_utts',0))" 2>/dev/null || echo 0)
    if [ "$n" -ge 2600 ]; then
      echo "=== SKIP existing $out n_utts=$n ==="
      return 0
    fi
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

# Reuse already-finished identical recipes.
if [ ! -f "$OUTDIR/whisper_enc_attn_test-clean.json" ]; then
  :
fi
cp -n results/w4a4_test-clean_log_log_token_nolm.json \
  "$OUTDIR/whisper_nolm_test-clean.json" 2>/dev/null || true
cp -n results/qwen3_w4a4_test-clean_log_log_token_encoder_fake.json \
  "$OUTDIR/qwen_encoder_test-clean.json" 2>/dev/null || true
cp -n results/qwen3_w4a4_test-clean_log_log_token_nolm_fake.json \
  "$OUTDIR/qwen_nolm_test-clean.json" 2>/dev/null || true

PY=/venv/main/bin/python

for split in test-clean test-other; do
  for scope in enc_attn encoder decoder nolm; do
    out="$OUTDIR/whisper_${scope}_${split}.json"
    run_or_skip "$out" $PY src/run_eval.py \
      --mode w4a4 --w-mode log --a-mode log_token --scope "$scope" \
      --group-size 0 --backend fake --split "$split" --batch-size 16 \
      --out "$out"
  done
done

for split in test-clean test-other; do
  for scope in enc_attn encoder decoder nolm; do
    out="$OUTDIR/qwen_${scope}_${split}.json"
    run_or_skip "$out" $PY src/run_eval_qwen3.py \
      --mode w4a4 --model-id Qwen/Qwen3-ASR-1.7B-hf \
      --w-mode log --a-mode log_token --scope "$scope" \
      --group-size 0 --backend fake --split "$split" --batch-size 8 \
      --out "$out"
  done
done

echo "=== ALL JOBS FINISHED ==="
$PY - <<'PY'
import json
from pathlib import Path
root = Path("/workspace/SpeechPTQ/results/ablation_w4a4")
print(f"{'model':<8} {'scope':<10} {'split':<12} {'n':>5} {'wer':>8}")
for model in ("whisper", "qwen"):
    for scope in ("enc_attn", "encoder", "decoder", "nolm"):
        for split in ("test-clean", "test-other"):
            p = root / f"{model}_{scope}_{split}.json"
            if not p.exists():
                print(f"{model:<8} {scope:<10} {split:<12} {'':>5} {'MISSING':>8}")
                continue
            d = json.loads(p.read_text())
            print(f"{model:<8} {scope:<10} {split:<12} {d.get('n_utts',0):>5} {d.get('wer', float('nan')):8.2f}")
PY
