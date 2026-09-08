#!/bin/bash
# Fill grouping cells for the W4A4 ablation table.
# Same recipe as results/ablation_w4a4/* (log W4A4, log_token, fake), group-size=64.
# Matches W4A8 (grouping) g=64. Qwen uses BF16 base (not GPTQ), like the g=0 rows.
set -u
source /venv/main/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/SpeechPTQ/src
cd /workspace/SpeechPTQ

OUT=results/ablation_w4a4
mkdir -p "$OUT" logs
LOG=logs/grouping_ablation.log
exec > >(tee -a "$LOG") 2>&1

PY=/venv/main/bin/python
G=64

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

echo "===== grouping ablation g=$G  $(date -Is) ====="

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

    run_or_skip "$OUT/whisper_${name}_g${G}_${split}.json" \
      $PY src/run_eval.py --mode w4a4 --split "$split" \
        --w-mode log --a-mode log_token --scope "$scope" \
        --group-size "$G" --backend fake --batch-size 16 \
        --max-new-tokens 224 --pack none \
        --out "$OUT/whisper_${name}_g${G}_${split}.json"

    run_or_skip "$OUT/qwen_${name}_g${G}_${split}.json" \
      $PY src/run_eval_qwen3.py --mode w4a4 --split "$split" \
        --model-id Qwen/Qwen3-ASR-1.7B-hf \
        --w-mode log --a-mode log_token --scope "$scope" \
        --group-size "$G" --backend fake --batch-size 8 \
        --max-new-tokens 256 \
        --out "$OUT/qwen_${name}_g${G}_${split}.json"
  done
done

echo "[grouping] ALL JOBS FINISHED"
$PY - <<'PY'
import json
from pathlib import Path
root = Path("/workspace/SpeechPTQ/results/ablation_w4a4")

def w(p):
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("wer")

print(f"{'row':<16} {'Wh g0 c/o':>16} {'Wh g64 c/o':>16} {'Qw g0 c/o':>16} {'Qw g64 c/o':>16}")
rows = (
    ("Attention", "enc_attn"),
    ("FFN", "enc_ffn"),
    ("Encoder", "encoder"),
    ("Decoder", "decoder"),
    ("W4A4 (ours)", "nolm"),
)
for label, key in rows:
    cells = []
    for model in ("whisper", "qwen"):
        for gtag, gfile in (("g0", f"{model}_{key}_test-{{split}}.json"),
                            ("g64", f"{model}_{key}_g64_test-{{split}}.json")):
            # g0 FFN lives outside ablation_w4a4 for whisper/qwen enc_ffn
            c = w(root / gfile.format(split="clean"))
            o = w(root / gfile.format(split="other"))
            def fmt(x):
                return "—" if x is None else f"{x:.2f}"
            cells.append(f"{fmt(c)}/{fmt(o)}")
    print(f"{label:<16} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16} {cells[3]:>16}")
PY
echo "[grouping] done $(date -Is)"
