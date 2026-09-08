#!/bin/bash
# OOD: FP16/BF16 vs locked Proposed, Whisper + Qwen3.
# Whisper Proposed: log W4, FFN A4 / rest A8, scope=nolm, g=0  (2.42 / 5.93)
# Qwen Proposed: GPTQ LLM + encoder log W4, FFN A4 / rest A8     (3.22 / 8.81)
# CommonVoice = ESB English test (not 27-locale CV).
set -u
source /venv/main/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/SpeechPTQ/src
cd /workspace/SpeechPTQ

OUTDIR=results/ood
mkdir -p "$OUTDIR" logs
LOG=logs/ood_fp16_proposed.log
exec > >(tee -a "$LOG") 2>&1

QWEN_GPTQ=/workspace/SpeechPTQ/models/qwen3-asr-1.7b-W4A16-G128-llm
PY=/venv/main/bin/python

done_ok() {
  local out="$1"
  local min_n="${2:-50}"
  [ -f "$out" ] || return 1
  $PY -c "
import json,sys
d=json.load(open(sys.argv[1]))
sys.exit(0 if d.get('wer') is not None and int(d.get('n_utts') or 0)>=int(sys.argv[2]) else 1)
" "$out" "$min_n"
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

echo "===== prepare missing OOD caches ====="
for ds in tedlium gigaspeech; do
  if [ -f "data/ood/${ds}/en/all.trans.txt" ]; then
    echo "=== SKIP cache $ds ==="
  else
    echo "=== PREP $ds ==="
    $PY src/ood_data.py "$ds" || echo "=== FAIL prep $ds ==="
  fi
done
if [ -f data/ood/commonvoice/en/all.trans.txt ]; then
  echo "=== SKIP cache commonvoice/en ==="
else
  echo "=== PREP commonvoice/en (ESB) ==="
  $PY src/ood_data.py commonvoice --lang en || echo "=== FAIL prep commonvoice ==="
fi

# dataset w_bs q_bs
# AMI stitch-30 already cached.
for spec in \
  "ami 8 4" \
  "earnings22 4 2" \
  "voxpopuli 8 4" \
  "tedlium 8 4" \
  "gigaspeech 4 2" \
  "commonvoice 8 4"
do
  set -- $spec
  ds=$1 wbs=$2 qbs=$3
  lang_args=()
  ami_w=()
  ami_q=(--ami-stitch-sec 0)
  if [ "$ds" = "commonvoice" ]; then
    lang_args=(--lang en)
  fi
  if [ "$ds" = "ami" ]; then
    ami_w=(--ami-stitch-sec 30)
    ami_q=(--ami-stitch-sec 30)
  fi

  echo "[ood-prop] dataset=$ds"

  run_or_skip "$OUTDIR/whisper_fp16_${ds}.json" \
    $PY src/run_eval.py --mode fp16 --dataset "$ds" --split test \
      --batch-size "$wbs" --max-new-tokens 384 --pack none \
      --out "$OUTDIR/whisper_fp16_${ds}.json" \
      "${ami_w[@]}" "${lang_args[@]}"

  run_or_skip "$OUTDIR/whisper_proposed_${ds}.json" \
    $PY src/run_eval.py --mode w4a4 --dataset "$ds" --split test \
      --w-mode log --a-mode mixed_log_encffn_a8 --scope nolm \
      --group-size 0 --backend fake \
      --batch-size "$wbs" --max-new-tokens 384 --pack none \
      --out "$OUTDIR/whisper_proposed_${ds}.json" \
      "${ami_w[@]}" "${lang_args[@]}"

  run_or_skip "$OUTDIR/qwen_fp16_${ds}.json" \
    $PY src/run_eval_qwen3.py --mode fp16 --dataset "$ds" --split test \
      --model-id Qwen/Qwen3-ASR-1.7B-hf \
      --batch-size "$qbs" --max-new-tokens 384 \
      --out "$OUTDIR/qwen_fp16_${ds}.json" \
      "${ami_q[@]}" "${lang_args[@]}"

  run_or_skip "$OUTDIR/qwen_proposed_${ds}.json" \
    $PY src/run_eval_qwen3.py --mode w4a4 --dataset "$ds" --split test \
      --model-id "$QWEN_GPTQ" \
      --w-mode log --a-mode mixed_log_encffn_enc_a8 --scope encoder \
      --group-size 0 --backend fake \
      --batch-size "$qbs" --max-new-tokens 384 \
      --out "$OUTDIR/qwen_proposed_${ds}.json" \
      "${ami_q[@]}" "${lang_args[@]}"

  echo "[ood-prop] finished $ds"
done

echo "[ood-prop] ALL JOBS FINISHED"
$PY - <<'PY'
import json
from pathlib import Path
root = Path("/workspace/SpeechPTQ/results/ood")
print(f"{'dataset':<14} {'whisper_fp16':>12} {'whisper_prop':>12} {'qwen_fp16':>12} {'qwen_prop':>12}")
for ds in ("ami", "earnings22", "voxpopuli", "tedlium", "gigaspeech", "commonvoice"):
    row = [ds]
    for stem in (f"whisper_fp16_{ds}", f"whisper_proposed_{ds}", f"qwen_fp16_{ds}", f"qwen_proposed_{ds}"):
        p = root / f"{stem}.json"
        if not p.exists():
            row.append("MISSING")
            continue
        d = json.loads(p.read_text())
        row.append(f"{d.get('wer', float('nan')):.2f}")
    print(f"{row[0]:<14} {row[1]:>12} {row[2]:>12} {row[3]:>12} {row[4]:>12}")
PY
