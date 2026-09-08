#!/bin/bash
# Ablation vs proposed (uniform W g64 + log A per-tensor, SQ 0.5):
#  1) log W grouped, uniform A grouping-free
#  2) log W grouped, log A grouping-free
#  3) proposed W4A4 on encoder only (Whisper decoder FP16)
# Wait for the other agent's Qwen GPTQ W4A8 test-other to finish first.
set -u
source /venv/main/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/SpeechPTQ/src
cd /workspace/SpeechPTQ

PY=/venv/main/bin/python
QWEN_GPTQ=/workspace/SpeechPTQ/models/qwen3-asr-1.7b-W4A16-G128-llm
OUT=results/ablation_w4a4
mkdir -p "$OUT" logs
LOG=logs/log_placement_ablation.log
exec > >(tee -a "$LOG") 2>&1

done_ok() {
  local out="$1"
  local min_n="${2:-2000}"
  [ -f "$out" ] || return 1
  $PY -c "
import json,sys
d=json.load(open(sys.argv[1]))
sys.exit(0 if d.get('wer') is not None and int(d.get('n_utts') or 0)>=int(sys.argv[2]) else 1)
" "$out" "$min_n"
}

gpu_busy() {
  pgrep -f 'src/run_w4a8_method_compare.py --run' >/dev/null && return 0
  pgrep -f 'src/run_eval.py' >/dev/null && return 0
  pgrep -f 'src/run_eval_qwen3.py' >/dev/null && return 0
  pgrep -f 'src/quant_qwen3_enc_gptq.py' >/dev/null && return 0
  return 1
}

echo "===== wait for other-agent Qwen GPTQ W4A8 test-other  $(date -Is) ====="
GATE=results/compare/w4a8/qwen_enc_gptq_w4a8_test-other.json
while ! done_ok "$GATE" 2000 || gpu_busy; do
  sleep 20
done
echo "===== GPU free, start ablation  $(date -Is) ====="

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

COMMON=(--mode w4a4 --group-size 64 --act-group-size 0 --smoothquant-alpha 0.5 --backend fake)

for split in test-clean test-other; do
  # 1) log W grouped, uniform A grouping-free
  run_or_skip "$OUT/whisper_logW64_unifA_a0_nolm_${split}.json" \
    $PY src/run_eval.py "${COMMON[@]}" --w-mode log --a-mode uniform_dyn --scope nolm \
      --split "$split" --batch-size 16 --max-new-tokens 224 \
      --out "$OUT/whisper_logW64_unifA_a0_nolm_${split}.json"

  run_or_skip "$OUT/qwen_gptq_enc_logW64_unifA_a0_${split}.json" \
    $PY src/run_eval_qwen3.py "${COMMON[@]}" --w-mode log --a-mode uniform_dyn --scope encoder \
      --model-id "$QWEN_GPTQ" --split "$split" --batch-size 8 --max-new-tokens 256 \
      --out "$OUT/qwen_gptq_enc_logW64_unifA_a0_${split}.json"

  # 2) log W grouped, log A grouping-free
  run_or_skip "$OUT/whisper_logW64_logA_a0_nolm_${split}.json" \
    $PY src/run_eval.py "${COMMON[@]}" --w-mode log --a-mode log_token --scope nolm \
      --split "$split" --batch-size 16 --max-new-tokens 224 \
      --out "$OUT/whisper_logW64_logA_a0_nolm_${split}.json"

  run_or_skip "$OUT/qwen_gptq_enc_logW64_logA_a0_${split}.json" \
    $PY src/run_eval_qwen3.py "${COMMON[@]}" --w-mode log --a-mode log_token --scope encoder \
      --model-id "$QWEN_GPTQ" --split "$split" --batch-size 8 --max-new-tokens 256 \
      --out "$OUT/qwen_gptq_enc_logW64_logA_a0_${split}.json"

  # 3) proposed W4A4 encoder-only (Whisper). Qwen proposed is already encoder-only.
  run_or_skip "$OUT/whisper_unifW64_logA_a0_encoder_${split}.json" \
    $PY src/run_eval.py "${COMMON[@]}" --w-mode uniform --a-mode log_token --scope encoder \
      --split "$split" --batch-size 16 --max-new-tokens 224 \
      --out "$OUT/whisper_unifW64_logA_a0_encoder_${split}.json"
done

echo "===== ALL ABLATION JOBS FINISHED $(date -Is) ====="
$PY - <<'PY'
import json
from pathlib import Path
root = Path("/workspace/SpeechPTQ/results/ablation_w4a4")
final = Path("/workspace/SpeechPTQ/results/final")

def w(p):
    p = Path(p)
    if not p.exists():
        return "MISSING"
    d = json.loads(p.read_text())
    wer = d.get("wer")
    return "FAIL" if wer is None else f"{wer:.2f}"

rows = [
    ("proposed  unifW+logA  full/enc",
     final / "whisper_unifW64_logA_a0_test-clean.json",
     final / "whisper_unifW64_logA_a0_test-other.json",
     root / "qwen_gptq_enc_wuni_g64_alog_a0_test-clean.json",
     root / "qwen_gptq_enc_wuni_g64_alog_a0_test-other.json"),
    ("1 logW + unifA  full/enc",
     root / "whisper_logW64_unifA_a0_nolm_test-clean.json",
     root / "whisper_logW64_unifA_a0_nolm_test-other.json",
     root / "qwen_gptq_enc_logW64_unifA_a0_test-clean.json",
     root / "qwen_gptq_enc_logW64_unifA_a0_test-other.json"),
    ("2 logW + logA  full/enc",
     root / "whisper_logW64_logA_a0_nolm_test-clean.json",
     root / "whisper_logW64_logA_a0_nolm_test-other.json",
     root / "qwen_gptq_enc_logW64_logA_a0_test-clean.json",
     root / "qwen_gptq_enc_logW64_logA_a0_test-other.json"),
    ("3 proposed encoder-only",
     root / "whisper_unifW64_logA_a0_encoder_test-clean.json",
     root / "whisper_unifW64_logA_a0_encoder_test-other.json",
     root / "qwen_gptq_enc_wuni_g64_alog_a0_test-clean.json",
     root / "qwen_gptq_enc_wuni_g64_alog_a0_test-other.json"),
]
print(f"{'recipe':<32} {'W-cln':>8} {'W-oth':>8} {'Q-cln':>8} {'Q-oth':>8}")
for name, a, b, c, d in rows:
    print(f"{name:<32} {w(a):>8} {w(b):>8} {w(c):>8} {w(d):>8}")
PY
