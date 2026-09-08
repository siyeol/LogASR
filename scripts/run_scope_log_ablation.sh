#!/bin/bash
# Ablation vs proposed (uniform W g64 + log A group-free, SQ 0.5):
#  1) log W g64, uniform A8 (W4A8)
#  2) log W g64 + log A group-free (W4A4)
#  3) encoder proposed W4A4, rest W4A8
#  4) encoder attn proposed W4A4, rest W4A8
#  5) encoder FFN proposed W4A4, rest W4A8
#  6) decoder proposed W4A4, rest W4A8
# Qwen LLM stays GPTQ W4A16 except (6), where proposed is applied to the LLM.
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
LOG=logs/scope_log_ablation.log
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
echo "===== scope/log ablation start $(date -Is) ====="

for split in test-clean test-other; do
  # 1) log W grouped, uniform A8
  run_or_skip "$OUT/whisper_logW64_a8_nolm_${split}.json" \
    $PY src/run_eval.py "${COMMON[@]}" --w-mode log --a-mode uniform_a8_dyn --scope nolm \
      --split "$split" --batch-size 16 --max-new-tokens 224 \
      --out "$OUT/whisper_logW64_a8_nolm_${split}.json"

  run_or_skip "$OUT/qwen_gptq_enc_logW64_a8_${split}.json" \
    $PY src/run_eval_qwen3.py "${COMMON[@]}" --w-mode log --a-mode encoder_a8 --scope encoder \
      --model-id "$QWEN_GPTQ" --split "$split" --batch-size 8 --max-new-tokens 256 \
      --out "$OUT/qwen_gptq_enc_logW64_a8_${split}.json"

  # 2) log W grouped + log A group-free (Whisper clean already 2.76)
  run_or_skip "$OUT/whisper_logW64_logA_a0_nolm_${split}.json" \
    $PY src/run_eval.py "${COMMON[@]}" --w-mode log --a-mode log_token --scope nolm \
      --split "$split" --batch-size 16 --max-new-tokens 224 \
      --out "$OUT/whisper_logW64_logA_a0_nolm_${split}.json"

  run_or_skip "$OUT/qwen_gptq_enc_logW64_logA_a0_${split}.json" \
    $PY src/run_eval_qwen3.py "${COMMON[@]}" --w-mode log --a-mode log_token --scope encoder \
      --model-id "$QWEN_GPTQ" --split "$split" --batch-size 8 --max-new-tokens 256 \
      --out "$OUT/qwen_gptq_enc_logW64_logA_a0_${split}.json"

  # 3) encoder proposed W4A4, rest W4A8. Qwen = existing 2.22/5.43
  run_or_skip "$OUT/whisper_unifW64_enc_logA4_resta8_${split}.json" \
    $PY src/run_eval.py "${COMMON[@]}" --w-mode uniform --a-mode mixed_log_encoder_a8 --scope nolm \
      --split "$split" --batch-size 16 --max-new-tokens 224 \
      --out "$OUT/whisper_unifW64_enc_logA4_resta8_${split}.json"

  # 4) encoder attn proposed W4A4, rest W4A8
  run_or_skip "$OUT/whisper_unifW64_encattn_logA4_resta8_${split}.json" \
    $PY src/run_eval.py "${COMMON[@]}" --w-mode uniform --a-mode mixed_log_encattn_a8 --scope nolm \
      --split "$split" --batch-size 16 --max-new-tokens 224 \
      --out "$OUT/whisper_unifW64_encattn_logA4_resta8_${split}.json"

  run_or_skip "$OUT/qwen_gptq_encattn_logA4_resta8_${split}.json" \
    $PY src/run_eval_qwen3.py "${COMMON[@]}" --w-mode uniform --a-mode mixed_log_encattn_enc_a8 --scope encoder \
      --model-id "$QWEN_GPTQ" --split "$split" --batch-size 8 --max-new-tokens 256 \
      --out "$OUT/qwen_gptq_encattn_logA4_resta8_${split}.json"

  # 5) encoder FFN proposed W4A4, rest W4A8
  run_or_skip "$OUT/whisper_unifW64_encffn_logA4_resta8_${split}.json" \
    $PY src/run_eval.py "${COMMON[@]}" --w-mode uniform --a-mode mixed_log_encffn_a8 --scope nolm \
      --split "$split" --batch-size 16 --max-new-tokens 224 \
      --out "$OUT/whisper_unifW64_encffn_logA4_resta8_${split}.json"

  run_or_skip "$OUT/qwen_gptq_encffn_logA4_resta8_${split}.json" \
    $PY src/run_eval_qwen3.py "${COMMON[@]}" --w-mode uniform --a-mode mixed_log_encffn_enc_a8 --scope encoder \
      --model-id "$QWEN_GPTQ" --split "$split" --batch-size 8 --max-new-tokens 256 \
      --out "$OUT/qwen_gptq_encffn_logA4_resta8_${split}.json"

  # 6) decoder proposed W4A4, rest W4A8. Qwen: proposed on LLM, encoder A8.
  run_or_skip "$OUT/whisper_unifW64_dec_logA4_resta8_${split}.json" \
    $PY src/run_eval.py "${COMMON[@]}" --w-mode uniform --a-mode mixed_log_decoder_a8 --scope nolm \
      --split "$split" --batch-size 16 --max-new-tokens 224 \
      --out "$OUT/whisper_unifW64_dec_logA4_resta8_${split}.json"

  run_or_skip "$OUT/qwen_gptq_dec_logA4_enc_a8_${split}.json" \
    $PY src/run_eval_qwen3.py "${COMMON[@]}" --w-mode uniform --a-mode mixed_log_decoder_a8 --scope nolm \
      --model-id "$QWEN_GPTQ" --split "$split" --batch-size 8 --max-new-tokens 256 \
      --out "$OUT/qwen_gptq_dec_logA4_enc_a8_${split}.json"
done

echo "===== ALL ABLATION JOBS FINISHED $(date -Is) ====="
$PY - <<'PY'
import json
from pathlib import Path
r = Path("/workspace/SpeechPTQ/results/ablation_w4a4")
f = Path("/workspace/SpeechPTQ/results/final")

def w(p):
    p = Path(p)
    if not p.exists():
        return "MISSING"
    d = json.loads(p.read_text())
    wer = d.get("wer")
    return "FAIL" if wer is None else f"{wer:.2f}"

rows = [
    ("proposed unifW+logA",
     f/"whisper_unifW64_logA_a0_test-clean.json",
     f/"whisper_unifW64_logA_a0_test-other.json",
     r/"qwen_gptq_enc_wuni_g64_alog_a0_test-clean.json",
     r/"qwen_gptq_enc_wuni_g64_alog_a0_test-other.json"),
    ("1 logW + A8",
     r/"whisper_logW64_a8_nolm_test-clean.json",
     r/"whisper_logW64_a8_nolm_test-other.json",
     r/"qwen_gptq_enc_logW64_a8_test-clean.json",
     r/"qwen_gptq_enc_logW64_a8_test-other.json"),
    ("2 logW + logA W4A4",
     r/"whisper_logW64_logA_a0_nolm_test-clean.json",
     r/"whisper_logW64_logA_a0_nolm_test-other.json",
     r/"qwen_gptq_enc_logW64_logA_a0_test-clean.json",
     r/"qwen_gptq_enc_logW64_logA_a0_test-other.json"),
    ("3 enc W4A4 / rest A8",
     r/"whisper_unifW64_enc_logA4_resta8_test-clean.json",
     r/"whisper_unifW64_enc_logA4_resta8_test-other.json",
     r/"qwen_gptq_enc_wuni_g64_alog_a0_test-clean.json",
     r/"qwen_gptq_enc_wuni_g64_alog_a0_test-other.json"),
    ("4 enc-attn W4A4 / rest A8",
     r/"whisper_unifW64_encattn_logA4_resta8_test-clean.json",
     r/"whisper_unifW64_encattn_logA4_resta8_test-other.json",
     r/"qwen_gptq_encattn_logA4_resta8_test-clean.json",
     r/"qwen_gptq_encattn_logA4_resta8_test-other.json"),
    ("5 enc-FFN W4A4 / rest A8",
     r/"whisper_unifW64_encffn_logA4_resta8_test-clean.json",
     r/"whisper_unifW64_encffn_logA4_resta8_test-other.json",
     r/"qwen_gptq_encffn_logA4_resta8_test-clean.json",
     r/"qwen_gptq_encffn_logA4_resta8_test-other.json"),
    ("6 dec W4A4 / rest A8",
     r/"whisper_unifW64_dec_logA4_resta8_test-clean.json",
     r/"whisper_unifW64_dec_logA4_resta8_test-other.json",
     r/"qwen_gptq_dec_logA4_enc_a8_test-clean.json",
     r/"qwen_gptq_dec_logA4_enc_a8_test-other.json"),
]
print(f"{'recipe':<28} {'W-cln':>8} {'W-oth':>8} {'Q-cln':>8} {'Q-oth':>8}")
for name, a, b, c, d in rows:
    print(f"{name:<28} {w(a):>8} {w(b):>8} {w(c):>8} {w(d):>8}")
PY
