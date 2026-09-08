#!/bin/bash
# After scope/log ablation: proposed W4A4 on OOD.
# Whisper already in results/final (skip-if-done). Qwen encoder proposed + GPTQ LLM is new.
set -u
source /venv/main/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/SpeechPTQ/src
cd /workspace/SpeechPTQ

PY=/venv/main/bin/python
QWEN_GPTQ=/workspace/SpeechPTQ/models/qwen3-asr-1.7b-W4A16-G128-llm
OUT=results/final
mkdir -p "$OUT" logs
LOG=logs/proposed_ood.log
exec > >(tee -a "$LOG") 2>&1

echo "===== wait for scope/log ablation  $(date -Is) ====="
while pgrep -f 'scripts/run_scope_log_ablation.sh' >/dev/null; do
  sleep 30
done
if ! grep -q 'ALL ABLATION JOBS FINISHED' logs/scope_log_ablation.log 2>/dev/null; then
  echo "=== WARN ablation script exited without ALL ABLATION JOBS FINISHED ==="
fi
echo "===== start proposed OOD  $(date -Is) ====="

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
  echo "=== RUN $(date -Is) $* ==="
  echo "    -> $out"
  if "$@"; then
    echo "=== DONE $(date -Is) $out ==="
  else
    echo "=== FAIL $out exit=$? $(date -Is) ==="
    return 0
  fi
}

W_COMMON=(--mode w4a4 --w-mode uniform --a-mode log_token --scope nolm
          --group-size 64 --act-group-size 0 --smoothquant-alpha 0.5 --backend fake)
Q_COMMON=(--mode w4a4 --w-mode uniform --a-mode log_token --scope encoder
          --group-size 64 --act-group-size 0 --smoothquant-alpha 0.5 --backend fake
          --model-id "$QWEN_GPTQ")

# dataset whisper_bs qwen_bs extra whisper/qwen args
run_or_skip "$OUT/whisper_unifW64_logA_a0_ami.json" \
  $PY src/run_eval.py "${W_COMMON[@]}" --dataset ami --split test --batch-size 8 --max-new-tokens 384 \
    --ami-stitch-sec 30 --out "$OUT/whisper_unifW64_logA_a0_ami.json"
run_or_skip "$OUT/qwen_unifW64_logA_a0_ami.json" \
  $PY src/run_eval_qwen3.py "${Q_COMMON[@]}" --dataset ami --split test --batch-size 4 --max-new-tokens 384 \
    --ami-stitch-sec 30 --out "$OUT/qwen_unifW64_logA_a0_ami.json"

run_or_skip "$OUT/whisper_unifW64_logA_a0_earnings22.json" \
  $PY src/run_eval.py "${W_COMMON[@]}" --dataset earnings22 --split test --batch-size 4 --max-new-tokens 384 \
    --out "$OUT/whisper_unifW64_logA_a0_earnings22.json"
run_or_skip "$OUT/qwen_unifW64_logA_a0_earnings22.json" \
  $PY src/run_eval_qwen3.py "${Q_COMMON[@]}" --dataset earnings22 --split test --batch-size 2 --max-new-tokens 384 \
    --out "$OUT/qwen_unifW64_logA_a0_earnings22.json"

run_or_skip "$OUT/whisper_unifW64_logA_a0_voxpopuli.json" \
  $PY src/run_eval.py "${W_COMMON[@]}" --dataset voxpopuli --split test --batch-size 8 --max-new-tokens 384 \
    --out "$OUT/whisper_unifW64_logA_a0_voxpopuli.json"
run_or_skip "$OUT/qwen_unifW64_logA_a0_voxpopuli.json" \
  $PY src/run_eval_qwen3.py "${Q_COMMON[@]}" --dataset voxpopuli --split test --batch-size 4 --max-new-tokens 384 \
    --out "$OUT/qwen_unifW64_logA_a0_voxpopuli.json"

run_or_skip "$OUT/whisper_unifW64_logA_a0_gigaspeech.json" \
  $PY src/run_eval.py "${W_COMMON[@]}" --dataset gigaspeech --split test --batch-size 4 --max-new-tokens 384 \
    --out "$OUT/whisper_unifW64_logA_a0_gigaspeech.json"
run_or_skip "$OUT/qwen_unifW64_logA_a0_gigaspeech.json" \
  $PY src/run_eval_qwen3.py "${Q_COMMON[@]}" --dataset gigaspeech --split test --batch-size 2 --max-new-tokens 384 \
    --out "$OUT/qwen_unifW64_logA_a0_gigaspeech.json"

run_or_skip "$OUT/whisper_unifW64_logA_a0_commonvoice.json" \
  $PY src/run_eval.py "${W_COMMON[@]}" --dataset commonvoice --split test --batch-size 8 --max-new-tokens 384 \
    --lang en --out "$OUT/whisper_unifW64_logA_a0_commonvoice.json"
run_or_skip "$OUT/qwen_unifW64_logA_a0_commonvoice.json" \
  $PY src/run_eval_qwen3.py "${Q_COMMON[@]}" --dataset commonvoice --split test --batch-size 4 --max-new-tokens 384 \
    --lang en --out "$OUT/qwen_unifW64_logA_a0_commonvoice.json"

echo "===== PROPOSED OOD FINISHED $(date -Is) ====="
$PY - <<'PY'
import json
from pathlib import Path
root = Path("/workspace/SpeechPTQ/results/final")
print(f"{'set':<14} {'whisper':>10} {'qwen':>10}")
for ds in ("ami", "earnings22", "voxpopuli", "gigaspeech", "commonvoice"):
    def fmt(stem):
        p = root / f"{stem}_{ds}.json"
        if not p.exists():
            return "MISSING"
        d = json.loads(p.read_text())
        wer = d.get("wer")
        return "FAIL" if wer is None else f"{wer:.2f}"
    print(f"{ds:<14} {fmt('whisper_unifW64_logA_a0'):>10} {fmt('qwen_unifW64_logA_a0'):>10}")
PY
