#!/bin/bash
# After log-W test-clean finishes, pick the better of:
#   1) uniform W g64 + log A per-tensor
#   2) log W g64     + log A per-tensor
# then eval the winner on Whisper (LS other + OOD) and Qwen encoder + GPTQ LLM.
set -u
source /venv/main/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/SpeechPTQ/src
cd /workspace/SpeechPTQ

PY=/venv/main/bin/python
UNIF_JSON=results/ablation_w4a4/whisper_unifW64_logA_a0_nolm_test-clean.json
LOG_JSON=results/ablation_w4a4/whisper_logW64_logA_a0_nolm_test-clean.json
QWEN_GPTQ=/workspace/SpeechPTQ/models/qwen3-asr-1.7b-W4A16-G128-llm
OUTDIR=results/final
mkdir -p "$OUTDIR" logs
LOG=logs/final_proposed_suite.log
exec > >(tee -a "$LOG") 2>&1

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

echo "===== wait for log-W test-clean  $(date -Is) ====="
while ! done_ok "$LOG_JSON" 2000; do
  if ! kill -0 7529 2>/dev/null && ! pgrep -f 'whisper_logW64_logA_a0_nolm_test-clean' >/dev/null; then
    if ! done_ok "$LOG_JSON" 2000; then
      echo "=== FAIL log-W eval exited without a valid JSON ==="
      exit 1
    fi
  fi
  sleep 30
done

echo "===== pick winner on LS test-clean  $(date -Is) ====="
WINNER_META=$OUTDIR/winner.json
$PY - <<PY
import json
from pathlib import Path
unif = json.loads(Path("$UNIF_JSON").read_text())
log = json.loads(Path("$LOG_JSON").read_text())
u, l = float(unif["wer"]), float(log["wer"])
print(f"{'recipe':<44} {'wer':>8} {'g_w':>4} {'g_a':>4} {'sq':>4} {'w':>8} {'a':>10}")
print(f"{'1 uniform-W g64 + log-A per-tensor':<44} {u:8.2f} {unif.get('group_size'):>4} {unif.get('act_group_size'):>4} {unif.get('smoothquant_alpha'):>4} {unif.get('w_mode'):>8} {unif.get('a_mode'):>10}")
print(f"{'2 log-W g64 + log-A per-tensor':<44} {l:8.2f} {log.get('group_size'):>4} {log.get('act_group_size'):>4} {log.get('smoothquant_alpha'):>4} {log.get('w_mode'):>8} {log.get('a_mode'):>10}")
# Lower WER wins. Tie -> uniform (standard grouped codebook).
if l + 1e-6 < u:
    w_mode, tag, wer = "log", "logW64_logA_a0", l
    src = "$LOG_JSON"
else:
    w_mode, tag, wer = "uniform", "unifW64_logA_a0", u
    src = "$UNIF_JSON"
print(f"WINNER w_mode={w_mode} tag={tag} wer={wer:.2f}")
Path("$WINNER_META").write_text(json.dumps({
    "w_mode": w_mode,
    "a_mode": "log_token",
    "group_size": 64,
    "act_group_size": 0,
    "smoothquant_alpha": 0.5,
    "tag": tag,
    "ls_clean_wer": wer,
    "uniform_wer": u,
    "log_wer": l,
    "winner_src": src,
}, indent=2) + "\n")
PY

W_MODE=$($PY -c "import json; print(json.load(open('$WINNER_META'))['w_mode'])")
TAG=$($PY -c "import json; print(json.load(open('$WINNER_META'))['tag'])")
echo "===== final recipe: w=$W_MODE a=log_token g_w=64 g_a=0 sq=0.5 tag=$TAG ====="

# Copy Whisper LS clean winner into results/final
run_or_skip "$OUTDIR/whisper_${TAG}_test-clean.json" \
  cp -f "$( $PY -c "import json; print(json.load(open('$WINNER_META'))['winner_src'])" )" \
     "$OUTDIR/whisper_${TAG}_test-clean.json"

COMMON=(--mode w4a4 --w-mode "$W_MODE" --a-mode log_token
        --group-size 64 --act-group-size 0 --smoothquant-alpha 0.5 --backend fake)

# --- Whisper ---
run_or_skip "$OUTDIR/whisper_${TAG}_test-other.json" \
  $PY src/run_eval.py "${COMMON[@]}" --scope nolm --dataset librispeech --split test-other \
    --batch-size 16 --max-new-tokens 224 \
    --out "$OUTDIR/whisper_${TAG}_test-other.json"

run_or_skip "$OUTDIR/whisper_${TAG}_ami.json" \
  $PY src/run_eval.py "${COMMON[@]}" --scope nolm --dataset ami --split test \
    --batch-size 8 --max-new-tokens 384 --ami-stitch-sec 30 \
    --out "$OUTDIR/whisper_${TAG}_ami.json"

run_or_skip "$OUTDIR/whisper_${TAG}_earnings22.json" \
  $PY src/run_eval.py "${COMMON[@]}" --scope nolm --dataset earnings22 --split test \
    --batch-size 4 --max-new-tokens 384 \
    --out "$OUTDIR/whisper_${TAG}_earnings22.json"

run_or_skip "$OUTDIR/whisper_${TAG}_voxpopuli.json" \
  $PY src/run_eval.py "${COMMON[@]}" --scope nolm --dataset voxpopuli --split test \
    --batch-size 8 --max-new-tokens 384 \
    --out "$OUTDIR/whisper_${TAG}_voxpopuli.json"

run_or_skip "$OUTDIR/whisper_${TAG}_gigaspeech.json" \
  $PY src/run_eval.py "${COMMON[@]}" --scope nolm --dataset gigaspeech --split test \
    --batch-size 4 --max-new-tokens 384 \
    --out "$OUTDIR/whisper_${TAG}_gigaspeech.json"

run_or_skip "$OUTDIR/whisper_${TAG}_commonvoice.json" \
  $PY src/run_eval.py "${COMMON[@]}" --scope nolm --dataset commonvoice --split test \
    --batch-size 8 --max-new-tokens 384 --lang en \
    --out "$OUTDIR/whisper_${TAG}_commonvoice.json"

# --- Qwen: GPTQ W4A16 LLM + encoder-only winner recipe ---
run_or_skip "$OUTDIR/qwen_${TAG}_test-clean.json" \
  $PY src/run_eval_qwen3.py "${COMMON[@]}" --scope encoder --model-id "$QWEN_GPTQ" \
    --dataset librispeech --split test-clean --batch-size 8 --max-new-tokens 256 \
    --out "$OUTDIR/qwen_${TAG}_test-clean.json"

run_or_skip "$OUTDIR/qwen_${TAG}_test-other.json" \
  $PY src/run_eval_qwen3.py "${COMMON[@]}" --scope encoder --model-id "$QWEN_GPTQ" \
    --dataset librispeech --split test-other --batch-size 8 --max-new-tokens 256 \
    --out "$OUTDIR/qwen_${TAG}_test-other.json"

run_or_skip "$OUTDIR/qwen_${TAG}_ami.json" \
  $PY src/run_eval_qwen3.py "${COMMON[@]}" --scope encoder --model-id "$QWEN_GPTQ" \
    --dataset ami --split test --batch-size 4 --max-new-tokens 384 --ami-stitch-sec 30 \
    --out "$OUTDIR/qwen_${TAG}_ami.json"

run_or_skip "$OUTDIR/qwen_${TAG}_earnings22.json" \
  $PY src/run_eval_qwen3.py "${COMMON[@]}" --scope encoder --model-id "$QWEN_GPTQ" \
    --dataset earnings22 --split test --batch-size 2 --max-new-tokens 384 \
    --out "$OUTDIR/qwen_${TAG}_earnings22.json"

run_or_skip "$OUTDIR/qwen_${TAG}_voxpopuli.json" \
  $PY src/run_eval_qwen3.py "${COMMON[@]}" --scope encoder --model-id "$QWEN_GPTQ" \
    --dataset voxpopuli --split test --batch-size 4 --max-new-tokens 384 \
    --out "$OUTDIR/qwen_${TAG}_voxpopuli.json"

run_or_skip "$OUTDIR/qwen_${TAG}_gigaspeech.json" \
  $PY src/run_eval_qwen3.py "${COMMON[@]}" --scope encoder --model-id "$QWEN_GPTQ" \
    --dataset gigaspeech --split test --batch-size 2 --max-new-tokens 384 \
    --out "$OUTDIR/qwen_${TAG}_gigaspeech.json"

run_or_skip "$OUTDIR/qwen_${TAG}_commonvoice.json" \
  $PY src/run_eval_qwen3.py "${COMMON[@]}" --scope encoder --model-id "$QWEN_GPTQ" \
    --dataset commonvoice --split test --batch-size 4 --max-new-tokens 384 --lang en \
    --out "$OUTDIR/qwen_${TAG}_commonvoice.json"

echo "===== ALL JOBS FINISHED $(date -Is) ====="
$PY - <<'PY'
import json
from pathlib import Path
meta = json.loads(Path("/workspace/SpeechPTQ/results/final/winner.json").read_text())
tag = meta["tag"]
root = Path("/workspace/SpeechPTQ/results/final")
print(f"winner w_mode={meta['w_mode']}  LS-clean Whisper WER={meta['ls_clean_wer']:.2f}")
print(f"  (uniform {meta['uniform_wer']:.2f} vs log {meta['log_wer']:.2f})")
print(f"{'set':<16} {'whisper':>10} {'qwen':>10}")
rows = [
    ("test-clean", f"whisper_{tag}_test-clean.json", f"qwen_{tag}_test-clean.json"),
    ("test-other", f"whisper_{tag}_test-other.json", f"qwen_{tag}_test-other.json"),
    ("ami", f"whisper_{tag}_ami.json", f"qwen_{tag}_ami.json"),
    ("earnings22", f"whisper_{tag}_earnings22.json", f"qwen_{tag}_earnings22.json"),
    ("voxpopuli", f"whisper_{tag}_voxpopuli.json", f"qwen_{tag}_voxpopuli.json"),
    ("gigaspeech", f"whisper_{tag}_gigaspeech.json", f"qwen_{tag}_gigaspeech.json"),
    ("commonvoice", f"whisper_{tag}_commonvoice.json", f"qwen_{tag}_commonvoice.json"),
]
for name, w, q in rows:
    def fmt(p):
        path = root / p
        if not path.exists():
            return "MISSING"
        d = json.loads(path.read_text())
        wer = d.get("wer")
        return "FAIL" if wer is None else f"{wer:.2f}"
    print(f"{name:<16} {fmt(w):>10} {fmt(q):>10}")
PY
