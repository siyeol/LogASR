#!/bin/bash
# After Qwen GPTQ+encoder uniform A4 baseline finishes, measure the four
# efficiency metrics for FP16 / W4A8 baseline / W4A4 ours.
set -u
source /venv/main/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/SpeechPTQ/src
cd /workspace/SpeechPTQ

mkdir -p logs results/diagnostics
LOG=logs/eff_table_four.log
exec > >(tee -a "$LOG") 2>&1

PY=/venv/main/bin/python
OUT=results/ablation_w4a4

done_ok() {
  local out="$1"
  [ -f "$out" ] || return 1
  $PY -c "
import json,sys
d=json.load(open(sys.argv[1]))
sys.exit(0 if d.get('wer') is not None else 1)
" "$out"
}

echo "===== wait for Qwen W4A4 baseline  $(date -Is) ====="
Q1=$OUT/qwen_gptq_enc_wuni_g64_auni_a0_test-clean.json
Q2=$OUT/qwen_gptq_enc_wuni_g64_auni_a0_test-other.json
while ! done_ok "$Q1" || ! done_ok "$Q2"; do
  echo "  waiting Qwen W4A4 baseline JSONs  $(date -Is)"
  sleep 30
done
while pgrep -f 'scripts/run_qwen_gptq_enc_uni_a4_after_w4a8.sh' >/dev/null; do
  echo "  waiting Qwen W4A4 baseline script exit  $(date -Is)"
  sleep 10
done
sleep 20
echo "===== Qwen W4A4 baseline complete  $(date -Is) ====="
$PY -c "
import json
for p in [
  'results/ablation_w4a4/qwen_gptq_enc_wuni_g64_auni_a0_test-clean.json',
  'results/ablation_w4a4/qwen_gptq_enc_wuni_g64_auni_a0_test-other.json',
]:
    d=json.load(open(p))
    print(f\"  {p}: WER={d.get('wer')} RTF={d.get('rtf')}\")
"

echo "===== efficiency four-metric table  $(date -Is) ====="
$PY src/bench_eff_four.py
echo "ALL JOBS FINISHED $(date -Is)"
