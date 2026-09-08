#!/bin/bash
# Resume leftover ablation (skip-if-done), then robustness sim with fewer workers.
# Previous run "finished" because FAIL was ignored after thread exhaustion
# from simulate_robustness.py --workers 14.
set -u
source /venv/main/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/SpeechPTQ/src
cd /workspace/SpeechPTQ
mkdir -p logs
LOG=logs/resume_ablation_then_robust.log
exec >>"$LOG" 2>&1

echo "===== resume start $(date -Is) ====="

# Hung CPU pool was why GPU evals died with "can't start new thread".
pkill -f 'src/simulate_robustness.py' 2>/dev/null || true
sleep 3
if pgrep -f 'src/simulate_robustness.py' >/dev/null; then
  pkill -9 -f 'src/simulate_robustness.py' 2>/dev/null || true
  sleep 2
fi
echo "[resume] simulate_robustness killed $(date -Is)"

missing() {
  /venv/main/bin/python - "$@" <<'PY'
import json, sys
from pathlib import Path
root = Path("/workspace/SpeechPTQ")
need = []
for rel in sys.argv[1:]:
    p = root / rel
    ok = False
    if p.is_file():
        try:
            d = json.loads(p.read_text())
            ok = d.get("wer") is not None and int(d.get("n_utts") or 0) >= 2000
        except Exception:
            ok = False
    if not ok:
        need.append(rel)
sys.exit(0 if need else 1)
PY
}

SCOPE=(
  results/ablation_w4a4/whisper_logW64_a8_nolm_test-clean.json
  results/ablation_w4a4/whisper_logW64_a8_nolm_test-other.json
  results/ablation_w4a4/qwen_gptq_enc_logW64_a8_test-clean.json
  results/ablation_w4a4/qwen_gptq_enc_logW64_a8_test-other.json
  results/ablation_w4a4/whisper_logW64_logA_a0_nolm_test-clean.json
  results/ablation_w4a4/whisper_logW64_logA_a0_nolm_test-other.json
  results/ablation_w4a4/qwen_gptq_enc_logW64_logA_a0_test-clean.json
  results/ablation_w4a4/qwen_gptq_enc_logW64_logA_a0_test-other.json
  results/ablation_w4a4/whisper_unifW64_enc_logA4_resta8_test-clean.json
  results/ablation_w4a4/whisper_unifW64_enc_logA4_resta8_test-other.json
  results/ablation_w4a4/whisper_unifW64_encattn_logA4_resta8_test-clean.json
  results/ablation_w4a4/whisper_unifW64_encattn_logA4_resta8_test-other.json
  results/ablation_w4a4/qwen_gptq_encattn_logA4_resta8_test-clean.json
  results/ablation_w4a4/qwen_gptq_encattn_logA4_resta8_test-other.json
  results/ablation_w4a4/whisper_unifW64_encffn_logA4_resta8_test-clean.json
  results/ablation_w4a4/whisper_unifW64_encffn_logA4_resta8_test-other.json
  results/ablation_w4a4/qwen_gptq_encffn_logA4_resta8_test-clean.json
  results/ablation_w4a4/qwen_gptq_encffn_logA4_resta8_test-other.json
  results/ablation_w4a4/whisper_unifW64_dec_logA4_resta8_test-clean.json
  results/ablation_w4a4/whisper_unifW64_dec_logA4_resta8_test-other.json
  results/ablation_w4a4/qwen_gptq_dec_logA4_enc_a8_test-clean.json
  results/ablation_w4a4/qwen_gptq_dec_logA4_enc_a8_test-other.json
)
PLACE=(
  results/ablation_w4a4/whisper_logW64_unifA_a0_nolm_test-clean.json
  results/ablation_w4a4/whisper_logW64_unifA_a0_nolm_test-other.json
  results/ablation_w4a4/qwen_gptq_enc_logW64_unifA_a0_test-clean.json
  results/ablation_w4a4/qwen_gptq_enc_logW64_unifA_a0_test-other.json
  results/ablation_w4a4/whisper_unifW64_logA_a0_encoder_test-clean.json
  results/ablation_w4a4/whisper_unifW64_logA_a0_encoder_test-other.json
)

while missing "${SCOPE[@]}"; do
  echo "[resume] scope ablation pass $(date -Is)"
  bash scripts/run_scope_log_ablation.sh || true
done
echo "[resume] scope ablation complete $(date -Is)"

while missing "${PLACE[@]}"; do
  echo "[resume] log-placement pass $(date -Is)"
  bash scripts/run_log_placement_ablation.sh || true
done
echo "[resume] log-placement complete $(date -Is)"

echo "[resume] start robustness sim workers=4 $(date -Is)"
/venv/main/bin/python src/simulate_robustness.py --workers 4
echo "[resume] ALL DONE $(date -Is)"
