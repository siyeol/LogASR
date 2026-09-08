#!/bin/bash
# Quick W4A4 recipe sweep on a 96-utt slice.
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /venv/main/bin/activate
cd /workspace/SpeechPTQ/src

run() {
  echo "========== $* =========="
  python run_eval.py --mode w4a4 --split test-clean --batch-size 16 --limit 96 "$@"
}

# 1) log-A4 only on encoder fc2 (does the codebook work where ASR breaks?)
run --w-mode none --a-mode log_dyn --scope enc_fc2
# 2) W4 log + A4 log on encoder fc2
run --w-mode log --a-mode log_dyn --scope enc_fc2
# 3) encoder FFN
run --w-mode log --a-mode log_dyn --scope enc_ffn
# 4) full encoder
run --w-mode log --a-mode log_dyn --scope encoder
# 5) hybrid codebook: log on enc fc2, uniform A4 elsewhere, skip lm_head
run --w-mode log --a-mode hybrid --scope nolm
# 6) all blocks except lm_head, dynamic log
run --w-mode log --a-mode log_dyn --scope nolm
