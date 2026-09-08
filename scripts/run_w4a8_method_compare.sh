#!/bin/bash
# W4A8 method table. Does nothing unless EXECUTE=1.
#   bash scripts/run_w4a8_method_compare.sh          # plan only
#   EXECUTE=1 bash scripts/run_w4a8_method_compare.sh
set -u
source /venv/main/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/SpeechPTQ/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/SpeechPTQ
mkdir -p logs results/compare/w4a8
if [ "${EXECUTE:-0}" != "1" ]; then
  /venv/main/bin/python src/run_w4a8_method_compare.py
  exit 0
fi
exec > >(tee -a logs/w4a8_method_compare.log) 2>&1
/venv/main/bin/python src/run_w4a8_method_compare.py --run
