#!/usr/bin/env bash
set -euo pipefail

# Compute metrics for one selected codec reconstruction output.
# Examples:
#   MODEL=dac bash scripts/launchers/eval_test.sh
#   MODEL=wnac_8kbps bash scripts/launchers/eval_test.sh
#   MODEL=wnac_5.2kbps bash scripts/launchers/eval_test.sh
#   MODEL=wnac_2.5kbps bash scripts/launchers/eval_test.sh
#   MODEL=wavescale N_PROC=32 TORCH_THREADS=1 bash scripts/launchers/eval_test.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/codec_model_presets.sh"
resolve_codec_model "${MODEL:-wnac_8kbps}"

INPUT="${INPUT:-eval_set/general}"
OUT_ROOT="${OUT_ROOT:-results/codec_compare_general}"
RECONS="${RECONS:-${OUT_ROOT}/${MODEL_NAME}/recons}"
N_PROC="${N_PROC:-16}"
TORCH_THREADS="${TORCH_THREADS:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "[eval] model=${MODEL} name=${MODEL_NAME}"
echo "[eval] input=${INPUT} recons=${RECONS} n_proc=${N_PROC} torch_threads=${TORCH_THREADS}"

python -m scripts.evaluation.codec \
  --input "${INPUT}" \
  --output "${RECONS}" \
  --n_proc "${N_PROC}" \
  --torch_threads "${TORCH_THREADS}"

echo "[eval] done"
