#!/usr/bin/env bash
set -euo pipefail

# Compute ViSQOL MOS-LQO scores for one selected codec reconstruction output.
# Examples:
#   MODEL=dac bash scripts/launchers/eval_visqol.sh
#   MODEL=wnac_8kbps bash scripts/launchers/eval_visqol.sh
#   MODEL=wnac_5.2kbps bash scripts/launchers/eval_visqol.sh
#   MODEL=wnac_2.5kbps bash scripts/launchers/eval_visqol.sh
#   MODEL=wavescale MODE=speech bash scripts/launchers/eval_visqol.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/codec_model_presets.sh"
resolve_codec_model "${MODEL:-wnac_8kbps}"

INPUT="${INPUT:-eval_set/general}"
OUT_ROOT="${OUT_ROOT:-results/codec_compare_general}"
RECONS="${RECONS:-${OUT_ROOT}/${MODEL_NAME}/recons}"
MODE="${MODE:-audio}"
N_PROC="${N_PROC:-4}"
MAX_PAIRS="${MAX_PAIRS:-}"

echo "[eval_visqol] model=${MODEL} name=${MODEL_NAME}"
echo "[eval_visqol] input=${INPUT} recons=${RECONS} mode=${MODE} n_proc=${N_PROC} max_pairs=${MAX_PAIRS:-all}"

cmd=(python -m scripts.eval_visqol
  --input "${INPUT}"
  --output "${RECONS}"
  --mode "${MODE}"
  --n_proc "${N_PROC}")

if [[ -n "${MAX_PAIRS}" ]]; then
  cmd+=(--max_pairs "${MAX_PAIRS}")
fi

"${cmd[@]}"

echo "[eval_visqol] done"
