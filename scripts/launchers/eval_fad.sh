#!/usr/bin/env bash
set -euo pipefail

# Compute VGGish FAD for one selected codec reconstruction output.
# Examples:
#   MODEL=dac bash scripts/launchers/eval_fad.sh
#   MODEL=wnac_8kbps bash scripts/launchers/eval_fad.sh
#   MODEL=wnac_5.2kbps bash scripts/launchers/eval_fad.sh
#   MODEL=wnac_2.5kbps bash scripts/launchers/eval_fad.sh
#   MODEL=wavescale MAX_PAIRS=3000 bash scripts/launchers/eval_fad.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/codec_model_presets.sh"
resolve_codec_model "${MODEL:-wnac_8kbps}"

INPUT="${INPUT:-eval_set/general}"
OUT_ROOT="${OUT_ROOT:-results/codec_compare_general}"
RECONS="${RECONS:-${OUT_ROOT}/${MODEL_NAME}/recons}"
DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_PAIRS="${MAX_PAIRS:-}"
JSON_PATH="${JSON_PATH:-${RECONS}/fad.json}"
CSV_PATH="${CSV_PATH:-${RECONS}/fad_files.csv}"

export CUDA_VISIBLE_DEVICES

echo "[eval_fad] model=${MODEL} name=${MODEL_NAME}"
echo "[eval_fad] input=${INPUT} recons=${RECONS}"
echo "[eval_fad] device=${DEVICE} cuda_visible=${CUDA_VISIBLE_DEVICES} batch_size=${BATCH_SIZE} max_pairs=${MAX_PAIRS:-all}"
echo "[eval_fad] json=${JSON_PATH} csv=${CSV_PATH}"

if ! python -c "import torchvggish" >/dev/null 2>&1; then
  echo "[eval_fad] ERROR: missing Python package 'torchvggish'." >&2
  echo "[eval_fad] Install it in this environment, then rerun:" >&2
  echo "[eval_fad]   python -m pip install torchvggish==0.2" >&2
  exit 1
fi

cmd=(python -m scripts.eval_fad
  --input "${INPUT}"
  --output "${RECONS}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --json_path "${JSON_PATH}"
  --csv_path "${CSV_PATH}")

if [[ -n "${MAX_PAIRS}" ]]; then
  cmd+=(--max_pairs "${MAX_PAIRS}")
fi

"${cmd[@]}"

echo "[eval_fad] done"
