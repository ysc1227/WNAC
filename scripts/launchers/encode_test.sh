#!/usr/bin/env bash
set -euo pipefail

# Encode eval audio with one selected codec checkpoint.
# Examples:
#   MODEL=dac bash scripts/launchers/encode_test.sh
#   MODEL=wnac_8kbps bash scripts/launchers/encode_test.sh
#   MODEL=wnac_5.2kbps bash scripts/launchers/encode_test.sh
#   MODEL=wnac_2.5kbps bash scripts/launchers/encode_test.sh
#   MODEL=wavescale INPUT=eval_set/music bash scripts/launchers/encode_test.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/codec_model_presets.sh"
resolve_codec_model "${MODEL:-wnac_8kbps}"

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
INPUT="${INPUT:-eval_set/general}"
OUT_ROOT="${OUT_ROOT:-results/codec_compare_general}"
SAMPLE_RATE="${SAMPLE_RATE:-44100}"
WIN_DURATION="${WIN_DURATION:-${MODEL_DEFAULT_WIN_DURATION:-10}}"
OUTPUT="${OUTPUT:-${OUT_ROOT}/${MODEL_NAME}/codes}"

export CUDA_VISIBLE_DEVICES

echo "[encode] model=${MODEL} name=${MODEL_NAME}"
echo "[encode] input=${INPUT} output=${OUTPUT} weights=${WEIGHTS_PATH}"
echo "[encode] device=${DEVICE} cuda_visible=${CUDA_VISIBLE_DEVICES}"

python -m emac encode "${INPUT}" \
  --output "${OUTPUT}" \
  --weights_path "${WEIGHTS_PATH}" \
  --device "${DEVICE}" \
  --win_duration "${WIN_DURATION}" \
  --sample_rate "${SAMPLE_RATE}"

echo "[encode] done"
