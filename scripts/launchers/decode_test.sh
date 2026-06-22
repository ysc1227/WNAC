#!/usr/bin/env bash
set -euo pipefail

# Decode encoded files with one selected codec checkpoint.
# Examples:
#   MODEL=dac bash scripts/launchers/decode_test.sh
#   MODEL=wnac_8kbps bash scripts/launchers/decode_test.sh
#   MODEL=wnac_5.2kbps bash scripts/launchers/decode_test.sh
#   MODEL=wnac_2.5kbps bash scripts/launchers/decode_test.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/codec_model_presets.sh"
resolve_codec_model "${MODEL:-wnac_8kbps}"

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUT_ROOT="${OUT_ROOT:-results/codec_compare_general}"
DEPTH="${DEPTH:-full}"
CODES="${CODES:-${OUT_ROOT}/${MODEL_NAME}/codes}"
OUTPUT="${OUTPUT:-${OUT_ROOT}/${MODEL_NAME}/recons}"

export CUDA_VISIBLE_DEVICES

echo "[decode] model=${MODEL} name=${MODEL_NAME}"
echo "[decode] codes=${CODES} output=${OUTPUT} weights=${WEIGHTS_PATH}"
echo "[decode] device=${DEVICE} cuda_visible=${CUDA_VISIBLE_DEVICES}"

python -m emac decode "${CODES}" \
  --output "${OUTPUT}" \
  --weights_path "${WEIGHTS_PATH}" \
  --device "${DEVICE}" \
  --depth "${DEPTH}"

echo "[decode] done"
