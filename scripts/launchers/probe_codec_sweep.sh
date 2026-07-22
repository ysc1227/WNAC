#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-1}"
NUM_ITERS="${NUM_ITERS:-5000}"
VALID_FREQ="${VALID_FREQ:-500}"
BATCH_SIZE="${BATCH_SIZE:-12}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"

configs=(
  "wavescale_13_1024:conf/8.00kbps/wavescale_13_1024.yml"
  "learned_upsample:conf/8.00kbps/wavescale_13_1024_learned_upsample.yml"
  "fractional_upsample:conf/8.00kbps/wavescale_13_1024_fractional_upsample.yml"
  "fractional_upsample_context:conf/8.00kbps/wavescale_13_1024_fractional_upsample_context.yml"
  "learned_scale_codec_fractional:conf/8.00kbps/wavescale_13_1024_learned_scale_codec_fractional.yml"
)

for entry in "${configs[@]}"; do
  name="${entry%%:*}"
  config="${entry#*:}"
  CONFIG="${config}" \
  NAME="${name}_seed${SEED}_${NUM_ITERS}it" \
  GPU="${GPU}" \
  NUM_ITERS="${NUM_ITERS}" \
  VALID_FREQ="${VALID_FREQ}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  SEED="${SEED}" \
    bash scripts/launchers/probe_codec.sh
done
