#!/usr/bin/env bash
set -euo pipefail

# Train the original SAT codec through the native EMAC/audiotools training stack.
#
# Usage:
#   GPUS=1,2 bash scripts/launchers/train_sat_codec.sh
#   CONFIG=conf/4.60kbps/sat_codec.yml SAVE_PATH=runs/4.60kbps/sat_codec_original_quantizer bash scripts/launchers/train_sat_codec.sh
#   CONFIG=conf/4.60kbps/sat_codec_original_recipe.yml SAVE_PATH=runs/4.60kbps/sat_codec_original_recipe bash scripts/launchers/train_sat_codec.sh

CONFIG="${CONFIG:-conf/4.60kbps/sat_codec.yml}"
SAVE_PATH="${SAVE_PATH:-runs/4.60kbps/sat_codec}"
SEED="${SEED:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-gpu}"
RESUME="${RESUME:-1}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-1,2}}"

cmd=(
  torchrun
  --nproc_per_node "${NPROC_PER_NODE}"
  -m scripts.training.emac
  --args.load "${CONFIG}"
  --save_path "${SAVE_PATH}"
  --seed "${SEED}"
)

if [[ "${RESUME}" != "0" ]]; then
  cmd+=(--resume)
fi

echo "[SATCodec] config=${CONFIG}"
echo "[SATCodec] save_path=${SAVE_PATH}"
echo "[SATCodec] gpus=${GPUS}, nproc=${NPROC_PER_NODE}, resume=${RESUME}"
printf '[SATCodec] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}" CUDA_VISIBLE_DEVICES="${GPUS}" "${cmd[@]}"
