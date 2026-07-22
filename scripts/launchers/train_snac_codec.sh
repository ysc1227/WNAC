#!/usr/bin/env bash
set -euo pipefail

# Train the SNAC-44k style codec through the native EMAC/audiotools stack.
#
# Usage:
#   GPUS=1,2 bash scripts/launchers/train_snac_codec.sh
#   CONFIG=conf/2.60kbps/snac_codec.yml SAVE_PATH=runs/2.60kbps/snac_codec bash scripts/launchers/train_snac_codec.sh

CONFIG="${CONFIG:-conf/2.60kbps/snac_codec.yml}"
SAVE_PATH="${SAVE_PATH:-runs/2.60kbps/snac_codec}"
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

echo "[SNACCodec] config=${CONFIG}"
echo "[SNACCodec] save_path=${SAVE_PATH}"
echo "[SNACCodec] gpus=${GPUS}, nproc=${NPROC_PER_NODE}, resume=${RESUME}"
printf '[SNACCodec] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}" CUDA_VISIBLE_DEVICES="${GPUS}" "${cmd[@]}"
