#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-conf/8.00kbps/wavescale_13_1024_order_upscale.yml}"
SAVE_PATH="${SAVE_PATH:-runs/8.00kbps/wavescale_13_1024_order_upscale}"
SEED="${SEED:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-gpu}"
RESUME="${RESUME:-1}"

export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"

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

"${cmd[@]}"
