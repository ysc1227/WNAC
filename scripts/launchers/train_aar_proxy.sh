#!/usr/bin/env bash
set -euo pipefail

# Short proxy AAR comparison runner.
# Usage:
#   MODE=blockwise bash train_aar_proxy.sh
#   MODE=reordered bash train_aar_proxy.sh
#   MODE=original bash train_aar_proxy.sh
#   MODE=upscale bash train_aar_proxy.sh
#
# Optional overrides:
#   GPUS=1,2 NPROC=gpu ITERS=50000 bash train_aar_proxy.sh
#
# Training hyperparameters such as seed/use_ema/ema_decay/vae_path live in YAML.

MODE="${MODE:-blockwise}"
GPUS="${GPUS:-1,2}"
NPROC="${NPROC:-gpu}"
ITERS="${ITERS:-30000}"

case "${MODE}" in
  blockwise)
    CONFIG="conf/aar/wnac_1_proxy_blockwise.yml"
    SAVE_PATH="runs/proxy/aar/wnac_1_512_blockwise_scalewise_30k"
    ;;
  reordered|reorder)
    CONFIG="conf/aar/wnac_1_proxy_reordered.yml"
    SAVE_PATH="runs/proxy/aar/wnac_1_512_reordered_sequential_30k"
    ;;
  original|sequential|seq)
    CONFIG="conf/aar/wnac_1_proxy_sequential.yml"
    SAVE_PATH="runs/proxy/aar/wnac_1_512_original_sequential_30k"
    ;;
  upscale|upscale_rvq|rvq_upscale)
    CONFIG="conf/aar/wnac_1_proxy_sequential.yml"
    SAVE_PATH="runs/proxy/aar/wnac_1_512_upscale_rvq_30k"
    ;;
  *)
    echo "Unknown MODE='${MODE}'. Use MODE=original, MODE=reordered, MODE=blockwise, or MODE=upscale." >&2
    exit 1
    ;;
esac

# Keep save path descriptive if ITERS is overridden.
SAVE_PATH="${SAVE_PATH/30k/${ITERS}}"

echo "[AAR proxy] mode=${MODE}"
echo "[AAR proxy] config=${CONFIG}"
echo "[AAR proxy] save_path=${SAVE_PATH}"
echo "[AAR proxy] gpus=${GPUS}, nproc=${NPROC}, iters=${ITERS}"

TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --nproc_per_node "${NPROC}" \
  -m scripts.training.aar \
  --args.load "${CONFIG}" \
  --num_iters "${ITERS}" \
  --save_path "${SAVE_PATH}"