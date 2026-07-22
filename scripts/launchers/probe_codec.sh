#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-conf/8.00kbps/wavescale_13_1024_fractional_upsample.yml}"
NAME="${NAME:-$(basename "${CONFIG}" .yml)}"
SAVE_PATH="${SAVE_PATH:-runs/probes/8.00kbps/${NAME}}"
SEED="${SEED:-0}"
GPU="${GPU:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_ITERS="${NUM_ITERS:-5000}"
VALID_FREQ="${VALID_FREQ:-500}"
SAMPLE_FREQ="${SAMPLE_FREQ:-5000}"
BATCH_SIZE="${BATCH_SIZE:-12}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RESUME="${RESUME:-0}"

export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU}}"

cmd=(
  torchrun
  --nproc_per_node "${NPROC_PER_NODE}"
  -m scripts.training.emac
  --args.load "${CONFIG}"
  --save_path "${SAVE_PATH}"
  --seed "${SEED}"
  --num_iters "${NUM_ITERS}"
  --valid_freq "${VALID_FREQ}"
  --sample_freq "${SAMPLE_FREQ}"
  --save_iters "[]"
  --batch_size "${BATCH_SIZE}"
  --val_batch_size "${VAL_BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --skip_initial_eval
  --upsampler_stage_metric_freq "${VALID_FREQ}"
)

if [[ "${RESUME}" != "0" ]]; then
  cmd+=(--resume)
fi

"${cmd[@]}"
