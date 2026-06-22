#!/usr/bin/env bash
set -euo pipefail

# Simple AAR training launcher.
#
# Usage:
#   bash scripts/launchers/train_aar.sh conf/aar/wnac_10_auto_blockwise.yml
#   bash scripts/launchers/train_aar.sh conf/aar/wnac_10_auto_blockwise.yml --gpus 1,2 --iters 50000
#   bash scripts/launchers/train_aar.sh conf/aar/wnac_10_auto_blockwise.yml --save-path runs/aar/my_run
#   bash scripts/launchers/train_aar.sh conf/aar/wnac_10_auto_blockwise.yml -- --batch_size 8 --valid_freq 500
#
# Training hyperparameters such as seed/use_ema/ema_decay/vae_path live in YAML.

usage() {
  cat <<'USAGE'
Usage: bash train_aar.sh CONFIG.yml [options] [-- extra train_aar.py args]

Arguments:
  CONFIG.yml                AAR YAML config passed to --args.load

Options:
  --resume                  Resume from SAVE_PATH/TAG and pass --resume to train_aar.py
  --tag TAG                 Checkpoint tag for resume. Default: latest
  --save-path PATH          Output run directory. Default: runs/aar/<CONFIG_STEM>
  --gpus IDS                CUDA_VISIBLE_DEVICES value. Default: 1,2
  --nproc VALUE             torchrun --nproc_per_node value. Default: gpu
  --iters N                 Override --num_iters
  --python                  Run with python instead of torchrun
  --dry-run                 Print command without executing
  -h, --help                Show this help

Environment overrides are also supported:
  GPUS=0 NPROC=1 ITERS=50000 bash train_aar.sh conf/aar/wnac_10_auto_blockwise.yml
USAGE
}

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

CONFIG="${CONFIG:-}"
RESUME="${RESUME:-0}"
TAG="${TAG:-latest}"
SAVE_PATH="${SAVE_PATH:-}"
GPUS="${GPUS:-1,2}"
NPROC="${NPROC:-gpu}"
ITERS="${ITERS:-}"
LAUNCHER="${LAUNCHER:-torchrun}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS=()

if [[ $# -gt 0 && "$1" != -* ]]; then
  CONFIG="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --tag) TAG="$2"; shift 2 ;;
    --save-path|--save_path) SAVE_PATH="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --nproc) NPROC="$2"; shift 2 ;;
    --iters|--num-iters|--num_iters) ITERS="$2"; shift 2 ;;
    --python) LAUNCHER=python; shift ;;
    --torchrun) LAUNCHER=torchrun; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "${CONFIG}" ]]; then
  echo "Missing CONFIG.yml" >&2
  usage >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing config file: ${CONFIG}" >&2
  exit 1
fi

CONFIG_STEM="$(basename "${CONFIG}")"
CONFIG_STEM="${CONFIG_STEM%.*}"
SAVE_PATH="${SAVE_PATH:-runs/aar/${CONFIG_STEM}}"

if [[ "${LAUNCHER}" == "python" ]]; then
  CMD=(python -m scripts.training.aar)
else
  CMD=(torchrun --nproc_per_node "${NPROC}" -m scripts.training.aar)
fi

CMD+=(--args.load "${CONFIG}")

if truthy "${RESUME}"; then
  if [[ ! -d "${SAVE_PATH}/${TAG}/aar" ]]; then
    echo "Missing checkpoint directory: ${SAVE_PATH}/${TAG}/aar" >&2
    echo "Use --save-path/--tag to select an existing checkpoint, or omit --resume." >&2
    exit 1
  fi
  CMD+=(--resume --tag "${TAG}")
fi

if [[ -n "${ITERS}" ]]; then
  CMD+=(--num_iters "${ITERS}")
fi

CMD+=(--save_path "${SAVE_PATH}")
CMD+=("${EXTRA_ARGS[@]}")

echo "[AAR] config=${CONFIG}"
echo "[AAR] save_path=${SAVE_PATH}"
echo "[AAR] tag=${TAG}"
echo "[AAR] gpus=${GPUS}, launcher=${LAUNCHER}, nproc=${NPROC}, iters=${ITERS:-config/default}"
printf '[AAR] command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if truthy "${DRY_RUN}"; then
  exit 0
fi

TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES="${GPUS}" "${CMD[@]}"
