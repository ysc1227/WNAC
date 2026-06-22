#!/usr/bin/env bash
set -euo pipefail

# Tiny pre-training probe for guided upsampling.
# Example:
#   MODEL=wavescale STEPS=80 N_SAMPLES=4 bash scripts/launchers/guided_upsample_probe.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/codec_model_presets.sh"
resolve_codec_model "${MODEL:-wavescale}"

INPUT="${INPUT:-eval_set/general}"
STEPS="${STEPS:-80}"
N_SAMPLES="${N_SAMPLES:-4}"
DURATION="${DURATION:-0.5}"
DEVICE="${DEVICE:-auto}"
LR="${LR:-0.001}"
SAVE_JSON="${SAVE_JSON:-results/guided_upsample_probe/${MODEL_NAME}.json}"

python -m scripts.analysis.guided_upsample_probe \
  --folder "${INPUT}" \
  --model_path "${WEIGHTS_PATH}" \
  --device "${DEVICE}" \
  --n_samples "${N_SAMPLES}" \
  --duration "${DURATION}" \
  --steps "${STEPS}" \
  --lr "${LR}" \
  --save_json "${SAVE_JSON}"
