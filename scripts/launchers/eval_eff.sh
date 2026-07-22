#!/usr/bin/env bash
set -euo pipefail

# Compute codebook entropy efficiency for one selected codec checkpoint.
# Examples:
#   MODEL=dac bash scripts/launchers/eval_eff.sh
#   MODEL=wnac_8kbps bash scripts/launchers/eval_eff.sh
#   MODEL=wnac_5.2kbps bash scripts/launchers/eval_eff.sh
#   MODEL=wnac_2.5kbps bash scripts/launchers/eval_eff.sh
#   MODEL=wavescale N_SAMPLES=3000 CROP_MODE=salient bash scripts/launchers/eval_eff.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/codec_model_presets.sh"

if [[ -n "${WEIGHTS_PATH:-}" ]]; then
  prepare_codec_runtime_cache_dirs
  MODEL="${MODEL:-custom}"
  MODEL_NAME="${MODEL_NAME:-${MODEL}}"
else
  resolve_codec_model "${MODEL:-wnac_8kbps}"
fi

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
INPUT="${INPUT:-eval_set/general}"
OUT_ROOT="${OUT_ROOT:-results/codec_compare_general}"
N_SAMPLES="${N_SAMPLES:-3000}"
DURATION="${DURATION:-1.0}"
CROP_MODE="${CROP_MODE:-salient}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-1024}"
SCALE="${SCALE:-}"
IS_WAVE="${IS_WAVE:-}"
JSON_PATH="${JSON_PATH:-${OUT_ROOT}/${MODEL_NAME}/codebook_entropy_n${N_SAMPLES}.json}"

export CUDA_VISIBLE_DEVICES

echo "[eval_eff] model=${MODEL} name=${MODEL_NAME}"
echo "[eval_eff] input=${INPUT} weights=${WEIGHTS_PATH}"
echo "[eval_eff] device=${DEVICE} cuda_visible=${CUDA_VISIBLE_DEVICES}"
echo "[eval_eff] n_samples=${N_SAMPLES} duration=${DURATION} crop_mode=${CROP_MODE}"
echo "[eval_eff] json=${JSON_PATH}"

if [[ -n "${EXTERNAL_CODEC:-}" ]]; then
  if [[ "${EXTERNAL_CODEC}" == "snac_official" ]]; then
    if ! python -c "import snac" >/dev/null 2>&1; then
      echo "[eval_eff] ERROR: missing Python package 'snac' for MODEL=snac_official." >&2
      echo "[eval_eff] Install it in this environment, then rerun:" >&2
      echo "[eval_eff]   python -m pip install snac" >&2
      exit 1
    fi
  fi

  cmd=(python -m scripts.evaluation.external_codebook_entropy
    --codec "${EXTERNAL_CODEC}"
    --folder "${INPUT}"
    --n-samples "${N_SAMPLES}"
    --duration "${DURATION}"
    --crop-mode "${CROP_MODE}"
    --batch-size "${BATCH_SIZE}"
    --device "${DEVICE}"
    --save-json "${JSON_PATH}")

  if [[ "${EXTERNAL_CODEC}" == "sat_official" ]]; then
    cmd+=(--aar-repo "${SAT_OFFICIAL_AAR_REPO}" --sat-checkpoint "${WEIGHTS_PATH}")
  fi
else
  cmd=(python -m scripts.evaluation.codebook_entropy
    --folder "${INPUT}"
    --model_path "${WEIGHTS_PATH}"
    --n_samples "${N_SAMPLES}"
    --device "${DEVICE}"
    --codebook_size "${CODEBOOK_SIZE}"
    --duration "${DURATION}"
    --crop_mode "${CROP_MODE}"
    --save_json "${JSON_PATH}")

  if [[ -n "${SCALE}" ]]; then
    cmd+=(--scale "${SCALE}")
  fi
  if [[ -n "${IS_WAVE}" ]]; then
    cmd+=(--is_wave "${IS_WAVE}")
  fi
fi

"${cmd[@]}"

echo "[eval_eff] done"
