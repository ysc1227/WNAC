#!/usr/bin/env bash
set -euo pipefail

# Compute VGGish FAD for one selected codec reconstruction output.
# Examples:
#   MODEL=dac bash scripts/launchers/eval_fad.sh
#   MODEL=wnac_8kbps bash scripts/launchers/eval_fad.sh
#   MODEL=wnac_5.2kbps bash scripts/launchers/eval_fad.sh
#   MODEL=wnac_2.5kbps bash scripts/launchers/eval_fad.sh
#   MODEL=wavescale MAX_PAIRS=3000 bash scripts/launchers/eval_fad.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/codec_model_presets.sh"
resolve_codec_model "${MODEL:-wnac_8kbps}"

INPUT="${INPUT:-eval_set/general}"
OUT_ROOT="${OUT_ROOT:-results/codec_compare_general}"
RECONS="${RECONS:-${OUT_ROOT}/${MODEL_NAME}/recons}"
DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_PAIRS="${MAX_PAIRS:-}"
FAD_BACKEND="${FAD_BACKEND:-torcheval_vggish_no_relu}"
FAD_OUTPUT_DIR="${FAD_OUTPUT_DIR:-results/fad_torcheval_vggish_no_relu}"
VGGISH_WEIGHTS="${VGGISH_WEIGHTS:-}"

if [[ "${FAD_BACKEND}" == "torchvggish" ]]; then
  JSON_PATH="${JSON_PATH:-${RECONS}/fad.json}"
  CSV_PATH="${CSV_PATH:-${RECONS}/fad_files.csv}"
else
  JSON_PATH="${JSON_PATH:-${RECONS}/fad_torcheval_vggish_no_relu.json}"
  CSV_PATH="${CSV_PATH:-${FAD_OUTPUT_DIR}/summary.csv}"
fi

export CUDA_VISIBLE_DEVICES

echo "[eval_fad] model=${MODEL} name=${MODEL_NAME}"
echo "[eval_fad] input=${INPUT} recons=${RECONS}"
echo "[eval_fad] backend=${FAD_BACKEND}"
echo "[eval_fad] device=${DEVICE} cuda_visible=${CUDA_VISIBLE_DEVICES} batch_size=${BATCH_SIZE} max_pairs=${MAX_PAIRS:-all}"
echo "[eval_fad] json=${JSON_PATH} csv=${CSV_PATH}"
if [[ -n "${VGGISH_WEIGHTS}" ]]; then
  echo "[eval_fad] vggish_weights=${VGGISH_WEIGHTS}"
fi

case "${FAD_BACKEND}" in
  torcheval|torcheval_vggish|torcheval_vggish_no_relu)
    if ! python -c "from torchaudio.prototype.pipelines._vggish._vggish_impl import VGGish" >/dev/null 2>&1; then
      echo "[eval_fad] ERROR: this environment does not expose TorchAudio VGGish." >&2
      echo "[eval_fad] Install/update torchaudio, then rerun." >&2
      exit 1
    fi

    cmd=(python -m scripts.evaluation.fad_torcheval_vggish
      --input "${INPUT}"
      --model "${MODEL_NAME}=${RECONS}"
      --output-dir "${FAD_OUTPUT_DIR}"
      --batch-size "${BATCH_SIZE}"
      --device "${DEVICE}")

    if [[ -n "${VGGISH_WEIGHTS}" ]]; then
      cmd+=(--weights-path "${VGGISH_WEIGHTS}")
    fi

    if [[ -n "${MAX_PAIRS}" ]]; then
      cmd+=(--max-pairs "${MAX_PAIRS}")
    fi

    "${cmd[@]}"

    default_json="${RECONS}/fad_torcheval_vggish_no_relu.json"
    if [[ "${JSON_PATH}" != "${default_json}" ]]; then
      mkdir -p "$(dirname "${JSON_PATH}")"
      cp "${default_json}" "${JSON_PATH}"
    fi
    ;;

  torchvggish|legacy)
    if ! python -c "import torchvggish" >/dev/null 2>&1; then
      echo "[eval_fad] ERROR: missing Python package 'torchvggish' for FAD_BACKEND=torchvggish." >&2
      echo "[eval_fad] Install it in this environment, then rerun:" >&2
      echo "[eval_fad]   python -m pip install torchvggish==0.2" >&2
      exit 1
    fi

    cmd=(python -m scripts.evaluation.fad
      --input "${INPUT}"
      --output "${RECONS}"
      --device "${DEVICE}"
      --batch_size "${BATCH_SIZE}"
      --json_path "${JSON_PATH}"
      --csv_path "${CSV_PATH}")

    if [[ -n "${MAX_PAIRS}" ]]; then
      cmd+=(--max_pairs "${MAX_PAIRS}")
    fi

    "${cmd[@]}"
    ;;

  *)
    echo "[eval_fad] ERROR: unknown FAD_BACKEND='${FAD_BACKEND}'." >&2
    echo "[eval_fad] Use torcheval_vggish_no_relu or torchvggish." >&2
    exit 2
    ;;
esac

echo "[eval_fad] done"
