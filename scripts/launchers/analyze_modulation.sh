#!/usr/bin/env bash
set -euo pipefail

# Derive WaveScale pivot/scale schedules from temporal modulation statistics.
#
# Examples:
#   bash scripts/launchers/analyze_modulation.sh
#   CONFIG=conf/analyze_modulation_eval_general.yml bash scripts/launchers/analyze_modulation.sh
#   CONFIG=conf/analyze_modulation_train.yml N_EXAMPLES=10000 OUT=results/mod_train_10k bash scripts/launchers/analyze_modulation.sh

CONFIG="${CONFIG:-conf/analyze_modulation_train.yml}"
OUT="${OUT:-}"
N_EXAMPLES="${N_EXAMPLES:-}"
N_PROC="${N_PROC:-}"
INPUT="${INPUT:-}"
DATASET_KEYS="${DATASET_KEYS:-}"

echo "[analyze_modulation] config=${CONFIG}"

cmd=(python -m scripts.analysis.modulation --args.load "${CONFIG}")

if [[ -n "${OUT}" ]]; then
  cmd+=(--output "${OUT}")
fi

if [[ -n "${N_EXAMPLES}" ]]; then
  cmd+=(--n_examples "${N_EXAMPLES}")
fi

if [[ -n "${N_PROC}" ]]; then
  cmd+=(--n_proc "${N_PROC}")
fi

if [[ -n "${INPUT}" ]]; then
  cmd+=(--input "${INPUT}")
fi

if [[ -n "${DATASET_KEYS}" ]]; then
  IFS=',' read -r -a keys <<< "${DATASET_KEYS}"
  cmd+=(--dataset_keys "${keys[@]}")
fi

"${cmd[@]}"

echo "[analyze_modulation] done"
