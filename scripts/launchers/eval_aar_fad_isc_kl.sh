#!/usr/bin/env bash
set -euo pipefail

# AAR paper-style sample generation + FAD / Inception Score / KL evaluation.
# Override variables from the shell, e.g.:
#   CKPT_FOLDER=runs/proxy/aar/wnac_1_512_blockwise_scalewise_30000_seed0/latest LIMIT=256 bash eval_aar_fad_isc_kl.sh

CKPT_FOLDER="${CKPT_FOLDER:-runs/proxy/aar/wnac_1_512_blockwise_scalewise_30000_seed0/latest}"
MANIFEST="${MANIFEST:-samples/audioset_val_manifest.csv}"
OUT="${OUT:-results/aar_paper_eval/proxy_blockwise_latest}"
LIMIT="${LIMIT:-256}"
SEED="${SEED:-0}"
CFG="${CFG:-2.0}"
TOP_K="${TOP_K:-200}"
TOP_P="${TOP_P:-0.95}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SPLITS="${SPLITS:-10}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
DEVICE="${DEVICE:-cuda}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"

cmd=(python -m scripts.evaluation.aar_audio
  --ckpt_folder "$CKPT_FOLDER"
  --manifest "$MANIFEST"
  --out "$OUT"
  --limit "$LIMIT"
  --seed "$SEED"
  --cfg "$CFG"
  --top_k "$TOP_K"
  --top_p "$TOP_P"
  --batch_size "$BATCH_SIZE"
  --splits "$SPLITS"
  --device "$DEVICE")

if [[ -n "$MAX_SAMPLES" ]]; then
  cmd+=(--max_samples "$MAX_SAMPLES")
fi
if [[ "$SKIP_GENERATE" == "1" ]]; then
  cmd+=(--skip_generate)
fi

cmd+=("$@")

"${cmd[@]}"