#!/usr/bin/env bash

# Shared model presets for codec encode/decode/eval launchers.
# Source this file, then call `resolve_codec_model "${MODEL:-wnac_8kbps}"`.

prepare_codec_runtime_cache_dirs() {
  local user_tag="${USER:-$(id -u)}"

  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig-${user_tag}}"
  XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache-${user_tag}}"
  NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba-cache-${user_tag}}"

  mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${NUMBA_CACHE_DIR}"
  export MPLCONFIGDIR XDG_CACHE_HOME NUMBA_CACHE_DIR
}

resolve_codec_model() {
  prepare_codec_runtime_cache_dirs

  local model="${1:-wnac_8kbps}"

  case "${model}" in
    dac_official|dac_8kbps_official)
      MODEL="dac_official"
      MODEL_NAME="dac_official"
      WEIGHTS_PATH="${DAC_OFFICIAL_WEIGHTS:-/home/seungchan/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth}"
      ;;
    dac_1s)
      MODEL="dac_1s"
      MODEL_NAME="dac_1s"
      WEIGHTS_PATH="runs/8.00kbps/dac_9_1s_24b/best/emac/weights.pth"
      ;;
    wnac|wnac_8kbps|wavescale|wavescale_best|wavescale_13_1024)
      MODEL="wnac_8kbps"
      MODEL_NAME="wnac_8kbps"
      WEIGHTS_PATH="runs/8.00kbps/wavescale_13_1024/best/emac/weights.pth"
      ;;
    sat_official|sat_aar_official)
      MODEL="sat_official"
      MODEL_NAME="sat_official"
      EXTERNAL_CODEC="sat_official"
      SAT_OFFICIAL_AAR_REPO="${SAT_OFFICIAL_AAR_REPO:-/tmp/qiuk2_AAR}"
      WEIGHTS_PATH="${SAT_OFFICIAL_WEIGHTS:-/home/seungchan/.cache/huggingface/hub/models--qiuk6--AAR/snapshots/cd8757dc8ac2da363ed86d9a82c9d4479577ab98/SAT_bs_1536_d1024_lat64.pth}"
      MODEL_DEFAULT_WIN_DURATION="1"
      ;;
    snac_official|snac_2.6kbps_official|snac_44khz_official)
      MODEL="snac_official"
      MODEL_NAME="snac_official"
      EXTERNAL_CODEC="snac_official"
      WEIGHTS_PATH="${SNAC_OFFICIAL_WEIGHTS:-hubertsiuzdak/snac_44khz}"
      MODEL_DEFAULT_WIN_DURATION="10"
      ;;
    sat_codec|sat_reproduced_emac|sat_native)
      MODEL="sat_codec"
      MODEL_NAME="sat_codec"
      WEIGHTS_PATH="runs/4.60kbps/sat_codec/best/satcodec/weights.pth"
      MODEL_DEFAULT_WIN_DURATION="1"
      ;;
    wavescale_sat_budget)
      MODEL="wavescale_sat_budget"
      MODEL_NAME="wavescale_sat_budget"
      WEIGHTS_PATH="runs/4.60kbps/wavescale_11_1024_sat_token_budget/best/emac/weights.pth"
      MODEL_DEFAULT_WIN_DURATION="1"
      ;;
    snac_reproduced|snac_4)
      MODEL="snac_reproduced"
      MODEL_NAME="snac_reproduced"
      WEIGHTS_PATH="runs/2.60kbps/snac_codec/best/snaccodec/weights.pth"
      ;;
    wavescale_snac_budget)
      MODEL="snac_reproduced"
      MODEL_NAME="snac_reproduced"
      WEIGHTS_PATH="runs/2.60kbps/wavescale_5_512_snac44/best/emac/weights.pth"
      ;;  
    *)
      echo "Unknown MODEL='${model}'. Use one of: dac, dac_1s, dac_official, snac_official, snac_reproduced, snac_codec, sat_codec, sat_official, wnac_8kbps, wnac_pivot005, upscale_pivot005, wnac_5.2kbps, wnac_2.5kbps" >&2
      return 2
      ;;
  esac

  export MODEL MODEL_NAME WEIGHTS_PATH MODEL_DEFAULT_WIN_DURATION EXTERNAL_CODEC SAT_OFFICIAL_AAR_REPO
}
