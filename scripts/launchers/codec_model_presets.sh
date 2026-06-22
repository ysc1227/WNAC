#!/usr/bin/env bash

# Shared model presets for codec encode/decode/eval launchers.
# Source this file, then call `resolve_codec_model "${MODEL:-wnac_8kbps}"`.

resolve_codec_model() {
  local model="${1:-wnac_8kbps}"

  case "${model}" in
    dac|dac_seed0|dac_seed0_best)
      MODEL="dac"
      MODEL_NAME="dac"
      WEIGHTS_PATH="runs/8.00kbps/dac_9/0/best/wnac/weights.pth"
      ;;
    wnac|wnac_8kbps|wavescale|wavescale_best|wavescale_13_1024)
      MODEL="wnac_8kbps"
      MODEL_NAME="wnac_8kbps"
      WEIGHTS_PATH="runs/8.00kbps/wavescale_13_1024_nodrop_noattn/best/emac/weights.pth"
      ;;
    wnac_5.2kbps|wnac_5_2kbps|wnac_5kbps|5.2kbps|5_2kbps)
      MODEL="wnac_5.2kbps"
      MODEL_NAME="wnac_5.2kbps"
      WEIGHTS_PATH="runs/5.20kbps/15_1024/best/emac/weights.pth"
      ;;
    wnac_2.5kbps|wnac_2_5kbps|wnac_2.52kbps|wnac_2_52kbps|2.5kbps|2_5kbps|2.52kbps|2_52kbps)
      MODEL="wnac_2.5kbps"
      MODEL_NAME="wnac_2.5kbps"
      WEIGHTS_PATH="runs/2.52kbps/5_512/best/emac/weights.pth"
      ;;
    *)
      echo "Unknown MODEL='${model}'. Use one of: dac, wnac_8kbps, wnac_8kbps_no_phi, wnac_8kbps_phi5, wnac_5.2kbps, wnac_2.5kbps" >&2
      return 2
      ;;
  esac

  export MODEL MODEL_NAME WEIGHTS_PATH
}
