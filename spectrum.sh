python -m scripts.spectrum samples/music/input \
    --weights_path runs/res_depth/15_6.04_u/best/wnac/weights.pth,runs/res_depth/15_6.04_d/best/wnac/weights.pth,runs/res_depth/15_6.04_w/best/wnac/weights.pth \
    --win_duration 10 \
    --plot_path plot/spec/total \
    --model_labels upscale,downscale,wavescale \
    --sample_rate 44100