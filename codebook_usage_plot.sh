python -m scripts.codebook_usage samples/general/input \
    --weights_path runs/res_depth/15_6.04_u/200k/wnac/weights.pth,runs/res_depth/15_6.04_d/200k/wnac/weights.pth,runs/res_depth/15_6.04_w/200k/wnac/weights.pth \
    --win_duration 10 \
    --plot_path plot/usage/res_depth \
    --scale '0.03,0.05,0.08,0.12,0.16,0.21,0.27,0.33,0.41,0.49,0.57,0.67,0.77,0.88,1|1,0.88,0.77,0.67,0.57,0.49,0.41,0.33,0.27,0.21,0.16,0.12,0.08,0.05,0.03|0.03,0.05,0.11,0.21,0.35,0.53,0.755,1' \
    --is_wave False,False,True \
    --legends upscale,downscale,wavescale\
    --sample_rate 44100