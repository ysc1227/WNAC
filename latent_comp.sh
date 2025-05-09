python -m scripts.latent_comp samples/environment/input \
    --weights_path runs/wavescale_16/best/wnac/weights.pth,runs/uwavescale_16/best/wnac/weights.pth \
    --win_duration 10 \
    --plot_path plot/loss_scale/environment \
    --scale '0.03,0.05,0.08,0.12,0.16,0.21,0.27,0.33,0.41,0.49,0.57,0.67,0.77,0.88,1|0.03,0.05,0.08,0.12,0.16,0.21,0.27,0.33,0.41,0.49,0.57,0.67,0.77,0.88,1' \
    --is_wave True,True \
    --legends 'wavescale(w/o waveloss)',wavescale \
    --sample_rate 44100