python -m wnac encode samples/general/input \
    --output results/encode/wavescale_16/general \
    --weights_path runs/wavescale_16/best/wnac/weights.pth \
    --win_duration 10 \
    --plot_path plot/wavescale_16/general \
    --sample_rate 44100