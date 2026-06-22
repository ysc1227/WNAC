python -m scripts.evaluation.codebook_entropy \
    --folder /home/seungchan/wnac/samples/general/input \
    --model_path runs/2.52kbps/5_4096/best/emac/weights.pth \
    --scale '0.10,0.16,1.0' \
    --is_wave True \
    --codebook_size 4096 \
    --n_samples 3000
    