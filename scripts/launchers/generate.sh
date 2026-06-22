python -m emac.utils.generate \
    --input_dir samples/val \
    --ckpt_dir ./runs/downstream/aar/snac_10_1_4096/200k/aar/weights.pth \
    --vae_dir ./runs/snac_4/best/emac/weights.pth \
    --output_dir ./results/generated/snac_10_1_4096/val \
    --n 1

# python -m emac.utils.generate \
#     --input_dir /home/seungchan/wnac/samples/environment/input \
#     --ckpt_dir ./runs/downstream/aar/snac_10_1_4096/200k/aar/weights.pth \
#     --vae_dir ./runs/snac_4/best/emac/weights.pth \
#     --output_dir ./results/generated/snac_10_1_4096/environment \
#     --n 1