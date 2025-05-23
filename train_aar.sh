torchrun --standalone --master_addr=0.0.0.0 --nproc_per_node=3 -m scripts.train_aar \
    --args.load conf/aar.yml --encoder_path runs/uwavescale_16/best/wnac/weights.pth --save_path runs/aar