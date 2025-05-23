CUDA_VISIBLE_DEVICES=0 python -m scripts.train_aar \
    --args.load conf/aar.yml --encoder_path runs/uwavescale_16/best/wnac/weights.pth --save_path runs/aar_test  --batch_size 1