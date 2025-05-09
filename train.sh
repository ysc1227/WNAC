torchrun --standalone --master_addr=0.0.0.0 --nproc_per_node=3 -m scripts.train --args.load conf/ablations/waveloss/15_6.04_0.1.yml --save_path runs/waveloss/15_6.04_0.1
