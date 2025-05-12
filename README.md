# WNAC: Wavescale Neural Audio Codec

<div align="center">

</div>
<p align="center" style="font-size: larger;">
  <a href="">Wavescale Neural Audio Codec: Bidirectional Multiscale Residual Quantization for High-Fidelity Audio Compression</a>
</p>

<br>

# Installation

- Install all packages via ```pip3 install -r requirements.txt```.


# Dataset

```
datasets
├── audioset
├── common_voice
├── daps
├── datasets_fullband
├── jamendo
├── musdb
└── vctk

```

---

## Training

```
torchrun --standalone --master_addr=0.0.0.0 --nproc_per_node=3 -m scripts.train --args.load conf/wavescale_16.yml --save_path runs/wavescale_16
```

## Inferencing

Sampling test dataset:
```
python -m scripts.save_test_set --sample_rate 44100 --output samples/general --args.load conf/base.yml
```

Encoding test:
```
python -m wnac encode samples/general \
    --output results/encode/wavescale_16/general \
    --weights_path runs/snac/best/wavescale_16/weights.pth \
    --win_duration 10 \
    --plot_path plot/wavescale_16/general \
    --sample_rate 44100
```

Decoding test:
```
python -m wnac decode results/encode/wavescale_16/general \
    --output results/decode/wavescale_16/general \
    --weights_path runs/wavescale_16/best/wnac/weights.pth
```

## Evaluating

```
python -m scripts.eval \
    --input samples/general \
    --output results/decode/wavescale_16/general \
    --n_proc 1 \
```

## Acknowledgements

Module definitions are adapted from the [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec).
