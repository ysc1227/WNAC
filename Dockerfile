ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

FROM ${BASE_IMAGE}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt /requirements.txt

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ffmpeg \
    git \
    libsndfile1 \
    libsox-fmt-all \
    pkg-config \
    sox \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /requirements.txt

RUN python -c "import torch, torchaudio, torchvision; \
assert torch.__version__.startswith('2.8.'), torch.__version__; \
assert torchaudio.__version__.startswith('2.8.'), torchaudio.__version__; \
assert torchvision.__version__.startswith('0.23.'), torchvision.__version__; \
import torchvggish, snac, visqol, torcheval; \
from torchaudio.prototype.pipelines._vggish._vggish_impl import VGGish"

COPY . /app
WORKDIR /app
