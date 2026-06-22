import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor
from audiotools import AudioSignal
from audiotools.core import util

from emac.model.aar import AAR
from emac.model.emac import EMAC


# -----------------------------
# Config
# -----------------------------
@dataclass
class GenConfig:
    seed: int = 1234
    sample_rate: int = 44100
    seconds: float = 1.0
    ckpt_dir: str = "./"
    vae_dir: str = "./"
    input_dir: str = "./"

    # generation
    num_samples: int = 1  # per input
    output_dir: str = "./gen_out"
    audio_path: str = "./"
    chunk_prev_condition_weight: float = 0.5
    chunk_position_weight: float = 0.1


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    # torch seed도 같이 잡는 게 안전 (autoregressive 샘플링이면 필수)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def generate_one(audio: AudioSignal, aar: AAR, cond_processor, cond_model, sr: int, cfg: GenConfig) -> AudioSignal:
    cond = cond_processor(
        audio=audio.resample(48000).audio_data.squeeze(0).cpu().numpy(),
        return_tensors="pt",
        sampling_rate=48000,
    )
    conditions = cond_model.get_audio_features(
        input_features=cond["input_features"].to("cuda"),
        is_longer=cond["is_longer"].to("cuda"),
    )

    target_samples = int(round(cfg.seconds * sr)) if cfg.seconds is not None else None
    out = aar.autoregressive_infer_cfg_chunked(
        B=len(conditions),
        label_B=conditions,
        target_samples=target_samples,
        cfg=2.0,
        top_k=200,
        top_p=0.95,
        prev_condition_weight=cfg.chunk_prev_condition_weight,
        position_weight=cfg.chunk_position_weight,
    )
    return AudioSignal(out.cpu(), sr)


def run(cfg: GenConfig):
    set_seed(cfg.seed)

    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    vae = EMAC.load(location=cfg.vae_dir).to("cuda").eval()
    aar = AAR.load(location=cfg.ckpt_dir, vae_local=vae).to("cuda").eval()

    cond_processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
    cond_model = ClapModel.from_pretrained("laion/larger_clap_general").to("cuda").eval()

    input_root = Path(cfg.input_dir)
    audio_files = util.find_audio(input_root)
    audio_files = sorted(audio_files)

    for i in tqdm(range(len(audio_files)), desc="Generating audio files..."):
        signal = AudioSignal(audio_files[i], sample_rate=cfg.sample_rate)

        # ✅ sample 폴더/파일명은 입력 파일명과 무관하게 i 기준으로 고정
        sample_name = f"sample_{i}"
        output_dir = out_root / sample_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for j in range(cfg.num_samples):
            # ✅ input별 / sample별 seed 분리
            sample_seed = cfg.seed + i * cfg.num_samples + j
            set_seed(sample_seed)

            wav = generate_one(signal, aar, cond_processor, cond_model, cfg.sample_rate, cfg)

            output_name = f"{sample_name}__s{j:03d}.wav"
            output_path = output_dir / output_name
            wav.write(output_path)

    print(f"Saved {len(audio_files)} inputs x {cfg.num_samples} samples to: {out_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--vae_dir", type=str, required=True)
    ap.add_argument("--input_dir", type=str, required=True)

    ap.add_argument("--output_dir", type=str, default="./gen_out")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=10)  # 기본 10개 생성
    ap.add_argument("--chunk_prev_condition_weight", type=float, default=0.5)
    ap.add_argument("--chunk_position_weight", type=float, default=0.1)

    args = ap.parse_args()

    cfg = GenConfig(
        seed=args.seed,
        sample_rate=args.sr,
        seconds=args.seconds,
        ckpt_dir=args.ckpt_dir,
        vae_dir=args.vae_dir,
        num_samples=args.n,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_prev_condition_weight=args.chunk_prev_condition_weight,
        chunk_position_weight=args.chunk_position_weight,
    )

    run(cfg)


if __name__ == "__main__":
    main()
