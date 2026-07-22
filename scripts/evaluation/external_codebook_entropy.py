"""Codebook entropy for official external codec checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio
from audiotools import AudioSignal
from audiotools.core import util
from tqdm import tqdm

from scripts.evaluation.codebook_entropy import (
    _entropy_bits,
    _flatten_to_1d_int,
    _usage_concentration,
)
from scripts.evaluation.reconstruct_external_codec import _load_sat, _load_snac


def _load_excerpt(
    path: Path,
    sample_rate: int,
    duration: float,
    device: torch.device,
    crop_mode: str,
) -> torch.Tensor:
    if crop_mode == "salient":
        sig = AudioSignal.salient_excerpt(path, loudness_cutoff=-20, duration=duration)
        sig = sig.resample(sample_rate).to(device)
        return sig.audio_data.mean(dim=1, keepdim=True)

    info = torchaudio.info(str(path))
    src_sr = int(info.sample_rate)
    n_src = int(round(duration * src_sr))
    if crop_mode == "center":
        offset = max(0, (int(info.num_frames) - n_src) // 2)
    elif crop_mode == "first":
        offset = 0
    else:
        raise ValueError(f"Unknown crop_mode: {crop_mode}")

    audio, sr = torchaudio.load(str(path), frame_offset=offset, num_frames=n_src)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    return audio.unsqueeze(0).to(device)


def _pad_batch(items: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(audio.shape[-1] for audio in items)
    padded = []
    for audio in items:
        if audio.shape[-1] < max_len:
            audio = torch.nn.functional.pad(audio, (0, max_len - audio.shape[-1]))
        padded.append(audio)
    return torch.cat(padded, dim=0)


def _codebook_size(model, codec: str) -> int:
    if codec == "sat_official":
        return int(model.quantizer.bins)
    return int(getattr(model.quantizer, "codebook_size", 4096))


def _labels(model, codec: str, codes: list[torch.Tensor]) -> list[str]:
    if codec == "sat_official":
        scales = [float(x) for x in model.quantizer.vq.scale]
        max_scale = max(scales)
        return [f"{scale / max_scale:g}_{idx}" for idx, scale in enumerate(scales)]

    lengths = [max(1, int(code.reshape(code.shape[0], -1).shape[-1])) for code in codes]
    max_len = max(lengths)
    return [f"{length / max_len:g}_{idx}" for idx, length in enumerate(lengths)]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", choices=["snac_official", "sat_official"], required=True)
    parser.add_argument("--folder", type=Path, default=Path("eval_set/general"))
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--crop-mode", choices=["center", "first", "salient"], default="center")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-json", type=Path, required=True)
    parser.add_argument("--aar-repo", type=Path, default=Path("/tmp/qiuk2_AAR"))
    parser.add_argument("--sat-checkpoint", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.codec == "sat_official":
        model, sample_rate = _load_sat(device, args.aar_repo, args.sat_checkpoint)
    else:
        model, sample_rate = _load_snac(device)
    model.eval()

    files = util.find_audio(args.folder)[: args.n_samples]
    if not files:
        raise RuntimeError(f"No audio files found under {args.folder}")

    codebook_size = _codebook_size(model, args.codec)
    capacity_bits = math.log2(codebook_size)
    aggregated_counts: dict[str, torch.Tensor | None] = defaultdict(lambda: None)
    total_tokens: dict[str, int] = defaultdict(int)
    labels = None
    total_seconds = 0.0

    batch_size = max(1, int(args.batch_size))
    for start in tqdm(range(0, len(files), batch_size), desc=f"Encoding {args.codec}"):
        chunk = [Path(path) for path in files[start : start + batch_size]]
        loaded = [
            _load_excerpt(path, sample_rate, args.duration, device, args.crop_mode)
            for path in chunk
        ]
        total_seconds += sum(float(audio.shape[-1]) / float(sample_rate) for audio in loaded)
        audio = _pad_batch(loaded)

        if args.codec == "sat_official":
            encoded = model.encoder(audio)
            _, codes, _ = model.quantizer(encoded)
        else:
            codes = model.encode(audio)

        if labels is None:
            labels = _labels(model, args.codec, codes)
            print("[INFO] codebook labels:", labels)

        for idx, code in enumerate(codes):
            code_1d = _flatten_to_1d_int(code.detach().cpu())
            counts = torch.bincount(code_1d, minlength=codebook_size)
            key = labels[idx]
            total_tokens[key] += int(code_1d.numel())
            if aggregated_counts[key] is None:
                aggregated_counts[key] = counts.to(torch.long)
            else:
                aggregated_counts[key] = aggregated_counts[key] + counts.to(torch.long)

    keys_sorted = list(labels or [])
    per_codebook = []
    sum_h = 0.0
    sum_capacity = 0.0
    token_weighted_h = 0.0
    token_weighted_capacity = 0.0
    entropy_kbps = 0.0
    nominal_kbps = 0.0

    for key in keys_sorted:
        counts = aggregated_counts[key]
        if counts is None:
            continue
        counts = counts.to(torch.float64)
        h_bits = _entropy_bits(counts)
        tokens = int(total_tokens[key])
        token_rate = tokens / total_seconds if total_seconds > 0 else 0.0
        eff = h_bits / capacity_bits if capacity_bits > 0 else 0.0
        perplexity = float(2**h_bits)
        used_codes = int((counts > 0).sum().item())

        per_codebook.append(
            {
                "key": key,
                "entropy_bits": h_bits,
                "capacity_bits": capacity_bits,
                "efficiency": eff,
                "perplexity": perplexity,
                "used_codes": used_codes,
                "codebook_size": codebook_size,
                "usage_ratio": used_codes / codebook_size,
                "num_tokens": tokens,
                "token_rate_hz": token_rate,
                **_usage_concentration(counts),
            }
        )
        sum_h += h_bits
        sum_capacity += capacity_bits
        token_weighted_h += tokens * h_bits
        token_weighted_capacity += tokens * capacity_bits
        entropy_kbps += token_rate * h_bits / 1000.0
        nominal_kbps += token_rate * capacity_bits / 1000.0

    overall_efficiency = sum_h / sum_capacity if sum_capacity > 0 else 0.0
    token_weighted_efficiency = (
        token_weighted_h / token_weighted_capacity if token_weighted_capacity > 0 else 0.0
    )

    payload = {
        "codec": args.codec,
        "folder": str(args.folder),
        "n_audio_files": len(files),
        "duration": args.duration,
        "crop_mode": args.crop_mode,
        "batch_size": batch_size,
        "sample_rate": sample_rate,
        "total_seconds": total_seconds,
        "codebook_size": codebook_size,
        "per_codebook": per_codebook,
        "overall_efficiency": overall_efficiency,
        "token_weighted_efficiency": token_weighted_efficiency,
        "sum_entropy_bits": sum_h,
        "sum_capacity_bits": sum_capacity,
        "token_weighted_sum_entropy_bits": token_weighted_h,
        "token_weighted_sum_capacity_bits": token_weighted_capacity,
        "entropy_kbps": entropy_kbps,
        "nominal_kbps": nominal_kbps,
    }

    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(json.dumps(payload, indent=2) + "\n")

    print("\n=== External codebook efficiency ===")
    for row in per_codebook:
        print(
            f"- Codebook {row['key']}: H={row['entropy_bits']:.4f} bits, "
            f"capacity={row['capacity_bits']:.4f} bits, "
            f"efficiency={row['efficiency'] * 100:.2f}%, "
            f"used={row['used_codes']}/{row['codebook_size']} "
            f"({row['usage_ratio'] * 100:.2f}%), "
            f"rate={row['token_rate_hz']:.2f} Hz"
        )
    print(f"> Overall efficiency = {overall_efficiency * 100:.2f}%")
    print(f"> Token-weighted efficiency = {token_weighted_efficiency * 100:.2f}%")
    print(f"> Entropy bitrate ~= {entropy_kbps:.3f} kbps vs nominal ~= {nominal_kbps:.3f} kbps")
    print(f"[INFO] Saved to {args.save_json}")


if __name__ == "__main__":
    main()
