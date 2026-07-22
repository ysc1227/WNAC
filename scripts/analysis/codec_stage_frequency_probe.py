import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import torch
import torch.nn.functional as F
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emac.utils import load_model


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}


def parse_model_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return path.parent.parent.parent.name if path.name == "weights.pth" else path.stem, path
    name, path = spec.split("=", 1)
    return name, Path(path)


def resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    for candidate in (
        path / "best" / "emac" / "weights.pth",
        path / "latest" / "emac" / "weights.pth",
        path / "emac" / "weights.pth",
        path / "weights.pth",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve checkpoint from {path}")


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def find_audio_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in AUDIO_EXTENSIONS else []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in AUDIO_EXTENSIONS)


def sample_files(files: list[Path], n_files: int, seed: int) -> list[Path]:
    if n_files <= 0 or n_files >= len(files):
        return list(files)
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)
    return sorted(files[:n_files])


def chunks(items: list[Path], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def load_audio(path: Path, sample_rate: int, duration: float) -> torch.Tensor:
    audio, sr = torchaudio.load(str(path))
    audio = audio.mean(dim=0, keepdim=True)
    if int(sr) != int(sample_rate):
        audio = torchaudio.functional.resample(audio, int(sr), int(sample_rate))
    n_samples = max(1, int(round(float(duration) * sample_rate)))
    if audio.shape[-1] < n_samples:
        audio = F.pad(audio, (0, n_samples - audio.shape[-1]))
    elif audio.shape[-1] > n_samples:
        start = (audio.shape[-1] - n_samples) // 2
        audio = audio[..., start : start + n_samples]
    return audio


def load_batch(files: list[Path], sample_rate: int, duration: float, device: str) -> torch.Tensor:
    audio = [load_audio(path, sample_rate, duration) for path in files]
    return torch.stack(audio, dim=0).to(device)


def parse_band_edges(text: str, sample_rate: int) -> list[tuple[float, float]]:
    raw_edges = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(raw_edges) < 2:
        raise ValueError("--bands-hz must contain at least two comma-separated edges")
    nyquist = float(sample_rate) / 2.0
    edges = []
    for edge in raw_edges:
        edge = min(max(0.0, edge), nyquist)
        if not edges or edge > edges[-1]:
            edges.append(edge)
    if edges[-1] < nyquist:
        edges.append(nyquist)
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1) if edges[i + 1] > edges[i]]


def band_name(band: tuple[float, float]) -> str:
    lo, hi = band
    return f"{lo:g}-{hi:g}Hz"


def stft_band_energy(
    audio: torch.Tensor,
    sample_rate: int,
    bands: list[tuple[float, float]],
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    x = audio.squeeze(1).float()
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    power = spec.abs().pow(2)
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate)).to(device=x.device)
    values = []
    for idx, (lo, hi) in enumerate(bands):
        if idx == len(bands) - 1:
            mask = (freqs >= lo) & (freqs <= hi)
        else:
            mask = (freqs >= lo) & (freqs < hi)
        if not bool(mask.any()):
            values.append(power.new_zeros(power.shape[0]))
        else:
            values.append(power[:, mask, :].sum(dim=(1, 2)))
    return torch.stack(values, dim=1)


def band_share(energy: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return energy / energy.sum(dim=1, keepdim=True).clamp_min(eps)


def rms(x: torch.Tensor) -> torch.Tensor:
    return x.float().pow(2).mean(dim=(1, 2)).sqrt()


@torch.no_grad()
def stage_latents(model, audio: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], list[dict]]:
    z = model.encoder(model.preprocess(audio, model.sample_rate))
    quantizer = model.quantizer
    total_steps = z.shape[-1]
    residual = z
    z_q = torch.zeros_like(z)
    prefixes = [z_q]
    stages = []

    for stage, stage_quantizer in enumerate(quantizer.quantizers):
        scale = max(1, int(float(quantizer.scale_factors[stage]) * total_steps))
        guide = z_q if stage > 0 else None
        conv = None
        if getattr(quantizer, "quant_resi", None) is not None:
            conv = lambda h, stage_idx=stage: quantizer._apply_quant_resi(stage_idx, h)

        z_e = stage_quantizer.in_proj(residual)
        inter_z = stage_quantizer.downsample_latents(z_e, scale)
        z_q_small, indices = stage_quantizer.decode_latents(inter_z)
        z_q_up = stage_quantizer.upsample_code(
            z_q_small,
            total_steps,
            guide=guide,
            use_guide=scale != total_steps,
        )
        z_q_code = conv(z_q_up) if conv is not None else z_q_up
        contribution = stage_quantizer.out_proj(z_q_code)
        z_q = z_q + contribution
        residual = residual - contribution
        prefixes.append(z_q)
        stages.append(
            {
                "stage": int(stage),
                "scale_factor": float(quantizer.scale_factors[stage]),
                "scale_steps": int(scale),
                "contribution": contribution,
                "indices": indices,
            }
        )
    return z, prefixes, stages


@torch.no_grad()
def decode_latents(model, latents: list[torch.Tensor], max_batch: int) -> list[torch.Tensor]:
    if not latents:
        return []
    batch = latents[0].shape[0]
    packed = torch.cat(latents, dim=0)
    decoded = []
    max_batch = max(1, int(max_batch))
    for start in range(0, packed.shape[0], max_batch):
        decoded.append(model.decode(packed[start : start + max_batch]))
    packed_audio = torch.cat(decoded, dim=0)
    return [packed_audio[i * batch : (i + 1) * batch] for i in range(len(latents))]


def add_stat(accum: dict, key: str, value: torch.Tensor):
    value = value.detach().float().cpu()
    if value.ndim == 0:
        value = value.reshape(1)
    count = int(value.shape[0])
    summed = value.sum(dim=0)
    if key not in accum:
        accum[key] = {"sum": torch.zeros_like(summed), "count": 0}
    accum[key]["sum"] += summed
    accum[key]["count"] += count


def finalize_stat(accum: dict, key: str):
    item = accum[key]
    value = item["sum"] / max(1, item["count"])
    if value.ndim == 0:
        return float(value.item())
    if value.numel() == 1:
        return float(value.reshape(()).item())
    return [float(x) for x in value.tolist()]


@torch.no_grad()
def probe_model(model, files: list[Path], args, bands: list[tuple[float, float]]) -> dict:
    model.eval()
    band_labels = [band_name(band) for band in bands]
    n_stages = len(model.quantizer.quantizers)
    stage_accum = [dict() for _ in range(n_stages)]
    eps = 1e-8

    for batch_idx, batch_files in enumerate(chunks(files, args.batch_size)):
        print(
            f"[probe] batch {batch_idx + 1}/{(len(files) + args.batch_size - 1) // args.batch_size}",
            flush=True,
        )
        audio = load_batch(batch_files, model.sample_rate, args.duration, args.device)
        target_band = stft_band_energy(
            audio,
            model.sample_rate,
            bands,
            args.n_fft,
            args.hop_length,
        ).clamp_min(eps)
        target_total = target_band.sum(dim=1).clamp_min(eps)

        _, prefixes, stages = stage_latents(model, audio)
        full_latent = prefixes[-1]
        prefix_audio = decode_latents(model, prefixes, args.decode_batch_size)
        prefix_audio = [decoded[..., : audio.shape[-1]] for decoded in prefix_audio]
        full_audio = prefix_audio[-1]
        omitted_audio = decode_latents(
            model,
            [full_latent - stage["contribution"] for stage in stages],
            args.decode_batch_size,
        )
        omitted_audio = [decoded[..., : audio.shape[-1]] for decoded in omitted_audio]

        full_err_band = stft_band_energy(
            full_audio - audio,
            model.sample_rate,
            bands,
            args.n_fft,
            args.hop_length,
        )
        full_err_norm = full_err_band / target_band
        full_err_totalnorm_band = full_err_band / target_total[:, None]
        full_err_total_norm = full_err_band.sum(dim=1) / target_total

        for stage, stage_info in enumerate(stages):
            prev_audio = prefix_audio[stage]
            curr_audio = prefix_audio[stage + 1]
            marginal_audio = curr_audio - prev_audio

            prev_err_band = stft_band_energy(
                prev_audio - audio,
                model.sample_rate,
                bands,
                args.n_fft,
                args.hop_length,
            )
            curr_err_band = stft_band_energy(
                curr_audio - audio,
                model.sample_rate,
                bands,
                args.n_fft,
                args.hop_length,
            )
            omit_err_band = stft_band_energy(
                omitted_audio[stage] - audio,
                model.sample_rate,
                bands,
                args.n_fft,
                args.hop_length,
            )
            marginal_band = stft_band_energy(
                marginal_audio,
                model.sample_rate,
                bands,
                args.n_fft,
                args.hop_length,
            )

            prev_err_norm = prev_err_band / target_band
            curr_err_norm = curr_err_band / target_band
            omit_err_norm = omit_err_band / target_band
            prev_err_totalnorm_band = prev_err_band / target_total[:, None]
            curr_err_totalnorm_band = curr_err_band / target_total[:, None]
            omit_err_totalnorm_band = omit_err_band / target_total[:, None]
            prev_err_total_norm = prev_err_band.sum(dim=1) / target_total
            curr_err_total_norm = curr_err_band.sum(dim=1) / target_total
            omit_err_total_norm = omit_err_band.sum(dim=1) / target_total

            accum = stage_accum[stage]
            add_stat(accum, "latent_contribution_rms", rms(stage_info["contribution"]))
            add_stat(accum, "marginal_audio_rms", rms(marginal_audio))
            add_stat(accum, "marginal_band_energy", marginal_band)
            add_stat(accum, "marginal_band_share", band_share(marginal_band))
            add_stat(accum, "prefix_error_before_band_norm", prev_err_norm)
            add_stat(accum, "prefix_error_after_band_norm", curr_err_norm)
            add_stat(accum, "prefix_error_drop_band_norm", prev_err_norm - curr_err_norm)
            add_stat(accum, "prefix_error_before_band_totalnorm", prev_err_totalnorm_band)
            add_stat(accum, "prefix_error_after_band_totalnorm", curr_err_totalnorm_band)
            add_stat(
                accum,
                "prefix_error_drop_band_totalnorm",
                prev_err_totalnorm_band - curr_err_totalnorm_band,
            )
            add_stat(accum, "prefix_error_before_total_norm", prev_err_total_norm)
            add_stat(accum, "prefix_error_after_total_norm", curr_err_total_norm)
            add_stat(accum, "prefix_error_drop_total_norm", prev_err_total_norm - curr_err_total_norm)
            add_stat(
                accum,
                "prefix_error_drop_total_frac",
                (prev_err_total_norm - curr_err_total_norm) / prev_err_total_norm.clamp_min(eps),
            )
            add_stat(accum, "omit_error_band_norm", omit_err_norm)
            add_stat(accum, "full_error_band_norm", full_err_norm)
            add_stat(accum, "omit_extra_band_norm", omit_err_norm - full_err_norm)
            add_stat(accum, "omit_error_band_totalnorm", omit_err_totalnorm_band)
            add_stat(accum, "full_error_band_totalnorm", full_err_totalnorm_band)
            add_stat(accum, "omit_extra_band_totalnorm", omit_err_totalnorm_band - full_err_totalnorm_band)
            add_stat(accum, "omit_error_total_norm", omit_err_total_norm)
            add_stat(accum, "full_error_total_norm", full_err_total_norm)
            add_stat(accum, "omit_extra_total_norm", omit_err_total_norm - full_err_total_norm)

    stage_rows = []
    band_rows = []
    for stage, accum in enumerate(stage_accum):
        base = {
            "stage": int(stage),
            "scale_factor": float(model.quantizer.scale_factors[stage]),
        }
        for key in (
            "latent_contribution_rms",
            "marginal_audio_rms",
            "prefix_error_before_total_norm",
            "prefix_error_after_total_norm",
            "prefix_error_drop_total_norm",
            "prefix_error_drop_total_frac",
            "omit_error_total_norm",
            "full_error_total_norm",
            "omit_extra_total_norm",
        ):
            base[key] = finalize_stat(accum, key)

        for key in (
            "marginal_band_energy",
            "marginal_band_share",
            "prefix_error_before_band_norm",
            "prefix_error_after_band_norm",
            "prefix_error_drop_band_norm",
            "prefix_error_before_band_totalnorm",
            "prefix_error_after_band_totalnorm",
            "prefix_error_drop_band_totalnorm",
            "omit_error_band_norm",
            "full_error_band_norm",
            "omit_extra_band_norm",
            "omit_error_band_totalnorm",
            "full_error_band_totalnorm",
            "omit_extra_band_totalnorm",
        ):
            base[key] = finalize_stat(accum, key)
        stage_rows.append(base)

        arrays = {
            key: base[key]
            for key in (
                "marginal_band_energy",
                "marginal_band_share",
                "prefix_error_before_band_norm",
                "prefix_error_after_band_norm",
                "prefix_error_drop_band_norm",
                "prefix_error_before_band_totalnorm",
                "prefix_error_after_band_totalnorm",
                "prefix_error_drop_band_totalnorm",
                "omit_error_band_norm",
                "full_error_band_norm",
                "omit_extra_band_norm",
                "omit_error_band_totalnorm",
                "full_error_band_totalnorm",
                "omit_extra_band_totalnorm",
            )
        }
        for band_idx, label in enumerate(band_labels):
            row = {
                "stage": int(stage),
                "scale_factor": float(model.quantizer.scale_factors[stage]),
                "band": label,
                "band_low_hz": float(bands[band_idx][0]),
                "band_high_hz": float(bands[band_idx][1]),
            }
            for key, values in arrays.items():
                row[key] = float(values[band_idx])
            band_rows.append(row)

    return {
        "files": [str(path) for path in files],
        "bands": [
            {"name": label, "low_hz": lo, "high_hz": hi}
            for label, (lo, hi) in zip(band_labels, bands)
        ],
        "stages": stage_rows,
        "stage_band_rows": band_rows,
    }


def band_sum(row: dict, bands: list[tuple[float, float]], key: str, lo_hz: float, hi_hz: float) -> float:
    values = row[key]
    total = 0.0
    for value, (lo, hi) in zip(values, bands):
        if lo >= lo_hz and hi <= hi_hz:
            total += float(value)
    return total


def print_summary(results: dict):
    for name, data in results["models"].items():
        bands = [(band["low_hz"], band["high_hz"]) for band in data["bands"]]
        print(f"\n[{name}]")
        for row in data["stages"]:
            low_share = band_sum(row, bands, "marginal_band_share", 0.0, 1000.0)
            high_share = band_sum(row, bands, "marginal_band_share", 4000.0, 24000.0)
            low_gain = band_sum(row, bands, "prefix_error_drop_band_totalnorm", 0.0, 1000.0)
            high_gain = band_sum(row, bands, "prefix_error_drop_band_totalnorm", 4000.0, 24000.0)
            low_omit = band_sum(row, bands, "omit_extra_band_totalnorm", 0.0, 1000.0)
            high_omit = band_sum(row, bands, "omit_extra_band_totalnorm", 4000.0, 24000.0)
            print(
                f"s{int(row['stage']):02d} scale={row['scale_factor']:.2f} "
                f"marginal_low1k={low_share:.3f} high4k={high_share:.3f} "
                f"gain_low1k={low_gain:.3g} gain_high4k={high_gain:.3g} "
                f"omit_low1k={low_omit:.3g} omit_high4k={high_omit:.3g}"
            )


def write_csv(path: Path, results: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name, data in results["models"].items():
        for row in data["stage_band_rows"]:
            rows.append({"model": model_name, **row})
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Stage-wise waveform frequency attribution probe.")
    parser.add_argument("--model", action="append", required=True, help="name=/path/to/run_or_weights.pth")
    parser.add_argument("--audio", type=Path, default=Path("eval_set/general"))
    parser.add_argument("--n-files", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--bands-hz", default="0,250,500,1000,2000,4000,8000,16000,22050")
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", type=Path, default=Path("runs/analysis/codec_stage_frequency_probe.json"))
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    files = sample_files(find_audio_files(args.audio), args.n_files, args.seed)
    if not files:
        raise RuntimeError(f"No audio files found under {args.audio}")

    results = {
        "settings": {
            "audio": str(args.audio),
            "n_files": len(files),
            "batch_size": args.batch_size,
            "decode_batch_size": args.decode_batch_size,
            "duration": args.duration,
            "bands_hz": args.bands_hz,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "seed": args.seed,
            "device": args.device,
        },
        "models": {},
    }

    for spec in args.model:
        name, raw_path = parse_model_spec(spec)
        checkpoint = resolve_checkpoint(raw_path.expanduser())
        print(f"[probe] loading {name}: {checkpoint}", flush=True)
        model = load_model(load_path=str(checkpoint)).to(args.device).eval()
        bands = parse_band_edges(args.bands_hz, model.sample_rate)
        results["models"][name] = {
            "checkpoint": str(checkpoint),
            **probe_model(model, files, args, bands),
        }
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.output_csv is not None:
        write_csv(args.output_csv, results)
    print_summary(results)
    print(f"\n[probe] wrote {args.output_json}")
    if args.output_csv is not None:
        print(f"[probe] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
