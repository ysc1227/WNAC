import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import torch
import torch.nn.functional as F
import torchaudio
from audiotools import AudioSignal

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emac.nn import loss as losses
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
    candidates = [
        path / "best" / "emac" / "weights.pth",
        path / "latest" / "emac" / "weights.pth",
        path / "emac" / "weights.pth",
        path / "weights.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve checkpoint from {path}")


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def find_audio_files(dirs: list[str]) -> list[Path]:
    files = []
    for item in dirs:
        root = Path(item).expanduser()
        if root.is_file() and root.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(root)
        elif root.exists():
            files.extend(path for path in root.rglob("*") if path.suffix.lower() in AUDIO_EXTENSIONS)
    return sorted(set(files))


def sample_files(files: list[Path], n_files: int, seed: int) -> list[Path]:
    if n_files <= 0 or n_files >= len(files):
        return list(files)
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)
    return sorted(files[:n_files])


def load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    audio, sr = torchaudio.load(str(path))
    audio = audio.mean(dim=0, keepdim=True)
    if int(sr) != int(sample_rate):
        audio = torchaudio.functional.resample(audio, int(sr), int(sample_rate))
    return audio


def crop_or_pad(audio: torch.Tensor, n_samples: int) -> torch.Tensor:
    if audio.shape[-1] < n_samples:
        return F.pad(audio, (0, n_samples - audio.shape[-1]))
    if audio.shape[-1] == n_samples:
        return audio
    start = (audio.shape[-1] - n_samples) // 2
    return audio[..., start : start + n_samples]


def load_batch(files: list[Path], sample_rate: int, duration: float, device: str) -> torch.Tensor:
    n_samples = max(1, int(round(float(duration) * sample_rate)))
    audio = [crop_or_pad(load_audio(path, sample_rate), n_samples) for path in files]
    return torch.stack(audio, dim=0).to(device)


def chunks(items: list[Path], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def make_loss_modules(device: str) -> dict[str, torch.nn.Module]:
    return {
        "mel": losses.MelSpectrogramLoss(
            n_mels=[5, 10, 20, 40, 80, 160, 320],
            window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
            mel_fmin=[0, 0, 0, 0, 0, 0, 0],
            mel_fmax=[None, None, None, None, None, None, None],
            pow=1.0,
            clamp_eps=1.0e-5,
            mag_weight=0.0,
        ).to(device),
        "stft": losses.MultiScaleSTFTLoss(window_lengths=[2048, 512]).to(device),
        "waveform": losses.L1Loss().to(device),
    }


def tensor_mean(value) -> float:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return float(value.detach().float().mean().cpu())


def rms(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().pow(2).mean().sqrt()


def entropy_stats(code: torch.Tensor, codebook_size: int) -> dict[str, float]:
    flat = code.detach().long().reshape(-1).cpu()
    counts = torch.bincount(flat, minlength=codebook_size).float()
    total = counts.sum().clamp_min(1.0)
    probs = counts / total
    nonzero = probs > 0
    entropy = float(-(probs[nonzero] * torch.log2(probs[nonzero])).sum())
    sorted_probs = torch.sort(probs, descending=True)[0]
    return {
        "entropy_bits": entropy,
        "entropy_eff": entropy / math.log2(float(codebook_size)),
        "used_codes": int((counts > 0).sum()),
        "top10_ratio": float(sorted_probs[: min(10, codebook_size)].sum()),
    }


@torch.no_grad()
def decode_audio(model, z_q: torch.Tensor) -> torch.Tensor:
    audio = model.decoder(z_q)
    return audio


@torch.no_grad()
def metric_losses(loss_modules, audio: torch.Tensor, recon: torch.Tensor, sample_rate: int) -> dict[str, float]:
    n = min(int(audio.shape[-1]), int(recon.shape[-1]))
    audio = audio[..., :n]
    recon = recon[..., :n]
    signal = AudioSignal(audio, sample_rate)
    recons = AudioSignal(recon, sample_rate)
    return {
        "mel": tensor_mean(loss_modules["mel"](recons, signal)),
        "stft": tensor_mean(loss_modules["stft"](recons, signal)),
        "waveform": tensor_mean(loss_modules["waveform"](recons, signal)),
    }


@torch.no_grad()
def trace_quantizer(model, audio: torch.Tensor) -> dict:
    z = model.encoder(model.preprocess(audio, model.sample_rate))
    quantizer = model.quantizer
    total_steps = z.shape[-1]
    z_q = 0
    residual = z
    stages = []

    for stage, stage_quantizer in enumerate(quantizer.quantizers):
        residual_before = residual
        scale_factor = float(quantizer.scale_factors[stage])
        scale = max(1, int(scale_factor * total_steps))
        guide = z_q if torch.is_tensor(z_q) else None
        conv = None
        if getattr(quantizer, "quant_resi", None) is not None:
            conv = lambda h, stage_idx=stage: quantizer._apply_quant_resi(stage_idx, h)
        z_q_i, commitment_i, codebook_i, indices_i, z_e_i = stage_quantizer(
            residual_before,
            scale,
            conv,
            guide=guide,
        )
        quant_error = residual_before - z_q_i
        z_q = z_q + z_q_i
        residual = residual_before - z_q_i
        residual_energy = residual_before.detach().float().pow(2).mean()
        error_energy = quant_error.detach().float().pow(2).mean()
        stage_energy = z_q_i.detach().float().pow(2).mean()
        lookup_energy = z_e_i.detach().float().pow(2).mean()
        stages.append(
            {
                "stage": stage,
                "scale_factor": scale_factor,
                "scale_steps": int(scale),
                "residual_energy": tensor_mean(residual_energy),
                "quant_error_energy": tensor_mean(error_energy),
                "relative_quant_error": tensor_mean(error_energy / residual_energy.clamp_min(1e-12)),
                "stage_contribution_energy": tensor_mean(stage_energy),
                "lookup_energy": tensor_mean(lookup_energy),
                "codebook_loss": tensor_mean(codebook_i),
                "commitment_loss": tensor_mean(commitment_i),
                "z_q_i": z_q_i.detach(),
                "quant_error": quant_error.detach(),
                "indices": indices_i.detach(),
            }
        )

    return {
        "z": z.detach(),
        "z_q": z_q.detach(),
        "latent_residual": residual.detach(),
        "stages": stages,
    }


def mean_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(sum(float(row[key]) for row in rows) / len(rows)) for key in keys}


def mean_stage_rows(batches: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    n_stages = len(batches[0])
    out = []
    for stage in range(n_stages):
        rows = [batch[stage] for batch in batches]
        keys = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
        mean = {key: float(sum(float(row[key]) for row in rows) / len(rows)) for key in keys}
        mean["stage"] = int(stage)
        out.append(mean)
    return out


@torch.no_grad()
def evaluate_model(model, files: list[Path], args, loss_modules) -> dict:
    model.eval()
    stage_batches = []
    full_losses = []
    for batch_files in chunks(files, args.batch_size):
        audio = load_batch(batch_files, model.sample_rate, args.duration, args.device)
        trace = trace_quantizer(model, audio)
        full_recon = decode_audio(model, trace["z_q"])
        full_loss = metric_losses(loss_modules, audio, full_recon, model.sample_rate)
        full_losses.append(full_loss)

        batch_stage_rows = []
        for stage_row in trace["stages"]:
            quant_error = stage_row["quant_error"]
            err_rms = tensor_mean(rms(quant_error))
            full_z_rms = tensor_mean(rms(trace["z_q"]))
            perturb_scale = float(args.perturb_rel) * full_z_rms / max(err_rms, 1e-12)
            perturbed_z = trace["z_q"] + quant_error * perturb_scale
            perturbed_recon = decode_audio(model, perturbed_z)
            perturbed_loss = metric_losses(loss_modules, audio, perturbed_recon, model.sample_rate)
            row = {
                key: value
                for key, value in stage_row.items()
                if key not in ("z_q_i", "quant_error", "indices")
            }
            row.update(entropy_stats(stage_row["indices"], int(model.codebook_size)))
            row["error_rms"] = err_rms
            row["error_rms_rel_to_full_z"] = err_rms / max(full_z_rms, 1e-12)
            row["perturb_rms_rel_to_full_z"] = float(args.perturb_rel)
            for metric_name in ("mel", "stft", "waveform"):
                base = full_loss[metric_name]
                perturbed = perturbed_loss[metric_name]
                row[f"{metric_name}_perturb_delta"] = perturbed - base
                row[f"{metric_name}_perturb_delta_rel"] = (perturbed - base) / max(base, 1e-12)
                row[f"{metric_name}_sensitivity"] = abs(perturbed - base) / max(float(args.perturb_rel), 1e-12)
            batch_stage_rows.append(row)
        stage_batches.append(batch_stage_rows)

    stages = mean_stage_rows(stage_batches)
    raw_sum = sum(row["codebook_loss"] + row["commitment_loss"] for row in stages)
    rel_sum = sum(row["relative_quant_error"] for row in stages)
    scale_rel_sum = sum(row["relative_quant_error"] / max(row["scale_factor"], 1e-6) for row in stages)
    sqrt_scale_rel_sum = sum(row["relative_quant_error"] / math.sqrt(max(row["scale_factor"], 1e-6)) for row in stages)
    for row in stages:
        raw = row["codebook_loss"] + row["commitment_loss"]
        rel = row["relative_quant_error"]
        row["raw_vq_share"] = raw / max(raw_sum, 1e-12)
        row["relative_vq_share"] = rel / max(rel_sum, 1e-12)
        row["scale_relative_vq_share"] = (rel / max(row["scale_factor"], 1e-6)) / max(scale_rel_sum, 1e-12)
        row["sqrt_scale_relative_vq_share"] = (
            rel / math.sqrt(max(row["scale_factor"], 1e-6))
        ) / max(sqrt_scale_rel_sum, 1e-12)

    return {
        "full_loss": mean_dicts(full_losses),
        "stages": stages,
        "files": [str(path) for path in files],
    }


def print_model_summary(name: str, result: dict):
    print(f"\n[{name}] full mel={result['full_loss']['mel']:.4g} "
          f"stft={result['full_loss']['stft']:.4g} wav={result['full_loss']['waveform']:.4g}")
    print("stage scale H eff rel_err raw% rel% scale% mel_d% stft_d% top10")
    for row in result["stages"]:
        print(
            f"{int(row['stage']):02d} {row['scale_factor']:.2f} "
            f"{row['entropy_bits']:.3f} {row['entropy_eff']:.3f} "
            f"{row['relative_quant_error']:.3g} "
            f"{100*row['raw_vq_share']:.1f} "
            f"{100*row['relative_vq_share']:.1f} "
            f"{100*row['scale_relative_vq_share']:.1f} "
            f"{100*row['mel_perturb_delta_rel']:.2f} "
            f"{100*row['stft_perturb_delta_rel']:.2f} "
            f"{row['top10_ratio']:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Diagnose scale-aware VQ loss candidates without training.")
    parser.add_argument("--model", action="append", required=True, help="name=/path/to/run_or_weights.pth")
    parser.add_argument("--audio", nargs="+", required=True)
    parser.add_argument("--n-files", type=int, default=32)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--perturb-rel", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", default="runs/analysis/codec_scale_loss_diagnostics_probe.json")
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    files = sample_files(find_audio_files(args.audio), args.n_files, args.seed)
    if not files:
        raise RuntimeError("No audio files found")

    loss_modules = make_loss_modules(args.device)
    results = {
        "settings": {
            "audio": args.audio,
            "n_files": args.n_files,
            "duration": args.duration,
            "batch_size": args.batch_size,
            "device": args.device,
            "seed": args.seed,
        },
        "models": {},
    }
    for spec in args.model:
        name, raw_path = parse_model_spec(spec)
        checkpoint = resolve_checkpoint(raw_path.expanduser())
        print(f"[probe] loading {name}: {checkpoint}", flush=True)
        model = load_model(load_path=str(checkpoint)).to(args.device).eval()
        result = evaluate_model(model, files, args, loss_modules)
        result["checkpoint"] = str(checkpoint)
        results["models"][name] = result
        print_model_summary(name, result)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[probe] wrote {output}")


if __name__ == "__main__":
    main()
