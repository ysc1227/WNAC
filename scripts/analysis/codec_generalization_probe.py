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


def entropy_stats(codes, codebook_size: int) -> list[dict[str, float]]:
    rows = []
    if isinstance(codes, torch.Tensor):
        code_iter = [codes[:, i] for i in range(codes.shape[1])]
    else:
        code_iter = list(codes)
    denom = math.log2(float(codebook_size))
    for stage, code in enumerate(code_iter):
        flat = code.detach().long().reshape(-1).cpu()
        counts = torch.bincount(flat, minlength=codebook_size).float()
        total = counts.sum().clamp_min(1.0)
        probs = counts / total
        nonzero = probs > 0
        entropy = float(-(probs[nonzero] * torch.log2(probs[nonzero])).sum())
        perplexity = float(2.0 ** entropy)
        rows.append(
            {
                "stage": stage,
                "tokens": int(flat.numel()),
                "entropy_bits": entropy,
                "entropy_norm": entropy / denom if denom > 0 else 0.0,
                "perplexity": perplexity,
                "active_codes": int((counts > 0).sum()),
                "active_frac": float((counts > 0).float().mean()),
            }
        )
    return rows


@torch.no_grad()
def quantizer_trace(model, audio: torch.Tensor) -> dict:
    z = model.encoder(model.preprocess(audio, model.sample_rate))
    quantizer = model.quantizer
    total_steps = z.shape[-1]
    z_q = 0
    residual = z
    stage_rows = []
    codes = []

    for stage, stage_quantizer in enumerate(quantizer.quantizers):
        scale = int(float(quantizer.scale_factors[stage]) * total_steps)
        scale = max(1, scale)
        guide = z_q if torch.is_tensor(z_q) else None
        conv = None
        if getattr(quantizer, "quant_resi", None) is not None:
            conv = lambda h, stage_idx=stage: quantizer._apply_quant_resi(stage_idx, h)
        z_q_i, commitment_i, codebook_i, indices_i, z_e_i = stage_quantizer(
            residual,
            scale,
            conv,
            guide=guide,
        )
        z_q = z_q + z_q_i
        residual = residual - z_q_i
        codes.append(indices_i)
        stage_rows.append(
            {
                "stage": stage,
                "scale_factor": float(quantizer.scale_factors[stage]),
                "scale_steps": int(scale),
                "commitment": tensor_mean(commitment_i),
                "codebook": tensor_mean(codebook_i),
                "stage_z_e_energy": tensor_mean(z_e_i.pow(2)),
                "stage_contribution_energy": tensor_mean(z_q_i.pow(2)),
                "residual_energy_after": tensor_mean(residual.pow(2)),
            }
        )

    z_energy = tensor_mean(z.pow(2))
    final_residual = tensor_mean(residual.pow(2))
    entropy_rows = entropy_stats(codes, int(model.codebook_size))
    for row, entropy in zip(stage_rows, entropy_rows):
        row.update(entropy)
    return {
        "latent_energy": z_energy,
        "latent_residual_mse": final_residual,
        "latent_residual_rel": final_residual / max(z_energy, 1e-12),
        "stages": stage_rows,
    }


@torch.no_grad()
def evaluate_model(model, files: list[Path], args, loss_modules: dict[str, torch.nn.Module]) -> dict:
    model.eval()
    recon_rows = []
    trace_rows = []
    for batch_files in chunks(files, args.batch_size):
        audio = load_batch(batch_files, model.sample_rate, args.duration, args.device)
        out = model(audio, model.sample_rate)
        recons = AudioSignal(out["audio"], model.sample_rate)
        signal = AudioSignal(audio, model.sample_rate)
        trace = quantizer_trace(model, audio)
        codebook = [tensor_mean(x) for x in out["vq/codebook_loss"]]
        commitment = [tensor_mean(x) for x in out["vq/commitment_loss"]]
        recon_rows.append(
            {
                "mel": tensor_mean(loss_modules["mel"](recons, signal)),
                "stft": tensor_mean(loss_modules["stft"](recons, signal)),
                "waveform": tensor_mean(loss_modules["waveform"](recons, signal)),
                "vq_codebook_sum": float(sum(codebook)),
                "vq_commitment_sum": float(sum(commitment)),
                "latent_energy": trace["latent_energy"],
                "latent_residual_mse": trace["latent_residual_mse"],
                "latent_residual_rel": trace["latent_residual_rel"],
            }
        )
        trace_rows.append(trace["stages"])
    return {
        "summary": mean_rows(recon_rows),
        "stages": mean_stage_rows(trace_rows),
        "files": [str(path) for path in files],
    }


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in keys}


def mean_stage_rows(batches: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    n_stages = len(batches[0])
    out = []
    for stage in range(n_stages):
        rows = [batch[stage] for batch in batches]
        numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (float, int))]
        mean = {key: float(sum(float(row[key]) for row in rows) / len(rows)) for key in numeric_keys}
        mean["stage"] = int(stage)
        out.append(mean)
    return out


def print_summary(results: dict):
    for model_name, model_results in results["models"].items():
        print(f"\n[{model_name}]")
        for split in ("train", "val"):
            summary = model_results[split]["summary"]
            print(
                f"{split:>5s} "
                f"mel={summary['mel']:.6g} "
                f"stft={summary['stft']:.6g} "
                f"wav={summary['waveform']:.6g} "
                f"latent_rel={summary['latent_residual_rel']:.6g} "
                f"codebook={summary['vq_codebook_sum']:.6g} "
                f"commit={summary['vq_commitment_sum']:.6g}"
            )
        train = model_results["train"]["summary"]
        val = model_results["val"]["summary"]
        print(
            "  gap "
            f"mel={val['mel'] / max(train['mel'], 1e-12):.3f}x "
            f"stft={val['stft'] / max(train['stft'], 1e-12):.3f}x "
            f"wav={val['waveform'] / max(train['waveform'], 1e-12):.3f}x "
            f"latent_rel={val['latent_residual_rel'] / max(train['latent_residual_rel'], 1e-12):.3f}x"
        )


def main():
    parser = argparse.ArgumentParser(description="Compare codec train/val generalization diagnostics.")
    parser.add_argument("--model", action="append", required=True, help="name=/path/to/run_or_weights.pth")
    parser.add_argument("--train_dirs", nargs="+", required=True)
    parser.add_argument("--val_dirs", nargs="+", required=True)
    parser.add_argument("--n_files", type=int, default=16)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", default="runs/analysis/codec_generalization_probe.json")
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    train_files = sample_files(find_audio_files(args.train_dirs), args.n_files, args.seed)
    val_files = sample_files(find_audio_files(args.val_dirs), args.n_files, args.seed + 1)
    if not train_files:
        raise RuntimeError("No train audio files found")
    if not val_files:
        raise RuntimeError("No val audio files found")

    loss_modules = make_loss_modules(args.device)
    results = {
        "settings": {
            "n_files": args.n_files,
            "duration": args.duration,
            "batch_size": args.batch_size,
            "device": args.device,
            "seed": args.seed,
            "train_dirs": args.train_dirs,
            "val_dirs": args.val_dirs,
        },
        "models": {},
    }

    for spec in args.model:
        name, raw_path = parse_model_spec(spec)
        checkpoint = resolve_checkpoint(raw_path.expanduser())
        print(f"[probe] loading {name}: {checkpoint}", flush=True)
        model = load_model(load_path=str(checkpoint)).to(args.device).eval()
        results["models"][name] = {
            "checkpoint": str(checkpoint),
            "train": evaluate_model(model, train_files, args, loss_modules),
            "val": evaluate_model(model, val_files, args, loss_modules),
        }
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print_summary(results)
    print(f"\n[probe] wrote {output}")


if __name__ == "__main__":
    main()
