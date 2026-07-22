import argparse
import json
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


def chunks(items: list[Path], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


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


def audio_losses(
    decoded: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int,
    loss_modules: dict[str, torch.nn.Module],
) -> dict[str, float]:
    decoded = decoded[..., : target.shape[-1]]
    recons = AudioSignal(decoded, sample_rate)
    signal = AudioSignal(target, sample_rate)
    return {
        "mel": tensor_mean(loss_modules["mel"](recons, signal)),
        "stft": tensor_mean(loss_modules["stft"](recons, signal)),
        "waveform": tensor_mean(loss_modules["waveform"](recons, signal)),
    }


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in keys}


def mean_stage_rows(rows: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    n_stages = len(rows[0])
    out = []
    for stage in range(n_stages):
        stage_rows = [row[stage] for row in rows]
        keys = [key for key, value in stage_rows[0].items() if isinstance(value, (float, int))]
        mean = {key: float(sum(float(row[key]) for row in stage_rows) / len(stage_rows)) for key in keys}
        mean["stage"] = int(stage)
        out.append(mean)
    return out


@torch.no_grad()
def cumulative_quantized_latents(model, audio: torch.Tensor) -> tuple[torch.Tensor, list[dict]]:
    z = model.encoder(model.preprocess(audio, model.sample_rate))
    quantizer = model.quantizer
    total_steps = z.shape[-1]
    z_q = 0
    residual = z
    rows = []

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
        latent_energy = tensor_mean(z.pow(2))
        residual_mse = tensor_mean(residual.pow(2))
        rows.append(
            {
                "stage": stage,
                "n_stages": stage + 1,
                "scale_factor": float(quantizer.scale_factors[stage]),
                "scale_steps": int(scale),
                "codebook": tensor_mean(codebook_i),
                "commitment": tensor_mean(commitment_i),
                "latent_residual_mse": residual_mse,
                "latent_residual_rel": residual_mse / max(latent_energy, 1e-12),
                "z_q": z_q,
            }
        )
    return z, rows


@torch.no_grad()
def evaluate_split(model, files: list[Path], args, loss_modules: dict[str, torch.nn.Module]) -> dict:
    model.eval()
    oracle_rows = []
    prefix_batches = []
    for batch_files in chunks(files, args.batch_size):
        audio = load_batch(batch_files, model.sample_rate, args.duration, args.device)
        z, prefix_rows = cumulative_quantized_latents(model, audio)

        oracle = model.decode(z)
        oracle_row = audio_losses(oracle, audio, model.sample_rate, loss_modules)
        oracle_rows.append(oracle_row)

        stage_rows = []
        for row in prefix_rows:
            z_q = row.pop("z_q")
            decoded = model.decode(z_q)
            stage_row = dict(row)
            stage_row.update(audio_losses(decoded, audio, model.sample_rate, loss_modules))
            stage_rows.append(stage_row)
        prefix_batches.append(stage_rows)

    prefix = mean_stage_rows(prefix_batches)
    return {
        "oracle": mean_rows(oracle_rows),
        "prefix": prefix,
        "full": prefix[-1],
        "files": [str(path) for path in files],
    }


def print_summary(results: dict):
    for model_name, model_results in results["models"].items():
        print(f"\n[{model_name}]")
        for split in ("train", "val"):
            oracle = model_results[split]["oracle"]
            full = model_results[split]["full"]
            prefix = model_results[split]["prefix"]
            print(
                f"{split:>5s} oracle "
                f"mel={oracle['mel']:.6g} stft={oracle['stft']:.6g} wav={oracle['waveform']:.6g}"
            )
            print(
                f"{split:>5s} full   "
                f"mel={full['mel']:.6g} stft={full['stft']:.6g} wav={full['waveform']:.6g} "
                f"latent_rel={full['latent_residual_rel']:.6g}"
            )
            print(
                "      q/oracle "
                f"mel={full['mel'] / max(oracle['mel'], 1e-12):.3f}x "
                f"stft={full['stft'] / max(oracle['stft'], 1e-12):.3f}x "
                f"wav={full['waveform'] / max(oracle['waveform'], 1e-12):.3f}x"
            )
            best = {metric: min(prefix, key=lambda row: row[metric]) for metric in ("mel", "stft", "waveform")}
            print(
                "      best-prefix "
                f"mel=s{int(best['mel']['stage'])}:{best['mel']['mel']:.6g} "
                f"stft=s{int(best['stft']['stage'])}:{best['stft']['stft']:.6g} "
                f"wav=s{int(best['waveform']['stage'])}:{best['waveform']['waveform']:.6g}"
            )


def main():
    parser = argparse.ArgumentParser(description="Oracle-decoder and stage-prefix codec probe.")
    parser.add_argument("--model", action="append", required=True, help="name=/path/to/run_or_weights.pth")
    parser.add_argument("--train_dirs", nargs="+", required=True)
    parser.add_argument("--val_dirs", nargs="+", required=True)
    parser.add_argument("--n_files", type=int, default=16)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", default="runs/analysis/codec_decoder_stage_probe.json")
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
            "train": evaluate_split(model, train_files, args, loss_modules),
            "val": evaluate_split(model, val_files, args, loss_modules),
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
