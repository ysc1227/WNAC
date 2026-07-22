import argparse
import csv
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emac.utils import load_model


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}


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


def load_audio(path: Path, sample_rate: int, duration: float, device: str) -> torch.Tensor:
    audio, sr = torchaudio.load(str(path))
    audio = audio.mean(dim=0, keepdim=True)
    if int(sr) != int(sample_rate):
        audio = torchaudio.functional.resample(audio, int(sr), int(sample_rate))
    n_samples = max(1, int(round(duration * sample_rate)))
    if audio.shape[-1] < n_samples:
        audio = F.pad(audio, (0, n_samples - audio.shape[-1]))
    elif audio.shape[-1] > n_samples:
        start = (audio.shape[-1] - n_samples) // 2
        audio = audio[..., start : start + n_samples]
    return audio.to(device)


def chunks(items: list[Path], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def tensor_energy(x: torch.Tensor) -> float:
    return float(x.detach().float().pow(2).mean().cpu())


def tensor_temporal_diff(x: torch.Tensor) -> float:
    if x.shape[-1] < 2:
        return 0.0
    return float((x[..., 1:] - x[..., :-1]).detach().float().pow(2).mean().cpu())


def covariance_stats(x: torch.Tensor) -> dict[str, float]:
    # x: B x C x T.  Codebook dim is small, so full eigendecomposition is cheap.
    flat = x.detach().float().permute(0, 2, 1).reshape(-1, x.shape[1])
    if flat.shape[0] < 2:
        return {"eff_rank": 0.0, "participation_rank": 0.0, "max_eig_frac": 0.0, "cond": 0.0}
    flat = flat - flat.mean(dim=0, keepdim=True)
    cov = flat.t().matmul(flat) / max(1, flat.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    total = eig.sum().clamp_min(1e-12)
    probs = eig / total
    nonzero = probs > 1e-12
    entropy = -(probs[nonzero] * torch.log(probs[nonzero])).sum()
    eff_rank = torch.exp(entropy)
    participation_rank = total.pow(2) / eig.pow(2).sum().clamp_min(1e-12)
    positive = eig[eig > eig.max().clamp_min(1e-12) * 1e-6]
    cond = eig.max() / positive.min().clamp_min(1e-12) if positive.numel() else eig.new_tensor(0.0)
    return {
        "eff_rank": float(eff_rank.cpu()),
        "participation_rank": float(participation_rank.cpu()),
        "max_eig_frac": float((eig.max() / total).cpu()),
        "cond": float(cond.cpu()),
    }


def direction_stats(x: torch.Tensor) -> dict[str, float]:
    # VQ lookup normalizes each token vector, so summarize the distribution on
    # the unit sphere directly.
    flat = x.detach().float().permute(0, 2, 1).reshape(-1, x.shape[1])
    if flat.numel() == 0:
        return {"mean_resultant": 0.0, "direction_max_eig_frac": 0.0}
    directions = F.normalize(flat, dim=1)
    mean_resultant = directions.mean(dim=0).norm()
    if directions.shape[0] < 2:
        return {
            "mean_resultant": float(mean_resultant.cpu()),
            "direction_max_eig_frac": 0.0,
        }
    centered = directions - directions.mean(dim=0, keepdim=True)
    cov = centered.t().matmul(centered) / max(1, centered.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    return {
        "mean_resultant": float(mean_resultant.cpu()),
        "direction_max_eig_frac": float((eig.max() / eig.sum().clamp_min(1e-12)).cpu()),
    }


def usage_stats(counts: torch.Tensor) -> dict[str, float | int]:
    counts = counts.double()
    total = counts.sum().clamp_min(1.0)
    probs = counts / total
    nonzero = probs > 0
    entropy = float(-(probs[nonzero] * torch.log2(probs[nonzero])).sum().cpu())
    sorted_counts = torch.sort(counts, descending=True).values
    n = counts.numel()
    ascending = torch.sort(counts).values
    index = torch.arange(1, n + 1, dtype=torch.float64)
    gini = float((((2 * index - n - 1) * ascending.cpu()).sum() / (n * total.cpu())).item())

    def top_ratio(k: int) -> float:
        return float((sorted_counts[: min(k, n)].sum() / total).cpu())

    return {
        "entropy_bits": entropy,
        "efficiency": entropy / math.log2(n),
        "used_codes": int((counts > 0).sum().item()),
        "unused_codes": int((counts == 0).sum().item()),
        "rare_codes_le_5": int((counts <= 5).sum().item()),
        "top1_ratio": top_ratio(1),
        "top10_ratio": top_ratio(10),
        "gini": gini,
    }


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    out = {}
    for key in rows[0].keys():
        out[key] = float(sum(float(row[key]) for row in rows) / len(rows))
    return out


@torch.no_grad()
def probe(model, files: list[Path], args) -> dict:
    quantizer = model.quantizer
    n_stages = len(quantizer.quantizers)
    codebook_size = int(getattr(model, "codebook_size", getattr(quantizer, "codebook_size", 1024)))
    accum: list[list[dict[str, float]]] = [[] for _ in range(n_stages)]
    counts = [torch.zeros(codebook_size, dtype=torch.long) for _ in range(n_stages)]

    for batch_files in chunks(files, args.batch_size):
        audio = torch.stack(
            [load_audio(path, model.sample_rate, args.duration, args.device) for path in batch_files],
            dim=0,
        )
        z = model.encoder(model.preprocess(audio, model.sample_rate))
        residual = z
        z_q = 0
        total_steps = z.shape[-1]

        for stage, stage_quantizer in enumerate(quantizer.quantizers):
            scale = max(1, int(float(quantizer.scale_factors[stage]) * total_steps))
            guide = z_q if torch.is_tensor(z_q) else None
            conv = None
            if getattr(quantizer, "quant_resi", None) is not None:
                conv = lambda h, stage_idx=stage: quantizer._apply_quant_resi(stage_idx, h)

            z_e = stage_quantizer.in_proj(residual)
            inter_z_base = stage_quantizer.downsample_latents_base(z_e, scale)
            inter_z = stage_quantizer.downsample_latents(z_e, scale)
            z_q_small, indices = stage_quantizer.decode_latents(inter_z)
            z_q_up = stage_quantizer.upsample_code(
                z_q_small,
                total_steps,
                guide=guide,
                use_guide=scale != total_steps,
            )
            z_q_i = conv(z_q_up) if conv is not None else z_q_up
            z_q_i = stage_quantizer.out_proj(z_q_i)

            residual_before = residual
            residual_after = residual_before - z_q_i
            flat_indices = indices.detach().long().reshape(-1).cpu()
            counts[stage] += torch.bincount(flat_indices, minlength=codebook_size)

            proj_stats = covariance_stats(z_e)
            inter_stats = covariance_stats(inter_z)
            inter_direction_stats = direction_stats(inter_z)
            base_delta = inter_z - inter_z_base
            accum[stage].append(
                {
                    "scale_factor": float(quantizer.scale_factors[stage]),
                    "scale_steps": float(scale),
                    "residual_energy_before": tensor_energy(residual_before),
                    "residual_energy_after": tensor_energy(residual_after),
                    "residual_energy_ratio_after": tensor_energy(residual_after)
                    / max(tensor_energy(residual_before), 1e-12),
                    "contribution_energy": tensor_energy(z_q_i),
                    "proj_energy": tensor_energy(z_e),
                    "inter_energy": tensor_energy(inter_z),
                    "inter_over_proj_energy": tensor_energy(inter_z) / max(tensor_energy(z_e), 1e-12),
                    "proj_temporal_diff": tensor_temporal_diff(z_e),
                    "inter_temporal_diff": tensor_temporal_diff(inter_z),
                    "inter_over_proj_temporal_diff": tensor_temporal_diff(inter_z)
                    / max(tensor_temporal_diff(z_e), 1e-12),
                    "downsampler_delta_rel": tensor_energy(base_delta) ** 0.5
                    / max(tensor_energy(inter_z_base) ** 0.5, 1e-12),
                    "proj_eff_rank": proj_stats["eff_rank"],
                    "proj_participation_rank": proj_stats["participation_rank"],
                    "proj_max_eig_frac": proj_stats["max_eig_frac"],
                    "proj_cond": proj_stats["cond"],
                    "inter_eff_rank": inter_stats["eff_rank"],
                    "inter_participation_rank": inter_stats["participation_rank"],
                    "inter_max_eig_frac": inter_stats["max_eig_frac"],
                    "inter_cond": inter_stats["cond"],
                    "inter_mean_resultant": inter_direction_stats["mean_resultant"],
                    "inter_direction_max_eig_frac": inter_direction_stats["direction_max_eig_frac"],
                }
            )
            z_q = z_q + z_q_i
            residual = residual_after

    stages = []
    for stage in range(n_stages):
        row = mean_rows(accum[stage])
        row["stage"] = stage
        row.update(usage_stats(counts[stage]))
        stages.append(row)
    return {
        "model_path": str(args.model),
        "folder": str(args.audio),
        "n_audio_files": len(files),
        "duration": args.duration,
        "stages": stages,
    }


def write_csv(path: Path, stages: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stages[0].keys()))
        writer.writeheader()
        writer.writerows(stages)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, default=Path("eval_set/general"))
    parser.add_argument("--n-files", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    args.device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto":
        args.device = "cpu"
    args.model = resolve_checkpoint(args.model)
    files = sample_files(find_audio_files(args.audio), args.n_files, args.seed)
    if not files:
        raise RuntimeError(f"No audio files found under {args.audio}")

    model = load_model(load_path=str(args.model)).to(args.device)
    model.eval()
    result = probe(model, files, args)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(result, f, indent=2)
    write_csv(args.output_csv, result["stages"])

    for row in result["stages"]:
        print(
            f"s{int(row['stage']):02d} scale={row['scale_factor']:.2f} "
            f"H={row['entropy_bits']:.3f} used={int(row['used_codes'])}/"
            f"{int(row['used_codes']) + int(row['unused_codes'])} "
            f"top10={row['top10_ratio']:.3f} gini={row['gini']:.3f} "
            f"proj_rank={row['proj_eff_rank']:.2f} inter_rank={row['inter_eff_rank']:.2f} "
            f"meanR={row['inter_mean_resultant']:.3f}->{row['lookup_mean_resultant']:.3f} "
            f"inter/proj_E={row['inter_over_proj_energy']:.3f}"
        )
    print(f"[probe] wrote {args.output_json}")
    print(f"[probe] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
