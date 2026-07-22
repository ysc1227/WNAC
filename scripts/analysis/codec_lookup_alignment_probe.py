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


def parse_strata_specs(specs: list[str] | None) -> dict[str, list[str]]:
    strata = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"Expected stratum=/path format, got: {spec}")
        name, path = spec.split("=", 1)
        strata.setdefault(name, []).append(path)
    return strata


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


def select_audio_files(
    dirs: list[str],
    n_files: int,
    strata_specs: list[str] | None,
    n_files_per_stratum: int,
    seed: int,
) -> tuple[list[Path], dict[str, str]]:
    strata = parse_strata_specs(strata_specs)
    if not strata:
        files = sample_files(find_audio_files(dirs), n_files, seed)
        return files, {str(path): "all" for path in files}

    selected = []
    labels = {}
    for stratum_idx, (name, stratum_dirs) in enumerate(strata.items()):
        candidates = find_audio_files(stratum_dirs)
        if len(candidates) < n_files_per_stratum:
            raise RuntimeError(
                f"Stratum '{name}' has {len(candidates)} files, fewer than "
                f"n_files_per_stratum={n_files_per_stratum}."
            )
        files = sample_files(
            candidates,
            n_files_per_stratum,
            seed + 1009 * stratum_idx,
        )
        for path in files:
            key = str(path)
            if key in labels:
                raise ValueError(f"Audio file appears in multiple strata: {path}")
            labels[key] = name
            selected.append(path)
    return selected, labels


def load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    audio, sr = torchaudio.load(str(path))
    audio = audio.mean(dim=0, keepdim=True)
    if int(sr) != int(sample_rate):
        audio = torchaudio.functional.resample(audio, int(sr), int(sample_rate))
    return audio


def crop_or_pad(
    audio: torch.Tensor,
    n_samples: int,
    start: int | None = None,
) -> torch.Tensor:
    if audio.shape[-1] < n_samples:
        return F.pad(audio, (0, n_samples - audio.shape[-1]))
    if audio.shape[-1] == n_samples:
        return audio
    max_start = audio.shape[-1] - n_samples
    start = max_start // 2 if start is None else max(0, min(int(start), max_start))
    return audio[..., start : start + n_samples]


def load_one(
    path: Path,
    sample_rate: int,
    duration: float,
    device: str,
    crop_mode: str = "center",
    crop_seed: int = 0,
) -> torch.Tensor:
    n_samples = max(1, int(round(float(duration) * sample_rate)))
    audio = load_audio(path, sample_rate)
    start = None
    if crop_mode == "random" and audio.shape[-1] > n_samples:
        rng = random.Random(f"{crop_seed}:{path}")
        start = rng.randint(0, audio.shape[-1] - n_samples)
    elif crop_mode != "center":
        raise ValueError(f"Unknown crop_mode: {crop_mode}")
    return crop_or_pad(audio, n_samples, start=start).unsqueeze(0).to(device)


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
        "sisdr": losses.SISDRLoss(reduction="none").to(device),
    }


def audio_metric(
    decoded: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int,
    metric: str,
    loss_modules: dict[str, torch.nn.Module],
) -> torch.Tensor:
    decoded = decoded[..., : target.shape[-1]]
    target = target.expand(decoded.shape[0], -1, -1)
    if metric == "waveform":
        return (decoded - target).abs().mean(dim=(1, 2))
    if metric == "sisdr":
        value = loss_modules["sisdr"](target, decoded)
        return value.reshape(value.shape[0], -1).mean(dim=1)
    if metric in {"mel", "stft"}:
        values = []
        for idx in range(decoded.shape[0]):
            recons = AudioSignal(decoded[idx : idx + 1], sample_rate)
            signal = AudioSignal(target[idx : idx + 1], sample_rate)
            values.append(loss_modules[metric](recons, signal).reshape(()))
        return torch.stack(values)
    raise ValueError(f"Unknown metric: {metric}")


def token_positions(length: int, n_positions: int) -> list[int]:
    n_positions = min(int(n_positions), int(length))
    if n_positions <= 0:
        return []
    if n_positions == 1:
        return [length // 2]
    return sorted(set(torch.linspace(0, length - 1, steps=n_positions).round().long().tolist()))


def lookup_codebook_weight(stage_quantizer) -> torch.Tensor:
    getter = getattr(stage_quantizer, "lookup_codebook_weight_tensor", None)
    if callable(getter):
        return getter()
    return stage_quantizer.codebook.weight


def select_token_positions(
    length: int,
    n_positions: int,
    mode: str,
    seed: int,
) -> list[int]:
    if mode == "evenly_spaced":
        return token_positions(length, n_positions)
    if mode != "random":
        raise ValueError(f"Unknown position_mode: {mode}")
    n_positions = min(int(n_positions), int(length))
    if n_positions <= 0:
        return []
    rng = random.Random(seed)
    return sorted(rng.sample(range(length), n_positions))


@torch.no_grad()
def encode_stage_cache(model, audio: torch.Tensor):
    z = model.encoder(model.preprocess(audio, model.sample_rate))
    quantizer = model.quantizer
    total_steps = z.shape[-1]
    z_q = 0
    residual = z
    stages = []

    for stage, stage_quantizer in enumerate(quantizer.quantizers):
        scale = int(float(quantizer.scale_factors[stage]) * total_steps)
        scale = max(1, scale)
        z_e = stage_quantizer.in_proj(residual)
        inter_z = stage_quantizer.downsample_latents(z_e, scale)
        z_q_small, indices = stage_quantizer.decode_latents(inter_z)
        guide = z_q if torch.is_tensor(z_q) else None
        use_guide = scale != total_steps
        z_q_up = stage_quantizer.upsample_code(
            z_q_small,
            total_steps,
            guide=guide,
            use_guide=use_guide,
        )
        z_q_code = quantizer._apply_quant_resi(stage, z_q_up)
        contribution = stage_quantizer.out_proj(z_q_code)
        stages.append(
            {
                "stage": stage,
                "scale_factor": float(quantizer.scale_factors[stage]),
                "scale_steps": int(scale),
                "quantizer": stage_quantizer,
                "inter_z": inter_z,
                "z_q_small": z_q_small,
                "indices": indices,
                "guide": guide,
                "use_guide": bool(use_guide),
                "contribution": contribution,
            }
        )
        z_q = z_q + contribution
        residual = residual - contribution
    return z_q, stages


def topk_candidates(stage_quantizer, inter_z: torch.Tensor, pos: int, topk: int):
    query = inter_z[0, :, pos].unsqueeze(0)
    if hasattr(stage_quantizer, "codebook_distances"):
        dist = stage_quantizer.codebook_distances(query)[0]
    else:
        query = F.normalize(query, dim=1)
        codebook = F.normalize(lookup_codebook_weight(stage_quantizer), dim=1)
        dist = (codebook - query).pow(2).sum(dim=1)
    values, indices = torch.topk(dist, k=min(topk, dist.numel()), largest=False)
    return indices, values


def rank_candidate_losses(
    metric_losses: torch.Tensor,
    candidates: torch.Tensor,
    distances: torch.Tensor,
) -> dict:
    best_idx = int(torch.argmin(metric_losses).item())
    top1_loss = float(metric_losses[0].detach().cpu())
    best_loss = float(metric_losses[best_idx].detach().cpu())
    regret_abs = top1_loss - best_loss
    regret_rel = regret_abs / max(abs(top1_loss), 1e-12)
    return {
        "hit_at_1": float(best_idx == 0),
        "oracle_rank": int(best_idx),
        "top1_loss": top1_loss,
        "best_loss": best_loss,
        "regret_abs": float(regret_abs),
        "regret_rel": float(regret_rel),
        "top1_distance": float(distances[0].detach().cpu()),
        "best_distance": float(distances[best_idx].detach().cpu()),
        "top1_code": int(candidates[0].detach().cpu()),
        "best_code": int(candidates[best_idx].detach().cpu()),
    }


def decode_candidate_codes(
    model,
    full_z_q: torch.Tensor,
    stage_info: dict,
    pos: int,
    candidates: torch.Tensor,
) -> torch.Tensor:
    stage_quantizer = stage_info["quantizer"]
    n_candidates = int(candidates.numel())
    base_indices = stage_info["indices"].repeat(n_candidates, 1)
    base_indices[torch.arange(n_candidates, device=base_indices.device), pos] = candidates
    z_q_small = stage_quantizer.decode_code(base_indices)
    guide = stage_info["guide"]
    if torch.is_tensor(guide):
        guide = guide.expand(n_candidates, -1, -1)
    z_q_up = stage_quantizer.upsample_code(
        z_q_small,
        full_z_q.shape[-1],
        guide=guide,
        use_guide=stage_info["use_guide"],
    )
    z_q_code = model.quantizer._apply_quant_resi(stage_info["stage"], z_q_up)
    contribution = stage_quantizer.out_proj(z_q_code)
    full = (
        full_z_q.expand(n_candidates, -1, -1)
        - stage_info["contribution"].expand(n_candidates, -1, -1)
        + contribution
    )
    return model.decode(full)


@torch.no_grad()
def evaluate_event(
    model,
    audio: torch.Tensor,
    full_z_q: torch.Tensor,
    stage_info: dict,
    pos: int,
    topk: int,
    metrics: list[str],
    loss_modules: dict[str, torch.nn.Module],
) -> dict:
    stage_quantizer = stage_info["quantizer"]
    candidates, distances = topk_candidates(stage_quantizer, stage_info["inter_z"], pos, topk)
    n_candidates = int(candidates.numel())
    decoded = decode_candidate_codes(model, full_z_q, stage_info, pos, candidates)

    return {
        "stage": int(stage_info["stage"]),
        "scale_factor": float(stage_info["scale_factor"]),
        "scale_steps": int(stage_info["scale_steps"]),
        "token_pos": int(pos),
        "n_candidates": n_candidates,
        "metrics": {
            metric: rank_candidate_losses(
                audio_metric(decoded, audio, model.sample_rate, metric, loss_modules),
                candidates,
                distances,
            )
            for metric in metrics
        },
    }


def aggregate(rows: list[dict], metric: str) -> dict:
    metric_rows = [row["metrics"][metric] for row in rows if metric in row["metrics"]]
    if not metric_rows:
        return {}
    return {
        "n_events": len(metric_rows),
        "hit_at_1": sum(row["hit_at_1"] for row in metric_rows) / len(metric_rows),
        "mean_oracle_rank": sum(row["oracle_rank"] for row in metric_rows) / len(metric_rows),
        "mean_regret_rel": sum(row["regret_rel"] for row in metric_rows) / len(metric_rows),
        "mean_regret_abs": sum(row["regret_abs"] for row in metric_rows) / len(metric_rows),
    }


def aggregate_by_stage(rows: list[dict], metric: str) -> list[dict]:
    stages = sorted({row["stage"] for row in rows})
    out = []
    for stage in stages:
        stage_rows = [row for row in rows if row["stage"] == stage]
        summary = aggregate(stage_rows, metric)
        if not summary:
            continue
        summary["stage"] = int(stage)
        summary["scale_factor"] = float(stage_rows[0]["scale_factor"])
        out.append(summary)
    return out


def aggregate_by_stratum(rows: list[dict], metric: str) -> list[dict]:
    strata = sorted({row.get("stratum", "all") for row in rows})
    out = []
    for stratum in strata:
        stratum_rows = [
            row for row in rows if row.get("stratum", "all") == stratum
        ]
        summary = aggregate(stratum_rows, metric)
        if not summary:
            continue
        summary["stratum"] = stratum
        out.append(summary)
    return out


@torch.no_grad()
def evaluate_model(model, files: list[Path], file_strata: dict[str, str], args, loss_modules):
    model.eval()
    rows = []
    for file_idx, path in enumerate(files):
        print(f"[probe] file {file_idx + 1}/{len(files)} {path}", flush=True)
        audio = load_one(
            path,
            model.sample_rate,
            args.duration,
            args.device,
            crop_mode=args.crop_mode,
            crop_seed=args.seed + file_idx,
        )
        full_z_q, stages = encode_stage_cache(model, audio)
        selected_stages = range(len(stages)) if args.stages == "all" else [int(x) for x in args.stages.split(",")]
        for stage in selected_stages:
            stage_info = stages[stage]
            positions = select_token_positions(
                stage_info["indices"].shape[-1],
                args.positions_per_stage,
                args.position_mode,
                args.seed + 100000 * file_idx + stage,
            )
            for pos in positions:
                row = evaluate_event(
                    model,
                    audio,
                    full_z_q,
                    stage_info,
                    pos,
                    args.topk,
                    args.metrics,
                    loss_modules,
                )
                row["file"] = str(path)
                row["file_index"] = int(file_idx)
                row["stratum"] = file_strata[str(path)]
                rows.append(row)
    return {
        "summary": {metric: aggregate(rows, metric) for metric in args.metrics},
        "by_stage": {metric: aggregate_by_stage(rows, metric) for metric in args.metrics},
        "by_stratum": {
            metric: aggregate_by_stratum(rows, metric)
            for metric in args.metrics
        },
        "events": rows,
    }


def print_summary(results: dict):
    for name, data in results["models"].items():
        for metric, summary in data["summary"].items():
            print(
                f"[{name}:{metric}] events={summary.get('n_events', 0)} "
                f"hit@1={summary.get('hit_at_1', 0.0):.3f} "
                f"rank={summary.get('mean_oracle_rank', 0.0):.3f} "
                f"regret_abs={summary.get('mean_regret_abs', 0.0):.4g} "
                f"regret_rel={summary.get('mean_regret_rel', 0.0):.4f}"
            )
            worst = sorted(data["by_stage"][metric], key=lambda row: row["mean_regret_rel"], reverse=True)[:5]
            text = " ".join(
                f"s{row['stage']}({row['scale_factor']:.2f}):hit={row['hit_at_1']:.2f},reg={row['mean_regret_rel']:.3g}"
                for row in worst
            )
            print(f"  worst {text}")
            strata = data.get("by_stratum", {}).get(metric, [])
            if strata:
                text = " ".join(
                    f"{row['stratum']}:hit={row['hit_at_1']:.2f},"
                    f"rank={row['mean_oracle_rank']:.2f},"
                    f"reg={row['mean_regret_abs']:.3g}"
                    for row in strata
                )
                print(f"  strata {text}")


def main():
    parser = argparse.ArgumentParser(description="Probe Euclidean VQ lookup vs decoder-oracle top-k code selection.")
    parser.add_argument("--model", action="append", required=True, help="name=/path/to/run_or_weights.pth")
    parser.add_argument("--dirs", nargs="*", default=[])
    parser.add_argument("--n_files", type=int, default=4)
    parser.add_argument(
        "--stratum",
        action="append",
        default=None,
        help="Balanced sampling group in name=/path form; repeat paths/groups as needed.",
    )
    parser.add_argument("--n_files_per_stratum", type=int, default=0)
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument("--crop_mode", choices=["center", "random"], default="center")
    parser.add_argument("--positions_per_stage", type=int, default=2)
    parser.add_argument(
        "--position_mode",
        choices=["evenly_spaced", "random"],
        default="evenly_spaced",
    )
    parser.add_argument("--topk", type=int, default=8)
    metric_choices = ["waveform", "mel", "stft", "sisdr"]
    parser.add_argument("--metric", choices=metric_choices, default=None)
    parser.add_argument("--metrics", nargs="+", choices=metric_choices, default=None)
    parser.add_argument("--stages", default="all", help="'all' or comma-separated stage indices")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", default="runs/analysis/codec_lookup_alignment_probe.json")
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    args.metrics = args.metrics or ([args.metric] if args.metric else ["waveform"])

    if args.stratum and args.n_files_per_stratum <= 0:
        raise ValueError("--n_files_per_stratum must be positive when --stratum is used.")
    files, file_strata = select_audio_files(
        args.dirs,
        args.n_files,
        args.stratum,
        args.n_files_per_stratum,
        args.seed,
    )
    if not files:
        raise RuntimeError("No audio files found")

    loss_modules = make_loss_modules(args.device)
    results = {
        "settings": {
            "dirs": args.dirs,
            "strata": parse_strata_specs(args.stratum),
            "n_files": len(files),
            "n_files_per_stratum": args.n_files_per_stratum,
            "duration": args.duration,
            "crop_mode": args.crop_mode,
            "positions_per_stage": args.positions_per_stage,
            "position_mode": args.position_mode,
            "topk": args.topk,
            "metrics": args.metrics,
            "stages": args.stages,
            "device": args.device,
            "seed": args.seed,
        },
        "files": [str(path) for path in files],
        "file_strata": file_strata,
        "models": {},
    }

    for spec in args.model:
        name, raw_path = parse_model_spec(spec)
        checkpoint = resolve_checkpoint(raw_path.expanduser())
        print(f"[probe] loading {name}: {checkpoint}", flush=True)
        model = load_model(load_path=str(checkpoint)).to(args.device).eval()
        results["models"][name] = {
            "checkpoint": str(checkpoint),
            **evaluate_model(model, files, file_strata, args, loss_modules),
        }
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print_summary(results)
    print(f"[probe] wrote {output}")


if __name__ == "__main__":
    main()
