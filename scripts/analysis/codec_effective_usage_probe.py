import argparse
import json
import random
from pathlib import Path

import torch

from codec_lookup_alignment_probe import (
    aggregate_by_stage,
    audio_metric,
    decode_candidate_codes,
    encode_stage_cache,
    find_audio_files,
    load_one,
    make_loss_modules,
    parse_model_spec,
    resolve_checkpoint,
    resolve_device,
    sample_files,
    select_token_positions,
    topk_candidates,
)
from emac.utils import load_model


def tensor_mean(x: torch.Tensor) -> float:
    return float(x.detach().float().mean().cpu())


def candidate_codes(stage_info: dict, pos: int, topk: int, rng: random.Random) -> dict[str, torch.Tensor]:
    actual = stage_info["indices"][0, pos].detach().long()
    nearest, _ = topk_candidates(stage_info["quantizer"], stage_info["inter_z"], pos, topk)
    nearest_alt = None
    for code in nearest:
        if int(code.detach().cpu()) != int(actual.detach().cpu()):
            nearest_alt = code.detach().long()
            break
    if nearest_alt is None:
        nearest_alt = (actual + 1) % int(stage_info["quantizer"].codebook_size)

    codebook_size = int(stage_info["quantizer"].codebook_size)
    random_alt_int = rng.randrange(codebook_size - 1)
    if random_alt_int >= int(actual.detach().cpu()):
        random_alt_int += 1
    random_alt = torch.tensor(random_alt_int, device=actual.device, dtype=actual.dtype)

    return {
        "actual": actual.reshape(1),
        "nearest": nearest_alt.reshape(1),
        "random": random_alt.reshape(1),
    }


@torch.no_grad()
def evaluate_event(
    model,
    audio: torch.Tensor,
    full_z_q: torch.Tensor,
    decoded_ref: torch.Tensor,
    ref_losses: dict[str, float],
    stage_info: dict,
    pos: int,
    args,
    loss_modules,
    rng: random.Random,
) -> dict:
    codes = candidate_codes(stage_info, pos, args.topk, rng)
    row = {
        "stage": int(stage_info["stage"]),
        "scale_factor": float(stage_info["scale_factor"]),
        "scale_steps": int(stage_info["scale_steps"]),
        "token_pos": int(pos),
        "actual_code": int(codes["actual"].item()),
        "alternatives": {},
    }

    for name in ("nearest", "random"):
        decoded = decode_candidate_codes(model, full_z_q, stage_info, pos, codes[name])
        decoded = decoded[..., : decoded_ref.shape[-1]]
        diff = decoded - decoded_ref
        alt = {
            "code": int(codes[name].item()),
            "waveform_delta_l1": tensor_mean(diff.abs()),
            "waveform_delta_l2": tensor_mean(diff.pow(2)).__float__(),
            "relative_waveform_delta_l1": tensor_mean(diff.abs())
            / max(tensor_mean(decoded_ref.abs()), 1e-12),
        }
        for metric in args.metrics:
            loss = tensor_mean(audio_metric(decoded, audio, model.sample_rate, metric, loss_modules))
            alt[f"{metric}_loss"] = loss
            alt[f"{metric}_loss_delta"] = loss - ref_losses[metric]
            alt[f"{metric}_loss_delta_rel"] = (loss - ref_losses[metric]) / max(abs(ref_losses[metric]), 1e-12)

        # Decoder-facing latent sensitivity for the same token replacement.
        stage_quantizer = stage_info["quantizer"]
        base_indices = stage_info["indices"].clone()
        base_indices[0, pos] = codes[name][0]
        z_q_small = stage_quantizer.decode_code(base_indices)
        guide = stage_info["guide"]
        z_q_up = stage_quantizer.upsample_code(
            z_q_small,
            full_z_q.shape[-1],
            guide=guide,
            use_guide=stage_info["use_guide"],
        )
        z_q_code = model.quantizer._apply_quant_resi(stage_info["stage"], z_q_up)
        contribution = stage_quantizer.out_proj(z_q_code)
        latent_delta = contribution - stage_info["contribution"]
        alt["latent_delta_l2"] = tensor_mean(latent_delta.pow(2))
        alt["relative_latent_delta_l2"] = tensor_mean(latent_delta.pow(2)) / max(
            tensor_mean(stage_info["contribution"].pow(2)),
            1e-12,
        )
        row["alternatives"][name] = alt
    return row


def aggregate(rows: list[dict], alt_name: str) -> dict:
    selected = [row["alternatives"][alt_name] for row in rows]
    if not selected:
        return {}
    keys = [key for key, value in selected[0].items() if isinstance(value, (int, float)) and key != "code"]
    return {
        "n_events": len(selected),
        **{key: sum(float(row[key]) for row in selected) / len(selected) for key in keys},
    }


def aggregate_by_stage_effective(rows: list[dict], alt_name: str) -> list[dict]:
    out = []
    for stage in sorted({row["stage"] for row in rows}):
        stage_rows = [row for row in rows if row["stage"] == stage]
        summary = aggregate(stage_rows, alt_name)
        summary["stage"] = int(stage)
        summary["scale_factor"] = float(stage_rows[0]["scale_factor"])
        out.append(summary)
    return out


@torch.no_grad()
def evaluate_model(model, files: list[Path], args, loss_modules):
    rows = []
    model.eval()
    for file_idx, path in enumerate(files):
        print(f"[effective] file {file_idx + 1}/{len(files)} {path}", flush=True)
        audio = load_one(
            path,
            model.sample_rate,
            args.duration,
            args.device,
            crop_mode=args.crop_mode,
            crop_seed=args.seed + file_idx,
        )
        full_z_q, stages = encode_stage_cache(model, audio)
        decoded_ref = model.decode(full_z_q)[..., : audio.shape[-1]]
        ref_losses = {
            metric: tensor_mean(audio_metric(decoded_ref, audio, model.sample_rate, metric, loss_modules))
            for metric in args.metrics
        }
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
                rng = random.Random(f"{args.seed}:{file_idx}:{stage}:{pos}")
                rows.append(
                    evaluate_event(
                        model,
                        audio,
                        full_z_q,
                        decoded_ref,
                        ref_losses,
                        stage_info,
                        pos,
                        args,
                        loss_modules,
                        rng,
                    )
                )
    return {
        "summary": {
            "nearest": aggregate(rows, "nearest"),
            "random": aggregate(rows, "random"),
        },
        "by_stage": {
            "nearest": aggregate_by_stage_effective(rows, "nearest"),
            "random": aggregate_by_stage_effective(rows, "random"),
        },
        "events": rows,
    }


def print_summary(results: dict):
    for name, data in results["models"].items():
        print(f"[{name}]")
        for alt_name, summary in data["summary"].items():
            print(
                f"  {alt_name}: events={summary.get('n_events', 0)} "
                f"wav_delta={summary.get('waveform_delta_l1', 0.0):.4g} "
                f"rel_wav_delta={summary.get('relative_waveform_delta_l1', 0.0):.4g} "
                f"latent_delta={summary.get('latent_delta_l2', 0.0):.4g} "
                f"mel_dloss={summary.get('mel_loss_delta', 0.0):.4g} "
                f"stft_dloss={summary.get('stft_loss_delta', 0.0):.4g}"
            )


def main():
    parser = argparse.ArgumentParser(description="Probe decoder-effective code usage by token replacement sensitivity.")
    parser.add_argument("--model", action="append", required=True, help="name=/path/to/run_or_weights.pth")
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--n_files", type=int, default=32)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--crop_mode", choices=["center", "random"], default="center")
    parser.add_argument("--positions_per_stage", type=int, default=4)
    parser.add_argument("--position_mode", choices=["evenly_spaced", "random"], default="random")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--metrics", nargs="+", default=["waveform", "mel", "stft"], choices=["waveform", "mel", "stft", "sisdr"])
    parser.add_argument("--stages", default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output_json", default="runs/analysis/codec_effective_usage_probe.json")
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    files = sample_files(find_audio_files(args.dirs), args.n_files, args.seed)
    if not files:
        raise RuntimeError("No audio files found")
    loss_modules = make_loss_modules(args.device)

    results = {
        "settings": {
            "dirs": args.dirs,
            "n_files": len(files),
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
        "models": {},
    }
    for spec in args.model:
        name, raw_path = parse_model_spec(spec)
        checkpoint = resolve_checkpoint(raw_path.expanduser())
        print(f"[effective] loading {name}: {checkpoint}", flush=True)
        model = load_model(load_path=str(checkpoint)).to(args.device).eval()
        results["models"][name] = {
            "checkpoint": str(checkpoint),
            **evaluate_model(model, files, args, loss_modules),
        }
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print_summary(results)
    print(f"[effective] wrote {output}")


if __name__ == "__main__":
    main()
