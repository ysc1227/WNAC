import argparse
import csv
import json
from pathlib import Path

import torch

from codec_stage_frequency_probe import (
    add_stat,
    chunks,
    decode_latents,
    finalize_stat,
    find_audio_files,
    load_batch,
    load_model,
    parse_model_spec,
    resolve_checkpoint,
    resolve_device,
    rms,
    sample_files,
    stage_latents,
    stft_band_energy,
)


def make_blocks(n_stages: int, n_blocks: int) -> list[dict]:
    blocks = [
        {
            "block": idx,
            "progress_low": idx / n_blocks,
            "progress_high": (idx + 1) / n_blocks,
            "progress_center": (idx + 0.5) / n_blocks,
            "stage_indices": [],
        }
        for idx in range(n_blocks)
    ]
    denom = max(1, n_stages - 1)
    for stage in range(n_stages):
        progress = stage / denom
        block_idx = min(n_blocks - 1, int(progress * n_blocks))
        blocks[block_idx]["stage_indices"].append(stage)
    return blocks


def parse_bands(sample_rate: int) -> list[tuple[float, float]]:
    nyquist = float(sample_rate) / 2.0
    edges = [0.0, nyquist]
    return [(edges[0], edges[1])]


def block_label(row: dict) -> str:
    low = int(round(row["progress_low"] * 100))
    high = int(round(row["progress_high"] * 100))
    return f"{low}-{high}%"


def empty_block_row(block: dict) -> dict:
    return {
        "block": int(block["block"]),
        "label": block_label(block),
        "progress_low": float(block["progress_low"]),
        "progress_high": float(block["progress_high"]),
        "progress_center": float(block["progress_center"]),
        "stage_indices": [],
        "stage_start": None,
        "stage_end": None,
        "num_stages": 0,
        "block_audio_rms": None,
        "prefix_error_before_total_norm": None,
        "prefix_error_after_total_norm": None,
        "prefix_error_drop_total_norm": None,
        "full_error_total_norm": None,
        "omit_error_total_norm": None,
        "omit_extra_total_norm": None,
    }


@torch.no_grad()
def probe_model(model, files: list[Path], args) -> dict:
    model.eval()
    n_stages = len(model.quantizer.quantizers)
    blocks = make_blocks(n_stages, args.n_blocks)
    nonempty_blocks = [block for block in blocks if block["stage_indices"]]
    block_accum = [dict() for _ in blocks]
    eps = 1e-8

    bands = parse_bands(model.sample_rate)

    for batch_idx, batch_files in enumerate(chunks(files, args.batch_size)):
        print(
            f"[block-probe] batch {batch_idx + 1}/{(len(files) + args.batch_size - 1) // args.batch_size}",
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

        prefix_latents = []
        omitted_latents = []
        for block in nonempty_blocks:
            indices = block["stage_indices"]
            start = indices[0]
            end = indices[-1] + 1
            prefix_latents.extend([prefixes[start], prefixes[end]])

            block_contribution = torch.zeros_like(full_latent)
            for stage in indices:
                block_contribution = block_contribution + stages[stage]["contribution"]
            omitted_latents.append(full_latent - block_contribution)

        decoded = decode_latents(
            model,
            [full_latent, *prefix_latents, *omitted_latents],
            args.decode_batch_size,
        )
        decoded = [item[..., : audio.shape[-1]] for item in decoded]
        full_audio = decoded[0]
        prefix_audio = decoded[1 : 1 + 2 * len(nonempty_blocks)]
        omitted_audio = decoded[1 + 2 * len(nonempty_blocks) :]

        full_err_band = stft_band_energy(
            full_audio - audio,
            model.sample_rate,
            bands,
            args.n_fft,
            args.hop_length,
        )
        full_err_total_norm = full_err_band.sum(dim=1) / target_total

        for decoded_idx, block in enumerate(nonempty_blocks):
            block_idx = int(block["block"])
            start_audio = prefix_audio[2 * decoded_idx]
            end_audio = prefix_audio[2 * decoded_idx + 1]
            block_audio = end_audio - start_audio

            before_err_band = stft_band_energy(
                start_audio - audio,
                model.sample_rate,
                bands,
                args.n_fft,
                args.hop_length,
            )
            after_err_band = stft_band_energy(
                end_audio - audio,
                model.sample_rate,
                bands,
                args.n_fft,
                args.hop_length,
            )
            omit_err_band = stft_band_energy(
                omitted_audio[decoded_idx] - audio,
                model.sample_rate,
                bands,
                args.n_fft,
                args.hop_length,
            )

            before_err_total_norm = before_err_band.sum(dim=1) / target_total
            after_err_total_norm = after_err_band.sum(dim=1) / target_total
            omit_err_total_norm = omit_err_band.sum(dim=1) / target_total

            accum = block_accum[block_idx]
            add_stat(accum, "block_audio_rms", rms(block_audio))
            add_stat(accum, "prefix_error_before_total_norm", before_err_total_norm)
            add_stat(accum, "prefix_error_after_total_norm", after_err_total_norm)
            add_stat(accum, "prefix_error_drop_total_norm", before_err_total_norm - after_err_total_norm)
            add_stat(accum, "full_error_total_norm", full_err_total_norm)
            add_stat(accum, "omit_error_total_norm", omit_err_total_norm)
            add_stat(accum, "omit_extra_total_norm", omit_err_total_norm - full_err_total_norm)

    rows = []
    for block, accum in zip(blocks, block_accum):
        if not block["stage_indices"]:
            rows.append(empty_block_row(block))
            continue
        row = {
            "block": int(block["block"]),
            "label": block_label(block),
            "progress_low": float(block["progress_low"]),
            "progress_high": float(block["progress_high"]),
            "progress_center": float(block["progress_center"]),
            "stage_indices": [int(stage) for stage in block["stage_indices"]],
            "stage_start": int(block["stage_indices"][0]),
            "stage_end": int(block["stage_indices"][-1]),
            "num_stages": int(len(block["stage_indices"])),
        }
        for key in (
            "block_audio_rms",
            "prefix_error_before_total_norm",
            "prefix_error_after_total_norm",
            "prefix_error_drop_total_norm",
            "full_error_total_norm",
            "omit_error_total_norm",
            "omit_extra_total_norm",
        ):
            row[key] = finalize_stat(accum, key)
        rows.append(row)
    return {"blocks": rows}


def write_csv(path: Path, results: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name, data in results["models"].items():
        for row in data["blocks"]:
            csv_row = {key: value for key, value in row.items() if key != "stage_indices"}
            csv_row["stage_indices"] = ",".join(str(stage) for stage in row["stage_indices"])
            rows.append({"model": model_name, **csv_row})
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(results: dict):
    for name, data in results["models"].items():
        print(f"\n[{name}]")
        for row in data["blocks"]:
            if row["num_stages"] == 0:
                print(f"{row['label']} stages=empty")
                continue
            print(
                f"{row['label']} stages={row['stage_start']}-{row['stage_end']} "
                f"contrib={row['block_audio_rms']:.4g} "
                f"prefix_drop={row['prefix_error_drop_total_norm']:.4g} "
                f"loo={row['omit_extra_total_norm']:.4g}"
            )


def main():
    parser = argparse.ArgumentParser(description="Block-wise RVQ contribution and leave-one-out probe.")
    parser.add_argument("--model", action="append", required=True, help="name=/path/to/run_or_weights.pth")
    parser.add_argument("--audio", type=Path, default=Path("eval_set/general"))
    parser.add_argument("--n-files", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--n-blocks", type=int, default=5)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("runs/analysis/codec_stage_block_probe.json"),
    )
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
            "n_blocks": args.n_blocks,
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
        print(f"[block-probe] loading {name}: {checkpoint}", flush=True)
        model = load_model(load_path=str(checkpoint)).to(args.device).eval()
        results["models"][name] = {
            "checkpoint": str(checkpoint),
            **probe_model(model, files, args),
        }
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.output_csv is not None:
        write_csv(args.output_csv, results)
    print_summary(results)
    print(f"\n[block-probe] wrote {args.output_json}")
    if args.output_csv is not None:
        print(f"[block-probe] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
