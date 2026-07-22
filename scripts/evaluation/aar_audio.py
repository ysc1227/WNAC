"""End-to-end AAR paper-style evaluation helper.

This script optionally generates paired reference/generated wavs from an AAR
checkpoint and then computes the folder-level metrics used in the FAD/ISc/KL
evaluation flow:

  1. FAD via ``scripts.metrics.fad.compute_fad_torchvggish``
  2. Inception Score and paired KL via CNN14 logits from ``scripts.metrics.isc_kl``

The expected folder layout is compatible with ``scripts/data/generate_aar_pairs.py``::

    OUT_DIR/
      ref/sample_00000.wav
      gen/sample_00000.wav
      metrics.json

If ``--skip_generate`` is set, ``--out/ref`` and ``--out/gen`` must already
exist and contain matching basenames.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from scripts.metrics.fad import compute_fad_torchvggish
from scripts.metrics.isc_kl import infer_logits, inception_score, load_cnn14, paired_kl, wavs


def run_generate(args):
    cmd = [
        sys.executable,
        "-m",
        "scripts.data.generate_aar_pairs",
        "--ckpt_folder",
        args.ckpt_folder,
        "--manifest",
        args.manifest,
        "--out",
        str(args.out),
        "--limit",
        str(args.limit),
        "--seed",
        str(args.seed),
        "--cfg",
        str(args.cfg),
        "--top_k",
        str(args.top_k),
        "--top_p",
        str(args.top_p),
    ]
    if args.excerpt_duration is not None:
        cmd += ["--excerpt_duration", str(args.excerpt_duration)]
    if args.target_duration is not None:
        cmd += ["--target_duration", str(args.target_duration)]
    cmd += [
        "--chunk_prev_condition_weight",
        str(args.chunk_prev_condition_weight),
        "--chunk_position_weight",
        str(args.chunk_position_weight),
    ]
    print("[generate]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def compute_isc_kl(ref_dir, gen_dir, device, sr, batch_size, splits, max_samples):
    ref = wavs(ref_dir)
    gen = wavs(gen_dir)
    keys = sorted(set(ref) & set(gen))
    if max_samples is not None:
        keys = keys[:max_samples]
    if not keys:
        raise RuntimeError(f"No paired files with matching basenames: ref={ref_dir}, gen={gen_dir}")

    model = load_cnn14(device)
    gen_logits = infer_logits([gen[k] for k in keys], model, sr, device, batch_size)
    ref_logits = infer_logits([ref[k] for k in keys], model, sr, device, batch_size)

    isc_mean, isc_std = inception_score(gen_logits, splits)
    kl_softmax, kl_sigmoid = paired_kl(gen_logits, ref_logits)
    return {
        "inception_score_mean": isc_mean,
        "inception_score_std": isc_std,
        "kullback_leibler_divergence_softmax": kl_softmax,
        "kullback_leibler_divergence_sigmoid": kl_sigmoid,
        "num_pairs": len(keys),
        "protocol": "AudioLDM-style: CNN14 logits, IS=softmax logits, KL=paired target||gen",
    }


def main():
    p = argparse.ArgumentParser(description="Generate AAR eval pairs and compute FAD/ISc/KL.")

    # Generation options, mirrored from scripts/data/generate_aar_pairs.py
    p.add_argument("--ckpt_folder", default=None, help="AAR checkpoint folder; required unless --skip_generate")
    p.add_argument("--manifest", default="samples/audioset_val_manifest.csv")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument(
        "--excerpt_duration",
        type=float,
        default=None,
        help="Crop reference first and use the same excerpt for CLAP conditioning/generation.",
    )
    p.add_argument(
        "--target_duration",
        type=float,
        default=None,
        help="Generated duration in seconds. Uses chunked AAR when longer than one RVQ frame window.",
    )
    p.add_argument("--chunk_prev_condition_weight", type=float, default=0.5)
    p.add_argument("--chunk_position_weight", type=float, default=0.1)
    p.add_argument("--skip_generate", action="store_true", help="reuse existing OUT/ref and OUT/gen")

    # Metric options
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sr", type=int, default=32000, help="CNN14 evaluation sample rate")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--splits", type=int, default=10)
    p.add_argument("--max_samples", type=int, default=None, help="metric max paired files")
    p.add_argument("--no_fad", action="store_true")
    p.add_argument("--no_isc_kl", action="store_true")
    p.add_argument("--json_path", default=None, help="defaults to OUT/metrics.json")
    args = p.parse_args()

    if not args.skip_generate:
        if not args.ckpt_folder:
            raise SystemExit("--ckpt_folder is required unless --skip_generate is used")
        run_generate(args)

    ref_dir = args.out / "ref"
    gen_dir = args.out / "gen"
    if not ref_dir.is_dir() or not gen_dir.is_dir():
        raise SystemExit(f"Expected paired folders not found: {ref_dir} and {gen_dir}")

    result = {
        "ref_dir": str(ref_dir),
        "gen_dir": str(gen_dir),
        "limit": args.limit,
        "seed": args.seed,
        "excerpt_duration": args.excerpt_duration,
        "target_duration": args.target_duration,
        "chunk_prev_condition_weight": args.chunk_prev_condition_weight,
        "chunk_position_weight": args.chunk_position_weight,
    }

    if not args.no_fad:
        result["fad"] = compute_fad_torchvggish(
            recon_dir=str(gen_dir),
            org_dir=str(ref_dir),
            device=args.device,
            max_files=args.max_samples,
        )

    if not args.no_isc_kl:
        result["isc_kl"] = compute_isc_kl(
            ref_dir=ref_dir,
            gen_dir=gen_dir,
            device=args.device,
            sr=args.sr,
            batch_size=args.batch_size,
            splits=args.splits,
            max_samples=args.max_samples,
        )

    text = json.dumps(result, indent=2)
    print(text)
    json_path = Path(args.json_path) if args.json_path else args.out / "metrics.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(text + "\n")


if __name__ == "__main__":
    main()
