"""Folder-level FAD evaluation for codec reconstructions.

This mirrors the codec/ViSQOL evaluation launchers: match reference and
reconstructed files by relative path, embed the matched set with VGGish, then
compute a Frechet distance between the two embedding distributions.
"""

import csv
import json
import time
from pathlib import Path

import argbind
import numpy as np
import torch
from audiotools.core import util
from scipy import linalg


def _compute_stats(x: np.ndarray):
    mu = np.mean(x, axis=0)
    sigma = np.cov(x, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Standard FID/FAD Frechet distance between two Gaussians."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    eye = np.eye(sigma1.shape[0])
    covmean, _ = linalg.sqrtm((sigma1 + eps * eye) @ (sigma2 + eps * eye), disp=False)

    if not np.isfinite(covmean).all():
        offset = (10 * eps) * eye
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def _match_pairs(input_root: Path, recons_root: Path, max_pairs: int = None):
    audio_files = util.find_audio(input_root)
    if len(audio_files) == 0:
        raise RuntimeError(f"No audio files found in input path: {input_root}")

    pairs = []
    missing = []
    for audio_file in audio_files:
        try:
            rel = audio_file.relative_to(input_root)
        except ValueError:
            rel = Path(audio_file.name)

        recons_path = recons_root / rel
        if recons_path.exists():
            pairs.append((audio_file, recons_path))
            continue

        flat_recons_path = recons_root / audio_file.name
        if flat_recons_path.exists():
            pairs.append((audio_file, flat_recons_path))
        else:
            missing.append((audio_file, recons_path))

    if not pairs:
        examples = "\n".join(f"  {src} -> expected {dst}" for src, dst in missing[:10])
        raise RuntimeError(
            f"No matching reconstructed files found under {recons_root}. "
            f"Expected files with the same relative paths as {input_root}.\n{examples}"
        )

    if max_pairs is not None:
        max_pairs = int(max_pairs)
        if max_pairs > 0:
            pairs = pairs[:max_pairs]

    return pairs, missing


def _load_vggish(device: str):
    try:
        from torchvggish import vggish, vggish_input
    except ImportError as exc:
        raise ImportError(
            "torchvggish is not installed. Install it before running FAD:\n"
            "  python -m pip install torchvggish==0.2"
        ) from exc

    model = vggish(postprocess=False).to(device).eval()
    return model, vggish_input


@torch.no_grad()
def _embed_file(path: Path, model, vggish_input, device: str, batch_size: int):
    examples = vggish_input.wavfile_to_examples(str(path))
    if examples is None or examples.numel() == 0:
        raise RuntimeError("VGGish produced no examples")

    outs = []
    batch_size = max(1, int(batch_size))
    for chunk in examples.split(batch_size, dim=0):
        chunk = chunk.to(device, non_blocking=True)
        outs.append(model(chunk).detach().float().cpu())

    emb = torch.cat(outs, dim=0).mean(dim=0)
    if not torch.isfinite(emb).all():
        raise RuntimeError("VGGish embedding contains non-finite values")
    return emb.numpy()


@argbind.bind(without_prefix=True)
@torch.no_grad()
def evaluate_fad(
    input: str = "samples/input",
    output: str = "samples/output",
    device: str = "cuda",
    batch_size: int = 16,
    max_pairs: int = None,
    json_path: str = None,
    csv_path: str = None,
):
    """Compute VGGish FAD for matched reference/reconstruction folders."""
    input_root = Path(input)
    recons_root = Path(output)
    recons_root.mkdir(parents=True, exist_ok=True)

    if device == "cuda" and not torch.cuda.is_available():
        print("[fad] CUDA requested but unavailable; falling back to CPU", flush=True)
        device = "cpu"

    pairs, missing = _match_pairs(input_root, recons_root, max_pairs=max_pairs)
    total = len(pairs)
    json_path = Path(json_path) if json_path else recons_root / "fad.json"
    csv_path = Path(csv_path) if csv_path else recons_root / "fad_files.csv"

    if missing:
        print(f"[WARN] Skipping {len(missing)} input files with no matching reconstruction.")

    print(f"[fad] Starting {total} matched pairs  |  device={device}  |  batch_size={batch_size}")
    print(f"[fad] JSON will be saved to {json_path}")
    print(f"[fad] File list will be saved to {csv_path}")

    model, vggish_input = _load_vggish(device)
    ref_embeddings = []
    gen_embeddings = []
    rows = []
    skipped = []
    t0 = time.time()

    for i, (ref_path, gen_path) in enumerate(pairs, 1):
        try:
            ref_emb = _embed_file(ref_path, model, vggish_input, device, batch_size)
            gen_emb = _embed_file(gen_path, model, vggish_input, device, batch_size)
        except Exception as exc:
            skipped.append((ref_path, gen_path, type(exc).__name__, str(exc)))
            rows.append(
                {
                    "path": str(ref_path),
                    "recons_path": str(gen_path),
                    "used": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[fad] [{i}/{total}] SKIP {ref_path.name}: {type(exc).__name__}: {exc}", flush=True)
            continue

        ref_embeddings.append(ref_emb)
        gen_embeddings.append(gen_emb)
        rows.append({"path": str(ref_path), "recons_path": str(gen_path), "used": True, "error": ""})

        elapsed = time.time() - t0
        eta = (elapsed / i) * (total - i)
        print(
            f"[fad] [{i}/{total}] embedded  "
            f"({elapsed:.0f}s elapsed, ~{eta:.0f}s left)  {ref_path.name}",
            flush=True,
        )

    if len(ref_embeddings) < 2 or len(gen_embeddings) < 2:
        raise RuntimeError(
            f"Need >=2 valid pairs for FAD, got ref={len(ref_embeddings)} gen={len(gen_embeddings)}"
        )

    x_ref = np.stack(ref_embeddings, axis=0)
    x_gen = np.stack(gen_embeddings, axis=0)
    mu_ref, sigma_ref = _compute_stats(x_ref)
    mu_gen, sigma_gen = _compute_stats(x_gen)
    fad = frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)

    result = {
        "FAD": fad,
        "N_real": int(x_ref.shape[0]),
        "N_gen": int(x_gen.shape[0]),
        "N_matched": int(total),
        "N_skipped_embedding": int(len(skipped)),
        "N_missing_reconstruction": int(len(missing)),
        "D": int(x_ref.shape[1]),
        "input_dir": str(input_root),
        "output_dir": str(recons_root),
        "device": device,
        "batch_size": int(batch_size),
        "max_pairs": None if max_pairs is None else int(max_pairs),
        "protocol": "torchvggish postprocess=False embeddings, matched reference/reconstruction folders",
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "recons_path", "used", "error"])
        writer.writeheader()
        writer.writerows(rows)

    elapsed_total = time.time() - t0
    print(f"\n[fad] Done  |  FAD={fad:.6f}  |  valid={x_ref.shape[0]}/{total}  |  {elapsed_total:.1f}s total")
    print(f"[fad] Results saved to {json_path}")
    print(f"[fad] File list saved to {csv_path}")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        evaluate_fad()
