"""AAR/torcheval-style VGGish FAD for codec reconstructions.

This uses TorchAudio's VGGish implementation with the final ReLU removed,
matching the feature extractor used by ``torcheval.metrics.FrechetAudioDistance
.with_vggish()``.  Embeddings are accumulated at the 0.96 s VGGish patch level.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torchaudio
from audiotools.core import util
from scipy import linalg

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

from torchaudio.prototype.pipelines._vggish._vggish_impl import (  # noqa: E402
    VGGish,
    VGGishInputProcessor,
    _SAMPLE_RATE,
)


def _match_pairs(input_root: Path, recons_root: Path, max_pairs: int | None):
    audio_files = util.find_audio(input_root)
    if max_pairs:
        audio_files = audio_files[: int(max_pairs)]

    pairs = []
    missing = []
    for audio_file in audio_files:
        rel = audio_file.relative_to(input_root)
        recons_path = recons_root / rel.with_suffix(".wav")
        flat_recons_path = recons_root / audio_file.with_suffix(".wav").name
        if recons_path.exists():
            pairs.append((Path(audio_file), recons_path))
        elif flat_recons_path.exists():
            pairs.append((Path(audio_file), flat_recons_path))
        else:
            missing.append(str(audio_file))

    if not pairs:
        raise RuntimeError(f"No matched wav pairs for {input_root} -> {recons_root}")
    return pairs, missing


def _candidate_weight_paths() -> list[Path]:
    paths = []
    env_path = os.getenv("VGGISH_WEIGHTS")
    if env_path:
        paths.append(Path(env_path).expanduser())
    paths.extend(
        [
            Path(torch.hub.get_dir()) / "torchaudio" / "models" / "vggish.pt",
            Path.home() / ".cache" / "torch" / "hub" / "torchaudio" / "models" / "vggish.pt",
            Path("/home/seungchan/.cache/torch/hub/torchaudio/models/vggish.pt"),
            Path("/root/.cache/torch/hub/torchaudio/models/vggish.pt"),
            Path("/u/home/.cache/torch/hub/torchaudio/models/vggish.pt"),
        ]
    )
    return paths


def _resolve_weights_path(weights_path: str | None) -> Path:
    candidates = [Path(weights_path).expanduser()] if weights_path else []
    candidates.extend(_candidate_weight_paths())
    for path in candidates:
        if path.exists():
            return path

    if weights_path:
        raise FileNotFoundError(f"VGGish weights not found: {weights_path}")
    return Path(torchaudio.utils.download_asset("models/vggish.pt"))


def _load_model(device: str, weights_path: str | None):
    model = VGGish().to(device).eval()
    weights_path = _resolve_weights_path(weights_path)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.embedding_network = torch.nn.Sequential(
        *list(model.embedding_network.children())[:-1]
    ).to(device).eval()
    return model, VGGishInputProcessor(), weights_path


def _embed_file(path: Path, model, processor, device: str, batch_size: int):
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0)
    if sr != _SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, _SAMPLE_RATE)
    examples = processor(wav)
    if examples.numel() == 0:
        raise RuntimeError(f"VGGish produced no examples for {path}")

    outs = []
    with torch.no_grad():
        for chunk in examples.split(batch_size, dim=0):
            outs.append(model(chunk.to(device)).detach().float().cpu())
    return torch.cat(outs, dim=0).numpy()


def _fingerprint(paths: list[Path], max_pairs: int | None):
    h = hashlib.sha1()
    h.update(str(max_pairs).encode())
    for path in paths:
        h.update(str(path).encode())
        try:
            stat = path.stat()
            h.update(str(stat.st_mtime_ns).encode())
            h.update(str(stat.st_size).encode())
        except FileNotFoundError:
            h.update(b"missing")
    return h.hexdigest()[:16]


def _load_or_embed(
    paths: list[Path],
    cache_path: Path,
    model,
    processor,
    device: str,
    batch_size: int,
    label: str,
):
    if cache_path.exists():
        data = np.load(cache_path)
        return data["embeddings"], int(data["n_files"])

    embeddings = []
    t0 = time.time()
    for i, path in enumerate(paths, 1):
        embeddings.append(_embed_file(path, model, processor, device, batch_size))
        if i == 1 or i % 250 == 0 or i == len(paths):
            elapsed = time.time() - t0
            eta = elapsed / i * (len(paths) - i)
            print(f"[fad-vggish] {label}: {i}/{len(paths)}  {elapsed:.1f}s elapsed  ~{eta:.1f}s left", flush=True)

    x = np.concatenate(embeddings, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=x, n_files=len(paths))
    return x, len(paths)


def _compute_stats(x: np.ndarray):
    return np.mean(x, axis=0), np.cov(x, rowvar=False)


def _frechet(mu_x, sigma_x, mu_y, sigma_y, eps=1e-6):
    mu_x = np.atleast_1d(mu_x)
    mu_y = np.atleast_1d(mu_y)
    sigma_x = np.atleast_2d(sigma_x)
    sigma_y = np.atleast_2d(sigma_y)

    eye = np.eye(sigma_x.shape[0])
    covmean, _ = linalg.sqrtm((sigma_x + eps * eye) @ (sigma_y + eps * eye), disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((sigma_x + 10 * eps * eye) @ (sigma_y + 10 * eps * eye))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu_x - mu_y
    return float(diff @ diff + np.trace(sigma_x) + np.trace(sigma_y) - 2.0 * np.trace(covmean))


def _parse_model_arg(raw: str):
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--model must be NAME=RECONS_DIR")
    name, path = raw.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("model name cannot be empty")
    return name, Path(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("eval_set/general"))
    parser.add_argument("--model", action="append", type=_parse_model_arg, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/fad_torcheval_vggish_no_relu"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--weights-path", default=None)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[fad-vggish] CUDA unavailable; falling back to CPU", flush=True)
        device = "cpu"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, processor, weights_path = _load_model(device, args.weights_path)
    max_pairs = args.max_pairs or None

    summary_rows = []
    for name, recons_root in args.model:
        pairs, missing = _match_pairs(args.input, recons_root, max_pairs)
        ref_paths = [p[0] for p in pairs]
        gen_paths = [p[1] for p in pairs]

        pair_fp = _fingerprint(ref_paths + gen_paths, max_pairs)
        ref_fp = _fingerprint(ref_paths, max_pairs)
        ref_cache = args.output_dir / "cache" / f"reference_{ref_fp}.npz"
        gen_cache = args.output_dir / "cache" / f"{name}_{pair_fp}.npz"

        print(f"[fad-vggish] model={name} pairs={len(pairs)} missing={len(missing)}", flush=True)
        ref_emb, n_ref_files = _load_or_embed(
            ref_paths, ref_cache, model, processor, device, args.batch_size, "reference"
        )
        gen_emb, n_gen_files = _load_or_embed(
            gen_paths, gen_cache, model, processor, device, args.batch_size, name
        )

        mu_ref, sigma_ref = _compute_stats(ref_emb)
        mu_gen, sigma_gen = _compute_stats(gen_emb)
        fad = _frechet(mu_ref, sigma_ref, mu_gen, sigma_gen)

        result = {
            "FAD": fad,
            "N_real": int(ref_emb.shape[0]),
            "N_gen": int(gen_emb.shape[0]),
            "N_real_files": int(n_ref_files),
            "N_gen_files": int(n_gen_files),
            "N_matched": int(len(pairs)),
            "N_missing_reconstruction": int(len(missing)),
            "D": int(ref_emb.shape[1]),
            "input_dir": str(args.input),
            "output_dir": str(recons_root),
            "device": device,
            "batch_size": int(args.batch_size),
            "max_pairs": max_pairs,
            "vggish_weights": str(weights_path),
            "protocol": (
                "torcheval-like VGGish FAD: TorchAudio VGGish with final ReLU removed, "
                "0.96s patch-level embeddings, Frechet distance"
            ),
            "reference_cache": str(ref_cache),
            "generated_cache": str(gen_cache),
        }
        json_path = recons_root / "fad_torcheval_vggish_no_relu.json"
        json_path.write_text(json.dumps(result, indent=2) + "\n")
        summary_rows.append({"model": name, "FAD": fad, "json_path": str(json_path), **result})
        print(f"[fad-vggish] done {name}: FAD={fad:.6f}", flush=True)

    summary_csv = args.output_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "FAD",
                "N_real",
                "N_gen",
                "N_real_files",
                "N_gen_files",
                "N_matched",
                "N_missing_reconstruction",
                "D",
                "input_dir",
                "output_dir",
                "device",
                "batch_size",
                "max_pairs",
                "json_path",
                "vggish_weights",
                "reference_cache",
                "generated_cache",
                "protocol",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[fad-vggish] summary: {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
