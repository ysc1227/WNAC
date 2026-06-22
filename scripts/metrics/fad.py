import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from torchvggish import vggish, vggish_input
from scipy import linalg

@torch.no_grad()
def folder_embeddings_torchvggish(wav_dir, device="cuda", max_files=None):
    files = sorted([f for f in os.listdir(wav_dir) if f.endswith(".wav")])
    if len(files) == 0:
        raise RuntimeError(f"No .wav files in {wav_dir}")
    if max_files is not None:
        files = files[:max_files]

    # IMPORTANT: disable postprocess (PCA/quantize) to avoid CPU tensors inside torchvggish
    model = vggish(postprocess=False).to(device).eval()

    embs = []
    for f in tqdm(files, desc=f"Embedding ({os.path.basename(wav_dir)})"):
        p = os.path.join(wav_dir, f)

        try:
            ex = vggish_input.wavfile_to_examples(p)  # CPU tensor
        except Exception:
            continue

        if ex is None or ex.numel() == 0:
            continue

        ex = ex.to(device, non_blocking=True)  # (num_windows, 1, 96, 64) typically

        e = model(ex)  # (num_windows, 128)
        if e is None or e.numel() == 0:
            continue

        e = e.mean(dim=0)  # (128,)
        if torch.isfinite(e).all():
            embs.append(e.detach().float().cpu().numpy())

    if len(embs) < 2:
        raise RuntimeError(f"Need >=2 valid wavs in {wav_dir}, got {len(embs)}")

    return np.stack(embs, axis=0)  # (N, 128)

def _compute_stats(x: np.ndarray):
    # x: (N, D)
    mu = np.mean(x, axis=0)
    sigma = np.cov(x, rowvar=False)  # (D, D)
    return mu, sigma

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Standard FID Frechet distance between two Gaussians.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be singular; add eps to diagonals
    covmean, _ = linalg.sqrtm((sigma1 + eps*np.eye(sigma1.shape[0])) @ (sigma2 + eps*np.eye(sigma2.shape[0])), disp=False)

    if not np.isfinite(covmean).all():
        # stronger regularization
        offset = (10 * eps) * np.eye(sigma1.shape[0])
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # sqrtm may return complex due to numerical error
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)

def compute_fad_torchvggish(recon_dir, org_dir, device="cuda", max_files=None):
    x_recon = folder_embeddings_torchvggish(recon_dir, device=device, max_files=max_files)
    x_org   = folder_embeddings_torchvggish(org_dir, device=device, max_files=max_files)

    mu_r, sig_r = _compute_stats(x_org)
    mu_g, sig_g = _compute_stats(x_recon)

    return {
        "FAD": frechet_distance(mu_r, sig_r, mu_g, sig_g),
        "N_real": int(x_org.shape[0]),
        "N_gen": int(x_recon.shape[0]),
        "D": int(x_org.shape[1]),
    }

def main():
    p = argparse.ArgumentParser(description="Compute FAD with torchvggish embeddings.")
    p.add_argument("--input_dir", required=True, help="ground-truth/reference wav directory")
    p.add_argument("--output_dir", required=True, help="generated/reconstructed wav directory")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--json_path", default=None)
    args = p.parse_args()

    res = compute_fad_torchvggish(
        recon_dir=args.output_dir,
        org_dir=args.input_dir,
        device=args.device,
        max_files=args.max_samples,
    )
    res["protocol"] = "torchvggish embeddings, Frechet distance between reference and generated folders"

    text = json.dumps(res, indent=2)
    print(text)
    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_path).write_text(text + "\n")


if __name__ == "__main__":
    main()
