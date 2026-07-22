"""Reconstruct eval audio with official external codec checkpoints.

The common metric scripts operate on folders of reconstructed wav files.  This
helper creates those folders for codecs whose official checkpoints are not EMAC
checkpoints.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torchaudio
from audiotools.core import util
from tqdm import tqdm


def _load_audio(path: Path, sample_rate: int) -> tuple[torch.Tensor, int]:
    audio, sr = torchaudio.load(str(path))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    return audio.unsqueeze(0), sample_rate


def _pad_batch(items: list[tuple[Path, torch.Tensor]]) -> tuple[torch.Tensor, list[int], list[Path]]:
    lengths = [audio.shape[-1] for _, audio in items]
    max_len = max(lengths)
    batch = []
    paths = []
    for path, audio in items:
        pad = max_len - audio.shape[-1]
        if pad:
            audio = torch.nn.functional.pad(audio, (0, pad))
        batch.append(audio)
        paths.append(path)
    return torch.cat(batch, dim=0), lengths, paths


def _write_audio(path: Path, audio: torch.Tensor, sample_rate: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio.detach().float().cpu()
    if audio.dim() == 3:
        audio = audio[0]
    torchaudio.save(str(path), audio.clamp(-1.0, 1.0), sample_rate)


def _reconstruct_sat_chunked(model, audio: torch.Tensor, lengths: list[int], sample_rate: int) -> torch.Tensor:
    """Run SAT official reconstruction in the 1 s geometry used by its scales."""
    chunk_samples = int(sample_rate)
    outputs = []

    for item, length in zip(audio, lengths):
        item = item[..., :length]
        chunks = []
        for start in range(0, length, chunk_samples):
            seg_len = min(chunk_samples, length - start)
            segment = item[..., start : start + seg_len]
            if seg_len < chunk_samples:
                segment = torch.nn.functional.pad(segment, (0, chunk_samples - seg_len))
            recon, _, _ = model(segment.unsqueeze(0))
            chunks.append(recon[..., :seg_len])
        outputs.append(torch.cat(chunks, dim=-1))

    max_len = max(out.shape[-1] for out in outputs)
    padded = [
        torch.nn.functional.pad(out, (0, max_len - out.shape[-1]))
        if out.shape[-1] < max_len
        else out
        for out in outputs
    ]
    return torch.cat(padded, dim=0)


def _load_snac(device: torch.device):
    try:
        from snac import SNAC
    except ImportError as exc:
        raise ImportError(
            "The official SNAC checkpoint requires the Python package 'snac'. "
            "Install it in this environment with: python -m pip install snac"
        ) from exc

    model = SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval().to(device)
    return model, 44100


def _load_sat(device: torch.device, aar_repo: Path, checkpoint: str | None):
    if not aar_repo.exists():
        raise FileNotFoundError(
            f"AAR repo not found: {aar_repo}. Clone https://github.com/qiuk2/AAR "
            "and pass --aar-repo."
        )
    sys.path.insert(0, str(aar_repo))
    sat_spec = importlib.util.spec_from_file_location("aar_sat_model", aar_repo / "model" / "SAT.py")
    if sat_spec is None or sat_spec.loader is None:
        raise RuntimeError(f"Could not load SAT.py from {aar_repo}")
    sat_module = importlib.util.module_from_spec(sat_spec)
    sat_spec.loader.exec_module(sat_module)
    SAT = sat_module.SAT

    if checkpoint is None:
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download(
            repo_id="qiuk6/AAR",
            filename="SAT_bs_1536_d1024_lat64.pth",
        )

    model = SAT(
        sample_rate=24000,
        channels=1,
        causal=False,
        model_norm="weight_norm",
        audio_normalize=False,
        ratios=[8, 5, 4, 2],
        multi_scale=[1, 2, 4, 6, 9, 12, 16, 20, 25, 31, 37, 43, 50, 58, 66, 75],
        phi_kernel=[9, 9, 9, 9, 9, 9],
        dimension=1024,
        latent_dim=64,
    ).to(device)

    state = torch.load(checkpoint, map_location="cpu")
    state_dict = state.get("generator_state_dict", state)
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, 24000


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", choices=["snac_official", "sat_official"], required=True)
    parser.add_argument("--input", type=Path, default=Path("eval_set/general"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--aar-repo", type=Path, default=Path("/tmp/qiuk2_AAR"))
    parser.add_argument("--sat-checkpoint", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.codec == "snac_official":
        model, sample_rate = _load_snac(device)
    else:
        model, sample_rate = _load_sat(device, args.aar_repo, args.sat_checkpoint)

    audio_files = util.find_audio(args.input)
    if args.limit:
        audio_files = audio_files[: args.limit]
    if not audio_files:
        raise RuntimeError(f"No audio files found in {args.input}")

    pending = []
    for audio_path in audio_files:
        rel = audio_path.relative_to(args.input)
        out_path = args.output / rel.with_suffix(".wav")
        if not out_path.exists():
            pending.append((Path(audio_path), out_path))

    batch_size = max(1, int(args.batch_size))
    progress = tqdm(range(0, len(pending), batch_size), desc=f"Reconstructing {args.codec}")
    for start in progress:
        chunk = pending[start : start + batch_size]
        loaded = [(out_path, _load_audio(audio_path, sample_rate)[0]) for audio_path, out_path in chunk]
        audio, lengths, out_paths = _pad_batch(loaded)
        audio = audio.to(device)

        if args.codec == "snac_official":
            recon, _ = model(audio)
        else:
            recon = _reconstruct_sat_chunked(model, audio, lengths, sample_rate)

        for idx, (out_path, n) in enumerate(zip(out_paths, lengths)):
            _write_audio(out_path, recon[idx : idx + 1, ..., :n], sample_rate)


if __name__ == "__main__":
    main()
