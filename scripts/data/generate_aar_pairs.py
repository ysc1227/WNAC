import argparse, csv, random
from pathlib import Path

import torch
from audiotools import AudioSignal
from transformers import ClapModel, ClapProcessor

import emac
from emac.utils.aar_utils import salient_excerpt


def rows(csv_path, limit, seed):
    with open(csv_path, newline="") as f:
        xs = list(csv.DictReader(f))
    random.Random(seed).shuffle(xs)
    return xs[:limit]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_folder", required=True)
    p.add_argument("--manifest", default="samples/audioset_val_manifest.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument(
        "--excerpt_duration",
        type=float,
        default=None,
        help="If set, crop each source audio to this duration first, then use the same excerpt for ref and CLAP conditioning.",
    )
    p.add_argument(
        "--target_duration",
        type=float,
        default=None,
        help="Generated audio duration in seconds. If it exceeds the AAR RVQ frame window, chunked generation is used.",
    )
    p.add_argument("--chunk_prev_condition_weight", type=float, default=0.5)
    p.add_argument("--chunk_position_weight", type=float, default=0.1)
    args = p.parse_args()

    device = "cuda"
    out = Path(args.out)
    gen_dir, ref_dir = out / "gen", out / "ref"
    gen_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    model, _ = emac.model.AAR.load_from_folder(
        folder=args.ckpt_folder, map_location="cpu", package=True, weights_only=False
    )
    model = model.to(device).eval()
    model.vae_proxy[0].to(device).eval()
    model.vae_quant_proxy[0].to(device)
    sr = model.vae_proxy[0].sample_rate

    proc = ClapProcessor.from_pretrained("laion/larger_clap_general")
    clap = ClapModel.from_pretrained("laion/larger_clap_general").to(device).eval()

    for i, r in enumerate(rows(args.manifest, args.limit, args.seed)):
        src = r["src_path"]
        name = f"sample_{i:05d}.wav"
        sig = AudioSignal(src, sample_rate=sr)
        if args.excerpt_duration is not None:
            sig = salient_excerpt(sig, duration=args.excerpt_duration, state=args.seed + i)
        sig.write(ref_dir / name)
        cond = proc(
            audio=sig.resample(48000).audio_data.squeeze(0).cpu().numpy(),
            return_tensors="pt", sampling_rate=48000,
        )
        feat = clap.get_audio_features(
            input_features=cond["input_features"].to(device),
            is_longer=cond["is_longer"].to(device),
        )
        if not torch.is_tensor(feat):
            feat = feat.pooler_output if hasattr(feat, "pooler_output") else feat[0]
        if feat.shape[0] != 1:
            feat = feat.mean(dim=0, keepdim=True)
        target_samples = None if args.target_duration is None else int(round(args.target_duration * sr))
        if target_samples is not None:
            wav = model.autoregressive_infer_cfg_chunked(
                B=1,
                label_B=feat,
                target_samples=target_samples,
                g_seed=args.seed + i,
                cfg=args.cfg,
                top_k=args.top_k,
                top_p=args.top_p,
                prev_condition_weight=args.chunk_prev_condition_weight,
                position_weight=args.chunk_position_weight,
            )
        else:
            wav = model.autoregressive_infer_cfg(
                B=1, label_B=feat, g_seed=args.seed + i,
                cfg=args.cfg, top_k=args.top_k, top_p=args.top_p,
            )
        AudioSignal(wav.cpu(), sr).write(gen_dir / name)
        print(i + 1, name, flush=True)


if __name__ == "__main__":
    main()