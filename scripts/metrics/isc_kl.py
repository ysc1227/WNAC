"""AudioLDM-style ISc/KL without installing audioldm_eval."""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class PANNsAudioTaggingAsLogits:
    """Small adapter around panns_inference.AudioTagging.

    The qiuqiangkong/audioset_tagging_cnn GitHub repository does not provide a
    torch.hub ``hubconf.py`` in the current checkout, so ``torch.hub.load(...,
    "Cnn14")`` can fail even after clearing the hub cache.  The project already
    depends on ``panns_inference``, which ships the same Cnn14 architecture and
    checkpoint handling.  Its public inference API returns probabilities, so this
    adapter converts clipwise probabilities back to logits to keep the rest of
    the AudioLDM-style IS/KL code unchanged.
    """

    def __init__(self, device):
        from panns_inference import AudioTagging

        self.audio_tagger = AudioTagging(device=device)
        self.device = self.audio_tagger.device

    def to(self, device):
        # AudioTagging chooses cuda/cpu internally during construction.
        return self

    def eval(self):
        self.audio_tagger.model.eval()
        return self

    def __call__(self, x):
        prob, _ = self.audio_tagger.inference(x.detach().cpu().numpy())
        prob = torch.from_numpy(prob).to(x.device).float().clamp(1e-6, 1 - 1e-6)
        return torch.logit(prob)


def wavs(root):
    exts = {".wav", ".flac", ".mp3", ".ogg"}
    return {p.name: p for p in sorted(Path(root).iterdir()) if p.suffix.lower() in exts}


def load_audio(path, sr):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def pad_batch(xs):
    n = max(len(x) for x in xs)
    out = np.zeros((len(xs), n), dtype=np.float32)
    for i, x in enumerate(xs):
        out[i, : len(x)] = x
    return torch.from_numpy(out)


def load_cnn14(device):
    try:
        return PANNsAudioTaggingAsLogits(device).to(device).eval()
    except Exception as e:
        print(f"[warn] panns_inference AudioTagging failed ({type(e).__name__}: {e}); trying torch.hub", flush=True)
        model = torch.hub.load(
            "qiuqiangkong/audioset_tagging_cnn",
            "Cnn14",
            pretrained=True,
            trust_repo=True,
        )
        return model.to(device).eval()


@torch.no_grad()
def infer_logits(paths, model, sr, device, batch_size):
    logits_all = []
    for i in tqdm(range(0, len(paths), batch_size), desc="CNN14"):
        batch = [load_audio(p, sr) for p in paths[i : i + batch_size]]
        x = pad_batch(batch).to(device)
        out = model(x)
        if isinstance(out, dict):
            if "logits" in out:
                logits = out["logits"]
            else:
                prob = out["clipwise_output"].clamp(1e-6, 1 - 1e-6)
                logits = torch.logit(prob)
        else:
            logits = out
        logits_all.append(logits.detach().cpu())
    return torch.cat(logits_all, dim=0)


def inception_score(logits, splits):
    n = logits.shape[0]
    splits = max(1, min(splits, n))
    scores = []
    for i in range(splits):
        z = logits[i * n // splits : (i + 1) * n // splits]
        p = z.softmax(dim=1)
        log_p = z.log_softmax(dim=1)
        q = p.mean(dim=0, keepdim=True).clamp_min(1e-12)
        scores.append(torch.exp((p * (log_p - q.log())).sum(dim=1).mean()).item())
    return float(np.mean(scores)), float(np.std(scores))


def paired_kl(gen_logits, ref_logits):
    kl_softmax = F.kl_div(
        gen_logits.softmax(dim=1).clamp_min(1e-6).log(),
        ref_logits.softmax(dim=1),
        reduction="sum",
    ) / gen_logits.shape[0]
    kl_sigmoid = F.kl_div(
        gen_logits.sigmoid().clamp_min(1e-6).log(),
        ref_logits.sigmoid(),
        reduction="sum",
    ) / gen_logits.shape[0]
    return float(kl_softmax), float(kl_sigmoid)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="ground-truth wav dir")
    p.add_argument("--output_dir", required=True, help="generated wav dir")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sr", type=int, default=32000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--splits", type=int, default=10)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--json_path", default=None)
    args = p.parse_args()

    ref = wavs(args.input_dir)
    gen = wavs(args.output_dir)
    keys = sorted(set(ref) & set(gen))
    if args.max_samples:
        keys = keys[: args.max_samples]
    if not keys:
        raise SystemExit("No paired files with matching basenames.")

    model = load_cnn14(args.device)
    gen_logits = infer_logits([gen[k] for k in keys], model, args.sr, args.device, args.batch_size)
    ref_logits = infer_logits([ref[k] for k in keys], model, args.sr, args.device, args.batch_size)

    isc_mean, isc_std = inception_score(gen_logits, args.splits)
    kl_softmax, kl_sigmoid = paired_kl(gen_logits, ref_logits)
    result = {
        "inception_score_mean": isc_mean,
        "inception_score_std": isc_std,
        "kullback_leibler_divergence_softmax": kl_softmax,
        "kullback_leibler_divergence_sigmoid": kl_sigmoid,
        "num_pairs": len(keys),
        "protocol": "AudioLDM-style: CNN14 logits, IS=softmax logits, KL=paired target||gen",
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.json_path:
        Path(args.json_path).write_text(text + "\n")


if __name__ == "__main__":
    main()