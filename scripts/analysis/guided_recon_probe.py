import argparse
import inspect
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import torch
import torch.nn.functional as F
import torchaudio
from audiotools import AudioSignal

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emac.model.emac import EMAC
from emac.nn import loss as losses


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}


def checkpoint_kwargs_and_state(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError(f"Unsupported checkpoint format: {path}")

    kwargs = ckpt.get("metadata", {}).get("kwargs", {}) or {}
    valid_args = set(inspect.signature(EMAC.__init__).parameters.keys()) - {"self"}
    kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
    state_dict = ckpt["state_dict"]

    quantizer_ids = sorted(
        {
            int(k.split("quantizer.quantizers.")[1].split(".")[0])
            for k in state_dict.keys()
            if "quantizer.quantizers." in k and "codebook.weight" in k
        }
    )
    scale_factor = kwargs.get("scale_factor", None)
    if scale_factor is None:
        kwargs["scale_factor"] = [1.0] * int(kwargs.get("n_codebooks", len(quantizer_ids)))
        kwargs["use_wavescale"] = bool(kwargs.get("use_wavescale", False))
    else:
        expected_wavescale = len(scale_factor) * 2 - 1
        if len(quantizer_ids) == expected_wavescale and expected_wavescale != len(scale_factor):
            kwargs["use_wavescale"] = True

    return kwargs, state_dict


def resolve_device(device: str):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_model(args):
    kwargs, state_dict = checkpoint_kwargs_and_state(args.model_path)
    kwargs.update(
        quantizer_dropout=0.0,
        quantizer_loss_target=args.quantizer_loss_target,
        guided_upsample=True,
        guided_upsample_hidden_dim=args.hidden_dim or None,
        guided_upsample_kernel=args.kernel,
        guided_upsample_detach_guide=not args.attach_guide_grad,
        guided_upsample_init_scale=args.init_scale,
        guided_upsample_after_pivot_only=args.guide_mode == "pivot",
        guided_upsample_decoder_grad_alpha=args.decoder_grad_alpha,
    )
    model = EMAC(**kwargs)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if "guided_upsampler" not in k]
    unexpected = [k for k in unexpected if "guided_upsampler" not in k]
    if missing or unexpected:
        raise RuntimeError(f"Unexpected checkpoint mismatch. missing={missing[:8]} unexpected={unexpected[:8]}")
    return model


def configure_guided_stages(model: EMAC, mode: str):
    model.quantizer.guided_upsample = True
    model.quantizer.guided_upsample_after_pivot_only = mode == "pivot"
    return model.quantizer.configure_guided_upsample_stages()


def set_guided_enabled(model: EMAC, enabled: bool, stage_mask=None):
    if stage_mask is None:
        stage_mask = getattr(model.quantizer, "guided_upsample_stage_mask", None)
    for i, quantizer in enumerate(model.quantizer.quantizers):
        active = bool(enabled)
        if stage_mask is not None:
            active = active and bool(stage_mask[i])
        if hasattr(quantizer, "use_guided_upsample"):
            quantizer.use_guided_upsample = active


def set_only_guided_trainable(model: EMAC, stage_mask):
    for param in model.parameters():
        param.requires_grad_(False)
    for i, quantizer in enumerate(model.quantizer.quantizers):
        if not stage_mask[i]:
            continue
        upsampler = getattr(quantizer, "guided_upsampler", None)
        if upsampler is None:
            continue
        for param in upsampler.parameters():
            param.requires_grad_(True)


def trainable_parameters(model: EMAC):
    return [p for p in model.parameters() if p.requires_grad]


def find_audio_files(folder: str):
    root = Path(folder)
    if root.is_file() and root.suffix.lower() in AUDIO_EXTENSIONS:
        return [root]
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in AUDIO_EXTENSIONS)


def split_files(files: list[Path], n_train: int, n_val: int, seed: int):
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)
    needed = n_train + n_val
    if len(files) < needed:
        raise RuntimeError(f"Need at least {needed} audio files, found {len(files)}")
    return files[:n_train], files[n_train:needed]


def load_audio(path: Path, sample_rate: int):
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0, keepdim=True)
    if int(sr) != int(sample_rate):
        wav = torchaudio.functional.resample(wav, int(sr), int(sample_rate))
    return wav


def crop_or_pad(wav: torch.Tensor, target_len: int, generator=None, random_crop: bool = False):
    if wav.shape[-1] < target_len:
        return F.pad(wav, (0, target_len - wav.shape[-1]))
    if wav.shape[-1] == target_len:
        return wav
    max_start = wav.shape[-1] - target_len
    if random_crop:
        start = int(torch.randint(0, max_start + 1, (1,), generator=generator).item())
    else:
        start = max_start // 2
    return wav[..., start : start + target_len]


def load_batch(
    files: list[Path],
    sample_rate: int,
    duration: float,
    batch_size: int,
    device: str,
    generator=None,
    random_crop: bool = False,
    step: int = 0,
):
    target_len = max(1, int(round(float(duration) * sample_rate)))
    if random_crop:
        idx = torch.randint(0, len(files), (batch_size,), generator=generator).tolist()
    else:
        idx = [(step * batch_size + i) % len(files) for i in range(batch_size)]
    audio = [crop_or_pad(load_audio(files[i], sample_rate), target_len, generator, random_crop) for i in idx]
    return torch.stack(audio, dim=0).to(device)


def mean_dict(rows: Iterable[dict[str, float]]):
    rows = list(rows)
    keys = rows[0].keys()
    return {k: float(sum(row[k] for row in rows) / len(rows)) for k in keys}


def make_loss_modules(device: str):
    return {
        "mel": losses.MelSpectrogramLoss().to(device),
        "stft": losses.MultiScaleSTFTLoss().to(device),
        "waveform": losses.L1Loss().to(device),
    }


def scalar_losses(out):
    commitment = torch.stack([x.float() for x in out["vq/commitment_loss"]]).sum()
    codebook = torch.stack([x.float() for x in out["vq/codebook_loss"]]).sum()
    aux = out["vq/aux_loss"]
    if not torch.is_tensor(aux):
        aux = torch.as_tensor(aux, device=codebook.device, dtype=codebook.dtype)
    return commitment, codebook, aux.float()


def reconstruction_losses(model: EMAC, audio: torch.Tensor, loss_modules: dict):
    out = model(audio, model.sample_rate)
    recons = AudioSignal(out["audio"], model.sample_rate)
    signal = AudioSignal(audio, model.sample_rate)
    commitment, codebook, aux = scalar_losses(out)
    mel = loss_modules["mel"](recons, signal)
    stft = loss_modules["stft"](recons, signal)
    waveform = loss_modules["waveform"](recons, signal)
    return {
        "mel": mel,
        "stft": stft,
        "waveform": waveform,
        "commitment": commitment,
        "codebook": codebook,
        "aux": aux,
    }


def weighted_train_loss(model: EMAC, audio: torch.Tensor, loss_modules: dict, args):
    use_recon_loss = args.mel_weight != 0.0 or args.stft_weight != 0.0 or args.waveform_weight != 0.0
    if use_recon_loss:
        vals = reconstruction_losses(model, audio, loss_modules)
        loss = (
            args.mel_weight * vals["mel"]
            + args.stft_weight * vals["stft"]
            + args.waveform_weight * vals["waveform"]
            + args.commitment_weight * vals["commitment"]
            + args.codebook_weight * vals["codebook"]
            + args.aux_weight * vals["aux"]
        )
        return loss, vals

    audio = model.preprocess(audio, model.sample_rate)
    _, _, _, commitment_loss, codebook_loss, aux_loss = model.encode(audio)
    commitment = torch.stack([x.float() for x in commitment_loss]).sum()
    codebook = torch.stack([x.float() for x in codebook_loss]).sum()
    if not torch.is_tensor(aux_loss):
        aux_loss = torch.as_tensor(aux_loss, device=codebook.device, dtype=codebook.dtype)
    vals = {
        "commitment": commitment,
        "codebook": codebook,
        "aux": aux_loss.float(),
    }
    loss = (
        args.commitment_weight * commitment
        + args.codebook_weight * codebook
        + args.aux_weight * aux_loss.float()
    )
    return loss, vals


@torch.no_grad()
def evaluate(model, files, batch_size, duration, device, loss_modules, guided_enabled, stage_mask):
    model.eval()
    set_guided_enabled(model, guided_enabled, stage_mask)
    rows = []
    n_batches = (len(files) + batch_size - 1) // batch_size
    for step in range(n_batches):
        current_batch = min(batch_size, len(files) - step * batch_size)
        audio = load_batch(
            files,
            model.sample_rate,
            duration,
            current_batch,
            device,
            random_crop=False,
            step=step,
        )
        vals = reconstruction_losses(model, audio, loss_modules)
        rows.append({k: float(v.detach().cpu()) for k, v in vals.items()})
    return mean_dict(rows)


def relative_delta(base: dict[str, float], value: dict[str, float]):
    return {f"{k}_rel_delta": (base[k] - value[k]) / max(abs(base[k]), 1e-12) for k in base}


def main():
    parser = argparse.ArgumentParser(description="Held-out reconstruction probe for guided WaveScale upsampling.")
    parser.add_argument("--folder", default="eval_set/general")
    parser.add_argument("--model_path", default="runs/8.00kbps/wavescale_13_1024/best/emac/weights.pth")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n_train", type=int, default=32)
    parser.add_argument("--n_val", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=0)
    parser.add_argument("--kernel", type=int, default=5)
    parser.add_argument("--guide_mode", choices=["full", "pivot"], default="full")
    parser.add_argument("--quantizer_loss_target", choices=["full", "lowpass", "bandpass", "residual_band"], default="full")
    parser.add_argument("--attach_guide_grad", action="store_true")
    parser.add_argument("--init_scale", type=float, default=1e-3)
    parser.add_argument("--decoder_grad_alpha", type=float, default=0.0)
    parser.add_argument("--mel_weight", type=float, default=0.0)
    parser.add_argument("--stft_weight", type=float, default=0.0)
    parser.add_argument("--waveform_weight", type=float, default=0.0)
    parser.add_argument("--commitment_weight", type=float, default=0.0)
    parser.add_argument("--codebook_weight", type=float, default=1.0)
    parser.add_argument("--aux_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_json", default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    device = resolve_device(args.device)

    files = find_audio_files(args.folder)
    train_files, val_files = split_files(files, args.n_train, args.n_val, args.seed)

    model = build_model(args).to(device)
    stage_mask = configure_guided_stages(model, args.guide_mode)
    set_only_guided_trainable(model, stage_mask)
    params = trainable_parameters(model)
    if not params:
        raise RuntimeError("No guided upsampler parameters are trainable.")

    loss_modules = make_loss_modules(device)

    linear_val = evaluate(model, val_files, args.batch_size, args.duration, device, loss_modules, False, stage_mask)
    initial_val = evaluate(model, val_files, args.batch_size, args.duration, device, loss_modules, True, stage_mask)
    print(f"[probe] guided_stage_mask={stage_mask}")
    print(f"[probe] val linear={linear_val}")
    print(f"[probe] val initial_guided={initial_val}")
    print(f"[probe] initial_rel_delta={relative_delta(linear_val, initial_val)}")

    opt = torch.optim.AdamW(params, lr=args.lr)
    set_guided_enabled(model, True, stage_mask)
    for step in range(args.steps):
        model.train()
        set_guided_enabled(model, True, stage_mask)
        audio = load_batch(
            train_files,
            model.sample_rate,
            args.duration,
            args.batch_size,
            device,
            generator=rng,
            random_crop=True,
        )
        opt.zero_grad(set_to_none=True)
        loss, train_vals = weighted_train_loss(model, audio, loss_modules, args)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 100.0)
        opt.step()
        if step == 0 or (step + 1) % max(1, args.log_every) == 0:
            metric_text = " ".join(
                f"{key}={float(value.detach().cpu()):.6g}" for key, value in train_vals.items()
            )
            print(f"[probe] step={step + 1}/{args.steps} train_loss={float(loss.detach().cpu()):.6g} {metric_text}")

    final_train = evaluate(model, train_files, args.batch_size, args.duration, device, loss_modules, True, stage_mask)
    final_val = evaluate(model, val_files, args.batch_size, args.duration, device, loss_modules, True, stage_mask)
    print(f"[probe] train final_guided={final_train}")
    print(f"[probe] val final_guided={final_val}")
    print(f"[probe] final_rel_delta={relative_delta(linear_val, final_val)}")

    summary = {
        "model_path": args.model_path,
        "folder": args.folder,
        "guide_mode": args.guide_mode,
        "quantizer_loss_target": args.quantizer_loss_target,
        "decoder_grad_alpha": args.decoder_grad_alpha,
        "mel_weight": args.mel_weight,
        "stft_weight": args.stft_weight,
        "waveform_weight": args.waveform_weight,
        "guided_stage_mask": stage_mask,
        "n_train": len(train_files),
        "n_val": len(val_files),
        "linear_val": linear_val,
        "initial_val": initial_val,
        "final_train": final_train,
        "final_val": final_val,
        "initial_rel_delta": relative_delta(linear_val, initial_val),
        "final_rel_delta": relative_delta(linear_val, final_val),
    }
    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[probe] wrote {out}")


if __name__ == "__main__":
    main()
