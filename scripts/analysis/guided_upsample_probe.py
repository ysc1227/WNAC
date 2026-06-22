import argparse
import inspect
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

from emac.model.emac import EMAC


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}


def _checkpoint_kwargs_and_state(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError(f"Unsupported checkpoint format: {path}")

    kwargs = ckpt.get("metadata", {}).get("kwargs", {}) or {}
    state_dict = ckpt["state_dict"]
    valid_args = set(inspect.signature(EMAC.__init__).parameters.keys()) - {"self"}
    kwargs = {k: v for k, v in kwargs.items() if k in valid_args}

    quantizer_ids = sorted({
        int(k.split("quantizer.quantizers.")[1].split(".")[0])
        for k in state_dict.keys()
        if "quantizer.quantizers." in k and "codebook.weight" in k
    })
    scale_factor = kwargs.get("scale_factor", None)
    if scale_factor is None:
        kwargs["scale_factor"] = [1.0] * int(kwargs.get("n_codebooks", len(quantizer_ids)))
        kwargs["use_wavescale"] = bool(kwargs.get("use_wavescale", False))
    else:
        expected_wavescale = len(scale_factor) * 2 - 1
        if len(quantizer_ids) == expected_wavescale and expected_wavescale != len(scale_factor):
            kwargs["use_wavescale"] = True

    return kwargs, state_dict


def load_guided_model(
    model_path: str,
    device: str,
    hidden_dim: int | None,
    kernel: int,
    detach_guide: bool,
    init_scale: float,
):
    kwargs, state_dict = _checkpoint_kwargs_and_state(model_path)
    kwargs.update(
        guided_upsample=True,
        guided_upsample_hidden_dim=hidden_dim,
        guided_upsample_kernel=kernel,
        guided_upsample_detach_guide=detach_guide,
        guided_upsample_init_scale=init_scale,
    )
    model = EMAC(**kwargs)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unexpected = [k for k in unexpected if "guided_upsampler" not in k]
    missing = [k for k in missing if "guided_upsampler" not in k]
    if missing or unexpected:
        raise RuntimeError(f"Unexpected checkpoint mismatch. missing={missing[:8]} unexpected={unexpected[:8]}")

    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    for name, param in model.named_parameters():
        if "guided_upsampler" in name:
            param.requires_grad_(True)
    return model


def resolve_device(device: str):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def set_guided_enabled(model: EMAC, enabled: bool):
    for quantizer in model.quantizer.quantizers:
        if hasattr(quantizer, "use_guided_upsample"):
            quantizer.use_guided_upsample = bool(enabled)


def force_zero_guide(model: EMAC):
    for quantizer in model.quantizer.quantizers:
        upsampler = getattr(quantizer, "guided_upsampler", None)
        if upsampler is None:
            continue
        with torch.no_grad():
            upsampler.guide_proj.weight.zero_()
            if upsampler.guide_proj.bias is not None:
                upsampler.guide_proj.bias.zero_()
        for param in upsampler.guide_proj.parameters():
            param.requires_grad_(False)


def find_audio_files(folder: str):
    root = Path(folder)
    if root.is_file() and root.suffix.lower() in AUDIO_EXTENSIONS:
        return [root]
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in AUDIO_EXTENSIONS)


def load_audio_batch(folder: str, model: EMAC, n_samples: int, duration: float, device: str):
    files = find_audio_files(folder)[:n_samples]
    if len(files) == 0:
        raise RuntimeError(f"No audio files found under {folder}")

    target_len = max(1, int(round(float(duration) * model.sample_rate)))
    xs = []
    for path in files:
        wav, sample_rate = torchaudio.load(str(path))
        x = wav.mean(dim=0, keepdim=True)
        if int(sample_rate) != int(model.sample_rate):
            x = torchaudio.functional.resample(x, int(sample_rate), int(model.sample_rate))
        if x.shape[-1] < target_len:
            x = F.pad(x, (0, target_len - x.shape[-1]))
        x = x[..., :target_len]
        xs.append(x)

    audio = torch.stack(xs, dim=0).to(device)
    return model.preprocess(audio, sample_rate=model.sample_rate)


def scalar_losses(losses):
    return torch.stack([loss.detach().float().cpu() for loss in losses])


@torch.no_grad()
def evaluate_losses(model: EMAC, audio: torch.Tensor, enabled: bool):
    set_guided_enabled(model, enabled)
    _, _, _, _, codebook_loss, _ = model.encode(audio)
    return scalar_losses(codebook_loss)


def stage_rows(model: EMAC, base, initial, final):
    rows = []
    scales = list(getattr(model.quantizer, "scale_factors", []))
    for i, b in enumerate(base.tolist()):
        init = float(initial[i])
        fin = float(final[i])
        gain = (b - fin) / max(abs(b), 1e-12)
        rows.append(
            {
                "stage": i,
                "scale": float(scales[i]) if i < len(scales) else None,
                "linear_loss": b,
                "guided_initial_loss": init,
                "guided_final_loss": fin,
                "relative_gain": gain,
            }
        )
    return rows


def print_rows(rows):
    print("stage scale    linear      guided_init guided_final rel_gain")
    for row in rows:
        print(
            f"{row['stage']:>5d} "
            f"{row['scale'] if row['scale'] is not None else float('nan'):>6.3f} "
            f"{row['linear_loss']:>11.6g} "
            f"{row['guided_initial_loss']:>11.6g} "
            f"{row['guided_final_loss']:>12.6g} "
            f"{100.0 * row['relative_gain']:>7.2f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="Tiny guided-upsample overfit probe.")
    parser.add_argument("--folder", default="eval_set/general")
    parser.add_argument("--model_path", default="runs/8.00kbps/wavescale_phi5_13_1024/best/emac/weights.pth")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=0)
    parser.add_argument("--kernel", type=int, default=5)
    parser.add_argument("--attach_guide_grad", action="store_true")
    parser.add_argument("--zero_guide", action="store_true", help="Ablate guidance by zeroing and freezing guide projection.")
    parser.add_argument("--init_scale", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_json", default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(True)
    device = resolve_device(args.device)
    model = load_guided_model(
        model_path=args.model_path,
        device=device,
        hidden_dim=args.hidden_dim or None,
        kernel=args.kernel,
        detach_guide=not args.attach_guide_grad,
        init_scale=args.init_scale,
    )
    if args.zero_guide:
        force_zero_guide(model)
    audio = load_audio_batch(args.folder, model, args.n_samples, args.duration, device)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No guided upsampler parameters are trainable.")

    base = evaluate_losses(model, audio, enabled=False)
    initial = evaluate_losses(model, audio, enabled=True)

    opt = torch.optim.AdamW(params, lr=args.lr)
    set_guided_enabled(model, True)
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        _, _, _, _, codebook_loss, _ = model.encode(audio)
        loss = torch.stack(codebook_loss).sum()
        loss.backward()
        opt.step()
        if step == 0 or (step + 1) % max(1, args.steps // 4) == 0:
            print(f"[probe] step={step + 1}/{args.steps} loss={float(loss.detach().cpu()):.6g}")

    final = evaluate_losses(model, audio, enabled=True)
    rows = stage_rows(model, base, initial, final)
    print_rows(rows)

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"[probe] wrote {out}")


if __name__ == "__main__":
    main()
