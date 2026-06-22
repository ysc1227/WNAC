#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argbind
import torch
from audiotools.core import util as at_util

from scripts.train import EMAC, build_dataset


def tensor_summary(x):
    finite = torch.isfinite(x)
    safe = torch.nan_to_num(x.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    total = x.numel()
    finite_count = int(finite.sum().item())
    return {
        "ok": bool(finite.all().item()),
        "finite_ratio": float(finite_count / total) if total else 1.0,
        "finite_count": finite_count,
        "bad_count": int(total - finite_count),
        "shape": list(x.shape),
        "mean": float(safe.mean().item()),
        "absmax": float(safe.abs().max().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--config", default="conf/debug_gpu0_train_samples.yml")
    ap.add_argument("--mode", choices=["eval", "train"], default="train")
    ap.add_argument("--autocast", action="store_true")
    ap.add_argument("--grad", action="store_true")
    args_cli = ap.parse_args()

    summary_path = Path(args_cli.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    out = {
        "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
    }

    try:
        at_util.seed(0)
        # argbind.parse_args() reads from sys.argv and expects an ArgumentParser
        # object as its first optional argument.  Keep this diagnostic script's
        # CLI separate from argbind by temporarily replacing sys.argv with the
        # equivalent training-config load arguments.
        old_argv = sys.argv[:]
        sys.argv = [old_argv[0], "--args.load", args_cli.config]
        try:
            args = argbind.parse_args()
        finally:
            sys.argv = old_argv
        with argbind.scope(args):
            model = EMAC().cuda()
        model.train(args_cli.mode == "train")
        with argbind.scope(args, "train"):
            ds = build_dataset(model.sample_rate)
        batch_size = int(args.get("batch_size", 24))
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=0, collate_fn=ds.collate)
        batch = next(iter(loader))
        batch = at_util.prepare_batch(batch, torch.device("cuda"))
        grad_ctx = torch.enable_grad() if args_cli.grad else torch.no_grad()
        with grad_ctx:
            signal = ds.transform(batch["signal"].clone(), **batch["transform_args"])
            if args_cli.autocast:
                with torch.autocast(device_type="cuda"):
                    result = model(signal.audio_data, signal.sample_rate)
            else:
                result = model(signal.audio_data, signal.sample_rate)
        out["mode"] = args_cli.mode
        out["autocast"] = bool(args_cli.autocast)
        out["grad"] = bool(args_cli.grad)
        out["batch_size"] = batch_size
        out["signal"] = tensor_summary(signal.audio_data)
        out["z"] = tensor_summary(result["z"])
        out["audio"] = tensor_summary(result["audio"])
        out["status"] = "ok"
    except Exception as e:
        out["status"] = "exception"
        out["exception"] = repr(e)

    summary_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(str(summary_path))


if __name__ == "__main__":
    main()