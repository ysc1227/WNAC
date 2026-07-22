#!/usr/bin/env python3
"""Prepare the upstream AAR SAT trainer for reproduced training.

The original AAR SAT training code is tied to AudioSet CSV rows named
``Y<youtube_id>.mp3``.  This helper patches a local clone to also support a
plain recursive folder dataset, then writes a SAT config matching the official
architecture.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


BUILD_PY = '''"""Dataset factory patched for local SAT reproduction."""

try:
    from .local_audio import LocalAudioDataset
except ImportError:
    LocalAudioDataset = None


def create_dataset(dataset_name, args, split="train"):
    configured_name = getattr(args, "dataset_name", dataset_name)
    if configured_name in {"local_audio", "folder", "folders"}:
        if LocalAudioDataset is None:
            raise RuntimeError("LocalAudioDataset is unavailable; missing datasets/local_audio.py")
        return LocalAudioDataset(args, transform=None, mode=split)

    from .audioset import AudioSet
    return AudioSet(args, transform=None, mode=split)
'''


LOCAL_AUDIO_PY = '''import os
import random
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus"}


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v)]
    if isinstance(value, str):
        sep = ":" if ":" in value else ","
        return [v.strip() for v in value.split(sep) if v.strip()]
    return [str(value)]


def _is_main_process():
    rank = os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", "0"))
    return str(rank) == "0"


class LocalAudioDataset(Dataset):
    def __init__(self, config, transform=None, mode="train"):
        self.mode = mode
        self.transform = transform
        self.fixed_length = int(getattr(config, "fixed_length", 0) or 0)
        self.tensor_cut = int(getattr(config, "tensor_cut", 24000) or 0)
        self.sample_rate = int(getattr(config, "sample_rate", 24000) or 24000)
        self.channels = int(getattr(config, "channels", 1) or 1)

        split_dirs = getattr(config, f"{mode}_dirs", None)
        roots = _as_list(split_dirs)
        if not roots:
            roots = _as_list(getattr(config, "train_dirs", None))
        if not roots:
            roots = _as_list(getattr(config, "train_dir", None))

        existing_roots = []
        audio_files = []
        for root in roots:
            root_path = Path(os.path.expanduser(root))
            if not root_path.exists():
                if _is_main_process():
                    print(f"[LocalAudioDataset] skip missing root: {root_path}")
                continue
            existing_roots.append(root_path)
            if root_path.is_file() and root_path.suffix.lower() in AUDIO_EXTENSIONS:
                audio_files.append(root_path)
                continue
            for path in root_path.rglob("*"):
                if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                    audio_files.append(path)

        if not audio_files:
            searched = ", ".join(str(r) for r in roots)
            raise RuntimeError(f"No audio files found for LocalAudioDataset. searched={searched}")

        seed = int(getattr(config, "seed", 0) or 0)
        rng = random.Random(seed)
        audio_files = sorted(set(audio_files))
        rng.shuffle(audio_files)
        if self.fixed_length > 0 and len(audio_files) > self.fixed_length:
            audio_files = audio_files[: self.fixed_length]
        self.audio_files = audio_files

        if _is_main_process():
            print(
                f"[LocalAudioDataset] mode={mode}, files={len(self.audio_files)}, "
                f"roots={len(existing_roots)}, sample_rate={self.sample_rate}, tensor_cut={self.tensor_cut}"
            )

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        path = self.audio_files[idx % len(self.audio_files)]
        waveform, _ = librosa.load(
            str(path),
            sr=self.sample_rate,
            mono=(self.channels == 1),
        )
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=0)

        if self.tensor_cut > 0:
            if waveform.size > self.tensor_cut:
                start = random.randint(0, waveform.size - self.tensor_cut)
                waveform = waveform[start : start + self.tensor_cut]
            elif waveform.size < self.tensor_cut:
                waveform = np.pad(waveform, (0, self.tensor_cut - waveform.size), "constant")

        waveform = np.asarray(waveform, dtype=np.float32)
        if self.transform:
            waveform = self.transform(waveform)
        return torch.from_numpy(waveform)
'''


def patch_train_script(aar_repo: Path) -> None:
    train_path = aar_repo / "train_SAT_mpi.py"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing AAR training script: {train_path}")

    backup_path = aar_repo / "train_SAT_mpi.py.sat_reproduced.bak"
    if backup_path.exists():
        text = backup_path.read_text(encoding="utf-8")
    else:
        text = train_path.read_text(encoding="utf-8")
        backup_path.write_text(text, encoding="utf-8")

    text = text.replace("import wandb\n", "from torch.utils.tensorboard import SummaryWriter\n")
    text = text.replace(
        "def train_epoch(audiovae, disc, dataloader, optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D, progress_bar, rank, args):\n",
        "def train_epoch(audiovae, disc, dataloader, optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D, progress_bar, rank, args, writer=None):\n",
    )

    old_setup = '''def setup(args):
    args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    args.gpus = args.world_size
    args.gpu = int(os.environ['OMPI_COMM_WORLD_RANK'])
    dist.init_process_group(backend='nccl', rank=args.rank, world_size=args.world_size)
    # dist.barrier()
'''
    new_setup = '''def setup(args):
    args.rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", 0)))
    args.world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("WORLD_SIZE", 1)))
    args.local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("LOCAL_RANK", 0)))
    args.gpus = args.world_size
    args.gpu = args.local_rank
    dist.init_process_group(backend='nccl', rank=args.rank, world_size=args.world_size)
    # dist.barrier()
'''
    text = text.replace(old_setup, new_setup)
    text = text.replace(
        '''    for batch_idx, batch in enumerate(dataloader):
''',
        '''    for batch_idx, batch in enumerate(dataloader):
        if getattr(args, "max_train_steps", 0) and args.completed_steps >= args.max_train_steps:
            break
''',
        1,
    )
    text = text.replace(
        '''    device = torch.device(f"cuda:{os.environ['OMPI_COMM_WORLD_LOCAL_RANK']}")
    # seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
''',
        '''    device = torch.device(f"cuda:{args.local_rank}")
    # seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs")) if args.rank == 0 else None
''',
    )
    text = text.replace(
        '''    args.num_update_steps_per_epoch = len(dataloader)
    args.max_train_steps = args.num_epochs * args.num_update_steps_per_epoch
''',
        '''    args.num_update_steps_per_epoch = len(dataloader)
    requested_max_steps = int(getattr(args, "max_train_steps", 0) or 0)
    if requested_max_steps > 0:
        args.max_train_steps = requested_max_steps
        args.num_epochs = math.ceil(args.max_train_steps / args.num_update_steps_per_epoch)
    else:
        args.max_train_steps = args.num_epochs * args.num_update_steps_per_epoch
''',
    )
    text = text.replace(
        '''    if args.rank == 0:
        wandb_dir = './wandb'
        if not os.path.exists(wandb_dir):
            os.makedirs(wandb_dir)
        os.environ["WANDB_CONFIG_DIR"] = './wandb'
        os.environ["WANDB_CACHE_DIR"] = './wandb'
        os.environ["WANDB_DIR"] = './wandb'
        wandb.login()
        if args.debug:
            wandb.init(project="Debug")
        else:
            wandb.init(project="AudioVAR")


''',
        '''    if args.rank == 0:
        print(f"TensorBoard logs: {os.path.join(args.output_dir, 'logs')}")


''',
    )
    text = text.replace(
        '''        progress_bar.set_description(f"train/loss: {generator_loss.item()}")
''',
        '''        disc_for_log = discriminator_loss.item() if update_disc else np.nan
        progress_bar.set_description("train")
        if rank == 0:
            progress_bar.set_postfix(
                loss=f"{generator_loss.item():.4f}",
                mel=f"{loss_g['l_f'].item():.4f}",
                wav=f"{loss_g['l_t'].item():.4f}",
                adv=f"{loss_g['l_g'].item():.4f}",
                feat=f"{loss_g['l_feat'].item():.4f}",
                commit=f"{commit_loss.item():.4f}",
                disc=f"{disc_for_log:.4f}" if np.isfinite(disc_for_log) else "nan",
                lr=f"{optimizer_G.param_groups[0]['lr']:.2e}",
            )
''',
    )
    text = text.replace(
        '''                loss_data = {
                    **{f"train_generator/{key}": value.item() for key, value in loss_g.items()},
                    "train_disc": discriminator_loss.item() if update_disc else np.nan,
                    "commit_loss": commit_loss.item(),
                    "lr": optimizer_G.param_groups[0]['lr']
                }
                wandb.log(
                    loss_data,
                    step = args.completed_steps
                )
                input_audio = input_wav[0].flatten().float().cpu().numpy()
                output_audio = output_wav[0].flatten().float().cpu().detach().numpy()
                wandb.log({"input_audio": wandb.Audio(input_audio, sample_rate=args.sample_rate, caption="Input Audio")})
                wandb.log({"output_audio": wandb.Audio(output_audio, sample_rate=args.sample_rate, caption="Output Audio")})
''',
        '''                if writer is not None:
                    step = args.completed_steps
                    disc_value = disc_for_log
                    writer.add_scalar("train/loss", generator_loss.item(), step)
                    writer.add_scalar("train/generator_loss", generator_loss.item(), step)
                    writer.add_scalar("train/discriminator_loss", disc_value, step)
                    writer.add_scalar("train/waveform/loss", loss_g["l_t"].item(), step)
                    writer.add_scalar("train/mel/loss", loss_g["l_f"].item(), step)
                    writer.add_scalar("train/adv/gen_loss", loss_g["l_g"].item(), step)
                    writer.add_scalar("train/adv/feat_loss", loss_g["l_feat"].item(), step)
                    writer.add_scalar("train/vq/commitment_loss", commit_loss.item(), step)
                    writer.add_scalar("train/lr", optimizer_G.param_groups[0]["lr"], step)
                    writer.add_audio("train/input_audio", input_wav[0].detach().float().cpu(), step, sample_rate=args.sample_rate)
                    writer.add_audio("train/output_audio", output_wav[0].detach().float().cpu(), step, sample_rate=args.sample_rate)
                    writer.flush()
                    with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as log_f:
                        log_f.write(
                            f"step={step} train/loss={generator_loss.item():.6f} "
                            f"train/mel/loss={loss_g['l_f'].item():.6f} "
                            f"train/waveform/loss={loss_g['l_t'].item():.6f} "
                            f"train/adv/gen_loss={loss_g['l_g'].item():.6f} "
                            f"train/adv/feat_loss={loss_g['l_feat'].item():.6f} "
                            f"train/vq/commitment_loss={commit_loss.item():.6f} "
                            f"train/discriminator_loss={disc_value:.6f} "
                            f"train/lr={optimizer_G.param_groups[0]['lr']:.8e}\\n"
                        )
''',
    )
    text = text.replace(
        '''        if args.save_interval == 'epoch' and args.rank == 0:
            save_checkpoint(audiovae, discriminator, optimizer_G, optimizer_D, epoch, args.completed_steps, args.output_dir)
''',
        '''        if args.save_interval == 'epoch' and args.rank == 0:
            save_checkpoint(audiovae, discriminator, optimizer_G, optimizer_D, epoch, args.completed_steps, args.output_dir)

        if args.completed_steps >= args.max_train_steps:
            break
''',
    )
    text = text.replace(
        '''        train_epoch(audiovae, discriminator, dataloader, optimizer_G, optimizer_D, scheduler_G, 
                    scheduler_D, progress_bar, args.rank, args)
''',
        '''        train_epoch(audiovae, discriminator, dataloader, optimizer_G, optimizer_D, scheduler_G, 
                    scheduler_D, progress_bar, args.rank, args, writer)
''',
    )
    text = text.replace(
        '''    cleanup()
''',
        '''    if writer is not None:
        writer.close()

    cleanup()
''',
    )

    train_path.write_text(text, encoding="utf-8")


def parse_dirs(value: str) -> list[str]:
    return [item.strip() for item in value.split(":") if item.strip()]


def yaml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def yaml_list(values: list[object]) -> str:
    return "[" + ", ".join(yaml_scalar(v) for v in values) + "]"


def write_config(args: argparse.Namespace, train_dirs: list[str]) -> None:
    cfg = {
        "dataset_name": "local_audio",
        "train_dirs": train_dirs,
        "train_dir": train_dirs[0] if train_dirs else "",
        "train_csv": "",
        "sample_rate": args.sample_rate,
        "channels": 1,
        "tensor_cut": args.tensor_cut,
        "fixed_length": args.fixed_length,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "use_prefetcher": args.use_prefetcher,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "resume": args.resume,
        "seed": args.seed,
        "clap_process": False,
        "num_epochs": args.num_epochs,
        "max_train_steps": args.max_steps,
        "log_interval": args.log_interval,
        "optimizer": "adam",
        "learning_rate": args.learning_rate,
        "lr_scheduler": args.lr_scheduler,
        "warmup_epoch": args.warmup_epoch,
        "save_interval": "epoch",
        "mixed_precision": args.mixed_precision,
        "model_norm": "weight_norm",
        "ratios": [8, 5, 4, 2],
        "multi_scale": [1, 2, 4, 6, 9, 12, 16, 20, 25, 31, 37, 43, 50, 58, 66, 75],
        "phi_kernel": [9, 9, 9, 9, 9, 9],
        "dimension": 1024,
        "latent_dim": 64,
        "disc_win_lengths": [1024, 2048, 512, 256, 128],
        "disc_hop_lengths": [256, 512, 128, 64, 32],
        "disc_n_ffts": [1024, 2048, 512, 256, 128],
    }

    lines = [
        "# Auto-generated by scripts/utils/prepare_sat_reproduced.py",
        "# SAT architecture follows qiuk2/AAR config/train/SAT.yaml.",
    ]
    for key, value in cfg.items():
        if isinstance(value, list):
            lines.append(f"{key}: {yaml_list(value)}")
        else:
            lines.append(f"{key}: {yaml_scalar(value)}")
    lines.append("")

    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text("\n".join(lines), encoding="utf-8")


def patch_aar_repo(aar_repo: Path) -> None:
    datasets_dir = aar_repo / "datasets"
    if not datasets_dir.exists():
        raise FileNotFoundError(f"Missing AAR datasets directory: {datasets_dir}")

    build_path = datasets_dir / "build.py"
    backup_path = datasets_dir / "build.py.sat_reproduced.bak"
    if build_path.exists() and not backup_path.exists():
        backup_path.write_text(build_path.read_text(encoding="utf-8"), encoding="utf-8")

    build_path.write_text(BUILD_PY, encoding="utf-8")
    (datasets_dir / "local_audio.py").write_text(LOCAL_AUDIO_PY, encoding="utf-8")
    patch_train_script(aar_repo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aar-repo", type=Path, required=True)
    parser.add_argument("--config-out", type=Path, required=True)
    parser.add_argument("--train-dirs", required=True, help="Colon-separated recursive audio roots.")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--tensor-cut", type=int, default=24000)
    parser.add_argument("--fixed-length", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=250000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--warmup-epoch", type=int, default=2)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16", "fp8"], default="no")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--resume", default="latest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-prefetcher", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    aar_repo = args.aar_repo.resolve()
    if not aar_repo.exists():
        raise FileNotFoundError(f"AAR repo not found: {aar_repo}")

    train_dirs = parse_dirs(args.train_dirs)
    if not train_dirs:
        raise ValueError("--train-dirs must contain at least one path")

    patch_aar_repo(aar_repo)
    write_config(args, train_dirs)
    print(f"[SAT reproduced] patched AAR repo: {aar_repo}")
    print(f"[SAT reproduced] wrote config: {args.config_out}")


if __name__ == "__main__":
    main()
