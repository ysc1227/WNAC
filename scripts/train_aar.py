import math
import os
import torch

from pathlib import Path
import sys

from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from audiotools import AudioSignal
from audiotools.core import util
from audiotools import ml
from audiotools.ml.decorators import Tracker
from audiotools.data.datasets import AudioDataset as AD
from audiotools.data.datasets import AudioLoader as AL
from audiotools.data.datasets import ConcatDataset
from wnac.datasets import ConditionDataset
from audiotools.data import transforms
from audiotools.ml.decorators import when
from audiotools.ml.decorators import timer
import argbind
import wnac
from wnac.utils import load_model
from transformers import ClapModel


AdamW = argbind.bind(torch.optim.AdamW, "regressor")
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)

@argbind.bind("regressor")
def CosineLinearWarmupLR(
    optimizer,
    num_warmup_steps: int = 1000,
    num_training_steps: int = 200000,
    scheme: str = 'lin0', # 'cosine', 'lin', 'lin0', 'exp', ...
    start_lr_ratio: float = 0.005,
    end_lr_ratio: float = 0.001,
):
    def lr_lambda(current_step):
        wp_it = num_warmup_steps
        max_it = num_training_steps
        cur_it = current_step

        if cur_it < wp_it:
            return start_lr_ratio + (1.0 - start_lr_ratio) * (cur_it / wp_it)

        pasd = (cur_it - wp_it) / (max_it - 1 - wp_it)
        rest = 1 - pasd

        if scheme == 'cosine':
            return end_lr_ratio + (1.0 - end_lr_ratio) * (0.5 + 0.5 * math.cos(math.pi * pasd))
        elif scheme == 'lin':
            T = 0.15
            if pasd < T:
                return 1.0
            else:
                return end_lr_ratio + (1.0 - end_lr_ratio) * rest / (1 - T)
        elif scheme == 'lin0':
            T = 0.05
            if pasd < T:
                return 1.0
            else:
                return end_lr_ratio + (1.0 - end_lr_ratio) * rest / (1 - T)
        elif scheme == 'lin00':
            return end_lr_ratio + (1.0 - end_lr_ratio) * rest
        elif scheme.startswith('lin'):
            T = float(scheme[3:])
            max_rest = 1 - T
            wpe_mid = (1.0 + (end_lr_ratio + (1.0 - end_lr_ratio) * max_rest)) / 2
            if pasd < T:
                return 1.0 + (wpe_mid - 1.0) * (pasd / T)
            else:
                return end_lr_ratio + (wpe_mid - end_lr_ratio) * rest / max_rest
        elif scheme == 'exp':
            T = 0.15
            max_rest = 1 - T
            if pasd < T:
                return 1.0
            else:
                expo = (pasd - T) / max_rest * math.log(end_lr_ratio)
                return math.exp(expo)
        else:
            raise NotImplementedError(f'unknown scheme: {scheme}')

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Data
AudioDataset = argbind.bind(AD, "train", "val")
AudioLoader = argbind.bind(AL, "train", "val")

# Transforms
filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "BaseTransform",
    "Compose",
    "Choose",
]
tfm = argbind.bind_module(transforms, "train", "val", filter_fn=filter_fn)

# Loss
filter_fn = lambda fn: hasattr(fn, "forward") and "Loss" in fn.__name__
losses = argbind.bind_module(torch.nn, filter_fn=filter_fn)

WNAC = argbind.bind(wnac.model.WNAC)
AAR = argbind.bind(wnac.model.AAR)

def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch

@dataclass
class State:
    regressor: wnac.model.AAR
    optimizer: torch.optim.AdamW
    scheduler: CosineLinearWarmupLR

    ce_loss: torch.nn.CrossEntropyLoss
    train_data: AD
    val_data: AL

    tracker: Tracker

@argbind.bind("train", "val")
def build_transform(
    augment_prob: float = 1.0,
    preprocess: list = ["Identity"],
    augment: list = ["Identity"],
    postprocess: list = ["Identity"],
):
    to_tfm = lambda l: [getattr(tfm, x)(prob=augment_prob / len(augment)) for x in l]
    preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
    augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
    postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    transform = transforms.Compose(preprocess, augment, postprocess)
    return transform

@argbind.bind("train", "val")
def build_dataset(
    sample_rate: int,
    folders: dict = None,
):
    # Give one loader per key/value of dictionary, where
    # value is a list of folders. Create a dataset for each one.
    # Concatenate the datasets with ConcatDataset, which
    # cycles through them.
    datasets = []
    for _, v in folders.items():
        loader = AudioLoader(sources=v)
        transform = build_transform()
        dataset = AudioDataset(loader, sample_rate, transform=transform)
        datasets.append(dataset)

    dataset = ConditionDataset(ConcatDataset(datasets))
    dataset.transform = transform
    return dataset

def load_encoder(
    path,
    model_tag: str = "latest",
    model_bitrate: str = "8kbps",
    device: str = "cuda",
    model_type: str = "44khz",
):
    print(path)
    encoder = load_model(
        model_type=model_type,
        model_bitrate=model_bitrate,
        tag=model_tag,
        load_path=path,
    )
    
    encoder.to(device)
    encoder.eval()
    
    return encoder

@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    encoder: wnac.model.WNAC,
    resume: bool = False,
    tag: str = "latest",
    load_weights: bool = False
):
    regressor, r_extra = None, {}

    if resume:
        kwargs = {
            "folder": f"{save_path}/{tag}",
            "map_location": "cpu",
            "package": not load_weights,
            "weights_only": False
        }
        tracker.print(f"Resuming from {str(Path('.').absolute())}/{kwargs['folder']}")
        if (Path(kwargs["folder"]) / "aar").exists():
            regressor, r_extra = AAR.load_from_folder(**kwargs)

    max_patch_size = encoder.compress(AudioSignal(torch.Tensor(1, 1, args['win_dur'] * encoder.sample_rate), sample_rate=encoder.sample_rate), win_duration=args['win_dur']).codes[0][-1].shape[-1]
    regressor = AAR(max_patch_size=max_patch_size, vae_local=encoder) if regressor is None else regressor
    tracker.print(regressor)
    regressor = accel.prepare_model(regressor)

    with argbind.scope(args, "regressor"):
        optimizer = AdamW(regressor.parameters(), use_zero=accel.use_ddp)
        scheduler = CosineLinearWarmupLR(optimizer)

    if "optimizer.pth" in r_extra:
        optimizer.load_state_dict(r_extra["optimizer.pth"])
    if "scheduler.pth" in r_extra:
        scheduler.load_state_dict(r_extra["scheduler.pth"])
    if "tracker.pth" in r_extra:
        tracker.load_state_dict(r_extra["tracker.pth"])

    sample_rate = accel.unwrap(regressor).sample_rate
    
    with argbind.scope(args, "train"):
        train_data = build_dataset(sample_rate)
    with argbind.scope(args, "val"):
        val_data = build_dataset(sample_rate)

    ce_loss = losses.CrossEntropyLoss(reduction='none')

    return State(
        regressor=regressor,
        optimizer=optimizer,
        scheduler=scheduler,
        ce_loss=ce_loss,
        tracker=tracker,
        train_data=train_data,
        val_data=val_data,
    )

def checkpoint(state, save_iters, save_freq, save_path):
    metadata = {"logs": state.tracker.history}

    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")
    if state.tracker.is_best("val", "ce_loss"):
        state.tracker.print(f"Best generator so far")
        tags.append("best")
    if state.tracker.step in save_iters or (save_freq and state.tracker.step % save_freq == 0):
        tags.append(f"{state.tracker.step // 1000}k")

    for tag in tags:
        regressor_extra = {
            "optimizer.pth": state.optimizer.state_dict(),
            "scheduler.pth": state.scheduler.state_dict(),
            "tracker.pth": state.tracker.state_dict(),
            "metadata.pth": metadata,
        }
        accel.unwrap(state.regressor).metadata = metadata
        accel.unwrap(state.regressor).save_to_folder(
            f"{save_path}/{tag}", regressor_extra
        )

@torch.no_grad()
def save_samples(state, val_idx, cond_model, writer, guidance_scale=4.0, top_k=900, top_p=0.95, seed=42):
    state.tracker.print("Saving audio samples to TensorBoard")
    state.regressor.eval()
    cond_model.eval()

    samples = [state.val_data[idx] for idx in val_idx]
    batch = state.val_data.collate(samples)
    batch = util.prepare_batch(batch, accel.device)
    signal = state.train_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )
    cond = batch["condition"]

    condition = cond_model.get_audio_features(**cond)
    audios = state.regressor.module.autoregressive_infer_cfg(B=len(condition), label_B=condition, cfg=guidance_scale, top_k=top_k, top_p=top_p, g_seed=seed)

    audio_dict = {"generated": AudioSignal(audios, sample_rate=signal.sample_rate)}
    if state.tracker.step == 0:
        audio_dict["signal"] = signal

    for k, v in audio_dict.items():
        for nb in range(v.batch_size):
            v[nb].cpu().write_audio_to_tb(
                f"{k}/sample_{nb}.wav", writer, state.tracker.step
            )

def validate(state, val_dataloader, encoder, cond_model, accel, win_dur):
    for batch in val_dataloader:
        output = val_loop(batch, state, encoder, cond_model, accel, win_dur)
    # Consolidate state dicts if using ZeroRedundancyOptimizer
    if hasattr(state.optimizer, "consolidate_state_dict"):
        state.optimizer.consolidate_state_dict()
    return output

@argbind.bind(without_prefix=True)
def train(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    save_path: str = "ckpt",
    encoder_path: str = "",
    num_iters: int = 250000,
    save_iters: list = [10000, 50000, 100000, 200000],
    save_freq: int = None,
    sample_freq: int = 10000,
    valid_freq: int = 1000,
    batch_size: int = 48,
    val_batch_size: int = 10,
    num_workers: int = 8,
    val_idx: list = [0, 1, 2, 3, 4, 5, 6, 7],
    win_dur: int = 1
):
    util.seed(seed)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    writer = (
        SummaryWriter(log_dir=f"{save_path}/logs") if accel.local_rank == 0 else None
    )
    tracker = Tracker(
        writer=writer, log_file=f"{save_path}/log.txt", rank=accel.local_rank
    )

    encoder = load_encoder(encoder_path, accel, 'best')
    cond_model = ClapModel.from_pretrained("laion/larger_clap_general").to("cuda")
    state = load(args, accel, tracker, save_path, encoder, win_dur=win_dur)
    train_dataloader = accel.prepare_dataloader(
        state.train_data,
        start_idx=state.tracker.step * batch_size,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.train_data.collate
    )
    
    train_dataloader = get_infinite_loader(train_dataloader)
    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=val_batch_size,
        collate_fn=state.val_data.collate,
        persistent_workers=True if num_workers > 0 else False
    )

    # Wrap the functions so that they neatly track in TensorBoard + progress bars
    # and only run when specific conditions are met.
    global train_loop, val_loop, validate, save_samples, checkpoint
    train_loop = tracker.log("train", "value", history=False)(
        tracker.track("train", num_iters, completed=state.tracker.step)(train_loop)
    )
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate)

    # These functions run only on the 0-rank process
    save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)
    
    with tracker.live:
        for tracker.step, batch in enumerate(train_dataloader, start=tracker.step):
            train_loop(state, encoder, cond_model, batch, accel, win_dur)

            last_iter = (
                tracker.step == num_iters - 1 if num_iters is not None else False
            )
            if tracker.step % sample_freq == 0 or last_iter:
                save_samples(state, val_idx, cond_model, writer)

            if tracker.step % valid_freq == 0 or last_iter:
                validate(state, val_dataloader, encoder, cond_model, accel, win_dur)
                checkpoint(state, save_iters, save_freq, save_path)
                # Reset validation progress bar, print summary since last validation.
                tracker.done("val", f"Iteration {tracker.step}")

            if last_iter:
                break

@timer()
@torch.no_grad()
def val_loop(batch, state, encoder, cond_model, accel, win_dur):
    state.regressor.eval()
    batch = util.prepare_batch(batch, accel.device)
    signal = state.val_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )
    cond = batch["condition"]
    conditions = cond_model.get_audio_features(**cond)
    
    labels_list = encoder.compress(signal, win_duration=win_dur).codes[0]
    latent = encoder.quantizer.idx_to_var_input(labels_list)
    logits = state.regressor(conditions, torch.concat(latent, dim=1)) # (B, E), (B, c_len, Cvae)
    
    b, l, v = logits.size()
    logits = logits.view(-1, v)
    labels = torch.cat(labels_list, dim=1)
    labels = labels.view(-1)
    loss = state.ce_loss(logits, labels).view(b, -1)
    loss = loss.mul(1.0 / l).sum(dim=-1).mean()
    
    return {
        "ce_loss": loss,
    }

@timer()
def train_loop(state, encoder, cond_model, batch, accel, win_dur):
    state.regressor.train()
    output = {}

    batch = util.prepare_batch(batch, accel.device)
    with torch.no_grad():
        signal = state.train_data.transform(
            batch["signal"].clone(), **batch["transform_args"]
        )
        cond = batch["condition"]
        conditions = cond_model.get_audio_features(**cond)
        labels_list = encoder.compress(signal, win_duration=win_dur).codes[0]
        latent = encoder.quantizer.idx_to_var_input(labels_list)
        
    with accel.autocast():
        logits = state.regressor(conditions, torch.concat(latent, dim=1)) # (B, E), (B, c_len, Cvae) -> (B, c_len, vocab)

    b, l, v = logits.size()
    logits = logits.view(-1, v)
    labels = torch.cat(labels_list, dim=1)
    labels = labels.view(-1)
    loss = state.ce_loss(logits, labels).view(b, -1)
    loss = loss.mul(1.0 / l).sum(dim=-1).mean()
    output["ce_loss"] = loss
    
    state.optimizer.zero_grad()
    accel.backward(output["ce_loss"])

    accel.scaler.unscale_(state.optimizer)
    output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
        state.regressor.parameters(), 2
    )
    accel.step(state.optimizer)
    state.scheduler.step()
    accel.update()

    output["other/learning_rate"] = state.optimizer.param_groups[0]["lr"]
    output["other/batch_size"] = signal.batch_size * accel.world_size

    return {k: v for k, v in sorted(output.items())}

if __name__ == '__main__':
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            train(args, accel)