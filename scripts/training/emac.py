import os
import sys
import warnings
import math
import inspect
from dataclasses import dataclass
from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal
from audiotools import ml
from audiotools.core import util
from audiotools.data import transforms
from audiotools.data.datasets import AudioDataset
from audiotools.data.datasets import AudioLoader
from audiotools.data.datasets import ConcatDataset
from audiotools.ml.decorators import timer
from audiotools.ml.decorators import Tracker
from audiotools.ml.decorators import when
from torch.utils.tensorboard import SummaryWriter

import emac

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Enable cudnn autotuner to speed up training
# (can be altered by the funcs.seed function)
if os.getenv("EMAC_DISABLE_CUDNN", "0") == "1":
    torch.backends.cudnn.enabled = False
if os.getenv("EMAC_DISABLE_TF32", "0") == "1":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))
# Uncomment to trade memory for speed.

# Optimizers
AdamW = argbind.bind(torch.optim.AdamW, "generator", "discriminator")
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)


@argbind.bind("generator", "discriminator")
def ExponentialLR(optimizer, gamma: float = 1.0):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


# Models
EMAC = argbind.bind(emac.model.EMAC)
Discriminator = argbind.bind(emac.model.Discriminator)

# Data
AudioDataset = argbind.bind(AudioDataset, "train", "val")
AudioLoader = argbind.bind(AudioLoader, "train", "val")

# Transforms
filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "BaseTransform",
    "Compose",
    "Choose",
]
tfm = argbind.bind_module(transforms, "train", "val", filter_fn=filter_fn)

# Loss
filter_fn = lambda fn: hasattr(fn, "forward") and "Loss" in fn.__name__
losses = argbind.bind_module(emac.nn.loss, filter_fn=filter_fn)


def _nan_probe_enabled():
    return os.getenv("EMAC_NAN_PROBE", "0") == "1"


def _finite_summary(name, x, step=None):
    if not _nan_probe_enabled():
        return True
    try:
        if isinstance(x, AudioSignal):
            x = x.audio_data
        if isinstance(x, (list, tuple)):
            ok = True
            for i, v in enumerate(x):
                ok = _finite_summary(f"{name}[{i}]", v, step) and ok
            return ok
        if not torch.is_tensor(x):
            print(f"[NAN_PROBE step={step}] {name}: non_tensor {type(x).__name__}", flush=True)
            return True
        with torch.no_grad():
            finite = torch.isfinite(x)
            ok = bool(finite.all().item())
            xf = x.detach().float()
            finite_ratio = float(finite.float().mean().item())
            safe = torch.nan_to_num(xf, nan=0.0, posinf=0.0, neginf=0.0)
            print(
                f"[NAN_PROBE step={step}] {name}: ok={ok} "
                f"finite_ratio={finite_ratio:.6f} shape={tuple(x.shape)} "
                f"mean={float(safe.mean().item()):.6g} "
                f"absmax={float(safe.abs().max().item()):.6g}",
                flush=True,
            )
            return ok
    except Exception as e:
        print(f"[NAN_PROBE step={step}] {name}: probe_error={repr(e)}", flush=True)
        return False


def _module_param_probe(name, module, step=None, max_bad=3):
    if not _nan_probe_enabled():
        return True
    ok = True
    bad = 0
    with torch.no_grad():
        for pn, p in module.named_parameters():
            if not torch.isfinite(p).all().item():
                ok = False
                bad += 1
                pf = torch.nan_to_num(p.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
                print(
                    f"[NAN_PROBE step={step}] PARAM_BAD {name}.{pn}: "
                    f"shape={tuple(p.shape)} mean={float(pf.mean().item()):.6g} "
                    f"absmax={float(pf.abs().max().item()):.6g}",
                    flush=True,
                )
                if bad >= max_bad:
                    break
    print(f"[NAN_PROBE step={step}] PARAM_SUMMARY {name}: ok={ok} bad_count_shown={bad}", flush=True)
    return ok


def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def _vq_stage_label(generator, stage_idx: int) -> str:
    quantizer = getattr(generator, "quantizer", None)
    scales = getattr(quantizer, "scale_factors", None)
    if scales is None or stage_idx >= len(scales):
        return f"stage_{stage_idx:02d}"
    scale = float(scales[stage_idx])
    return f"stage_{stage_idx:02d}_scale_{scale:.4g}"


def _arg_get(args, key, default=None):
    if isinstance(args, dict):
        return args.get(key, default)
    return default


def _infer_training_rvq_frame_size(args, generator):
    """Infer the EMAC/RVQ latent frame window used by the training dataset."""
    explicit = _arg_get(args, "EMAC.rvq_frame_size", None)
    if explicit is None:
        explicit = _arg_get(args, "generator/EMAC.rvq_frame_size", None)
    if explicit is not None:
        return int(explicit)

    duration = _arg_get(args, "train/AudioDataset.duration", None)
    if duration is None:
        duration = _arg_get(args, "AudioDataset.duration", None)
    if duration is None:
        return None

    sample_rate = int(getattr(generator, "sample_rate", 1))
    hop_length = int(getattr(generator, "hop_length", 1))
    return int(math.ceil(float(duration) * sample_rate / max(1, hop_length)))


def _attach_rvq_frame_size(generator, rvq_frame_size):
    if rvq_frame_size is None:
        return
    rvq_frame_size = int(rvq_frame_size)
    setattr(generator, "rvq_frame_size", rvq_frame_size)
    if hasattr(generator, "quantizer"):
        setattr(generator.quantizer, "rvq_frame_size", rvq_frame_size)


def _configure_guided_upsample_stages(generator):
    quantizer = getattr(generator, "quantizer", None)
    configure = getattr(quantizer, "configure_guided_upsample_stages", None)
    if callable(configure):
        return configure()
    return None


def _sync_emac_runtime_metadata(generator, metadata=None):
    """Keep saved constructor kwargs aligned with mutable runtime modules.

    BaseModel.save() records kwargs by looking for attributes on the model with
    the same names as __init__ arguments.  Older checkpoints may not have model
    attributes for fields that live inside the quantizer, so mirror the actual
    runtime values before packaging or writing metadata.pth.
    """
    quantizer = getattr(generator, "quantizer", None)
    if quantizer is not None:
        mirrored = {
            "quantizer_dropout": getattr(quantizer, "quantizer_dropout", None),
            "quantizer_dropout_mode": getattr(quantizer, "quantizer_dropout_mode", None),
            "quantizer_pooling": getattr(quantizer, "quantizer_pooling", None),
            "quantizer_pooling_alpha": getattr(quantizer, "quantizer_pooling_alpha", None),
            "quantizer_pooling_power": getattr(quantizer, "quantizer_pooling_power", None),
            "quantizer_loss_target": getattr(quantizer, "quantizer_loss_target", None),
            "guided_upsample": getattr(quantizer, "guided_upsample", None),
            "guided_upsample_hidden_dim": getattr(quantizer, "guided_upsample_hidden_dim", None),
            "guided_upsample_kernel": getattr(quantizer, "guided_upsample_kernel", None),
            "guided_upsample_detach_guide": getattr(quantizer, "guided_upsample_detach_guide", None),
            "guided_upsample_init_scale": getattr(quantizer, "guided_upsample_init_scale", None),
            "guided_upsample_after_pivot_only": getattr(quantizer, "guided_upsample_after_pivot_only", None),
            "guided_upsample_decoder_grad_alpha": getattr(
                quantizer, "guided_upsample_decoder_grad_alpha", None
            ),
        }
        for key, value in mirrored.items():
            if value is not None:
                setattr(generator, key, value)

    if metadata is None:
        return None

    kwargs = {}
    signature = inspect.signature(generator.__class__)
    for key, parameter in signature.parameters.items():
        if key == "self":
            continue
        if hasattr(generator, key):
            kwargs[key] = getattr(generator, key)
        elif parameter.default is not inspect.Parameter.empty:
            kwargs[key] = parameter.default
    metadata["kwargs"] = kwargs
    return metadata


@argbind.bind("train", "val")
def build_transform(
    augment_prob: float = 1.0,
    preprocess: list = ["Identity"],
    augment: list = ["Identity"],
    postprocess: list = ["Identity"],
):
    def to_tfm(names):
        out = []
        for x in names:
            # 1) 커스텀에 있으면 커스텀 사용, 없으면 기본 tfm에서 찾기
            out.append(getattr(tfm, x)())
        return out

    preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
    augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
    postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    return transforms.Compose(preprocess, augment, postprocess)


@argbind.bind("train", "val", "test")
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

    dataset = ConcatDataset(datasets)
    dataset.transform = transform
    return dataset


@dataclass
class State:
    generator: EMAC # type:ignore
    optimizer_g: AdamW # type:ignore
    scheduler_g: ExponentialLR # type:ignore

    discriminator: Discriminator # type:ignore
    optimizer_d: AdamW # type:ignore
    scheduler_d: ExponentialLR

    stft_loss: losses.MultiScaleSTFTLoss # type:ignore
    mel_loss: losses.MelSpectrogramLoss # type:ignore
    gan_loss: losses.GANLoss # type:ignore
    waveform_loss: losses.L1Loss # type:ignore

    train_data: AudioDataset # type:ignore
    val_data: AudioDataset # type:ignore

    tracker: Tracker


@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    resume: bool = False,
    tag: str = "latest",
    load_weights: bool = False,
):
    generator, g_extra = None, {}
    discriminator, d_extra = None, {}
    
    if resume:
        kwargs = {
            "folder": f"{save_path}/{tag}",
            "map_location": "cpu",
            "package": not load_weights,
            "weights_only": False
        }
        tracker.print(f"Resuming from {str(Path('.').absolute())}/{kwargs['folder']}")
        if (Path(kwargs["folder"]) / "emac").exists():
            generator, g_extra = EMAC.load_from_folder(**kwargs)
        if (Path(kwargs["folder"]) / "discriminator").exists():
            discriminator, d_extra = Discriminator.load_from_folder(**kwargs)

    generator = EMAC() if generator is None else generator
    guided_mask = _configure_guided_upsample_stages(generator)
    discriminator = Discriminator() if discriminator is None else discriminator

    tracker.print(generator)
    tracker.print(discriminator)
    if guided_mask is not None and any(guided_mask):
        tracker.print(f"Guided upsample trainable stages: {guided_mask}")
            
    generator = accel.prepare_model(generator)
    discriminator = accel.prepare_model(discriminator)

    with argbind.scope(args, "generator"):
        optimizer_g = AdamW(generator.parameters(), use_zero=accel.use_ddp)
        scheduler_g = ExponentialLR(optimizer_g)
    with argbind.scope(args, "discriminator"):
        optimizer_d = AdamW(discriminator.parameters(), use_zero=accel.use_ddp)
        scheduler_d = ExponentialLR(optimizer_d)

    if "optimizer.pth" in g_extra:
        optimizer_g.load_state_dict(g_extra["optimizer.pth"])
    if "scheduler.pth" in g_extra:
        scheduler_g.load_state_dict(g_extra["scheduler.pth"])
    if "tracker.pth" in g_extra:
        tracker.load_state_dict(g_extra["tracker.pth"])

    if "optimizer.pth" in d_extra:
        optimizer_d.load_state_dict(d_extra["optimizer.pth"])
    if "scheduler.pth" in d_extra:
        scheduler_d.load_state_dict(d_extra["scheduler.pth"])

    sample_rate = accel.unwrap(generator).sample_rate
    
    with argbind.scope(args, "train"):
        train_data = build_dataset(sample_rate)
    with argbind.scope(args, "val"):
        val_data = build_dataset(sample_rate)

    rvq_frame_size = _infer_training_rvq_frame_size(args, accel.unwrap(generator))
    _attach_rvq_frame_size(accel.unwrap(generator), rvq_frame_size)
    if rvq_frame_size is not None:
        tracker.print(f"RVQ frame_size inferred for checkpoint metadata: {rvq_frame_size}")

    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()
    gan_loss = losses.GANLoss(discriminator)

    return State(
        generator=generator,
        optimizer_g=optimizer_g,
        scheduler_g=scheduler_g,
        discriminator=discriminator,
        optimizer_d=optimizer_d,
        scheduler_d=scheduler_d,
        waveform_loss=waveform_loss,
        stft_loss=stft_loss,
        mel_loss=mel_loss,
        gan_loss=gan_loss,
        tracker=tracker,
        train_data=train_data,
        val_data=val_data,
    )


@timer()
@torch.no_grad()
def val_loop(batch, state, accel):
    state.generator.eval()
    batch = util.prepare_batch(batch, accel.device)
    signal = state.val_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )

    out = state.generator(signal.audio_data, signal.sample_rate)
    recons = AudioSignal(out["audio"], signal.sample_rate)

    return {
        "loss": state.mel_loss(recons, signal),
        "mel/loss": state.mel_loss(recons, signal),
        "stft/loss": state.stft_loss(recons, signal),
        "waveform/loss": state.waveform_loss(recons, signal),
    }


@timer()
def train_loop(state, batch, accel, lambdas, tracker, num_iters):
    state.generator.train()
    state.discriminator.train()
    output = {}
    step = getattr(tracker, "step", None)
    
    batch = util.prepare_batch(batch, accel.device)
    _finite_summary("batch.signal.raw", batch.get("signal"), step)
    with torch.no_grad():
        signal = state.train_data.transform(
            batch["signal"].clone(), **batch["transform_args"]
        )
    _finite_summary("signal.after_transform", signal, step)

    with accel.autocast():
        out = state.generator(signal.audio_data, signal.sample_rate)
        recons = AudioSignal(out["audio"], signal.sample_rate)

        commitment_loss = out["vq/commitment_loss"]        
        codebook_loss = out["vq/codebook_loss"]
        aux_loss = out["vq/aux_loss"]
    _finite_summary("generator.out.audio", out["audio"], step)
    _finite_summary("recons.audio", recons, step)
    _finite_summary("vq.commitment_loss", commitment_loss, step)
    _finite_summary("vq.codebook_loss", codebook_loss, step)
    _finite_summary("vq.aux_loss", aux_loss, step)

    with accel.autocast():
        output["adv/disc_loss"] = state.gan_loss.discriminator_loss(recons, signal)
    _finite_summary("loss.adv_disc_before_backward", output["adv/disc_loss"], step)

    state.optimizer_d.zero_grad()
    accel.backward(output["adv/disc_loss"])
    accel.scaler.unscale_(state.optimizer_d)
    output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(
        state.discriminator.parameters(), 10.0
    )
    _finite_summary("grad_norm_d", output["other/grad_norm_d"], step)
    accel.step(state.optimizer_d)
    state.scheduler_d.step()
    _module_param_probe("discriminator.after_step", accel.unwrap(state.discriminator), step)

    with accel.autocast():
        output["stft/loss"] = state.stft_loss(recons, signal)
        output["mel/loss"] = state.mel_loss(recons, signal)
        output["waveform/loss"] = state.waveform_loss(recons, signal)
        (
            output["adv/gen_loss"],
            output["adv/feat_loss"],
        ) = state.gan_loss.generator_loss(recons, signal)
        pivot_idx = len(commitment_loss) // 2
        output["vq/c_loss_first"] = commitment_loss[0]
        output["vq/c_loss_pivot"] = commitment_loss[pivot_idx]
        output["vq/c_loss_last"] = commitment_loss[-1]
        output["vq/commitment_loss"] = sum(commitment_loss)
        output["vq/codebook_loss"] = sum(codebook_loss)
        output["vq/aux_loss"] = aux_loss
        output["loss"] = sum([v * output[k] for k, v in lambdas.items() if k in output])
    for _k in ["stft/loss", "mel/loss", "waveform/loss", "adv/gen_loss", "adv/feat_loss", "vq/commitment_loss", "vq/codebook_loss", "loss"]:
        _finite_summary(f"output.{_k}", output[_k], step)
        
    state.optimizer_g.zero_grad()
    accel.backward(output["loss"])
    accel.scaler.unscale_(state.optimizer_g)
    output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
        state.generator.parameters(), 1e3
    )
    _finite_summary("grad_norm_g", output["other/grad_norm"], step)
    accel.step(state.optimizer_g)
    state.scheduler_g.step()
    accel.update()
    _module_param_probe("generator.after_step", accel.unwrap(state.generator), step)

    output["other/learning_rate"] = state.optimizer_g.param_groups[0]["lr"]
    output["other/batch_size"] = signal.batch_size * accel.world_size

    return {k: v for k, v in sorted(output.items())}


def checkpoint(state, save_iters, save_path):
    generator = accel.unwrap(state.generator)
    rvq_frame_size = getattr(generator, "rvq_frame_size", None)
    metadata = {
        "logs": state.tracker.history,
        "rvq_frame_size": rvq_frame_size,
        "frame_size": rvq_frame_size,
    }
    _sync_emac_runtime_metadata(generator, metadata)

    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")
    if state.tracker.is_best("val", "mel/loss"):
        state.tracker.print(f"Best generator so far")
        tags.append("best")
    if state.tracker.step in save_iters:
        tags.append(f"{state.tracker.step // 1000}k")

    for tag in tags:
        generator_extra = {
            "optimizer.pth": state.optimizer_g.state_dict(),
            "scheduler.pth": state.scheduler_g.state_dict(),
            "tracker.pth": state.tracker.state_dict(),
            "metadata.pth": metadata,
        }
        generator.metadata = metadata
        generator.save_to_folder(
            f"{save_path}/{tag}", generator_extra
        )
        discriminator_extra = {
            "optimizer.pth": state.optimizer_d.state_dict(),
            "scheduler.pth": state.scheduler_d.state_dict(),
        }
        accel.unwrap(state.discriminator).save_to_folder(
            f"{save_path}/{tag}", discriminator_extra
        )


@torch.no_grad()
def save_samples(state, val_idx, writer):
    state.tracker.print("Saving audio samples to TensorBoard")
    state.generator.eval()

    samples = [state.val_data[idx] for idx in val_idx]
    batch = state.val_data.collate(samples)
    batch = util.prepare_batch(batch, accel.device)
    signal = state.train_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )

    out = state.generator(signal.audio_data, signal.sample_rate)
    recons = AudioSignal(out["audio"], signal.sample_rate)

    audio_dict = {"recons": recons}
    if state.tracker.step == 0:
        audio_dict["signal"] = signal

    for k, v in audio_dict.items():
        for nb in range(v.batch_size):
            v[nb].cpu().write_audio_to_tb(
                f"{k}/sample_{nb}.wav", writer, state.tracker.step
            )


def validate(state, val_dataloader, accel):
    for batch in val_dataloader:
        output = val_loop(batch, state, accel)
    # Consolidate state dicts if using ZeroRedundancyOptimizer
    if hasattr(state.optimizer_g, "consolidate_state_dict"):
        state.optimizer_g.consolidate_state_dict()
        state.optimizer_d.consolidate_state_dict()
    return output


@argbind.bind(without_prefix=True)
def train(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    save_path: str = "ckpt",
    num_iters: int = 250000,
    save_iters: list = [10000, 50000, 100000, 200000],
    sample_freq: int = 10000,
    valid_freq: int = 1000,
    batch_size: int = 12,
    val_batch_size: int = 10,
    num_workers: int = 8,
    val_idx: list = [0, 1, 2, 3, 4, 5, 6, 7],
    skip_initial_eval: bool = False,
    lambdas: dict = {
        "mel/loss": 100.0,
        "adv/feat_loss": 2.0,
        "adv/gen_loss": 1.0,
        "vq/commitment_loss": 0.25,
        "vq/codebook_loss": 1.0,
    },
):
    util.seed(seed)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    writer = (
        SummaryWriter(log_dir=f"{save_path}/logs") if accel.local_rank == 0 else None
    )
    tracker = Tracker(
        writer=writer, log_file=f"{save_path}/log.txt", rank=accel.local_rank
    )

    state = load(args, accel, tracker, save_path)
    train_dataloader = accel.prepare_dataloader(
        state.train_data,
        start_idx=state.tracker.step * batch_size,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.train_data.collate,
    )
    train_dataloader = get_infinite_loader(train_dataloader)
    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=val_batch_size,
        collate_fn=state.val_data.collate,
        persistent_workers=True if num_workers > 0 else False,
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
            train_loop(state, batch, accel, lambdas, tracker, num_iters)

            last_iter = (
                tracker.step == num_iters - 1 if num_iters is not None else False
            )
            do_initial_eval = not (skip_initial_eval and tracker.step == 0)
            if do_initial_eval and (tracker.step % sample_freq == 0 or last_iter):
                save_samples(state, val_idx, writer)

            if do_initial_eval and (tracker.step % valid_freq == 0 or last_iter):
                validate(state, val_dataloader, accel)
                checkpoint(state, save_iters, save_path)
                # Reset validation progress bar, print summary since last validation.
                tracker.done("val", f"Iteration {tracker.step}")

            if last_iter:
                break


if __name__ == "__main__":
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            train(args, accel)
