import os
import sys
import warnings
import math
import inspect
import json
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
SATCodec = argbind.bind(emac.model.SATCodec)
SNACCodec = argbind.bind(emac.model.SNACCodec)
Discriminator = argbind.bind(emac.model.Discriminator)
SATDiscriminator = argbind.bind(emac.model.SATDiscriminator)

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


@torch.no_grad()
def _dual_codebook_metrics(generator):
    quantizer = getattr(generator, "quantizer", None)
    stages = getattr(quantizer, "quantizers", None)
    if stages is None:
        return {}
    dual_stages = [
        stage for stage in stages
        if getattr(stage, "separate_lookup_codebook", False)
    ]
    if not dual_stages:
        return {}

    lookup_losses = [
        stage.last_lookup_loss.float()
        for stage in dual_stages
        if getattr(stage, "last_lookup_loss", None) is not None
    ]
    relative_distances = []
    cosine_similarities = []
    for stage in dual_stages:
        lookup = stage.lookup_codebook.weight.detach().float()
        reconstruction = stage.codebook.weight.detach().float()
        relative_distances.append(
            (lookup - reconstruction).norm()
            / reconstruction.norm().clamp_min(1e-8)
        )
        cosine_similarities.append(
            torch.nn.functional.cosine_similarity(
                lookup,
                reconstruction,
                dim=1,
            ).mean()
        )

    metrics = {
        "vq/codebook_separation_rel": torch.stack(relative_distances).mean(),
        "vq/codebook_cosine": torch.stack(cosine_similarities).mean(),
    }
    if lookup_losses:
        metrics["vq/lookup_loss"] = torch.stack(lookup_losses).sum()
    return metrics


def _optional_ddp_parameter_anchor(generator):
    """Attach optional quantizer modules to the loss graph without changing it.

    Some residual quantizer helpers, such as Phi residual convolutions, may only
    receive gradients through auxiliary losses. When those losses are disabled
    or detached for a particular configuration, DDP treats the trainable helper
    parameters as unused and stops at the next iteration. A zero-valued anchor
    keeps DDP reductions well-defined while preserving the exact objective.
    """
    quantizer = getattr(generator, "quantizer", None)
    if quantizer is None:
        return None

    optional_modules = [
        getattr(quantizer, "quant_resi", None),
    ]
    anchors = []
    for module in optional_modules:
        if module is None:
            continue
        anchors.extend(p.sum() * 0.0 for p in module.parameters() if p.requires_grad)

    if not anchors:
        return None

    anchor = anchors[0]
    for item in anchors[1:]:
        anchor = anchor + item
    return anchor


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
            "quantizer_upsample_mode": getattr(quantizer, "quantizer_upsample_mode", None),
            "quantizer_codebook_update": getattr(quantizer, "quantizer_codebook_update", None),
            "quantizer_decoder_grad_alpha": getattr(quantizer, "quantizer_decoder_grad_alpha", None),
            "quantizer_residual_detach": getattr(quantizer, "quantizer_residual_detach", None),
            "quantizer_shared_projection": getattr(quantizer, "quantizer_shared_projection", None),
            "quantizer_separate_lookup_codebook": getattr(
                quantizer, "quantizer_separate_lookup_codebook", None
            ),
            "quantizer_lookup_commitment_weight": getattr(
                quantizer, "quantizer_lookup_commitment_weight", None
            ),
            "quantizer_lookup_codebook_weight": getattr(
                quantizer, "quantizer_lookup_codebook_weight", None
            ),
            "ema_decay": getattr(quantizer, "ema_decay", None),
            "ema_epsilon": getattr(quantizer, "ema_epsilon", None),
            "ema_kmeans_init": getattr(quantizer, "ema_kmeans_init", None),
            "ema_kmeans_iters": getattr(quantizer, "ema_kmeans_iters", None),
            "ema_threshold_ema_dead_code": getattr(quantizer, "ema_threshold_ema_dead_code", None),
            "quantizer_scale_anneal": getattr(quantizer, "quantizer_scale_anneal", None),
            "quantizer_scale_anneal_steps": getattr(quantizer, "quantizer_scale_anneal_steps", None),
            "quantizer_scale_anneal_start": getattr(quantizer, "quantizer_scale_anneal_start", None),
            "quantizer_scale_anneal_mode": getattr(quantizer, "quantizer_scale_anneal_mode", None),
            "learned_downsample": getattr(quantizer, "learned_downsample", None),
            "learned_downsample_hidden_dim": getattr(quantizer, "learned_downsample_hidden_dim", None),
            "learned_downsample_kernel": getattr(quantizer, "learned_downsample_kernel", None),
            "learned_downsample_init_scale": getattr(quantizer, "learned_downsample_init_scale", None),
            "learned_downsample_max_scale": getattr(quantizer, "learned_downsample_max_scale", None),
            "guided_upsample": getattr(quantizer, "guided_upsample", None),
            "guided_upsample_hidden_dim": getattr(quantizer, "guided_upsample_hidden_dim", None),
            "guided_upsample_kernel": getattr(quantizer, "guided_upsample_kernel", None),
            "guided_upsample_detach_guide": getattr(quantizer, "guided_upsample_detach_guide", None),
            "guided_upsample_init_scale": getattr(quantizer, "guided_upsample_init_scale", None),
            "guided_upsample_after_pivot_only": getattr(quantizer, "guided_upsample_after_pivot_only", None),
            "guided_upsample_decoder_grad_alpha": getattr(
                quantizer, "guided_upsample_decoder_grad_alpha", None
            ),
            "learned_upsample": getattr(quantizer, "learned_upsample", None),
            "learned_upsample_mode": getattr(quantizer, "learned_upsample_mode", None),
            "learned_upsample_hidden_dim": getattr(quantizer, "learned_upsample_hidden_dim", None),
            "learned_upsample_kernel": getattr(quantizer, "learned_upsample_kernel", None),
            "learned_upsample_patch_size": getattr(quantizer, "learned_upsample_patch_size", None),
            "learned_upsample_init_scale": getattr(quantizer, "learned_upsample_init_scale", None),
            "learned_upsample_condition_guide": getattr(
                quantizer, "learned_upsample_condition_guide", None
            ),
            "learned_upsample_detach_guide": getattr(
                quantizer, "learned_upsample_detach_guide", None
            ),
            "learned_upsample_guide_space": getattr(quantizer, "learned_upsample_guide_space", None),
            "learned_upsample_max_scale": getattr(quantizer, "learned_upsample_max_scale", None),
            "learned_upsample_after_pivot_only": getattr(
                quantizer, "learned_upsample_after_pivot_only", None
            ),
            "learned_upsample_log_stages": getattr(quantizer, "learned_upsample_log_stages", None),
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
    generator: torch.nn.Module
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
    num_iters: int

    tracker: Tracker


def _generator_class(name: str):
    name = str(name).lower().replace("-", "_")
    aliases = {
        "default": "emac",
        "codec": "emac",
        "wnac": "emac",
        "wavescale": "emac",
        "sat": "sat",
        "satcodec": "sat",
        "sat_codec": "sat",
        "aar_sat": "sat",
        "snac": "snac",
        "snaccodec": "snac",
        "snac_codec": "snac",
        "snac44": "snac",
        "snac_44khz": "snac",
    }
    name = aliases.get(name, name)
    if name == "emac":
        return EMAC
    if name == "sat":
        return SATCodec
    if name == "snac":
        return SNACCodec
    raise ValueError(f"Unknown generator_model='{name}'. Expected 'emac', 'sat', or 'snac'.")


def _generator_checkpoint_folder(folder: str, Generator):
    folder = Path(folder)
    name = Generator.__name__.lower()
    if (folder / name).exists():
        return folder / name
    return None


def _discriminator_class(name: str):
    name = str(name).lower().replace("-", "_")
    aliases = {
        "default": "emac",
        "emac": "emac",
        "mrd": "emac",
        "wnac": "emac",
        "sat": "sat",
        "satdiscriminator": "sat",
        "sat_discriminator": "sat",
        "msstft": "sat",
        "ms_stft": "sat",
    }
    name = aliases.get(name, name)
    if name == "emac":
        return Discriminator
    if name == "sat":
        return SATDiscriminator
    raise ValueError(f"Unknown discriminator_model='{name}'. Expected 'emac' or 'sat'.")


def _frequency_loss_class(name: str):
    name = str(name).lower().replace("-", "_")
    aliases = {
        "default": "mel",
        "mel": "mel",
        "mel_spectrogram": "mel",
        "emac": "mel",
        "sat": "sat",
        "sat_frequency": "sat",
        "sat_lf": "sat",
        "aar": "sat",
    }
    name = aliases.get(name, name)
    if name == "mel":
        return losses.MelSpectrogramLoss
    if name == "sat":
        return losses.SATFrequencyLoss
    raise ValueError(f"Unknown frequency_loss='{name}'. Expected 'mel' or 'sat'.")


def _gan_loss_class(name: str):
    name = str(name).lower().replace("-", "_")
    aliases = {
        "default": "mse",
        "emac": "mse",
        "mse": "mse",
        "lsgan": "mse",
        "sat": "sat",
        "hinge": "sat",
        "sat_hinge": "sat",
    }
    name = aliases.get(name, name)
    if name == "mse":
        return losses.GANLoss
    if name == "sat":
        return losses.SATGANLoss
    raise ValueError(f"Unknown gan_loss='{name}'. Expected 'mse' or 'sat'.")


def _make_optimizer(params, optimizer_type: str, lr: float, betas: list, use_zero: bool):
    optimizer_type = str(optimizer_type).lower().replace("-", "_")
    betas = tuple(float(v) for v in betas)
    if optimizer_type == "adam":
        return torch.optim.Adam(params, lr=float(lr), betas=betas)
    if optimizer_type == "adamw":
        return AdamW(params, lr=float(lr), betas=betas, use_zero=use_zero)
    raise ValueError(f"Unknown optimizer_type='{optimizer_type}'. Expected 'adam' or 'adamw'.")


def _make_scheduler(
    optimizer,
    scheduler_type: str,
    total_steps: int,
    warmup_steps: int = 0,
    gamma: float = 0.999996,
):
    scheduler_type = str(scheduler_type).lower().replace("-", "_")
    if scheduler_type in {"exponential", "exp"}:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(gamma))
    if scheduler_type in {"cosine", "cosine_warmup"}:
        total_steps = max(1, int(total_steps))
        warmup_steps = max(0, int(warmup_steps))

        def lr_lambda(step):
            step = int(step)
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"Unknown scheduler_type='{scheduler_type}'. Expected 'exponential' or 'cosine'.")


def _should_update_discriminator(step, pattern: str):
    pattern = str(pattern).lower().replace("-", "_")
    if pattern in {"always", "default", "emac"}:
        return True
    if pattern in {"sat", "skip_first_of_three"}:
        return int(step or 0) % 3 != 0
    raise ValueError(f"Unknown discriminator_update_pattern='{pattern}'. Expected 'always' or 'sat'.")


def _discriminator_updates_before_generator(order: str):
    order = str(order).lower().replace("-", "_")
    if order in {"before_generator", "before_g", "default", "emac"}:
        return True
    if order in {"after_generator", "after_g", "sat"}:
        return False
    raise ValueError(
        f"Unknown discriminator_update_order='{order}'. "
        "Expected 'before_generator' or 'after_generator'."
    )


def _source_example_count(dataset):
    """Estimate the finite source count behind an audiotools dataset."""
    if hasattr(dataset, "datasets"):
        counts = [_source_example_count(d) for d in dataset.datasets]
        counts = [c for c in counts if c is not None]
        return sum(counts) if counts else None

    loaders = getattr(dataset, "loaders", None)
    if isinstance(loaders, dict):
        count = 0
        for loader in loaders.values():
            if hasattr(loader, "audio_indices"):
                count += len(loader.audio_indices)
            elif hasattr(loader, "audio_lists"):
                count += sum(len(src) for src in loader.audio_lists)
        return count or None
    return None


@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    generator_model: str = "emac",
    discriminator_model: str = "emac",
    frequency_loss: str = "mel",
    gan_loss: str = "mse",
    optimizer_type: str = "adamw",
    optimizer_lr: float = 1e-4,
    optimizer_betas: list = [0.8, 0.99],
    scheduler_type: str = "exponential",
    scheduler_gamma: float = 0.999996,
    scheduler_warmup_steps: int = 0,
    scheduler_warmup_epochs: int = 0,
    num_epochs: int = 0,
    resume: bool = False,
    tag: str = "latest",
    load_weights: bool = False,
):
    generator, g_extra = None, {}
    discriminator, d_extra = None, {}
    Generator = _generator_class(generator_model)
    DiscriminatorClass = _discriminator_class(discriminator_model)

    if resume:
        kwargs = {
            "folder": f"{save_path}/{tag}",
            "map_location": "cpu",
            "package": not load_weights,
            "weights_only": False
        }
        tracker.print(f"Resuming from {str(Path('.').absolute())}/{kwargs['folder']}")
        generator_folder = _generator_checkpoint_folder(kwargs["folder"], Generator)
        if generator_folder is not None:
            tracker.print(f"Found generator checkpoint folder: {generator_folder.name}")
            generator, g_extra = Generator.load_from_folder(**kwargs)
        discriminator_folder = _generator_checkpoint_folder(kwargs["folder"], DiscriminatorClass)
        if discriminator_folder is not None:
            tracker.print(f"Found discriminator checkpoint folder: {discriminator_folder.name}")
            discriminator, d_extra = DiscriminatorClass.load_from_folder(**kwargs)

    generator = Generator() if generator is None else generator
    guided_mask = _configure_guided_upsample_stages(generator)
    discriminator = DiscriminatorClass() if discriminator is None else discriminator

    tracker.print(generator)
    tracker.print(discriminator)
    if guided_mask is not None and any(guided_mask):
        tracker.print(f"Guided upsample trainable stages: {guided_mask}")
            
    generator = accel.prepare_model(generator)
    discriminator = accel.prepare_model(discriminator)

    sample_rate = accel.unwrap(generator).sample_rate
    
    with argbind.scope(args, "train"):
        train_data = build_dataset(sample_rate)
    with argbind.scope(args, "val"):
        val_data = build_dataset(sample_rate)

    rvq_frame_size = _infer_training_rvq_frame_size(args, accel.unwrap(generator))
    _attach_rvq_frame_size(accel.unwrap(generator), rvq_frame_size)
    if rvq_frame_size is not None:
        tracker.print(f"RVQ frame_size inferred for checkpoint metadata: {rvq_frame_size}")

    source_examples = _source_example_count(train_data)
    epoch_examples = source_examples if source_examples is not None else len(train_data)
    batch_size = int(_arg_get(args, "batch_size", 12) or 12)
    steps_per_epoch = math.ceil(epoch_examples / max(1, batch_size * max(1, accel.world_size)))
    configured_num_iters = int(_arg_get(args, "num_iters", 250000) or 0)
    if int(num_epochs or 0) > 0:
        num_iters = int(num_epochs) * max(1, steps_per_epoch)
    else:
        num_iters = configured_num_iters if configured_num_iters > 0 else 250000
    if int(scheduler_warmup_steps) <= 0 and int(scheduler_warmup_epochs) > 0:
        scheduler_warmup_steps = int(scheduler_warmup_epochs) * max(1, steps_per_epoch)
    tracker.print(
        f"Optimizer={optimizer_type}, lr={optimizer_lr}, betas={optimizer_betas}; "
        f"scheduler={scheduler_type}, warmup_steps={scheduler_warmup_steps}, "
        f"steps_per_epoch={steps_per_epoch}, total_steps={num_iters}"
    )

    optimizer_g = _make_optimizer(
        generator.parameters(),
        optimizer_type=optimizer_type,
        lr=optimizer_lr,
        betas=optimizer_betas,
        use_zero=accel.use_ddp and str(optimizer_type).lower() == "adamw",
    )
    optimizer_d = _make_optimizer(
        discriminator.parameters(),
        optimizer_type=optimizer_type,
        lr=optimizer_lr,
        betas=optimizer_betas,
        use_zero=accel.use_ddp and str(optimizer_type).lower() == "adamw",
    )
    scheduler_g = _make_scheduler(
        optimizer_g,
        scheduler_type=scheduler_type,
        total_steps=num_iters,
        warmup_steps=scheduler_warmup_steps,
        gamma=scheduler_gamma,
    )
    scheduler_d = _make_scheduler(
        optimizer_d,
        scheduler_type=scheduler_type,
        total_steps=num_iters,
        warmup_steps=scheduler_warmup_steps,
        gamma=scheduler_gamma,
    )

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

    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss_cls = _frequency_loss_class(frequency_loss)
    mel_loss = mel_loss_cls(sample_rate=sample_rate) if mel_loss_cls is losses.SATFrequencyLoss else mel_loss_cls()
    tracker.print(f"Using frequency loss: {mel_loss.__class__.__name__}")
    gan_loss_cls = _gan_loss_class(gan_loss)
    gan_loss = gan_loss_cls(discriminator)
    tracker.print(f"Using GAN loss: {gan_loss.__class__.__name__}")

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
        num_iters=num_iters,
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

    output = {
        "loss": state.mel_loss(recons, signal),
        "mel/loss": state.mel_loss(recons, signal),
        "stft/loss": state.stft_loss(recons, signal),
        "waveform/loss": state.waveform_loss(recons, signal),
    }
    return output


def _lambda_warmup_scale(step, schedule):
    if not schedule:
        return 1.0
    if isinstance(schedule, (int, float)):
        start = 0
        duration = float(schedule)
        min_scale = 0.0
    else:
        start = int(schedule.get("start", 0))
        duration = float(schedule.get("duration", schedule.get("steps", 0)))
        min_scale = float(schedule.get("min", schedule.get("min_scale", 0.0)))
    if duration <= 0:
        return 1.0
    if step is None:
        return min_scale
    progress = (float(step) - float(start)) / duration
    progress = max(0.0, min(1.0, progress))
    return min_scale + (1.0 - min_scale) * progress


@timer()
def train_loop(
    state,
    batch,
    accel,
    lambdas,
    lambda_warmups,
    tracker,
    num_iters,
    discriminator_update_pattern,
    discriminator_update_order,
    generator_grad_clip,
    discriminator_grad_clip,
):
    state.generator.train()
    state.discriminator.train()
    output = {}
    step = getattr(tracker, "step", None)
    generator = accel.unwrap(state.generator)
    set_step = getattr(getattr(generator, "quantizer", None), "set_training_step", None)
    if callable(set_step):
        set_step(step)
    
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
    _finite_summary("generator.out.audio", out["audio"], step)
    _finite_summary("recons.audio", recons, step)
    _finite_summary("vq.commitment_loss", commitment_loss, step)
    _finite_summary("vq.codebook_loss", codebook_loss, step)

    update_discriminator = _should_update_discriminator(step, discriminator_update_pattern)
    discriminator_before_generator = _discriminator_updates_before_generator(
        discriminator_update_order
    )
    output["adv/disc_update"] = signal.audio_data.new_tensor(float(update_discriminator))

    def discriminator_step():
        state.optimizer_d.zero_grad()
        if update_discriminator:
            with accel.autocast():
                output["adv/disc_loss"] = state.gan_loss.discriminator_loss(recons, signal)
            _finite_summary("loss.adv_disc_before_backward", output["adv/disc_loss"], step)

            accel.backward(output["adv/disc_loss"])
            accel.scaler.unscale_(state.optimizer_d)
            if float(discriminator_grad_clip) > 0:
                output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(
                    state.discriminator.parameters(), float(discriminator_grad_clip)
                )
            else:
                output["other/grad_norm_d"] = signal.audio_data.new_tensor(0.0)
            _finite_summary("grad_norm_d", output["other/grad_norm_d"], step)
            accel.step(state.optimizer_d)
        else:
            output["adv/disc_loss"] = signal.audio_data.new_tensor(0.0)
            output["other/grad_norm_d"] = signal.audio_data.new_tensor(0.0)
        state.scheduler_d.step()
        _module_param_probe("discriminator.after_step", accel.unwrap(state.discriminator), step)

    if discriminator_before_generator:
        discriminator_step()

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
        output.update({k: v for k, v in out.items() if k.startswith("downsampler/")})
        output.update({k: v for k, v in out.items() if k.startswith("scale_anneal/")})
        output.update({k: v for k, v in out.items() if k.startswith("upsampler/")})
        output.update(_dual_codebook_metrics(accel.unwrap(state.generator)))
        effective_lambdas = {}
        for name, value in lambdas.items():
            scale = _lambda_warmup_scale(step, (lambda_warmups or {}).get(name))
            effective_lambdas[name] = value * scale
            if name in output and scale != 1.0:
                output[f"lambda/{name}"] = output[name].new_tensor(effective_lambdas[name])
            if name in output:
                output[f"weighted/{name.replace('/', '_')}"] = (
                    output[name] * effective_lambdas[name]
                )
        output["loss"] = sum(
            [v * output[k] for k, v in effective_lambdas.items() if k in output]
        )
        ddp_anchor = _optional_ddp_parameter_anchor(accel.unwrap(state.generator))
        if ddp_anchor is not None:
            output["loss"] = output["loss"] + ddp_anchor
    for _k in ["stft/loss", "mel/loss", "waveform/loss", "adv/gen_loss", "adv/feat_loss", "vq/commitment_loss", "vq/codebook_loss", "loss"]:
        _finite_summary(f"output.{_k}", output[_k], step)
        
    state.optimizer_g.zero_grad()
    accel.backward(output["loss"])
    accel.scaler.unscale_(state.optimizer_g)
    if float(generator_grad_clip) > 0:
        output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
            state.generator.parameters(), float(generator_grad_clip)
        )
    else:
        output["other/grad_norm"] = signal.audio_data.new_tensor(0.0)
    _finite_summary("grad_norm_g", output["other/grad_norm"], step)
    accel.step(state.optimizer_g)
    generator = accel.unwrap(state.generator)
    state.scheduler_g.step()

    if not discriminator_before_generator:
        discriminator_step()

    accel.update()
    _module_param_probe("generator.after_step", generator, step)

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


def write_upsampler_stage_metrics(state, save_path: str, step: int, accel):
    generator = accel.unwrap(state.generator)
    quantizer = getattr(generator, "quantizer", None)
    stages = getattr(quantizer, "last_upsampler_stage_metrics", None)
    if not stages:
        return
    path = Path(save_path) / "upsampler_stage_metrics.jsonl"
    record = {"step": int(step), "stages": stages}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


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
    upsampler_stage_metric_freq: int = 100,
    discriminator_update_pattern: str = "always",
    discriminator_update_order: str = "before_generator",
    generator_grad_clip: float = 1e3,
    discriminator_grad_clip: float = 10.0,
    lambda_warmups: dict = None,
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
    if getattr(state, "num_iters", None) is not None:
        num_iters = int(state.num_iters)
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
            train_loop(
                state,
                batch,
                accel,
                lambdas,
                lambda_warmups,
                tracker,
                num_iters,
                discriminator_update_pattern,
                discriminator_update_order,
                generator_grad_clip,
                discriminator_grad_clip,
            )
            if (
                accel.local_rank == 0
                and upsampler_stage_metric_freq > 0
                and tracker.step % upsampler_stage_metric_freq == 0
            ):
                write_upsampler_stage_metrics(state, save_path, tracker.step, accel)

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
