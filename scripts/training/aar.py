import math
import os
import sys
import warnings
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
from transformers import ClapModel, ClapProcessor

import emac
from emac.utils import load_model
from emac.utils.aar_utils import salient_excerpt
from emac.nn.scheduler import WarmupLin0LrWdScheduler
from emac.utils.ema import ModelEMA

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Enable cudnn autotuner to speed up training
# (can be altered by the funcs.seed function)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))
# Uncomment to trade memory for speed.

# Optimizers
AdamW = argbind.bind(torch.optim.AdamW, "generator")
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)


@argbind.bind("generator")
def WarmupLin0LR(optimizer, 
    max_it, 
    peak_lr: float = 0.0001, 
    wp_ratio: float = 0.03, 
    wd: float = 0.05, 
    wd_end: float = 0, 
    wp0: float = 0.005, 
    wpe: float = 0.01, 
    last_epoch: int = -1
):
    return WarmupLin0LrWdScheduler(optimizer, max_it, peak_lr, wp_ratio, wd, wd_end, wp0, wpe, last_epoch)


# Models
AAR = argbind.bind(emac.model.AAR)

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

def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def iter_aar_code_chunks(codes, generator):
    """Yield full-size RVQ/AAR code chunks for fixed-frame AAR training.

    AAR is parameterized for one RVQ frame window (`generator.frame_size`).  When
    the audio segment is longer, split each RVQ stage along the underlying full
    latent timeline and drop the final partial chunk for stable fixed-shape
    training.
    """
    full_len = max(c.shape[-1] for c in codes)
    frame_size = int(generator.frame_size)
    if full_len < frame_size:
        return
    for start in range(0, full_len - frame_size + 1, frame_size):
        chunk = []
        for stage_idx, c in enumerate(codes):
            stage_len = c.shape[-1]
            # Match the quantizer's code length rule: each active stage gets at
            # least one token even when int(scale_factor * frame_size) rounds to
            # zero.  This keeps AAR logits length and target label length aligned
            # for very small scales / short frame_size settings.
            expected = max(1, int(generator.vae_quant_proxy[0].scale_factors[stage_idx] * frame_size))
            s = int(generator.vae_quant_proxy[0].scale_factors[stage_idx] * start)
            e = s + expected
            if e > stage_len:
                break
            chunk.append(c[:, s:e])
        if len(chunk) != len(codes):
            continue
        yield start // frame_size, max(1, full_len // frame_size), chunk


def forward_aar_code_chunks(generator, vae, conditions, codes, ce_loss):
    generator_cfg = generator.module if hasattr(generator, "module") else generator
    logits_chunks = []
    label_chunks = []
    ce_chunks = []
    prev_cond = None

    for chunk_idx, num_chunks, chunk_codes in iter_aar_code_chunks(codes, generator_cfg):
        chunk_cond = generator_cfg._compose_chunk_condition(
            label_B=conditions,
            prev_chunk_condition_BD=prev_cond,
            chunk_idx=chunk_idx,
            num_chunks=num_chunks,
            prev_condition_weight=0.5,
            position_weight=0.1,
        )
        aar_input = torch.concat(
            vae.quantizer.get_aar_input(
                chunk_codes,
                generator_cfg.use_offset,
                generator_cfg.use_scale_order,
                generator_cfg.use_blockwise,
            ),
            dim=1,
        )
        codes_list = vae.quantizer.get_aar_target_codes(
            chunk_codes,
            generator_cfg.use_offset,
            generator_cfg.use_scale_order,
        )
        logits = generator(chunk_cond, aar_input)
        labels = torch.cat(codes_list, dim=1)
        assert logits.shape[:2] == labels.shape, \
            f"chunk={chunk_idx}: logits={tuple(logits.shape)} labels={tuple(labels.shape)} shape mismatch"
        logits_chunks.append(logits)
        label_chunks.append(labels)
        ce_chunks.append(ce_loss(logits, codes_list))
        prev_cond = chunk_cond.detach()

    if not logits_chunks:
        raise RuntimeError(
            f"No full AAR chunks produced: max_code_len={max(c.shape[-1] for c in codes)}, "
            f"frame_size={generator_cfg.frame_size}"
        )

    return (
        torch.cat(logits_chunks, dim=1),
        torch.cat(label_chunks, dim=1),
        torch.cat(ce_chunks, dim=1),
    )


def backward_aar_code_chunks_streaming(generator, vae, conditions, codes, ce_loss, accel):
    """Forward/backward AAR chunks one at a time to avoid chunk-count OOM.

    ``forward_aar_code_chunks`` is convenient for validation metrics, but it keeps
    every chunk's logits/loss graph alive until the final backward pass.  For long
    audio this scales activation/logit memory roughly with the number of chunks.

    This training helper first counts the total number of target tokens, then runs
    each chunk independently and immediately backpropagates its token-normalized CE
    contribution.  Only scalar metric accumulators survive across chunks.
    """
    generator_cfg = generator.module if hasattr(generator, "module") else generator

    total_tokens = 0
    num_chunks_seen = 0
    for _, _, chunk_codes in iter_aar_code_chunks(codes, generator_cfg):
        codes_list = vae.quantizer.get_aar_target_codes(
            chunk_codes,
            generator_cfg.use_offset,
            generator_cfg.use_scale_order,
        )
        labels = torch.cat(codes_list, dim=1)
        total_tokens += labels.numel()
        num_chunks_seen += 1

    if total_tokens == 0:
        raise RuntimeError(
            f"No full AAR chunks produced: max_code_len={max(c.shape[-1] for c in codes)}, "
            f"frame_size={generator_cfg.frame_size}"
        )

    loss_sum_detached = conditions.new_zeros(())
    correct_detached = conditions.new_zeros(())
    prev_cond = None
    vocab_size = None

    for chunk_idx, num_chunks, chunk_codes in iter_aar_code_chunks(codes, generator_cfg):
        chunk_cond = generator_cfg._compose_chunk_condition(
            label_B=conditions,
            prev_chunk_condition_BD=prev_cond,
            chunk_idx=chunk_idx,
            num_chunks=num_chunks,
            prev_condition_weight=0.5,
            position_weight=0.1,
        )
        aar_input = torch.concat(
            vae.quantizer.get_aar_input(
                chunk_codes,
                generator_cfg.use_offset,
                generator_cfg.use_scale_order,
                generator_cfg.use_blockwise,
            ),
            dim=1,
        )
        codes_list = vae.quantizer.get_aar_target_codes(
            chunk_codes,
            generator_cfg.use_offset,
            generator_cfg.use_scale_order,
        )
        labels = torch.cat(codes_list, dim=1)
        logits = generator(chunk_cond, aar_input)
        assert logits.shape[:2] == labels.shape, \
            f"chunk={chunk_idx}: logits={tuple(logits.shape)} labels={tuple(labels.shape)} shape mismatch"

        ce_per_token = ce_loss(logits, codes_list)
        chunk_loss_sum = ce_per_token.sum()
        accel.backward(chunk_loss_sum / total_tokens)

        with torch.no_grad():
            loss_sum_detached += chunk_loss_sum.detach()
            correct_detached += (logits.argmax(dim=-1) == labels.to(logits.device)).float().sum()
            vocab_size = logits.shape[-1]

        prev_cond = chunk_cond.detach()

    loss = loss_sum_detached / total_tokens
    acc = correct_detached / total_tokens
    return loss, acc, int(total_tokens), int(vocab_size), int(num_chunks_seen)


@argbind.bind("train", "val")
def build_transform(
    augment_prob: float = 1.0,
    preprocess: list = ["Identity"],
    augment: list = ["Identity"],
    postprocess: list = ["Identity"],
):
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]
    preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
    augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
    postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    transform = transforms.Compose(preprocess, augment, postprocess)
    return transform


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
    generator: AAR # type:ignore
    vae: emac.model.EMAC # type:ignore
    cond_processor: ClapProcessor
    cond_model: ClapModel
    
    ce_loss: losses.CELoss # type:ignore
    
    optimizer_g: AdamW # type:ignore
    scheduler_g: WarmupLin0LR # type:ignore

    train_data: AudioDataset # type:ignore
    val_data: AudioDataset # type:ignore

    tracker: Tracker
    
    ema_g: ModelEMA


@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    vae_path: str,
    resume: bool = False,
    tag: str = "latest",
    load_weights: bool = False
):
    generator, g_extra = None, {}

    if resume:
        kwargs = {
            "folder": f"{save_path}/{tag}",
            "map_location": "cpu",
            "package": not load_weights,
            "weights_only": False
        }
        tracker.print(f"Resuming from {str(Path('.').absolute())}/{kwargs['folder']}")
        if (Path(kwargs["folder"]) / "aar").exists():
            generator, g_extra = AAR.load_from_folder(**kwargs)

    if generator is None:
        vae = load_model(load_path=vae_path)
        generator = AAR(vae_local=vae)
        generator.init_weights(init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1)
        
    cond_processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
    cond_model = ClapModel.from_pretrained("laion/larger_clap_general")

    tracker.print(generator)
    tracker.print(generator.vae_proxy[0])
    tracker.print(cond_processor)
    tracker.print(cond_model)
            
    generator = accel.prepare_model(generator)
    vae = accel.prepare_model(accel.unwrap(generator).vae_proxy[0])
    cond_model = accel.prepare_model(cond_model)
    
    ema_g = None
    
    if args["use_ema"]:
        ema_g = ModelEMA(accel.unwrap(generator), decay=args["ema_decay"])

    with argbind.scope(args, "generator"):
        optimizer_g = AdamW(generator.parameters(), use_zero=accel.use_ddp)
        scheduler_g = WarmupLin0LR(optimizer_g, args["num_iters"])

    if "optimizer.pth" in g_extra:
        optimizer_g.load_state_dict(g_extra["optimizer.pth"])
    if "scheduler.pth" in g_extra:
        scheduler_g.load_state_dict(g_extra["scheduler.pth"])
    if "tracker.pth" in g_extra:
        tracker.load_state_dict(g_extra["tracker.pth"])
    if "ema_g.pth" in g_extra:
        ema_g.load_state_dict(g_extra["ema_g.pth"])

    sample_rate = accel.unwrap(vae).sample_rate
    
    with argbind.scope(args, "train"):
        train_data = build_dataset(sample_rate)
    with argbind.scope(args, "val"):
        val_data = build_dataset(sample_rate)

    ce_loss = losses.CELoss(reduction='none')

    return State(
        generator=generator,
        vae=vae,
        cond_processor=cond_processor,
        cond_model=cond_model,
        ce_loss = ce_loss,
        optimizer_g=optimizer_g,
        scheduler_g=scheduler_g,
        tracker=tracker,
        train_data=train_data,
        val_data=val_data,
        ema_g=ema_g
    )

@timer()
@torch.no_grad()
def val_loop(batch, state, accel, duration, tracker):
    state.generator.eval()
    state.vae.eval()
    state.cond_model.eval()

    batch = util.prepare_batch(batch, accel.device)

    signal: AudioSignal = state.val_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )
    segment: AudioSignal = salient_excerpt(signal, duration=duration)

    vae = accel.unwrap(state.vae)
    cond_model = accel.unwrap(state.cond_model)
    generator = accel.unwrap(state.generator)

    # ---- cond ----
    cond = state.cond_processor(
        audio=[d.squeeze(0).cpu().numpy() for d in segment.resample(48000).audio_data],
        return_tensors="pt",
        sampling_rate=48000,
    )
    conditions = cond_model.get_audio_features(
        input_features=cond["input_features"].to(accel.device),
        is_longer=cond["is_longer"].to(accel.device),
    )

    # ---- codes ----
    _, codes, *_ = vae.encode(
        vae.preprocess(segment.resample(vae.sample_rate).audio_data, vae.sample_rate)
    )

    def compute_metrics(logits, labels, ce_per_token, prefix: str | None):
        """
        logits: [B, L, V]
        returns dict with keys prefixed
        """
        B, L, V = logits.shape

        loss = ce_per_token.mean()

        bpt = loss / math.log(2.0)

        K_eff = getattr(vae, "codebook_size", V)
        uniform_bpt = math.log2(float(K_eff))
        rel_gain = 1.0 - (bpt / uniform_bpt)

        logits_flat = logits.view(-1, V)
        labels_flat = labels.view(-1).to(logits_flat.device)

        assert logits_flat.size(0) == labels_flat.size(0), \
            f"{prefix}: logits_flat={logits_flat.size(0)} labels_flat={labels_flat.size(0)} mismatch"

        preds = logits_flat.argmax(dim=-1)
        acc = (preds == labels_flat).float().mean()

        if prefix is not None:
            out = {
                f"loss/{prefix}": loss.detach(),
                f"bpt/{prefix}/total": bpt.detach(),
                f"rel_gain/{prefix}": rel_gain.detach(),
                f"acc/{prefix}": acc.detach(),
            }
        else:
            out = {
                f"loss": loss.detach(),
                f"bpt/total": bpt.detach(),
                f"rel_gain": rel_gain.detach(),
                f"acc": acc.detach(),
            }

        return out

    # ---- RAW forward ----
    logits_raw, labels_raw, ce_raw = forward_aar_code_chunks(
        generator, vae, conditions, codes, state.ce_loss
    )
    out = compute_metrics(logits_raw, labels_raw, ce_raw, prefix=("raw" if state.ema_g is not None else None))

    # ---- EMA forward (있으면) ----
    if state.ema_g is not None:
        with state.ema_g.apply_to(generator):
            logits_ema, labels_ema, ce_ema = forward_aar_code_chunks(
                generator, vae, conditions, codes, state.ce_loss
            )
        out.update(compute_metrics(logits_ema, labels_ema, ce_ema, prefix="ema"))
    else:
        # EMA 없으면 raw만
        pass

    return out


@timer()
def train_loop(state, batch, accel, duration, tracker, num_iters):
    state.generator.train()
    state.vae.eval()
    state.cond_model.eval()
    output = {}
    
    batch = util.prepare_batch(batch, accel.device)
    with torch.no_grad():
        signal: AudioSignal = state.train_data.transform(
            batch["signal"].clone(), **batch["transform_args"]
        )
        segment: AudioSignal = salient_excerpt(signal, duration=duration)

        vae = accel.unwrap(state.vae)
        cond_model = accel.unwrap(state.cond_model)
        generator = accel.unwrap(state.generator)

        cond = state.cond_processor(audio=[data.squeeze(0).cpu().numpy() for data in segment.resample(48000).audio_data], return_tensors="pt", sampling_rate=48000)
        _, codes, _, _, _, _ = vae.encode(vae.preprocess(segment.audio_data, vae.sample_rate))
        
        conditions = cond_model.get_audio_features(input_features=cond['input_features'].to(accel.device), is_longer=cond['is_longer'].to(accel.device))

    state.optimizer_g.zero_grad()
    with accel.autocast():
        loss, acc, total_tokens, vocab_size, num_chunks_seen = backward_aar_code_chunks_streaming(
            state.generator, vae, conditions, codes, state.ce_loss, accel
        )
        output["loss"] = loss.detach()

        # 2-1. bits per token (CE 결과 재사용)
        bpt = loss / math.log(2.0)

        # 2-2. uniform 대비 relative gain
        K_eff = getattr(vae, "codebook_size", vocab_size)
        uniform_bpt = math.log2(float(K_eff))
        rel_gain = 1.0 - (bpt / uniform_bpt)

        output["bpt/total"] = bpt.detach()
        output["rel_gain"] = rel_gain.detach()
        output["acc"] = acc.detach()
        output["other/aar_tokens"] = torch.tensor(float(total_tokens), device=accel.device)
        output["other/aar_chunks"] = torch.tensor(float(num_chunks_seen), device=accel.device)

    accel.scaler.unscale_(state.optimizer_g)
    output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
        state.generator.parameters(), 2.
    )
    accel.step(state.optimizer_g)
    state.scheduler_g.step()
    accel.update()
    
    if state.ema_g is not None:
        state.ema_g.update(accel.unwrap(state.generator))

    output["other/learning_rate"] = state.optimizer_g.param_groups[0]["lr"]
    output["other/batch_size"] = signal.batch_size * accel.world_size

    return {k: v for k, v in sorted(output.items())}


def checkpoint(state, save_iters, save_path, use_ema):
    metadata = {"logs": state.tracker.history}

    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")
    if state.tracker.is_best("val", "loss/ema" if use_ema else "loss"):
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
        if state.ema_g is not None:
            generator_extra["ema_g.pth"] = state.ema_g.state_dict()
        accel.unwrap(state.generator).metadata = metadata
        accel.unwrap(state.generator).save_to_folder(
            f"{save_path}/{tag}", generator_extra
        )

@torch.no_grad()
def save_samples(state, duration, val_idx, writer, tracker):
    state.tracker.print("Saving audio samples to TensorBoard")
    state.generator.eval()
    state.cond_model.eval()

    samples = [state.val_data[idx] for idx in val_idx]
    batch = state.val_data.collate(samples)
    batch = util.prepare_batch(batch, accel.device)
    
    signal = state.train_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )
    
    segment: AudioSignal = salient_excerpt(signal, duration=duration)
    
    vae = accel.unwrap(state.vae)
    generator = accel.unwrap(state.generator)
    
    cond = state.cond_processor(audio=[data.squeeze(0).cpu().numpy() for data in segment.resample(48000).audio_data], return_tensors="pt", sampling_rate=48000)
    conditions = accel.unwrap(state.cond_model).get_audio_features(input_features=cond['input_features'].to(accel.device), is_longer=cond['is_longer'].to(accel.device))
    _, codes, _, _, _, _ = vae.encode(vae.preprocess(segment.audio_data, vae.sample_rate))
    first_chunk = next(iter_aar_code_chunks(codes, generator))[2]
    aar_input = torch.concat(
        vae.quantizer.get_aar_input(
            first_chunk,
            generator.use_offset,
            generator.use_scale_order,
            generator.use_blockwise,
        ),
        dim=1,
    )
    target_samples = int(round(duration * signal.sample_rate))

    if state.ema_g is not None:
        with state.ema_g.apply_to(generator):
            out = generator.autoregressive_infer_cfg_chunked(
                B=len(conditions), label_B=conditions.to(accel.device),
                target_samples=target_samples,
                cfg=4.0,
                top_k=max(1, int(accel.unwrap(state.vae).codebook_size * 0.06)),
                top_p=0.95, g_seed=42
            )
            out_tf = generator.autoregressive_infer_teacher_forcing(
                label_B=conditions.to(accel.device),
                x_BLCv_wo_first_l=aar_input,
            )
    else:
        out = generator.autoregressive_infer_cfg_chunked(B=len(conditions), label_B=conditions.to(accel.device), target_samples=target_samples, cfg=4.0, top_k=max(1, int(accel.unwrap(state.vae).codebook_size * 0.06)), top_p=0.95, g_seed=42)
        out_tf = generator.autoregressive_infer_teacher_forcing(
            label_B=conditions.to(accel.device),
            x_BLCv_wo_first_l=aar_input,
        )
    
    generated = AudioSignal(out, signal.sample_rate)
    generated_tf = AudioSignal(out_tf, signal.sample_rate)

    audio_dict = {"generated": generated, "generated_tf": generated_tf}
    if state.tracker.step == 0:
        audio_dict["signal"] = signal

    for k, v in audio_dict.items():
        for nb in range(v.batch_size):
            v[nb].cpu().write_audio_to_tb(
                f"{k}/sample_{nb}.wav", writer, state.tracker.step
            )

def validate(state, val_dataloader, accel, duration, tracker):
    # 1) 원래 CE/bpt/acc용 validation
    for batch in val_dataloader:
        output = val_loop(batch, state, accel, duration, tracker)

    if hasattr(state.optimizer_g, "consolidate_state_dict"):
        state.optimizer_g.consolidate_state_dict()

    return output

@argbind.bind(without_prefix=True)
def train(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    save_path: str = "ckpt",
    vae_path: str = "ckpt",
    num_iters: int = 250000,
    save_iters: list = [10000, 50000, 100000, 200000],
    sample_freq: int = 10000,
    valid_freq: int = 1000,
    batch_size: int = 12,
    val_batch_size: int = 10,
    num_workers: int = 8,
    val_idx: list = [0, 1, 2, 3, 4, 5, 6, 7]
):
    util.seed(seed)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    writer = (
        SummaryWriter(log_dir=f"{save_path}/logs") if accel.local_rank == 0 else None
    )
    tracker = Tracker(
        writer=writer, log_file=f"{save_path}/log.txt", rank=accel.local_rank
    )

    state = load(args, accel, tracker, save_path, vae_path)
    
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
            train_loop(state, batch, accel, args['duration'], tracker, num_iters)

            last_iter = (
                tracker.step == num_iters - 1 if num_iters is not None else False
            )
            if tracker.step % sample_freq == 0 or last_iter:
                save_samples(state, args['duration'], val_idx, writer, tracker)

            if tracker.step % valid_freq == 0 or last_iter:
                validate(state, val_dataloader, accel, args['duration'], tracker)
                checkpoint(state, save_iters, save_path, args["use_ema"])
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
