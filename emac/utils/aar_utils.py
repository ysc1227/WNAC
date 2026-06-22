import typing
import numpy as np
import torch.nn.functional as F
import math
import torch
from audiotools.core import util
from audiotools import AudioSignal

def compute_token_metrics(logits, target_codes, effective_vocab_size):
    # flatten
    B, T, K = logits.shape
    logits_flat = logits.reshape(B * T, K)
    targets_flat = target_codes.reshape(B * T)

    # CE in nats
    nll_nats = F.cross_entropy(logits_flat, targets_flat, reduction="mean")
    # bits per token
    bpt = nll_nats / math.log(2.0)            # [bits/token]
    # uniform baseline = log2(K)
    uniform_bpt = math.log2(effective_vocab_size)

    # 1 - (H / log2 K): uniform이면 0, 완벽히 맞추면 1에 근접
    rel_gain = 1.0 - (bpt / uniform_bpt)

    # accuracy
    preds = logits_flat.argmax(dim=-1)
    acc = (preds == targets_flat).float().mean()

    return {
        "bpt": bpt.detach(),
        "rel_gain": rel_gain,
        "acc": acc.detach(),
    }

@torch.no_grad()
def eval_clap_consistency(state, accel, num_batches=4):
    state.generator.eval()
    state.cond_model.eval()

    sims = []

    for _ in range(num_batches):
        batch = next(iter(state.val_loader))  # 혹은 sampler에서 가져오기
        batch = util.prepare_batch(batch, accel.device)

        # 원본 signal & cond
        signal = state.val_data.transform(
            batch["signal"].clone(), **batch["transform_args"]
        )
        sr = signal.sample_rate

        cond_orig = state.cond_processor(
            audio=[data.squeeze(0).cpu().numpy()
                   for data in signal.resample(48000).audio_data],
            return_tensors="pt",
            sampling_rate=48000,
        )
        cond_feat_orig = state.cond_model.get_audio_features(
            input_features=cond_orig["input_features"].to(accel.device),
            is_longer=cond_orig["is_longer"].to(accel.device),
        )  # [B, D]

        # AAR로 코드 샘플링 + 디코드
        vae = accel.unwrap(state.vae)
        generator = accel.unwrap(state.generator)

        # 여기선 condition으로 cond_feat_orig 사용 (지금 구조랑 맞춰서)
        out_codes = generator.autoregressive_infer_cfg(
            B=cond_feat_orig.shape[0],
            label_B=cond_feat_orig,
            cfg=4.0,
            top_k=max(1, int(vae.codebook_size * 0.06)),
            top_p=0.95,
            g_seed=42,
        )
        # out_codes -> waveform
        # 만약 autoregressive_infer_cfg가 이미 waveform을 리턴하는 거면 이 부분은 생략
        generated = AudioSignal(out_codes, sr)

        # 생성된 오디오를 다시 CLAP에 넣기
        cond_gen = state.cond_processor(
            audio=[data.squeeze(0).cpu().numpy()
                   for data in generated.resample(48000).audio_data],
            return_tensors="pt",
            sampling_rate=48000,
        )
        cond_feat_gen = state.cond_model.get_audio_features(
            input_features=cond_gen["input_features"].to(accel.device),
            is_longer=cond_gen["is_longer"].to(accel.device),
        )  # [B, D]

        # 코사인 유사도
        norm_orig = cond_feat_orig / cond_feat_orig.norm(dim=-1, keepdim=True)
        norm_gen  = cond_feat_gen  / cond_feat_gen.norm(dim=-1, keepdim=True)
        sim = (norm_orig * norm_gen).sum(dim=-1)  # [B]
        sims.append(sim)

    sims = torch.cat(sims, dim=0)
    return {
        "clap_sim/mean": sims.mean().item(),
        "clap_sim/std": sims.std().item(),
    }

def salient_excerpt(
    signal: AudioSignal,
    loudness_cutoff: float = None,
    num_tries: int = 8,
    duration: float = None,
    state: typing.Union[np.random.RandomState, int] = None
):
    """Similar to AudioSignal.excerpt, except it extracts excerpts only
    if they are above a specified loudness threshold, which is computed via
    a fast LUFS routine.

    Parameters
    ----------
    audio_path : typing.Union[str, Path]
        Path to audio file to grab excerpt from.
    loudness_cutoff : float, optional
        Loudness threshold in dB. Typical values are ``-40, -60``,
        etc, by default None
    num_tries : int, optional
        Number of tries to grab an excerpt above the threshold
        before giving up, by default 8.
    state : typing.Union[np.random.RandomState, int], optional
        RandomState or seed of random state, by default None
    kwargs : dict
        Keyword arguments to AudioSignal.excerpt

    Returns
    -------
    AudioSignal
        AudioSignal containing excerpt.


    .. warning::
        if ``num_tries`` is set to None, ``salient_excerpt`` may try forever, which can
        result in an infinite loop if ``audio_path`` does not have
        any loud enough excerpts.

    Examples
    --------
    >>> signal = AudioSignal.salient_excerpt(
            "path/to/audio",
            loudness_cutoff=-40,
            duration=5
        )
    """
    state = util.random_state(state)
    if loudness_cutoff is None:
        out = excerpt(signal, state=state, duration=duration)
    else:
        loudness = -np.inf
        num_try = 0
        while loudness <= loudness_cutoff:
            out = excerpt(signal, state=state, duration=duration)
            loudness = out.loudness()
            num_try += 1
            if num_tries is not None and num_try >= num_tries:
                break
    return out

def excerpt(
    signal: AudioSignal,
    offset: float = None,
    duration: float = None,
    state: typing.Union[np.random.RandomState, int] = None,
):
    """Randomly draw an excerpt of ``duration`` seconds from an
    audio file specified at ``audio_path``, between ``offset`` seconds
    and end of file. ``state`` can be used to seed the random draw.

    Parameters
    ----------
    audio_path : typing.Union[str, Path]
        Path to audio file to grab excerpt from.
    offset : float, optional
        Lower bound for the start time, in seconds drawn from
        the file, by default None.
    duration : float, optional
        Duration of excerpt, in seconds, by default None
    state : typing.Union[np.random.RandomState, int], optional
        RandomState or seed of random state, by default None

    Returns
    -------
    AudioSignal
        AudioSignal containing excerpt.

    Examples
    --------
    >>> signal = AudioSignal.excerpt("path/to/audio", duration=5)
    """
    total_duration = signal.duration

    state = util.random_state(state)
    lower_bound = 0 if offset is None else min(offset)
    upper_bound = max(total_duration - duration, 0)
    offset = min(state.uniform(lower_bound, upper_bound), total_duration - duration)
    
    out = AudioSignal(signal.audio_data[:, :, int(offset * signal.sample_rate):int((offset + duration) * signal.sample_rate)], sample_rate=signal.sample_rate)
    out.metadata["offset"] = offset
    out.metadata["duration"] = duration

    return out