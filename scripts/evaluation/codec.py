import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal
from audiotools.core import util
from audiotools.ml.decorators import Tracker
from scripts.training.emac import losses


@dataclass
class State:
    stft_loss: losses.MultiScaleSTFTLoss # type:ignore
    mel_loss: losses.MelSpectrogramLoss # type:ignore
    waveform_loss: losses.L1Loss # type:ignore
    sisdr_loss: losses.SISDRLoss # type:ignore


_WORKER_STATE = None


def _build_state():
    return State(
        waveform_loss=losses.L1Loss(),
        stft_loss=losses.MultiScaleSTFTLoss(),
        mel_loss=losses.MelSpectrogramLoss(),
        sisdr_loss=losses.SISDRLoss(),
    )


def _init_worker(torch_threads: int = 1):
    """Initialize expensive metric modules once per worker process.

    Passing the State object as an argument for every file forces repeated pickle /
    IPC overhead.  It also lets each worker inherit PyTorch's large default thread
    pools, which can massively oversubscribe CPU cores.  Build metric modules once
    per worker and keep per-worker torch thread usage bounded.
    """
    global _WORKER_STATE
    torch_threads = max(1, int(torch_threads))
    try:
        torch.set_num_threads(torch_threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        # Can only be set before inter-op work starts; ignore if already fixed.
        pass
    _WORKER_STATE = _build_state()


def _align_pair(signal: AudioSignal, recons: AudioSignal):
    """Make reconstructed/reference signals compatible for loss computation."""
    x = signal
    y = recons

    if x.audio_data.shape[1] != y.audio_data.shape[1]:
        # Prefer mono comparison when channel counts differ.
        x.audio_data = x.audio_data.mean(dim=1, keepdim=True)
        y.audio_data = y.audio_data.mean(dim=1, keepdim=True)

    n = min(x.shape[-1], y.shape[-1])
    if n <= 0:
        raise RuntimeError(f"Cannot compare empty audio: {signal.path_to_file}, {recons.path_to_file}")
    if x.shape[-1] != n:
        x = x[..., :n]
    if y.shape[-1] != n:
        y = y[..., :n]
    return x, y


def get_metrics(signal_path, recons_path, state=None):
    if state is None:
        state = _WORKER_STATE if _WORKER_STATE is not None else _build_state()
    output = {}
    signal = AudioSignal(signal_path)
    recons = AudioSignal(recons_path)
    for sr in [22050, 44100]:
        x = signal.clone().resample(sr)
        y = recons.clone().resample(sr)
        x, y = _align_pair(x, y)
        
        k = "22k" if sr == 22050 else "44k"
        output.update(
            {
                f"mel-{k}": state.mel_loss(x, y),
                f"stft-{k}": state.stft_loss(x, y),
                f"waveform-{k}": state.waveform_loss(x, y),
                # SISDRLoss is a loss, i.e. negative SI-SDR in dB.  Metrics CSV
                # should report SI-SDR itself, where higher is better.
                f"sisdr-{k}": -state.sisdr_loss(x, y),
                #f"visqol-audio-{k}": metrics.quality.visqol(x, y),
                #f"visqol-speech-{k}": metrics.quality.visqol(x, y, "speech")
            }
        )
    output["path"] = signal.path_to_file
    output["recons_path"] = str(recons_path)
    output.update(signal.metadata)
    return output

@argbind.bind(without_prefix=True)
@torch.no_grad()
def evaluate(
    input: str = "samples/input",
    output: str = "samples/output",
    n_proc: int = 1,
    torch_threads: int = 1,
):
    tracker = Tracker()
    input_root = Path(input)
    recons_root = Path(output)

    audio_files = util.find_audio(input_root)
    if len(audio_files) == 0:
        raise RuntimeError(f"No audio files found in input path: {input_root}")
    recons_root.mkdir(parents=True, exist_ok=True)

    @tracker.track("metrics", len(audio_files))
    def record(future, writer):
        o = future.result()
        for k, v in o.items():
            if torch.is_tensor(v):
                o[k] = v.item()
        writer.writerow(o)
        o.pop("path")
        return o

    futures = []
    pairs = []
    missing = []
    for audio_file in audio_files:
        try:
            rel = audio_file.relative_to(input_root)
        except ValueError:
            rel = Path(audio_file.name)
        recons_path = recons_root / rel
        if recons_path.exists():
            pairs.append((audio_file, recons_path))
        else:
            # Backward compatibility for old flat decode outputs.
            flat_recons_path = recons_root / audio_file.name
            if flat_recons_path.exists():
                pairs.append((audio_file, flat_recons_path))
            else:
                missing.append((audio_file, recons_path))

    if not pairs:
        examples = "\n".join(f"  {src} -> expected {dst}" for src, dst in missing[:10])
        raise RuntimeError(
            f"No matching reconstructed files found under {recons_root}. "
            f"Expected files with the same relative paths as {input_root}.\n{examples}"
        )

    if missing:
        print(f"[WARN] Skipping {len(missing)} input files with no matching reconstruction.")

    with tracker.live:
        with open(recons_root / "metrics.csv", "w") as csvfile:
            with ProcessPoolExecutor(
                n_proc,
                mp.get_context("fork"),
                initializer=_init_worker,
                initargs=(torch_threads,),
            ) as pool:
                for audio_path, recons_path in pairs:
                    future = pool.submit(
                        get_metrics, audio_path, recons_path
                    )
                    futures.append(future)
                keys = list(futures[0].result().keys())
                print(keys)
                writer = csv.DictWriter(csvfile, fieldnames=keys)
                writer.writeheader()

                for future in futures:
                    record(future, writer)

        tracker.done("test", f"N={len(pairs)} matched / {len(audio_files)} input")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        evaluate()