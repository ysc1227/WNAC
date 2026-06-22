"""ViSQOL (Virtual Speech Quality Objective Listener) evaluation.

Computes MOS-LQO scores for codec reconstructions using the ``visqol-python``
package.  Follows the same multiprocess executor pattern as ``codec.py``.

Install dependency:
    pip install visqol-python[accel]
"""

import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path

import argbind
import numpy as np
from audiotools import AudioSignal
from audiotools.core import util
from audiotools.ml.decorators import Tracker


# ---------------------------------------------------------------------------
# Worker state – heavy ViSQOL API object is created once per worker process.
# ---------------------------------------------------------------------------

_WORKER_API = None


def _init_worker(mode: str = "audio"):
    """Create a ViSQOL API instance once per worker process.

    Parameters
    ----------
    mode : str
        ``"audio"`` (48 kHz, general audio/music) or
        ``"speech"`` (16 kHz, telephony speech).
    """
    global _WORKER_API
    try:
        from visqol import VisqolApi
        api = VisqolApi()
        api.create(mode=mode)
        _WORKER_API = (api, mode)
        print(f"[visqol-worker] pid={os.getpid()} initialized mode={mode}", flush=True)
    except ImportError:
        raise ImportError(
            "visqol-python is not installed. "
            "Run:  pip install visqol-python[accel]"
        )


_VISQOL_SR = {"audio": 48000, "speech": 16000}


# ---------------------------------------------------------------------------
# Per-file metric computation
# ---------------------------------------------------------------------------

def _align_pair(signal: AudioSignal, recons: AudioSignal):
    """Make two AudioSignals compatible (mono, same length)."""
    x, y = signal, recons

    if x.audio_data.shape[1] != y.audio_data.shape[1]:
        x.audio_data = x.audio_data.mean(dim=1, keepdim=True)
        y.audio_data = y.audio_data.mean(dim=1, keepdim=True)

    n = min(x.shape[-1], y.shape[-1])
    if n <= 0:
        raise RuntimeError(
            f"Cannot compare empty audio: {signal.path_to_file}, "
            f"{recons.path_to_file}"
        )
    if x.shape[-1] != n:
        x = x[..., :n]
    if y.shape[-1] != n:
        y = y[..., :n]
    return x, y


def get_visqol(signal_path, recons_path):
    """Compute ViSQOL MOS-LQO for a single reference / reconstruction pair.

    The worker-local ViSQOL API is used (created via ``_init_worker``).
    Audio is resampled to the sample rate expected by the chosen mode
    (48 kHz for audio, 16 kHz for speech).
    """
    global _WORKER_API
    if _WORKER_API is None:
        _init_worker()
    api, mode = _WORKER_API
    target_sr = _VISQOL_SR[mode]

    signal = AudioSignal(signal_path)
    recons = AudioSignal(recons_path)

    # Resample to ViSQOL's expected sample rate
    signal = signal.resample(target_sr)
    recons = recons.resample(target_sr)
    signal, recons = _align_pair(signal, recons)

    # Convert to 1-D float64 numpy arrays (mono)
    ref_np = signal.audio_data.squeeze().cpu().numpy().astype(np.float64)
    deg_np = recons.audio_data.squeeze().cpu().numpy().astype(np.float64)

    result = api.measure_from_arrays(ref_np, deg_np, target_sr)

    output = {
        f"visqol-{mode}": result.moslqo,
        "path": str(signal_path),
        "recons_path": str(recons_path),
        "worker_pid": os.getpid(),
    }
    return output


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

@argbind.bind(without_prefix=True)
def evaluate_visqol(
    input: str = "samples/input",
    output: str = "samples/output",
    mode: str = "audio",
    n_proc: int = 4,
    max_pairs: int = None,
):
    """Compute ViSQOL MOS-LQO scores for all matched reference/reconstruction pairs.

    Parameters
    ----------
    input : str
        Path to the original (reference) audio directory.
    output : str
        Path to the reconstructed audio directory.
    mode : str
        ViSQOL mode: ``"audio"`` (48 kHz, for music/general) or
        ``"speech"`` (16 kHz).
    n_proc : int
        Number of parallel worker processes.
    """
    import time

    input_root = Path(input)
    recons_root = Path(output)

    audio_files = util.find_audio(input_root)
    if len(audio_files) == 0:
        raise RuntimeError(f"No audio files found in input path: {input_root}")
    recons_root.mkdir(parents=True, exist_ok=True)

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
            flat_recons_path = recons_root / audio_file.name
            if flat_recons_path.exists():
                pairs.append((audio_file, flat_recons_path))
            else:
                missing.append((audio_file, recons_path))

    if not pairs:
        examples = "\n".join(
            f"  {src} -> expected {dst}" for src, dst in missing[:10]
        )
        raise RuntimeError(
            f"No matching reconstructed files found under {recons_root}. "
            f"Expected files with the same relative paths as {input_root}.\n"
            f"{examples}"
        )

    if missing:
        print(
            f"[WARN] Skipping {len(missing)} input files with no matching "
            f"reconstruction."
        )

    if max_pairs is not None:
        max_pairs = int(max_pairs)
        if max_pairs > 0:
            pairs = pairs[:max_pairs]

    total = len(pairs)
    score_key = f"visqol-{mode}"
    csv_path = recons_root / f"visqol_{mode}.csv"
    t0 = time.time()

    print(f"[visqol] Starting {total} pairs  |  mode={mode}  |  workers={n_proc}")
    print(f"[visqol] Results will be saved to {csv_path}")

    with open(csv_path, "w", newline="") as csvfile:
        with ProcessPoolExecutor(
            n_proc,
            mp.get_context("fork"),
            initializer=_init_worker,
            initargs=(mode,),
        ) as pool:
            future_to_pair = {}
            for audio_path, recons_path in pairs:
                future = pool.submit(get_visqol, audio_path, recons_path)
                futures.append(future)
                future_to_pair[future] = (audio_path, recons_path)

            writer = None
            scores = []
            for i, future in enumerate(as_completed(futures), 1):
                audio_path, _ = future_to_pair[future]
                try:
                    o = future.result()
                except Exception as exc:
                    elapsed = time.time() - t0
                    print(
                        f"[visqol] [{i}/{total}] ERROR after {elapsed:.0f}s "
                        f"file={audio_path}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    raise

                # Initialize CSV writer with header from first result
                if writer is None:
                    keys = list(o.keys())
                    writer = csv.DictWriter(csvfile, fieldnames=keys)
                    writer.writeheader()

                writer.writerow(o)
                csvfile.flush()

                score = o[score_key]
                scores.append(score)
                elapsed = time.time() - t0
                eta = (elapsed / i) * (total - i)
                fname = Path(o["path"]).name
                print(
                    f"[visqol] [{i}/{total}]  {score:.4f}  "
                    f"({elapsed:.0f}s elapsed, ~{eta:.0f}s left)  "
                    f"pid={o.get('worker_pid')}  {fname}",
                    flush=True,
                )

    elapsed_total = time.time() - t0
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\n[visqol] Done  |  {total} files  |  {elapsed_total:.1f}s total")
    print(f"[visqol] Mean {score_key}: {avg:.4f}")
    print(f"[visqol] Results saved to {csv_path}")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        evaluate_visqol()
