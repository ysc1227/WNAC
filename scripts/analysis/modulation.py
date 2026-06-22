import csv
import json
import math
import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import argbind
import numpy as np
import soundfile as sf
from audiotools.data.datasets import AudioDataset, AudioLoader
from scipy.signal import stft

import scripts.training.emac as train


EPS = 1e-7
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
DEFAULT_AUDIO_GROUPS = [
    ["all", 0.0, 22050.0],
    ["low_0_1k", 0.0, 1000.0],
    ["mid_1_4k", 1000.0, 4000.0],
    ["high_4_12k", 4000.0, 12000.0],
    ["top_12_22k", 12000.0, 22050.0],
]
DEFAULT_QUANTILES = [0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98]
DEFAULT_CHECK_FREQS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 40]
DEFAULT_SCHEDULE_SPECS = [
    "dominant_q80_all_equalized:all:equalized:0.80",
    "dominant_q85_all_equalized:all:equalized:0.85",
    "balanced_q80_all_energy:all:energy_weighted:0.80",
    "high_q80_equalized:high_4_12k:equalized:0.80",
    "top_q75_equalized:top_12_22k:equalized:0.75",
]


def _arg_get(args, key, default=None):
    if isinstance(args, dict):
        return args.get(key, default)
    if hasattr(args, "get"):
        try:
            return args.get(key, default)
        except Exception:
            pass
    return getattr(args, key.replace("/", "_"), default)


def _flatten_sources(folders: dict, keys: Iterable[str]) -> List[str]:
    sources = []
    for key in keys:
        values = folders.get(key, [])
        if isinstance(values, (str, Path)):
            values = [values]
        sources.extend([str(v) for v in values])
    return sources


def _dataset_domain_keys(folders: dict, selected_keys: Iterable[str] = None):
    """Group dataset folder keys into broad audio domains.

    The training config stores dataset sources by quality/source buckets such as
    ``speech_hq`` and ``music_uq``.  For modulation analysis we want the broader
    domains used elsewhere in the project: speech, music, and environment.
    Unknown keys are kept as their own domain so custom configs still work.
    """
    selected_keys = [] if selected_keys is None else list(selected_keys)
    if len(selected_keys) == 0:
        keys = [str(k) for k in folders.keys()]
    else:
        keys = [str(k) for k in selected_keys]

    selected = set(keys)
    domains = OrderedDict()
    speech = [k for k in keys if k.startswith("speech") and k in selected]
    music = [k for k in keys if k.startswith("music") and k in selected]
    environment = [
        k
        for k in keys
        if (k in {"general", "environment"} or k.startswith("environment")) and k in selected
    ]
    consumed = set(speech + music + environment)
    if speech:
        domains["speech"] = speech
    if music:
        domains["music"] = music
    if environment:
        domains["environment"] = environment
    for key in keys:
        if key not in consumed:
            domains[key] = [key]
    return domains


def _audio_files(root: Path, max_files: Optional[int] = None) -> List[Path]:
    files = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if max_files is not None and int(max_files) > 0:
        files = files[: int(max_files)]
    return files


def _mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x


def _audio_signal_to_numpy(signal) -> np.ndarray:
    audio = signal.audio_data.detach().cpu().numpy()
    if audio.ndim == 3:
        audio = audio[0]
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    return np.asarray(audio, dtype=np.float32)


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def _make_mel_filter(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if fmax is None:
        fmax = sample_rate / 2
    mel_edges = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_edges = _mel_to_hz(mel_edges)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    filt = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = hz_edges[i], hz_edges[i + 1], hz_edges[i + 2]
        up = (freqs - left) / max(EPS, center - left)
        down = (right - freqs) / max(EPS, right - center)
        filt[i] = np.maximum(0.0, np.minimum(up, down))
        area = filt[i].sum()
        if area > 0:
            filt[i] /= area
    return filt, hz_edges[1:-1].astype(np.float32)


def _normalize_audio_groups(audio_groups, sample_rate: int):
    groups = audio_groups if audio_groups else DEFAULT_AUDIO_GROUPS
    out = []
    for group in groups:
        if len(group) != 3:
            raise ValueError(f"Audio group must be [name, low_hz, high_hz], got {group}")
        name, lo, hi = group
        out.append((str(name), float(lo), min(float(hi), sample_rate / 2)))
    return out


def _parse_schedule_spec(spec: str):
    parts = str(spec).split(":")
    if len(parts) != 4:
        raise ValueError(
            "schedule_specs entries must be `label:audio_group:mode:quantile`, "
            f"got {spec!r}"
        )
    label, group, mode, quantile = parts
    return label, group, mode, float(quantile)


def _solve_power_schedule(pivot: float, n_base_scales: int, target_sum: float):
    xs = np.linspace(0.0, 1.0, int(n_base_scales))
    target_base_sum = (float(target_sum) + float(pivot)) / 2.0

    def values(gamma):
        return pivot + (1.0 - pivot) * (xs ** gamma)

    def schedule_sum(gamma):
        return float(values(gamma).sum())

    lo, hi = 0.05, 8.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if schedule_sum(mid) > target_base_sum:
            lo = mid
        else:
            hi = mid

    gamma = (lo + hi) / 2.0
    base = values(gamma)
    actual = np.concatenate([base[::-1], base[1:]])
    return gamma, base, actual


class ModulationAccumulator:
    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        n_fft: int,
        n_mels: int,
        mod_bin_width: float,
        audio_groups,
    ):
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.n_mels = int(n_mels)
        self.mod_bin_width = float(mod_bin_width)
        self.audio_groups = _normalize_audio_groups(audio_groups, self.sample_rate)
        self.latent_frame_rate = self.sample_rate / self.hop_length
        self.max_mod_freq = self.latent_frame_rate / 2.0
        self.mod_edges = np.arange(
            0.0,
            self.max_mod_freq + self.mod_bin_width * 1.5,
            self.mod_bin_width,
            dtype=np.float64,
        )
        self.mod_centers = (self.mod_edges[:-1] + self.mod_edges[1:]) / 2.0
        self.mel_filter, self.mel_centers = _make_mel_filter(
            self.sample_rate,
            self.n_fft,
            self.n_mels,
            0.0,
            self.sample_rate / 2,
        )
        self.group_masks = [
            (self.mel_centers >= lo) & (self.mel_centers < hi)
            for _, lo, hi in self.audio_groups
        ]
        self.reset()

    def reset(self):
        shape = (len(self.audio_groups), 2, len(self.mod_centers))
        self.power = np.zeros(shape, dtype=np.float64)
        self.num_examples = 0
        self.total_frames = 0

    def process_array(self, audio: np.ndarray):
        audio = _mono(audio)
        if audio.size == 0:
            return
        _, _, spec = stft(
            audio,
            fs=self.sample_rate,
            window="hann",
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            nfft=self.n_fft,
            boundary=None,
            padded=False,
        )
        power = np.abs(spec).astype(np.float32) ** 2
        mel = self.mel_filter @ power
        logmel = np.log(mel + EPS).astype(np.float32)
        logmel = logmel - logmel.mean(axis=1, keepdims=True)
        std = logmel.std(axis=1, keepdims=True)
        equalized = logmel / (std + EPS)

        self.total_frames += int(logmel.shape[1])
        self.num_examples += 1

        freqs = np.fft.rfftfreq(logmel.shape[1], d=self.hop_length / self.sample_rate)[1:]
        for gi, mask in enumerate(self.group_masks):
            if not np.any(mask):
                continue
            for mi, data in enumerate([logmel, equalized]):
                temporal_fft = np.fft.rfft(data[mask], axis=1)[:, 1:]
                temporal_power = (np.abs(temporal_fft) ** 2).sum(axis=0)
                hist, _ = np.histogram(freqs, bins=self.mod_edges, weights=temporal_power)
                self.power[gi, mi] += hist

    def add_worker_result(self, power: np.ndarray, num_examples: int, total_frames: int):
        self.power += power
        self.num_examples += int(num_examples)
        self.total_frames += int(total_frames)

    def spectrum_rows(self):
        for i, center in enumerate(self.mod_centers):
            row = {
                "mod_freq_hz": center,
                "mod_freq_upper_hz": self.mod_edges[i + 1],
            }
            for gi, (group, _, _) in enumerate(self.audio_groups):
                row[f"{group}_energy_weighted"] = self.power[gi, 0, i]
                row[f"{group}_equalized"] = self.power[gi, 1, i]
            yield row

    def summary_rows(self, quantiles, check_freqs):
        rows = []
        modes = ["energy_weighted", "equalized"]
        for gi, (group, _, _) in enumerate(self.audio_groups):
            for mi, mode in enumerate(modes):
                values = self.power[gi, mi]
                total = float(values.sum())
                cdf = np.cumsum(values) / max(EPS, total)
                row = {
                    "audio_group": group,
                    "mode": mode,
                    "total_power": total,
                }
                for q in quantiles:
                    q_pct = int(round(float(q) * 100))
                    idx = int(np.searchsorted(cdf, float(q), side="left"))
                    idx = min(idx, len(self.mod_edges) - 2)
                    freq = float(self.mod_edges[idx + 1])
                    row[f"q{q_pct}_hz"] = freq
                    row[f"q{q_pct}_scale"] = min(1.0, 2.0 * freq / self.latent_frame_rate)
                for freq in check_freqs:
                    idx = np.searchsorted(self.mod_edges[1:], float(freq), side="right") - 1
                    row[f"frac_le_{freq}hz"] = float(cdf[idx]) if idx >= 0 else 0.0
                rows.append(row)
        return rows


_WORKER_ACCUMULATOR = None


def _init_worker(sample_rate, hop_length, n_fft, n_mels, mod_bin_width, audio_groups):
    global _WORKER_ACCUMULATOR
    _WORKER_ACCUMULATOR = ModulationAccumulator(
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_fft=n_fft,
        n_mels=n_mels,
        mod_bin_width=mod_bin_width,
        audio_groups=audio_groups,
    )


def _process_file_worker(path: str):
    global _WORKER_ACCUMULATOR
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != _WORKER_ACCUMULATOR.sample_rate:
        raise RuntimeError(f"{path}: expected sample_rate={_WORKER_ACCUMULATOR.sample_rate}, got {sr}")
    _WORKER_ACCUMULATOR.reset()
    _WORKER_ACCUMULATOR.process_array(y)
    return (
        _WORKER_ACCUMULATOR.power,
        _WORKER_ACCUMULATOR.num_examples,
        _WORKER_ACCUMULATOR.total_frames,
    )


def _analyze_folder(
    accumulator: ModulationAccumulator,
    input_root: Path,
    max_files: Optional[int],
    n_proc: int,
):
    files = _audio_files(input_root, max_files=max_files)
    if len(files) == 0:
        raise RuntimeError(f"No audio files found in {input_root}")

    n_proc = max(1, int(n_proc))
    if n_proc == 1:
        for i, path in enumerate(files, 1):
            y, sr = sf.read(path, dtype="float32", always_2d=False)
            if sr != accumulator.sample_rate:
                raise RuntimeError(f"{path}: expected sample_rate={accumulator.sample_rate}, got {sr}")
            accumulator.process_array(y)
            if i % 100 == 0 or i == len(files):
                print(f"[modulation] folder [{i}/{len(files)}]", flush=True)
        return len(files), [str(p) for p in files]

    with ProcessPoolExecutor(
        max_workers=n_proc,
        initializer=_init_worker,
        initargs=(
            accumulator.sample_rate,
            accumulator.hop_length,
            accumulator.n_fft,
            accumulator.n_mels,
            accumulator.mod_bin_width,
            accumulator.audio_groups,
        ),
    ) as executor:
        futures = [executor.submit(_process_file_worker, str(path)) for path in files]
        for i, future in enumerate(as_completed(futures), 1):
            power, num_examples, total_frames = future.result()
            accumulator.add_worker_result(power, num_examples, total_frames)
            if i % 100 == 0 or i == len(files):
                print(f"[modulation] folder [{i}/{len(files)}]", flush=True)
    return len(files), [str(p) for p in files]


def _analyze_dataset(
    args,
    accumulator: ModulationAccumulator,
    split: str,
    dataset_keys,
    n_examples: int,
    duration: float,
):
    folders = _arg_get(args, f"{split}/build_dataset.folders", None)
    if folders is None:
        raise RuntimeError(
            f"Missing `{split}/build_dataset.folders` in args/config. "
            "Pass a config that includes dataset folders."
        )

    if dataset_keys is None or len(dataset_keys) == 0:
        keys = list(folders.keys())
    else:
        keys = [str(k) for k in dataset_keys]
    sources = _flatten_sources(folders, keys)
    if len(sources) == 0:
        raise RuntimeError(f"No dataset sources found for keys={keys}")

    return _analyze_dataset_sources(
        args=args,
        accumulator=accumulator,
        split=split,
        sources=sources,
        n_examples=n_examples,
        duration=duration,
    )


def _analyze_dataset_sources(
    args,
    accumulator: ModulationAccumulator,
    split: str,
    sources,
    n_examples: int,
    duration: float,
):
    with argbind.scope(args, split):
        transform = train.build_transform()

    loader = AudioLoader(sources=sources)
    dataset = AudioDataset(
        loader,
        accumulator.sample_rate,
        n_examples=int(n_examples),
        duration=float(duration),
        transform=transform,
    )
    dataset.transform = transform

    for i in range(len(dataset)):
        batch = dataset[i]
        signal = dataset.transform(batch["signal"].clone(), **batch["transform_args"])
        accumulator.process_array(_audio_signal_to_numpy(signal))
        if (i + 1) % 100 == 0 or (i + 1) == len(dataset):
            print(f"[modulation] dataset/{split} [{i + 1}/{len(dataset)}]", flush=True)

    return len(dataset), list(sources)


def _analyze_dataset_by_domain(
    args,
    accumulator: ModulationAccumulator,
    output_root: Path,
    split: str,
    dataset_keys,
    n_examples: int,
    duration: float,
    output_kwargs,
):
    folders = _arg_get(args, f"{split}/build_dataset.folders", None)
    if folders is None:
        raise RuntimeError(
            f"Missing `{split}/build_dataset.folders` in args/config. "
            "Pass a config that includes dataset folders."
        )

    domains = _dataset_domain_keys(folders, dataset_keys)
    if len(domains) == 0:
        raise RuntimeError("No dataset domains found for modulation analysis")

    n_examples = int(n_examples)
    base = n_examples // len(domains)
    remainder = n_examples % len(domains)
    aggregate_sources = []
    domain_meta = []

    for domain_idx, (domain, keys) in enumerate(domains.items()):
        domain_n = base + (1 if domain_idx < remainder else 0)
        if domain_n <= 0:
            continue
        sources = _flatten_sources(folders, keys)
        if len(sources) == 0:
            continue

        domain_accumulator = ModulationAccumulator(
            sample_rate=accumulator.sample_rate,
            hop_length=accumulator.hop_length,
            n_fft=accumulator.n_fft,
            n_mels=accumulator.n_mels,
            mod_bin_width=accumulator.mod_bin_width,
            audio_groups=accumulator.audio_groups,
        )
        print(
            f"[modulation] domain={domain} keys={keys} n_examples={domain_n}",
            flush=True,
        )
        analyzed, used_sources = _analyze_dataset_sources(
            args=args,
            accumulator=domain_accumulator,
            split=split,
            sources=sources,
            n_examples=domain_n,
            duration=duration,
        )
        accumulator.add_worker_result(
            domain_accumulator.power,
            domain_accumulator.num_examples,
            domain_accumulator.total_frames,
        )
        aggregate_sources.extend(used_sources)
        domain_output = output_root / "domains" / domain
        domain_paths = _write_analysis_outputs(
            output_root=domain_output,
            accumulator=domain_accumulator,
            sources=used_sources,
            analyzed=analyzed,
            label=f"dataset/{split}/{domain}",
            **output_kwargs,
        )
        domain_meta.append(
            {
                "domain": domain,
                "keys": keys,
                "n_examples": analyzed,
                "sources": used_sources,
                "outputs": domain_paths,
            }
        )

    domains_path = output_root / "domains" / "metadata.json"
    domains_path.parent.mkdir(parents=True, exist_ok=True)
    domains_path.write_text(json.dumps(domain_meta, indent=2) + "\n")
    return accumulator.num_examples, aggregate_sources


def _make_schedule_rows(
    summary_rows,
    schedule_specs,
    target_sums,
    n_base_scales: int,
    latent_frame_rate: float,
):
    by_key = {(r["audio_group"], r["mode"]): r for r in summary_rows}
    rows = []
    for target_sum in target_sums:
        for spec in schedule_specs:
            label, group, mode, quantile = _parse_schedule_spec(spec)
            row = by_key[(group, mode)]
            q_pct = int(round(quantile * 100))
            bandwidth = float(row[f"q{q_pct}_hz"])
            pivot_raw = min(1.0, 2.0 * bandwidth / latent_frame_rate)
            pivot = round(pivot_raw, 2)
            gamma, base, actual = _solve_power_schedule(pivot, n_base_scales, float(target_sum))
            rows.append(
                {
                    "target_sum": float(target_sum),
                    "source": label,
                    "audio_group": group,
                    "mode": mode,
                    "quantile": quantile,
                    "mod_bandwidth_hz": bandwidth,
                    "pivot_raw": pivot_raw,
                    "pivot_rounded": pivot,
                    "gamma": gamma,
                    "base_scale_factor": json.dumps([round(float(x), 4) for x in base]),
                    "actual_schedule": json.dumps([round(float(x), 4) for x in actual]),
                    "actual_sum": float(actual.sum()),
                    "kbps": float(latent_frame_rate * 10.0 * actual.sum() / 1000.0),
                }
            )
    return rows


def _write_csv(path: Path, rows):
    rows = list(rows)
    if len(rows) == 0:
        raise RuntimeError(f"No rows to write for {path}")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_recommendation(
    path: Path,
    schedule_rows,
    recommended_source: str,
    recommended_target_sum: float,
):
    selected = None
    for row in schedule_rows:
        if row["source"] == recommended_source and float(row["target_sum"]) == float(recommended_target_sum):
            selected = row
            break
    if selected is None:
        selected = schedule_rows[0]

    lines = [
        "# Temporal Modulation Schedule Recommendation",
        "",
        f"- Selected source: `{selected['source']}`",
        f"- Target scale sum: `{selected['target_sum']}`",
        f"- Modulation bandwidth: `{float(selected['mod_bandwidth_hz']):.3f}` Hz",
        f"- Derived pivot: `{float(selected['pivot_raw']):.4f}` -> `{float(selected['pivot_rounded']):.2f}`",
        f"- Power-law gamma: `{float(selected['gamma']):.4f}`",
        f"- Estimated kbps with 1024-entry codebooks: `{float(selected['kbps']):.3f}`",
        "",
        "```yaml",
        f"EMAC.scale_factor: {selected['base_scale_factor']}",
        "EMAC.use_wavescale: True",
        "```",
        "",
        "Interpretation: a WaveScale stage at scale `s` has temporal modulation "
        "Nyquist `s * (sample_rate / hop_length) / 2`.  The pivot is derived from "
        "the selected modulation-energy quantile, and the remaining scales are "
        "allocated by a power-law curve constrained to the target bitrate.",
    ]
    path.write_text("\n".join(lines) + "\n")


def _selected_schedule(schedule_rows, recommended_source: str, recommended_target_sum: float):
    for row in schedule_rows:
        if row["source"] == recommended_source and float(row["target_sum"]) == float(recommended_target_sum):
            return row
    return schedule_rows[0]


def _load_json_array(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _plot_modulation_analysis(
    output_root: Path,
    spectrum_rows,
    summary_rows,
    schedule_rows,
    recommended_source: str,
    recommended_target_sum: float,
    latent_frame_rate: float,
):
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_root.mkdir(parents=True, exist_ok=True)
    selected = _selected_schedule(schedule_rows, recommended_source, recommended_target_sum)
    base = np.asarray(_load_json_array(selected["base_scale_factor"]), dtype=float)
    actual = np.asarray(_load_json_array(selected["actual_schedule"]), dtype=float)
    pivot = float(selected["pivot_rounded"])
    bandwidth = float(selected["mod_bandwidth_hz"])

    freqs = np.asarray([float(r["mod_freq_hz"]) for r in spectrum_rows])
    all_eq = np.asarray([float(r.get("all_equalized", 0.0)) for r in spectrum_rows])
    all_energy = np.asarray([float(r.get("all_energy_weighted", 0.0)) for r in spectrum_rows])
    top_eq = np.asarray([float(r.get("top_12_22k_equalized", 0.0)) for r in spectrum_rows])
    high_eq = np.asarray([float(r.get("high_4_12k_equalized", 0.0)) for r in spectrum_rows])

    def normalize(x):
        total = float(np.sum(x))
        return x / total if total > 0 else x

    def cdf(x):
        x = normalize(x)
        return np.cumsum(x)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), dpi=160)
    ax = axes[0, 0]
    ax.plot(freqs, normalize(all_energy), label="all, energy-weighted", linewidth=2.0)
    ax.plot(freqs, normalize(all_eq), label="all, band-equalized", linewidth=2.0)
    ax.plot(freqs, normalize(high_eq), label="4-12 kHz, equalized", linewidth=1.8)
    ax.plot(freqs, normalize(top_eq), label="12-22 kHz, equalized", linewidth=1.8)
    ax.axvline(bandwidth, color="black", linewidth=1.2, linestyle="--", label=f"selected {bandwidth:.1f} Hz")
    ax.set_xlim(0, min(30.0, latent_frame_rate / 2))
    ax.set_xlabel("Temporal modulation frequency (Hz)")
    ax.set_ylabel("Normalized modulation power")
    ax.set_title("Temporal Modulation Spectrum")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(freqs, cdf(all_energy), label="all, energy-weighted", linewidth=2.0)
    ax.plot(freqs, cdf(all_eq), label="all, band-equalized", linewidth=2.0)
    ax.plot(freqs, cdf(high_eq), label="4-12 kHz, equalized", linewidth=1.8)
    ax.plot(freqs, cdf(top_eq), label="12-22 kHz, equalized", linewidth=1.8)
    ax.axvline(bandwidth, color="black", linewidth=1.2, linestyle="--")
    for q in [0.75, 0.80, 0.85, 0.90]:
        ax.axhline(q, color="gray", linewidth=0.7, linestyle=":", alpha=0.65)
    ax.set_xlim(0, min(30.0, latent_frame_rate / 2))
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Temporal modulation frequency (Hz)")
    ax.set_ylabel("Cumulative modulation energy")
    ax.set_title("Modulation CDF")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    stage_idx = np.arange(len(actual))
    mod_nyquist = actual * latent_frame_rate / 2.0
    ax.bar(stage_idx, actual, color="#4c78a8", alpha=0.82, label="scale")
    ax.axhline(pivot, color="black", linewidth=1.2, linestyle="--", label=f"pivot {pivot:.2f}")
    ax.set_xlabel("Expanded WaveScale stage")
    ax.set_ylabel("Scale factor")
    ax.set_title("Recommended Expanded Schedule")
    ax.set_ylim(0, 1.08)
    ax.grid(True, axis="y", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(stage_idx, mod_nyquist, color="#f58518", marker="o", linewidth=1.8, label="modulation Nyquist")
    ax2.set_ylabel("Temporal modulation Nyquist (Hz)")
    ax2.set_ylim(0, latent_frame_rate / 2 * 1.08)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8, loc="lower center")

    ax = axes[1, 1]
    specs = [
        r
        for r in schedule_rows
        if float(r["target_sum"]) == float(recommended_target_sum)
    ]
    labels = [r["source"].replace("_", "\n") for r in specs]
    pivots = [float(r["pivot_rounded"]) for r in specs]
    bws = [float(r["mod_bandwidth_hz"]) for r in specs]
    colors = ["#54a24b" if r is selected else "#9ecae9" for r in specs]
    bars = ax.bar(np.arange(len(specs)), pivots, color=colors)
    ax.set_xticks(np.arange(len(specs)))
    ax.set_xticklabels(labels, rotation=0, fontsize=7)
    ax.set_ylabel("Derived pivot scale")
    ax.set_title(f"Candidate Pivots at Target Sum {recommended_target_sum}")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, bw in zip(bars, bws):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bw:.1f}Hz",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_ylim(0, max(0.5, max(pivots) + 0.08))

    fig.suptitle(
        f"Temporal Modulation Analysis | selected={selected['source']} | "
        f"scale_sum={float(selected['target_sum']):.1f} | kbps={float(selected['kbps']):.2f}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    overview_path = output_root / "modulation_analysis_overview.png"
    fig.savefig(overview_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=160)
    base_idx = np.arange(len(base))
    ax.plot(base_idx, base, marker="o", linewidth=2.0, color="#4c78a8")
    ax.fill_between(base_idx, base, pivot, color="#4c78a8", alpha=0.15)
    ax.axhline(pivot, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Base scale index")
    ax.set_ylabel("Scale factor")
    ax.set_title("Base Scale Factors for Config")
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.25)
    for i, value in enumerate(base):
        ax.text(i, value + 0.025, f"{value:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    schedule_path = output_root / "recommended_schedule.png"
    fig.savefig(schedule_path)
    plt.close(fig)

    return [overview_path, schedule_path]


def _write_analysis_outputs(
    output_root: Path,
    accumulator: ModulationAccumulator,
    sources,
    analyzed: int,
    label: str,
    source: str,
    input: str,
    split: str,
    dataset_keys: list,
    quantiles: list,
    check_freqs: list,
    target_sums: list,
    n_base_scales: int,
    schedule_specs: list,
    recommended_schedule_source: str,
    recommended_target_sum: float,
    make_plots: bool,
):
    output_root.mkdir(parents=True, exist_ok=True)

    spectrum_rows = list(accumulator.spectrum_rows())
    summary_rows = accumulator.summary_rows(quantiles=quantiles, check_freqs=check_freqs)
    schedule_rows = _make_schedule_rows(
        summary_rows=summary_rows,
        schedule_specs=schedule_specs,
        target_sums=target_sums,
        n_base_scales=n_base_scales,
        latent_frame_rate=accumulator.latent_frame_rate,
    )

    spectrum_path = output_root / "temporal_modulation_spectrum.csv"
    summary_path = output_root / "temporal_modulation_summary.csv"
    schedule_path = output_root / "recommended_schedules.csv"
    recommendation_path = output_root / "recommendation.md"
    meta_path = output_root / "meta.json"

    _write_csv(spectrum_path, spectrum_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(schedule_path, schedule_rows)
    _write_recommendation(
        recommendation_path,
        schedule_rows=schedule_rows,
        recommended_source=recommended_schedule_source,
        recommended_target_sum=recommended_target_sum,
    )
    plot_paths = []
    if make_plots:
        plot_paths = _plot_modulation_analysis(
            output_root=output_root,
            spectrum_rows=spectrum_rows,
            summary_rows=summary_rows,
            schedule_rows=schedule_rows,
            recommended_source=recommended_schedule_source,
            recommended_target_sum=recommended_target_sum,
            latent_frame_rate=accumulator.latent_frame_rate,
        )

    output_paths = [
        str(spectrum_path),
        str(summary_path),
        str(schedule_path),
        str(recommendation_path),
        *[str(path) for path in plot_paths],
        str(meta_path),
    ]
    meta = {
        "label": label,
        "source": source,
        "input": input,
        "split": split,
        "dataset_keys": dataset_keys,
        "num_analyzed": analyzed,
        "num_examples_accumulated": accumulator.num_examples,
        "total_frames": accumulator.total_frames,
        "sample_rate": accumulator.sample_rate,
        "hop_length": accumulator.hop_length,
        "latent_frame_rate_hz": accumulator.latent_frame_rate,
        "n_fft": accumulator.n_fft,
        "n_mels": accumulator.n_mels,
        "mod_bin_width": accumulator.mod_bin_width,
        "audio_groups": accumulator.audio_groups,
        "sources": list(sources),
        "outputs": output_paths,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"[modulation] wrote {summary_path}", flush=True)
    print(f"[modulation] wrote {schedule_path}", flush=True)
    print(f"[modulation] wrote {recommendation_path}", flush=True)
    for path in plot_paths:
        print(f"[modulation] wrote {path}", flush=True)
    return output_paths


@argbind.bind(without_prefix=True)
def analyze_modulation(
    args,
    source: str = "folder",
    input: str = "eval_set/general",
    output: str = "results/modulation_analysis",
    split: str = "train",
    dataset_keys: list = [],
    n_examples: int = 3000,
    max_files: int = None,
    duration: float = 10.0,
    sample_rate: int = 44100,
    hop_length: int = 512,
    n_fft: int = 2048,
    n_mels: int = 80,
    mod_bin_width: float = 0.1,
    audio_groups: list = None,
    quantiles: list = DEFAULT_QUANTILES,
    check_freqs: list = DEFAULT_CHECK_FREQS,
    target_sums: list = [9.0, 9.3],
    n_base_scales: int = 7,
    schedule_specs: list = DEFAULT_SCHEDULE_SPECS,
    recommended_schedule_source: str = "top_q75_equalized",
    recommended_target_sum: float = 9.3,
    make_plots: bool = True,
    domain_outputs: bool = False,
    n_proc: int = 24,
):
    output_root = Path(output)
    output_root.mkdir(parents=True, exist_ok=True)

    accumulator = ModulationAccumulator(
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_fft=n_fft,
        n_mels=n_mels,
        mod_bin_width=mod_bin_width,
        audio_groups=audio_groups,
    )

    print(
        f"[modulation] source={source} sample_rate={sample_rate} "
        f"hop={hop_length} latent_rate={accumulator.latent_frame_rate:.3f}Hz",
        flush=True,
    )

    output_kwargs = {
        "source": source,
        "input": input,
        "split": split,
        "dataset_keys": dataset_keys,
        "quantiles": quantiles,
        "check_freqs": check_freqs,
        "target_sums": target_sums,
        "n_base_scales": n_base_scales,
        "schedule_specs": schedule_specs,
        "recommended_schedule_source": recommended_schedule_source,
        "recommended_target_sum": recommended_target_sum,
        "make_plots": make_plots,
    }

    if source == "folder":
        analyzed, sources = _analyze_folder(
            accumulator,
            Path(input),
            max_files=max_files,
            n_proc=n_proc,
        )
    elif source == "dataset":
        if domain_outputs:
            analyzed, sources = _analyze_dataset_by_domain(
                args=args,
                accumulator=accumulator,
                output_root=output_root,
                split=split,
                dataset_keys=dataset_keys,
                n_examples=n_examples,
                duration=duration,
                output_kwargs=output_kwargs,
            )
        else:
            analyzed, sources = _analyze_dataset(
                args=args,
                accumulator=accumulator,
                split=split,
                dataset_keys=dataset_keys,
                n_examples=n_examples,
                duration=duration,
            )
    else:
        raise ValueError("source must be one of: folder, dataset")

    _write_analysis_outputs(
        output_root=output_root,
        accumulator=accumulator,
        sources=sources,
        analyzed=analyzed,
        label=f"{source}/{split}" if source == "dataset" else source,
        **output_kwargs,
    )


if __name__ == "__main__":
    parsed_args = argbind.parse_args()
    with argbind.scope(parsed_args):
        analyze_modulation(parsed_args)
