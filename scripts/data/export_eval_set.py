import csv
from collections import OrderedDict
from pathlib import Path

import argbind
import torch
from audiotools.core import util
from audiotools.data.datasets import AudioDataset, AudioLoader
from audiotools.ml.decorators import Tracker
from scripts.training.emac import Accelerator

import scripts.training.emac as train


@torch.no_grad()
def process(batch, accel, test_data):
    batch = util.prepare_batch(batch, accel.device)
    signal = test_data.transform(batch["signal"].clone(), **batch["transform_args"])
    return signal.cpu()


def _arg_get(args, key, default=None):
    """Read argbind args whether it behaves like dict or namespace."""
    if isinstance(args, dict):
        return args.get(key, default)
    if hasattr(args, "get"):
        try:
            return args.get(key, default)
        except Exception:
            pass
    return getattr(args, key.replace("/", "_"), default)


def _flatten_sources(folders: dict, keys):
    sources = []
    for key in keys:
        values = folders.get(key, [])
        if isinstance(values, (str, Path)):
            values = [values]
        sources.extend([str(v) for v in values])
    return sources


def _domain_sources(folders: dict):
    """Map eval-set yaml folder groups to requested export domains.

    - speech: all yaml groups whose name starts with "speech"
    - music: all yaml groups whose name starts with "music"
    - environment: old "general" group, renamed for output
    - general: mixture of speech + music + environment
    """
    speech_keys = [k for k in folders if str(k).startswith("speech")]
    music_keys = [k for k in folders if str(k).startswith("music")]
    environment_keys = [k for k in folders if str(k) in {"general", "environment"}]

    domains = OrderedDict()
    domains["speech"] = _flatten_sources(folders, speech_keys)
    domains["music"] = _flatten_sources(folders, music_keys)
    domains["environment"] = _flatten_sources(folders, environment_keys)
    domains["general"] = domains["speech"] + domains["music"] + domains["environment"]
    return domains


def _build_domain_dataset(args, sample_rate: int, sources, n_examples: int, duration: float):
    with argbind.scope(args, "test"):
        transform = train.build_transform()
    loader = AudioLoader(sources=sources)
    dataset = AudioDataset(
        loader,
        sample_rate,
        n_examples=n_examples,
        duration=duration,
        transform=transform,
    )
    dataset.transform = transform
    return dataset


@argbind.bind(without_prefix=True)
@torch.no_grad()
def save_test_set(
    args,
    accel,
    sample_rate: int = 44100,
    output: str = "samples/eval",
    samples_per_domain: int = 3000,
    duration: float = None,
):
    tracker = Tracker()
    folders = _arg_get(args, "test/build_dataset.folders", None)
    if folders is None:
        raise RuntimeError(
            "Missing `test/build_dataset.folders` in args/config. "
            "Pass e.g. `--args.load conf/base.yml`."
        )
    if duration is None:
        duration = float(_arg_get(args, "test/AudioDataset.duration", 10.0))

    domains = _domain_sources(folders)
    empty = [name for name, sources in domains.items() if len(sources) == 0]
    if empty:
        raise RuntimeError(
            f"No sources found for domains {empty}. Available yaml folder keys: {list(folders.keys())}"
        )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    with tracker.live:
        for domain, sources in domains.items():
            domain_data = _build_domain_dataset(
                args=args,
                sample_rate=sample_rate,
                sources=sources,
                n_examples=samples_per_domain,
                duration=duration,
            )

            global process
            tracked_process = tracker.track(f"process/{domain}", len(domain_data))(process)

            domain_dir = output / domain
            domain_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = domain_dir / "metadata.csv"
            keys = ["domain", "sample_index", "path", "original"]

            with open(metadata_path, "w") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=keys)
                writer.writeheader()

                for i in range(len(domain_data)):
                    signal = tracked_process(domain_data[i], accel, domain_data)
                    wav_path = domain_dir / f"sample_{i}.wav"
                    metadata = {
                        "domain": domain,
                        "sample_index": i,
                        "path": str(wav_path),
                        "original": str(signal.path_to_input_file),
                    }
                    writer.writerow(metadata)
                    signal.write(wav_path)

            summary_rows.append({
                "domain": domain,
                "n_samples": len(domain_data),
                "duration": duration,
                "sample_rate": sample_rate,
                "sources": ";".join(sources),
                "metadata": str(metadata_path),
            })

        with open(output / "metadata.csv", "w") as csvfile:
            keys = ["domain", "n_samples", "duration", "sample_rate", "sources", "metadata"]
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_rows)

        tracker.done(
            "test",
            f"domains={len(domains)} N/domain={samples_per_domain} total={samples_per_domain * len(domains)}",
        )


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        with Accelerator() as accel:
            save_test_set(args, accel)