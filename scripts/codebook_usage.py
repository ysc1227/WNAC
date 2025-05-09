import warnings
from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal
from audiotools.core import util
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import os
from collections import defaultdict

from wnac.utils import load_model

warnings.filterwarnings("ignore", category=UserWarning)


@argbind.bind(positional=True, without_prefix=True)
@torch.inference_mode()
@torch.no_grad()
def encode(
    input: str,
    weights_path: str = "",
    model_tag: str = "latest",
    model_bitrate: str = "8kbps",
    n_quantizers: int = None,
    device: str = "cuda",
    model_type: str = "44khz",
    win_duration: float = 5.0,
    verbose: bool = False,
    plot_path: str = "",
    sample_rate: int = 44100,
    scale: str = "",
    is_wave: str = "",
    legends: str = ""
):
    """Encode audio files in input path to .wnac format.

    Parameters
    ----------
    input : str
        Path to input audio file or directory
    output : str, optional
        Path to output directory, by default "". If `input` is a directory, the directory sub-tree relative to `input` is re-created in `output`.
    weights_path : str, optional
        Path to weights file, by default "". If not specified, the weights file will be downloaded from the internet using the
        model_tag and model_type.
    model_tag : str, optional
        Tag of the model to use, by default "latest". Ignored if `weights_path` is specified.
    model_bitrate: str
        Bitrate of the model. Must be one of "8kbps", or "16kbps". Defaults to "8kbps".
    n_quantizers : int, optional
        Number of quantizers to use, by default None. If not specified, all the quantizers will be used and the model will compress at maximum bitrate.
    device : str, optional
        Device to use, by default "cuda"
    model_type : str, optional
        The type of model to use. Must be one of "44khz", "24khz", or "16khz". Defaults to "44khz". Ignored if `weights_path` is specified.
    """
    
    weights_path = weights_path.split(',')
    scale = [[float(s) for s in sc.split(',')] for sc in scale.split('|')]
    is_wave = [w == "True" for w in is_wave.split(',')]
    
    model_counters = []
    model_length = [sc[::-1] + sc[1:] if wave else sc for wave, sc in zip(is_wave, scale)]
    model_labels = [
        lg for lg in legends.split(',')
    ]
    
    print(model_length)
    print(model_labels)
    
    for wp in weights_path:
        generator = load_model(
            model_type=model_type,
            model_bitrate=model_bitrate,
            tag=model_tag,
            load_path=wp,
        )
        generator.to(device)
        generator.eval()
        kwargs = {"n_quantizers": n_quantizers}

        # Find all audio files in input path
        input = Path(input)
        audio_files = util.find_audio(input)

        codes_counter = [Counter() for _ in range(len(generator.quantizer.quantizers))]
        for i in tqdm(range(len(audio_files)), desc="Encoding files"):
            signal = AudioSignal(audio_files[i], sample_rate=sample_rate)

            # Encode audio to .wnac format
            artifact = generator.compress(signal, win_duration, verbose=verbose, **kwargs)
            for k in artifact.codes:
                for l, counter in zip(k, codes_counter):
                    indices = l.flatten().tolist()
                    counter.update(indices)
            
        model_counters.append([{k: v / sum(cc.values()) for k, v in cc.items()} for cc in codes_counter])
        
    save_usage(
        model_counters, 
        model_length,
        model_labels,
        plot_path
    )
    
def save_usage(model_counters, model_lengths, model_labels, output_dir):
    """
    여러 모델의 unused code ratio vs code length 그래프를 한 plot에 그림

    Args:
        model_counters: List[List[Counter]] - 모델별 각 코드북의 카운터 리스트
        model_lengths: List[List[int]] - 모델별 코드북 길이 리스트
        model_labels: List[str] - 각 모델의 이름
        output_dir: str - 저장 디렉토리
    """
    plt.figure(figsize=(9, 6))

    for counters, lengths, label in zip(model_counters, model_lengths, model_labels):
        usage_by_length = defaultdict(list)

        for counter, length in zip(counters, lengths):
            max_index = max(counter.keys()) if counter else 0
            total_codes = max_index + 1
            used_indices = set(counter.keys())
            unused_count = total_codes - len(used_indices)
            unused_ratio = unused_count / total_codes if total_codes > 0 else 0.0
            usage_by_length[length].append(unused_ratio)

        sorted_lengths = sorted(usage_by_length.keys())
        avg_unused_ratios = [
            sum(usage_by_length[l]) / len(usage_by_length[l]) for l in sorted_lengths
        ]

        # 각 모델별 선 그래프 추가 (포인트 없음)
        plt.plot(sorted_lengths, avg_unused_ratios, label=label, linewidth=2)

    plt.xlabel("Code Length")
    plt.ylabel("Average Unused Code Ratio")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "unused_code_ratio_comparison.png")
    plt.savefig(filename)
    plt.close()

    print(f"Saved comparison plot: {filename}")

def save_usage_by_index(model_counters, model_labels, output_dir):
    """
    코드북 인덱스별 unused code ratio를 모델별로 비교하여 한 plot에 그림

    Args:
        model_counters: List[List[Counter]] - 모델별 각 코드북의 카운터 리스트
        model_labels: List[str] - 각 모델의 이름
        output_dir: str - 저장 디렉토리
    """
    plt.figure(figsize=(9, 6))

    for counters, label in zip(model_counters, model_labels):
        unused_ratios = []

        for counter in counters:
            max_index = max(counter.keys()) if counter else 0
            total_codes = max_index + 1
            used_indices = set(counter.keys())
            unused_count = total_codes - len(used_indices)
            unused_ratio = unused_count / total_codes if total_codes > 0 else 0.0
            unused_ratios.append(unused_ratio)

        plt.plot(range(len(unused_ratios)), unused_ratios, label=label, linewidth=2)

    plt.xlabel("Codebook Index")
    plt.ylabel("Unused Code Ratio")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "unused_code_ratio_by_index.png")
    plt.savefig(filename)
    plt.close()

    print(f"Saved index-wise usage plot: {filename}")

if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        encode()
        
