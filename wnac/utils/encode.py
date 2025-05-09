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


@argbind.bind(group="encode", positional=True, without_prefix=True)
@torch.inference_mode()
@torch.no_grad()
def encode(
    input: str,
    output: str = "",
    weights_path: str = "",
    model_tag: str = "latest",
    model_bitrate: str = "8kbps",
    n_quantizers: int = None,
    device: str = "cuda",
    model_type: str = "44khz",
    win_duration: float = 5.0,
    verbose: bool = False,
    plot_path: str = "",
    sample_rate: int = 44100
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

    generator = load_model(
        model_type=model_type,
        model_bitrate=model_bitrate,
        tag=model_tag,
        load_path=weights_path,
    )
    generator.to(device)
    generator.eval()
    kwargs = {"n_quantizers": n_quantizers}

    # Find all audio files in input path
    input = Path(input)
    audio_files = util.find_audio(input)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    
    scale = [0.03125, 0.05, 0.08, 0.12, 0.16, 0.21, 0.27, 0.33, 0.41, 0.49, 0.57, 0.67, 0.77, 0.88, 1]
    wave = False

    total_time = 0
    codes_counter = [Counter() for _ in range(len(generator.quantizer.quantizers))]
    for i in tqdm(range(len(audio_files)), desc="Encoding files"):
        signal = AudioSignal(audio_files[i], sample_rate=sample_rate)

        # Encode audio to .wnac format
        import time
        start = time.time()
        artifact = generator.compress(signal, win_duration, verbose=verbose, **kwargs)
        end = time.time()
        total_time += end - start
        for k in artifact.codes:
            for l, counter in zip(k, codes_counter):
                indices = l.flatten().tolist()
                counter.update(indices)
                
        # Compute output path
        relative_path = audio_files[i].relative_to(input)
        output_dir = output / relative_path.parent
        if not relative_path.name:
            output_dir = output
            relative_path = audio_files[i]
        output_name = relative_path.with_suffix(".wnac").name
        output_path = output_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        artifact.save(output_path)
        
    print(f"Ex: {total_time / len(audio_files):.4f}")
    usage_ratios = [{k: v / sum(cc.values()) for k, v in cc.items()} for cc in codes_counter]
    save_usage(
        usage_ratios, 
        scale[::-1] + scale[1:] if wave else scale,
        plot_path)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        encode()
        
def save_histograms(counter_list, output_dir):
    for i, counter in enumerate(counter_list):
        indices = list(counter.keys())
        counts = list(counter.values())

        # 히스토그램 그리기
        plt.figure(figsize=(8, 5))
        plt.bar(indices, counts, width=0.5, edgecolor='black', alpha=0.7)
        plt.xlabel("Codebook Index")
        plt.ylabel("Usage Count")
        plt.title(f"Codebook Usage Histogram for Counter {i + 1}")
        plt.xticks(indices)
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        # 파일 저장
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir, f"code_{i + 1}.png")
        plt.savefig(filename)
        plt.close()  # plt.show() 대신 플롯을 닫음

        print(f"Saved: {filename}")

def save_usage(counter_list, codebook_lengths, output_dir):
    """
    Args:
        counter_list: List[Counter] - 각 quantizer의 code index 사용 빈도
        codebook_lengths: List[int] - 각 quantizer의 code 길이
        output_dir: str - 결과 저장 경로
    """
    # code length별로 unused ratio들을 모은다
    usage_by_length = defaultdict(list)

    for counter, length in zip(counter_list, codebook_lengths):
        max_index = max(counter.keys()) if counter else 0
        total_codes = max_index + 1
        used_indices = set(counter.keys())
        unused_count = total_codes - len(used_indices)
        unused_ratio = unused_count / total_codes if total_codes > 0 else 0.0
        usage_by_length[length].append(unused_ratio)

    # 길이 오름차순 정렬 및 평균 계산
    sorted_lengths = sorted(usage_by_length.keys())
    avg_unused_ratios = [sum(usage_by_length[l]) / len(usage_by_length[l]) for l in sorted_lengths]

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_lengths, avg_unused_ratios, linestyle='-', linewidth=2)
    plt.xlabel("Code Length")
    plt.ylabel("Average Unused Code Ratio")
    plt.title("Unused Code Ratio vs Code Length")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "unused_code_ratio_by_length.png")
    plt.savefig(filename)
    plt.close()

    print(f"Saved unused code ratio plot by code length: {filename}")