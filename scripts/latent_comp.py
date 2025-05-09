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
    
    model_loss = []
    model_length = [sc for sc in scale]
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
        
        loss = {}

        for i in tqdm(range(len(audio_files)), desc="Encoding files"):
            signal = AudioSignal(audio_files[i], sample_rate=sample_rate)

            # Encode audio to .wnac format
            artifact = generator.compress(signal, win_duration, verbose=verbose, get_latent=True, **kwargs)

            for j in range(len(artifact) // 2 + 1):
                if j in loss:
                    loss[j] += torch.nn.functional.mse_loss(artifact[0], artifact[-(j+1)]).cpu()
                else:
                    loss[j] = torch.nn.functional.mse_loss(artifact[0], artifact[-(j+1)]).cpu()
        
        model_loss.append([v / len(audio_files) for v in list(loss.values())[::-1]])
        
    save_latent_comp(model_loss, model_length, model_labels, output_dir=plot_path)

    
def save_latent_comp(model_loss, model_lengths, model_labels, output_dir):
    """
    스케일별 reconstruction 차이를 모델별로 plot하고 저장합니다.

    Args:
        model_loss (List[List[float]]): 모델별 스케일별 reconstruction 차이값
        model_lengths (List[List[float]]): 모델별 스케일 값 리스트
        model_labels (List[str]): 모델 이름 리스트
        output_dir (str): 저장 디렉토리 경로
    """
    plt.figure(figsize=(9, 6))

    for losses, lengths, label in zip(model_loss, model_lengths, model_labels):
        plt.plot(lengths, losses, label=label, marker='o', linewidth=2)

    plt.xlabel("Scale")
    plt.ylabel("Metric Difference")
    plt.title("Scale-wise Reconstruction Difference Comparison")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "scalewise_recon_diff.png")
    plt.savefig(filename)
    plt.close()

    print(f"Saved plot to {filename}")

if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        encode()
        
