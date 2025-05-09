import time
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

from wnac.utils import load_model
from torcheval.metrics import FrechetAudioDistance

warnings.filterwarnings("ignore", category=UserWarning)

@argbind.bind(group="recon", positional=True, without_prefix=True)
@torch.inference_mode()
@torch.no_grad()
def recon(
    input: str,
    output: str = None,
    weights_path: str = "",
    model_tag: str = "latest",
    model_bitrate: str = "8kbps",
    n_quantizers: int = None,
    device: str = "cuda",
    model_type: str = "44khz",
    win_duration: float = 5.0,
    verbose: bool = False,
    plot_path: str = None,
    resume: bool = False,
    mono: bool = False,
    fad_score_file: str = None,
    seed: int = 0,
    sample_rate: int = 22050
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
    
    util.seed(seed)
    generator.to(device)
    generator.eval()
    kwargs = {"n_quantizers": n_quantizers}
    
    fad_metric = FrechetAudioDistance.with_vggish(device=device).to(device)
    fad_score = 0

    # Find all audio files in input path
    input = Path(input)
    audio_files = util.find_audio(input)
 
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)

    codes_counter = [Counter() for _ in range(len(generator.quantizer.quantizers))]
    progress = tqdm(range(len(audio_files)), desc="Reconstructing files", leave=False)
    
    if fad_score_file:
        os.makedirs(Path(fad_score_file).parent, exist_ok=True)
        with open(fad_score_file, 'w') as f:
            f.write("FAD Score per file\n")
    
    for i in progress:
        progress.set_postfix(file_path=audio_files[i], mean_FAD=fad_score)
        
        if output:
            relative_path = audio_files[i].relative_to(input)
            output_dir = output / relative_path.parent
            
            if not relative_path.name:
                output_dir = output
                relative_path = audio_files[i]
            output_name = relative_path.with_suffix(".wav").name
            output_path = output_dir / output_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
            if os.path.exists(output_path) and resume:
                continue
        
        # Load file
        signal = AudioSignal(audio_files[i], sample_rate=sample_rate, mono=mono)

        # Encode audio to .wnac format
        artifact = generator.compress(signal, win_duration, verbose=verbose, **kwargs)
        for k in artifact.codes:
            for l, counter in zip(k, codes_counter):
                indices = l.flatten().tolist()
                counter.update(indices)
                
        recons = generator.decompress(artifact, verbose=verbose)
        fad_metric.update(recons.audio_data.squeeze(0), signal.audio_data.squeeze(0))
        fad_score = fad_metric.compute()
        
        with open(fad_score_file, 'w') as f:
            f.write("FAD Score per file\n")
            f.write(f"{audio_files[i]} - fad_score: {fad_score}\n")
        
        if output:
            recons.cpu().write(output_path)
    
    usage_ratios = [{k: v / sum(cc.values()) for k, v in cc.items()} for cc in codes_counter]
   
    if plot_path:
        save_histograms(usage_ratios, plot_path)

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

if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        recon()
        

