import warnings
from pathlib import Path
import argbind
import librosa
from matplotlib import gridspec
import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.core import util
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import random

from wnac.utils import load_model

warnings.filterwarnings("ignore", category=UserWarning)

@argbind.bind(positional=True, without_prefix=True)
@torch.inference_mode()
@torch.no_grad()
def encode(
    input: str,
    weights_paths: str = "",
    model_labels: str = "",
    n_quantizers: int = None,
    device: str = "cuda",
    model_type: str = "44khz",
    win_duration: float = 5.0,
    verbose: bool = False,
    plot_path: str = "",
    sample_rate: int = 44100,
):
    # 여러 모델 정보 파싱
    weights_paths = weights_paths.split(',')
    model_labels = model_labels.split(',')

    input = Path(input)
    audio_files = util.find_audio(input)

    np.random.seed(5)
    random.seed(5)
    sampled_files = random.sample(audio_files, min(len(audio_files), 3))

    all_recon_lists = []  # 전체 모델별 리스트
    model_names = []

    for weights_path, model_label in zip(weights_paths, model_labels):
        print(f"Loading model: {model_label}")

        generator = load_model(
            model_type=model_type,
            model_bitrate="8kbps",
            tag='best',
            load_path=weights_path,
        )
        generator.to(device)
        generator.eval()
        kwargs = {"n_quantizers": n_quantizers}

        model_recons = []

        for file in tqdm(sampled_files, desc=f"Encoding with {model_label}"):
            signal = AudioSignal(file, sample_rate=sample_rate)
            artifact = generator.compress(signal, win_duration, verbose=verbose, **kwargs)
            map = generator.quantizer.from_codes(artifact.codes[0], scale_wise=True)
            recon_list = [generator.decode(m).cpu() for m in map]
            model_recons.append(recon_list)

        all_recon_lists.append(model_recons)
        model_names.append(model_label)

    plot_spectrogram_differences(
        all_recon_lists,
        model_names,
        plot_path,
        sample_rate=sample_rate
    )

def plot_spectrogram_differences(all_recon_lists, model_names, output_dir, sample_rate=22050, max_freq=6000, num_pairs_to_plot=6):
    os.makedirs(output_dir, exist_ok=True)

    num_models = len(all_recon_lists)
    num_samples = len(all_recon_lists[0])

    # 빈 줄 포함한 총 row 수
    total_rows = num_samples * (num_models + 1) - 1

    # height_ratios 설정
    height_ratios = []
    for sample_idx in range(num_samples):
        height_ratios.extend([1] * num_models)  # 모델별 normal 높이
        if sample_idx != num_samples - 1:
            height_ratios.append(0.05)  # 샘플 간격용 얇은 빈 줄

    fig = plt.figure(figsize=(3*num_pairs_to_plot, 2.5*total_rows))

    gs = gridspec.GridSpec(
        total_rows, num_pairs_to_plot,
        height_ratios=height_ratios,
        hspace=0.1, wspace=0.05
    )

    axs = np.empty((total_rows, num_pairs_to_plot), dtype=object)

    current_row = 0
    for sample_idx in range(num_samples):
        for model_idx, (recon_lists, model_name) in enumerate(zip(all_recon_lists, model_names)):
            recon_list = recon_lists[sample_idx]  # 샘플 우선
            row_idx = current_row

            total_scales = len(recon_list)
            interval = total_scales // num_pairs_to_plot
            indices = np.arange(0, total_scales, interval)

            if len(indices) > num_pairs_to_plot + 1:
                indices = indices[:num_pairs_to_plot+1]

            for col_idx in range(num_pairs_to_plot):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                axs[row_idx, col_idx] = ax

                i = indices[col_idx]
                j = indices[col_idx + 1]

                recon_i = recon_list[i].squeeze().cpu().detach().numpy()
                recon_j = recon_list[j].squeeze().cpu().detach().numpy()

                spec_i = librosa.stft(recon_i, n_fft=1024, hop_length=256)
                spec_j = librosa.stft(recon_j, n_fft=1024, hop_length=256)

                mag_i = np.abs(spec_i)
                mag_j = np.abs(spec_j)

                mag_diff = np.abs(mag_j - mag_i)
                mag_diff_db = librosa.amplitude_to_db(mag_diff, ref=np.max)

                freqs = np.linspace(0, sample_rate/2, mag_diff_db.shape[0])
                freq_mask = freqs <= max_freq
                mag_diff_db = mag_diff_db[freq_mask, :]

                img = ax.imshow(
                    mag_diff_db,
                    origin="lower",
                    aspect="auto",
                    cmap="plasma",
                    extent=[0, recon_i.shape[-1]/sample_rate, 0, max_freq],
                    vmin=-80, vmax=0
                )

                ax.set_xticks([])
                ax.set_yticks([])

            current_row += 1

        if sample_idx != num_samples - 1:
            current_row += 1  # 샘플 끝나면 빈 줄 하나 삽입

    # Column Titles
    for ax, start, end in zip(axs[0], indices[:-1], indices[1:]):
        ax.set_title(f"Scale {start}→{end}", fontsize=12)

    # Row Labels
    current_row = 0
    for sample_idx in range(num_samples):
        for model_idx, model_name in enumerate(model_names):
            idx = current_row
            axs[idx, 0].set_ylabel(f"Sample {sample_idx}\n{model_name}", fontsize=12, rotation=0, labelpad=50, ha='right', va='center')
            current_row += 1
        if sample_idx != num_samples - 1:
            current_row += 1  # 빈줄 넘기기

    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    # 공통 컬러바
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(img, cax=cbar_ax)

    filename = os.path.join(output_dir, "spectrogram_difference.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved spectrogram difference maps to {filename}")

if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        encode()
