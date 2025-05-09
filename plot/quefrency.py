import os
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from tqdm import tqdm
import matplotlib.patches as patches

# 도메인 경로 및 파라미터
domain_dirs = {
    'Speech': 'samples/speech/input',
    'Music': 'samples/music/input',
    'Environment': 'samples/environment/input'
}
sample_rate = 44100
frame_length = 2048
hop_length = 256
bands = [(100, 300), (300, 600), (600, 1200), (1200, 2400), (2400, 4800)]
quefrency_range = (int(sample_rate * 0.002), int(sample_rate * 0.015))  # 2ms~15ms

def bandpass_filter(x, sr, low, high):
    b, a = butter(N=4, Wn=[low/(sr/2), high/(sr/2)], btype='band')
    return lfilter(b, a, x)

# 결과 저장
band_labels = [f"{low}-{high} Hz" for (low, high) in bands]
peak_quefrencies = {band: {domain: [] for domain in domain_dirs} for band in band_labels}

# 분석
for domain, dir_path in domain_dirs.items():
    print(f"Processing: {domain}")
    audio_files = glob.glob(os.path.join(dir_path, "*.wav"))
    for file_path in tqdm(audio_files):
        y, sr = librosa.load(file_path, sr=sample_rate)

        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        max_frame_idx = np.argmax(rms)
        center = max_frame_idx * hop_length
        frame = y[center : center + frame_length]
        if len(frame) < frame_length:
            continue

        for (low, high), band_name in zip(bands, band_labels):
            y_band = bandpass_filter(frame, sr, low, high)
            spectrum = np.fft.fft(y_band)
            log_spectrum = np.log(np.abs(spectrum) + 1e-10)
            cepstrum = np.fft.ifft(log_spectrum).real

            # 피크 위치 (2ms ~ 15ms 내)
            start, end = quefrency_range
            peak_idx = np.argmax(cepstrum[start:end]) + start
            quef_time = peak_idx / sample_rate
            peak_quefrencies[band_name][domain].append(quef_time)

# === Violin plot 생성 ===
output_dir = "plots_quefrency"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10, 12))
for i, band_name in enumerate(band_labels):
    plt.subplot(len(band_labels), 1, i+1)
    data = [peak_quefrencies[band_name][domain] for domain in domain_dirs]
    ax = plt.gca()

    # violin plot with Q1, Q3
    vp = plt.violinplot(
        data,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        quantiles=[[0.25, 0.75]] * len(data)
    )
    
    # Q1-Q3 style
    vp['cquantiles'].set_linestyle('solid')
    vp['cquantiles'].set_color('gray')
    vp['cquantiles'].set_linewidth(1.0)
    
    # Median
    medians = [np.median(d) for d in data]
    for j, med in enumerate(medians):
        ax.hlines(med, j + 0.8, j + 1.2, color='black', linewidth=2.0, label='Median' if i == 0 and j == 0 else None)

    # Mean
    means = [np.mean(d) for d in data]
    for j, mean in enumerate(means):
        ax.hlines(mean, j + 0.8, j + 1.2, color='tab:red', linestyle='--', linewidth=2.0, label='Mean' if i == 0 and j == 0 else None)

    # Axes
    plt.xticks(np.arange(1, len(domain_dirs)+1), domain_dirs.keys())
    ax.set_ylabel(band_name, rotation=270, labelpad=18, va='center')
    ax.yaxis.set_label_position("right")
    plt.yticks([0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015],
           ['2.5', '5', '7.5', '10', '12.5', '15'])
    plt.grid(True)


handles, labels = plt.gca().get_legend_handles_labels()
if handles:
    plt.figlegend(handles, labels, loc='upper right', fontsize=10)

plt.figtext(0.02, 0.5, 'Quefrency (ms)', va='center', rotation='vertical', fontsize=12)
plt.tight_layout(rect=[0.03, 0, 1, 0.96])
plt.savefig(os.path.join(output_dir, "violin_quefrency_all_bands.png"))
plt.close()
