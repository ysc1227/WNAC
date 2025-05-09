import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import glob
import os

# 각 도메인별 오디오 파일 경로 설정
domain_dirs = {
    'Speech': 'samples/speech/input',  # 예: DAPS, DNS
    'Music': 'samples/music/input',    # 예: MUSDB, Jamendo
    'Environment': 'samples/environment/input'  # 예: Audioset
}

# 파라미터 설정
sample_rate = 44100
n_fft = 1024
hop_length = 256

# 결과 저장용
domain_spectrograms = {}

# 도메인별 평균 스펙트로그램 계산
for domain, path in domain_dirs.items():
    file_list = glob.glob(os.path.join(path, '*.wav'))
    specs = []

    for file in file_list:
        y, _ = librosa.load(file, sr=sample_rate)
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        log_S = librosa.amplitude_to_db(S, ref=np.max)
        specs.append(log_S)

    # 평균 스펙트로그램 저장
    domain_spectrograms[domain] = np.mean(np.stack(specs), axis=0)

# Plot
plt.figure(figsize=(15, 5))

for idx, (domain, spec) in enumerate(domain_spectrograms.items()):
    plt.subplot(1, 3, idx + 1)
    librosa.display.specshow(spec, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.title(f'{domain}')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

plt.savefig('domain_spectra.png')