import torch
import os
import librosa
import soundfile
from torch.nn import functional as F
from snac import SNAC
from tqdm import tqdm
import time
from audiotools import AudioSignal

model = SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval().cuda()

sample_dir = 'samples/general/input'
files = {file for file in os.listdir(sample_dir) if file.endswith(".wav")}
progress_bar = tqdm(files, desc="Processing")
total_time = 0

for file in progress_bar:
    path = os.path.join(sample_dir, file)
    signal = AudioSignal(path, sample_rate=44000)
    original_length = signal.signal_length
    channels = signal.shape[-2]
    signal.resample(44000)
    input_db = signal.loudness()
    signal.normalize(-16)
    signal.ensure_max_of_audio()
    T = signal.shape[-1]
    
    with torch.inference_mode():
        start = time.time()
        codes = model.encode(signal.audio_data.cuda())
        end = time.time()
        total_time += end - start
        audio_hat = AudioSignal(model.decode(codes).cpu(), sample_rate=44000)
        audio_hat.normalize(input_db)
        audio_hat.resample(44000)
        audio_hat = audio_hat[..., : T]
        audio_hat.loudness()
        
        audio_hat.audio_data = audio_hat.audio_data.reshape(
            -1, channels, T
        )
        
    audio_hat.write(f'results/decode/snac_checkpoint/general/{file}')

print(f"Latency: {(total_time/len(files)):.4f}")