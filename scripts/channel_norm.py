import os
import librosa
import subprocess

def find_wav_files(directory):
    """
    특정 디렉토리에서 모든 .wav 파일 경로를 반환.
    """
    wav_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files

def check_channels(file_path):
    """
    Librosa를 사용하여 파일의 채널 수를 확인.
    """
    audio_data, _ = librosa.load(file_path, sr=None, mono=False)
    return audio_data.shape[0] if audio_data.ndim > 1 else 1

def downmix_to_stereo(input_path, output_path):
    """
    FFmpeg를 사용하여 입력 파일을 2채널로 다운믹스하여 지정된 경로에 저장.
    """
    command = [
        "ffmpeg",
        "-i", input_path,
        "-ac", "2",  # 2채널로 강제 변환
        output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # FFmpeg 출력 숨김

def process_directory(directory):
    """
    디렉토리를 탐색하며 6채널 파일을 2채널로 변환.
    변환된 파일은 동일 폴더에 저장되고 원본 파일은 삭제 후 새 파일로 교체.
    """
    wav_files = find_wav_files(directory)
    print(f"Found {len(wav_files)} .wav files in {directory}")

    for file in wav_files:
        channels = check_channels(file)
        if channels == 6:
            print(f"Processing 6-channel file: {file}")

            # 2채널 파일의 임시 이름 생성
            output_path = file.replace(".wav", "_down.wav")

            # 2채널로 변환하여 임시 파일에 저장
            downmix_to_stereo(file, output_path)

            # 원본 파일 삭제 및 임시 파일을 원본 이름으로 이동
            os.remove(file)
            os.rename(output_path, file)

            print(f"Converted and saved: {file}")
        else:
            print(f"Skipping non-6-channel file: {file}")

# Example usage
directory = "datasets/audioset/wav"  # 변환할 파일들이 있는 디렉토리
process_directory(directory)

