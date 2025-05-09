import argbind
import audiotools as at
import numpy as np
import torch
import tqdm

import wnac


@argbind.bind(without_prefix=True, positional=True)
def main(
    folder: str,
    model_path: str,
    n_samples: int = 1024,
    device: str = "cuda",
):
    files = at.util.find_audio(folder)[:n_samples]
    signals = [
        at.AudioSignal.salient_excerpt(f, loudness_cutoff=-20, duration=1.0)
        for f in tqdm.tqdm(files)
    ]

    with torch.no_grad():
        model = wnac.model.WNAC.load(model_path).to(device)
        model.eval()

        all_codes = []
        for x in tqdm.tqdm(signals):
            x = x.to(model.device)
            x = model.preprocess(x.audio_data.mean(dim=1, keepdim=True), sample_rate=None)
            _, codes, _, _, _, _ = model.encode(x)
            all_codes.append([code.cpu() for code in codes])
        
        entropy = []

        
        for codes in all_codes:
            if len(codes) == 1:
                counts = [torch.bincount(code) for code in codes[0]]
            else:
                counts = [torch.bincount(code.squeeze(0)) for code in codes]
            max_length = max(tensor.size(0) for tensor in counts)
            counts = [
                torch.cat([tensor, torch.zeros(max_length - tensor.size(0))]) for tensor in counts
            ]
            counts = torch.stack(counts).sum(dim=0)
            counts = (counts / counts.sum()).clamp(1e-10)
            entropy.append(-(counts * counts.log()).sum().item() * np.log2(np.e))

        pct = sum(entropy) / (10 * len(entropy))
        print(f"Entropy for each codebook: {entropy}")
        print(f"Effective percentage: {pct * 100}%")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        main()