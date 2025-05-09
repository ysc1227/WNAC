import os
import torch
from torcheval.metrics import FrechetAudioDistance
import librosa
from tqdm import tqdm

# Compute FAD using torcheval.metrics
def compute_fad_with_vggish(recon_dir, org_dir):
    """
    Compute FAD for matching .wav files in two directories using VGGish embeddings.
    """
    # Collect all .wav files from both directories
    recon_files = {file for file in os.listdir(recon_dir) if file.endswith(".wav")}
    org_files = {file for file in os.listdir(org_dir) if file.endswith(".wav")}

    matching_files = recon_files.intersection(org_files)
    fad_metric = FrechetAudioDistance.with_vggish(device='cuda').to('cuda')

    result = 0
    
    progress_bar = tqdm(matching_files, desc="Processing")
    
    for file in progress_bar:
        recon_path = os.path.join(recon_dir, file)
        org_path = os.path.join(org_dir, file)
        
        recon, _ = librosa.load(recon_path)
        recon = torch.tensor(recon).unsqueeze(0).cuda()
        org, _ = librosa.load(org_path)
        org = torch.tensor(org).unsqueeze(0).cuda()
        
        fad_metric.update(recon, org)
        result = fad_metric.compute()
        progress_bar.set_postfix(mean_FAD=result)

    return result

# Example usage
recon_dir = "results/decode/snac/environment"
org_dir = "samples/environment/input"

fad_scores = compute_fad_with_vggish(recon_dir, org_dir)
print(fad_scores)
