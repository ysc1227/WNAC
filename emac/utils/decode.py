import warnings
from pathlib import Path

import argbind
import torch
from tqdm import tqdm

from emac import EMACFile
from emac.utils import load_model

from audiotools.core import util

warnings.filterwarnings("ignore", category=UserWarning)


@argbind.bind(group="decode", positional=True, without_prefix=True)
@torch.inference_mode()
@torch.no_grad()
def decode(
    input: str,
    output: str = "",
    weights_path: str = "",
    model_tag: str = "latest",
    model_bitrate: str = "8kbps",
    device: str = "cuda",
    model_type: str = "44khz",
    verbose: bool = False,
    seed: int = 0,
    depth: str = 'full'
):
    """Decode audio from codes.

    Parameters
    ----------
    input : str
        Path to input directory or file
    output : str, optional
        Path to output directory, by default "".
        If `input` is a directory, the directory sub-tree relative to `input` is re-created in `output`.
    weights_path : str, optional
        Path to weights file, by default "". If not specified, the weights file will be downloaded from the internet using the
        model_tag and model_type.
    model_tag : str, optional
        Tag of the model to use, by default "latest". Ignored if `weights_path` is specified.
    model_bitrate: str
        Bitrate of the model. Must be one of "8kbps", or "16kbps". Defaults to "8kbps".
    device : str, optional
        Device to use, by default "cuda". If "cpu", the model will be loaded on the CPU.
    model_type : str, optional
        The type of model to use. Must be one of "44khz", "24khz", or "16khz". Defaults to "44khz". Ignored if `weights_path` is specified.
    """
    util.seed(seed)
    generator = load_model(
        model_type=model_type,
        model_bitrate=model_bitrate,
        tag=model_tag,
        load_path=weights_path,
    )
    generator.to(device)
    generator.eval()

    # Find all encoded EMAC files in input directory. Older scripts sometimes
    # referred to these as .wnac, but EMACFile.save() canonicalizes to .emac.
    _input = Path(input)
    input_files = sorted(_input.glob("**/*.emac")) if _input.is_dir() else []

    # If input is a single encoded file, add it to the list.
    if _input.suffix in {".emac", ".wnac"}:
        input_files.append(_input)

    if len(input_files) == 0:
        raise RuntimeError(f"No .emac files found in input path: {input}")

    # Create output directory
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    total_time = 0
    progress = tqdm(range(len(input_files)), desc=f"Decoding files")
    for i in progress:
        progress.set_postfix(file_path=input_files[i])
        # Load file
        artifact = EMACFile.load(input_files[i])

        # Encode audio to .wnac format
        import time
        start = time.time()
        # Reconstruct audio from codes
        recons = generator.decompress(artifact, verbose=verbose, depth=depth)
        end = time.time()
        total_time += end - start
        
        
        # Compute output path
        relative_path = input_files[i].relative_to(_input) if _input.is_dir() else Path(input_files[i].name)
        output_dir = output / relative_path.parent
        output_name = relative_path.with_suffix(".wav").name
        output_path = output_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to file
        recons.write(output_path)
    
    print(f"Ex: {total_time / len(input_files):.4f}")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        decode()