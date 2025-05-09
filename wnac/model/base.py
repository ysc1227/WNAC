import math
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
import tqdm
from audiotools import AudioSignal
from torch import nn

from wnac.nn.attention import LocalMHA
from wnac.nn.quantize import Phi

SUPPORTED_VERSIONS = ["1.0.0"]


@dataclass
class WNACFile:
    codes: Union[torch.Tensor, list[list[torch.Tensor]]]

    # Metadata
    chunk_length: Union[int, list]
    original_length: int
    input_db: float
    channels: int
    sample_rate: int
    padding: bool
    wnac_version: str

    def save(self, path):
        artifacts = {
            "codes": self.codes.numpy().astype(np.uint16) if isinstance(self.codes, torch.Tensor) else [[c.cpu().numpy().astype(np.uint16) for c in code] for code in self.codes],
            "metadata": {
                "input_db": self.input_db.numpy().astype(np.float32),
                "original_length": self.original_length,
                "sample_rate": self.sample_rate,
                "chunk_length": self.chunk_length if isinstance(self.chunk_length, int) else np.array(self.chunk_length),
                "channels": self.channels,
                "padding": self.padding,
                "wnac_version": SUPPORTED_VERSIONS[-1],
            },
        }
        path = Path(path).with_suffix(".wnac")
        with open(path, "wb") as f:
            np.save(f, artifacts)
        return path

    @classmethod
    def load(cls, path):
        artifacts = np.load(path, allow_pickle=True)[()]
        codes = artifacts["codes"]
        
        if isinstance(codes, np.ndarray):
            codes = torch.from_numpy(artifacts["codes"].astype(int))
        else:
            codes = [[torch.from_numpy(c.astype(int)) for c in code] for code in codes]
            
        if artifacts["metadata"].get("wnac_version", None) not in SUPPORTED_VERSIONS:
            raise RuntimeError(
                f"Given file {path} can't be loaded with this version of descript-audio-codec."
            )
        return cls(codes=codes, **artifacts["metadata"])


class CodecMixin:
    @property
    def padding(self):
        if not hasattr(self, "_padding"):
            self._padding = True
        return self._padding

    @padding.setter
    def padding(self, value):
        assert isinstance(value, bool)

        layers = []
        for l in self.encoder.modules():
            if isinstance(l, (nn.Conv1d, nn.ConvTranspose1d)):
                layers.append(l)
            if isinstance(l, LocalMHA):
                break
        
        for l in list(self.decoder.modules())[::-1]:
            if isinstance(l, (nn.Conv1d, nn.ConvTranspose1d)):
                layers.append(l)
            if isinstance(l, LocalMHA):
                break

        for layer in layers:
            if not isinstance(layer, Phi):
                if value:
                    if hasattr(layer, "original_padding"):
                        layer.padding = layer.original_padding
                else:
                    layer.original_padding = layer.padding
                    layer.padding = tuple(0 for _ in range(len(layer.padding)))

        self._padding = value

    def get_delay(self):
        # Any number works here, delay is invariant to input length
        l_out = self.get_output_length(0)
        L = l_out

        layers = []
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                layers.append(layer)

        for layer in reversed(layers):
            d = layer.dilation[0]
            k = layer.kernel_size[0]
            s = layer.stride[0]

            if isinstance(layer, nn.ConvTranspose1d):
                L = ((L - d * (k - 1) - 1) / s) + 1
            elif isinstance(layer, nn.Conv1d):
                L = (L - 1) * s + d * (k - 1) + 1

            L = math.ceil(L)

        l_in = L

        return (l_in - l_out) // 2

    def get_output_length(self, input_length):
        L = input_length
        # Calculate output length
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                d = layer.dilation[0]
                k = layer.kernel_size[0]
                s = layer.stride[0]
                p = layer.padding[0]

                if isinstance(layer, nn.Conv1d):
                    L = (((L + 2 * p) - d * (k - 1) - 1) / s) + 1
                elif isinstance(layer, nn.ConvTranspose1d):
                    L = ((L + 2 * p) - 1) * s + d * (k - 1) + 1

                L = math.floor(L)
        return L
    
    def get_encoded_length(self, input_length, formula):
        L = input_length
        
        for operation in formula:
            L = operation(L)
        return L
    
    def adjust_input_length(self, input_length, attn_size):
        formula = []
        for layer in self.encoder.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                d = layer.dilation[0]
                k = layer.kernel_size[0]
                s = layer.stride[0]
                p = layer.padding[0]

                if isinstance(layer, nn.Conv1d):
                    formula.append(lambda L, p=p, d=d, k=k, s=s: math.floor((((L + 2 * p) - d * (k - 1) - 1) / s) + 1))
                elif isinstance(layer, nn.ConvTranspose1d):
                    formula.append(lambda L, p=p, d=d, k=k, s=s: math.floor((((L + 2 * p) - 1) * s + d * (k - 1) + 1)))

        while True:
            encoded_length = self.get_encoded_length(input_length, formula)
            if encoded_length % attn_size == 0:
                return input_length
            input_length += 1

    @torch.no_grad()
    def compress(
        self,
        audio_path_or_signal: Union[str, Path, AudioSignal],
        win_duration: float = 1.0,
        verbose: bool = False,
        normalize_db: float = -16,
        n_quantizers: int = None,
        get_latent: bool = False
    ) -> WNACFile:
        """Processes an audio signal from a file or AudioSignal object into
        discrete codes. This function processes the signal in short windows,
        using constant GPU memory.

        Parameters
        ----------
        audio_path_or_signal : Union[str, Path, AudioSignal]
            audio signal to reconstruct
        win_duration : float, optional
            window duration in seconds, by default 5.0
        verbose : bool, optional
            by default False
        normalize_db : float, optional
            normalize db, by default -16

        Returns
        -------
        WNACFile
            Object containing compressed codes and metadata
            required for decompression
        """
        audio_signal = audio_path_or_signal
        if isinstance(audio_signal, (str, Path)):
            audio_signal = AudioSignal.load_from_file_with_ffmpeg(str(audio_signal))
        
        self.eval()
        original_padding = self.padding
        original_device = audio_signal.device

        audio_signal = audio_signal.clone()
        original_sr = audio_signal.sample_rate

        resample_fn = audio_signal.resample
        loudness_fn = audio_signal.loudness

        # If audio is > 10 minutes long, use the ffmpeg versions
        if audio_signal.signal_duration >= 10 * 60 * 60:
            resample_fn = audio_signal.ffmpeg_resample
            loudness_fn = audio_signal.ffmpeg_loudness

        original_length = audio_signal.signal_length
        resample_fn(self.sample_rate)
        input_db = loudness_fn()

        if normalize_db is not None:
            audio_signal.normalize(normalize_db)
        audio_signal.ensure_max_of_audio()
        
        nb, nac, nt = audio_signal.audio_data.shape
        audio_signal.audio_data = audio_signal.audio_data.reshape(nb * nac, 1, nt)
        win_duration = (
            audio_signal.signal_duration if win_duration is None else win_duration
        )
        
        if audio_signal.signal_duration <= win_duration:
            # Unchunked compression (used if signal length < win duration)
            self.padding = True
            n_samples = nt
            hop = nt
        else:
            # Chunked inference
            self.padding = False
            # Zero-pad signal on either side by the delay
            audio_signal.zero_pad(self.delay, self.delay)
            n_samples = int(win_duration * self.sample_rate)
            # Round n_samples to nearest hop length multiple
            n_samples = int(math.ceil(n_samples / self.hop_length) * self.hop_length)
            n_samples = self.adjust_input_length(n_samples, self.attn_window_size) if self.attn_window_size is not None else n_samples
            hop = self.get_output_length(n_samples)
        codes = []
        chunk_length = []
        range_fn = range if not verbose else tqdm.trange
        
        for i in range_fn(0, nt, hop):
            x = audio_signal[..., i : i + n_samples]
            x = x.zero_pad(0, max(0, n_samples - x.shape[-1]))

            audio_data = x.audio_data.to(self.device)
            audio_data = self.preprocess(audio_data, self.sample_rate)
            _, c, l, _, _, _ = self.encode(audio_data, n_quantizers)

            if isinstance(c, torch.Tensor):
                codes.append(c.to(original_device))
                chunk_length = c.shape[-1]
            else:
                codes.append(c)
                chunk_length = [cc.shape[-1] for cc in c]
        
        if isinstance(chunk_length, int):
            codes = torch.cat(codes, dim=-1)

        wnac_file = WNACFile(
            codes=codes,
            chunk_length=chunk_length,
            original_length=original_length,
            input_db=input_db,
            channels=nac,
            sample_rate=original_sr,
            padding=self.padding,
            wnac_version=SUPPORTED_VERSIONS[-1],
        )

        if n_quantizers is not None:
            if isinstance(codes, torch.Tensor):
                codes = codes[:, :n_quantizers, :]
            else:
                codes = [code[:n_quantizers] for code in codes]

        self.padding = original_padding
        if get_latent:
            return l        
        return wnac_file

    @torch.no_grad()
    def decompress(
        self,
        obj: Union[str, Path, WNACFile],
        verbose: bool = False,
    ) -> AudioSignal:
        """Reconstruct audio from a given .wnac file

        Parameters
        ----------
        obj : Union[str, Path, WNACFile]
            .wnac file location or corresponding WNACFile object.
        verbose : bool, optional
            Prints progress if True, by default False

        Returns
        -------
        AudioSignal
            Object with the reconstructed audio
        """
        self.eval()        
        if isinstance(obj, (str, Path)):
            obj = WNACFile.load(obj)

        original_padding = self.padding
        self.padding = obj.padding

        range_fn = range if not verbose else tqdm.trange
        codes = obj.codes
        
        original_device = codes.device if isinstance(codes, torch.Tensor) else codes[0][0].device
        chunk_length = obj.chunk_length
        recons = []

        if isinstance(codes, torch.Tensor):
            for i in range_fn(0, codes.shape[-1], chunk_length):
                c = codes[..., i : i + chunk_length].to(self.device)
                z = self.quantizer.from_codes(c)[0]
                r = self.decode(z)
                recons.append(r.to(original_device))
        else:
            for code in codes:
                c = [co.to(self.device) for co in code]
                z = self.quantizer.from_codes(c)[0]                
                r = self.decode(z)
                recons.append(r.to(original_device))

        recons = torch.cat(recons, dim=-1)
        recons = AudioSignal(recons, self.sample_rate)

        resample_fn = recons.resample
        loudness_fn = recons.loudness

        # If audio is > 10 minutes long, use the ffmpeg versions
        if recons.signal_duration >= 10 * 60 * 60:
            resample_fn = recons.ffmpeg_resample
            loudness_fn = recons.ffmpeg_loudness

        recons.normalize(obj.input_db)
        resample_fn(obj.sample_rate)
        recons = recons[..., : obj.original_length]
        loudness_fn()

        recons.audio_data = recons.audio_data.reshape(
            -1, obj.channels, obj.original_length
        )

        self.padding = original_padding
        return recons