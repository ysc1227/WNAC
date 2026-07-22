import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from torch.nn.utils import weight_norm

from audiotools import ml
from audiotools.core.audio_signal import AudioSignal, STFTParams


def WNConv1d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv1d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


def WNConv2d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


class MPD(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                WNConv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
            ]
        )
        self.conv_post = WNConv2d(
            1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False
        )

    def pad_to_period(self, x):
        t = x.shape[-1]
        x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        return x

    def forward(self, x):
        fmap = []

        x = self.pad_to_period(x)
        x = rearrange(x, "b c (l p) -> b c l p", p=self.period)

        for layer in self.convs:
            x = layer(x)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return fmap


class MSD(nn.Module):
    def __init__(self, rate: int = 1, sample_rate: int = 44100):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                WNConv1d(1, 16, 15, 1, padding=7),
                WNConv1d(16, 64, 41, 4, groups=4, padding=20),
                WNConv1d(64, 256, 41, 4, groups=16, padding=20),
                WNConv1d(256, 1024, 41, 4, groups=64, padding=20),
                WNConv1d(1024, 1024, 41, 4, groups=256, padding=20),
                WNConv1d(1024, 1024, 5, 1, padding=2),
            ]
        )
        self.conv_post = WNConv1d(1024, 1, 3, 1, padding=1, act=False)
        self.sample_rate = sample_rate
        self.rate = rate

    def forward(self, x):
        x = AudioSignal(x, self.sample_rate)
        x.resample(self.sample_rate // self.rate)
        x = x.audio_data

        fmap = []

        for l in self.convs:
            x = l(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap


BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


class MRD(nn.Module):
    def __init__(
        self,
        window_length: int,
        hop_factor: float = 0.25,
        sample_rate: int = 44100,
        bands: list = BANDS,
    ):
        """Complex multi-band spectrogram discriminator.
        Parameters
        ----------
        window_length : int
            Window length of STFT.
        hop_factor : float, optional
            Hop factor of the STFT, defaults to ``0.25 * window_length``.
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run discriminator over.
        """
        super().__init__()

        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate
        self.stft_params = STFTParams(
            window_length=window_length,
            hop_length=int(window_length * hop_factor),
            match_stride=True
        )

        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands

        ch = 32
        convs = lambda: nn.ModuleList(
            [
                WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def spectrogram(self, x):
        x = AudioSignal(x, self.sample_rate, stft_params=self.stft_params)
        x = torch.view_as_real(x.stft())
        x = rearrange(x, "b 1 f t c -> (b 1) c t f")
        # Split into bands
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        
        return x_bands

    def forward(self, x):
        x_bands = self.spectrogram(x)
        fmap = []

        x_list = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
                
            x_list.append(band)    
        
        x = torch.cat(x_list, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap


class Discriminator(ml.BaseModel):
    def __init__(
        self,
        rates: list = [],
        periods: list = [2, 3, 5, 7, 11],
        fft_sizes: list = [2048, 1024, 512],
        sample_rate: int = 44100,
        bands: list = BANDS,
    ):
        """Discriminator that combines multiple discriminators.

        Parameters
        ----------
        rates : list, optional
            sampling rates (in Hz) to run MSD at, by default []
            If empty, MSD is not used.
        periods : list, optional
            periods (of samples) to run MPD at, by default [2, 3, 5, 7, 11]
        fft_sizes : list, optional
            Window sizes of the FFT to run MRD at, by default [2048, 1024, 512]
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run MRD at, by default `BANDS`
        """
        super().__init__()
        discs = []
        # discs += [MPD(p) for p in periods]
        # discs += [MSD(r, sample_rate=sample_rate) for r in rates]
        discs += [MRD(f, sample_rate=sample_rate, bands=bands) for f in fft_sizes]
        self.discriminators = nn.ModuleList(discs)

    def preprocess(self, y):
        # Remove DC offset
        y = y - y.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        y = 0.8 * y / (y.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return y

    def forward(self, x):
        x = self.preprocess(x)
        fmaps = [d(x) for d in self.discriminators]
        return fmaps


def _get_2d_padding(kernel_size, dilation=(1, 1)):
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )


def _SATNormConv2d(*args, norm: str = "weight_norm", **kwargs):
    conv = nn.Conv2d(*args, **kwargs)
    if norm == "weight_norm":
        conv = weight_norm(conv)
    elif norm not in {"none", None}:
        raise ValueError(f"Unsupported SAT discriminator norm='{norm}'")
    return conv


class SATSTFTDiscriminator(nn.Module):
    """STFT sub-discriminator matching the public SAT/AAR training recipe."""

    def __init__(
        self,
        filters: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        max_filters: int = 1024,
        filters_scale: int = 1,
        kernel_size: tuple = (3, 9),
        dilations: list = [1, 2, 4],
        stride: tuple = (1, 2),
        normalized: bool = True,
        norm: str = "weight_norm",
        activation: str = "LeakyReLU",
        activation_params: dict = {"negative_slope": 0.2},
    ):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.normalized = bool(normalized)
        self.activation = getattr(nn, activation)(**activation_params)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

        spec_channels = 2 * int(in_channels)
        self.convs = nn.ModuleList()
        self.convs.append(
            _SATNormConv2d(
                spec_channels,
                filters,
                kernel_size=kernel_size,
                padding=_get_2d_padding(kernel_size),
                norm=norm,
            )
        )
        in_chs = min(filters_scale * filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * filters, max_filters)
            self.convs.append(
                _SATNormConv2d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=(dilation, 1),
                    padding=_get_2d_padding(kernel_size, (dilation, 1)),
                    norm=norm,
                )
            )
            in_chs = out_chs
        out_chs = min((filters_scale ** (len(dilations) + 1)) * filters, max_filters)
        square_kernel = (kernel_size[0], kernel_size[0])
        self.convs.append(
            _SATNormConv2d(
                in_chs,
                out_chs,
                kernel_size=square_kernel,
                padding=_get_2d_padding(square_kernel),
                norm=norm,
            )
        )
        self.conv_post = _SATNormConv2d(
            out_chs,
            out_channels,
            kernel_size=square_kernel,
            padding=_get_2d_padding(square_kernel),
            norm=norm,
        )

    def forward(self, x: torch.Tensor):
        audio = x.squeeze(1)
        window = self.window.to(device=audio.device, dtype=audio.dtype)
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=False,
            return_complex=True,
        )
        if self.normalized:
            norm_factor = torch.sqrt(torch.tensor(self.n_fft / 2, device=audio.device, dtype=audio.dtype))
            spec = spec / norm_factor
        z = torch.stack([spec.real.transpose(1, 2), spec.imag.transpose(1, 2)], dim=1)

        fmap = []
        for layer in self.convs:
            z = self.activation(layer(z))
            fmap.append(z)
        z = self.conv_post(z)
        return z, fmap


class SATDiscriminator(ml.BaseModel):
    """Multi-scale STFT discriminator from the public SAT/AAR training code."""

    def __init__(
        self,
        filters: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
        n_ffts: list = [1024, 2048, 512, 256, 128],
        hop_lengths: list = [256, 512, 128, 64, 32],
        win_lengths: list = [1024, 2048, 512, 256, 128],
        **kwargs,
    ):
        super().__init__()
        if not (len(n_ffts) == len(hop_lengths) == len(win_lengths)):
            raise ValueError("n_ffts, hop_lengths, and win_lengths must have the same length")
        self.filters = filters
        self.n_ffts = n_ffts
        self.hop_lengths = hop_lengths
        self.win_lengths = win_lengths
        self.discriminators = nn.ModuleList(
            [
                SATSTFTDiscriminator(
                    filters,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_fft=n_ffts[i],
                    hop_length=hop_lengths[i],
                    win_length=win_lengths[i],
                    **kwargs,
                )
                for i in range(len(n_ffts))
            ]
        )

    def forward(self, x: torch.Tensor):
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps


if __name__ == "__main__":
    disc = Discriminator()
    x = torch.zeros(1, 1, 44100)
    results = disc(x)
    for i, result in enumerate(results):
        print(f"disc{i}")
        for i, r in enumerate(result):
            print(r.shape, r.mean(), r.min(), r.max())
        print()
