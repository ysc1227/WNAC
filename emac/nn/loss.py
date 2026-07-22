import typing
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from audiotools.core.audio_signal import AudioSignal, STFTParams


class L1Loss(nn.L1Loss):
    """L1 Loss between AudioSignals. Defaults
    to comparing ``audio_data``, but any
    attribute of an AudioSignal can be used.

    Parameters
    ----------
    attribute : str, optional
        Attribute of signal to compare, defaults to ``audio_data``.
    weight : float, optional
        Weight of this loss, defaults to 1.0.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/distance.py
    """

    def __init__(self, attribute: str = "audio_data", weight: float = 1.0, **kwargs):
        self.attribute = attribute
        self.weight = weight
        super().__init__(**kwargs)

    def forward(self, x: AudioSignal, y: AudioSignal):
        """
        Parameters
        ----------
        x : AudioSignal
            Estimate AudioSignal
        y : AudioSignal
            Reference AudioSignal

        Returns
        -------
        torch.Tensor
            L1 loss between AudioSignal attributes.
        """
        if isinstance(x, AudioSignal):
            x = getattr(x, self.attribute)
            y = getattr(y, self.attribute)
        return super().forward(x, y)


class SISDRLoss(nn.Module):
    """
    Computes the Scale-Invariant Source-to-Distortion Ratio between a batch
    of estimated and reference audio signals or aligned features.

    Parameters
    ----------
    scaling : int, optional
        Whether to use scale-invariant (True) or
        signal-to-noise ratio (False), by default True
    reduction : str, optional
        How to reduce across the batch (either 'mean',
        'sum', or none).], by default ' mean'
    zero_mean : int, optional
        Zero mean the references and estimates before
        computing the loss, by default True
    clip_min : int, optional
        The minimum possible loss value. Helps network
        to not focus on making already good examples better, by default None
    weight : float, optional
        Weight of this loss, defaults to 1.0.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/distance.py
    """

    def __init__(
        self,
        scaling: int = True,
        reduction: str = "mean",
        zero_mean: int = True,
        clip_min: int = None,
        weight: float = 1.0,
    ):
        self.scaling = scaling
        self.reduction = reduction
        self.zero_mean = zero_mean
        self.clip_min = clip_min
        self.weight = weight
        super().__init__()

    def forward(self, x: AudioSignal, y: AudioSignal):
        eps = 1e-8
        # nb, nc, nt
        if isinstance(x, AudioSignal):
            references = x.audio_data
            estimates = y.audio_data
        else:
            references = x
            estimates = y

        nb = references.shape[0]
        references = references.reshape(nb, 1, -1).permute(0, 2, 1)
        estimates = estimates.reshape(nb, 1, -1).permute(0, 2, 1)

        # samples now on axis 1
        if self.zero_mean:
            mean_reference = references.mean(dim=1, keepdim=True)
            mean_estimate = estimates.mean(dim=1, keepdim=True)
        else:
            mean_reference = 0
            mean_estimate = 0

        _references = references - mean_reference
        _estimates = estimates - mean_estimate

        references_projection = (_references**2).sum(dim=-2) + eps
        references_on_estimates = (_estimates * _references).sum(dim=-2) + eps

        scale = (
            (references_on_estimates / references_projection).unsqueeze(1)
            if self.scaling
            else 1
        )

        e_true = scale * _references
        e_res = _estimates - e_true

        signal = (e_true**2).sum(dim=1)
        noise = (e_res**2).sum(dim=1)
        sdr = -10 * torch.log10(signal / noise + eps)

        if self.clip_min is not None:
            sdr = torch.clamp(sdr, min=self.clip_min)

        if self.reduction == "mean":
            sdr = sdr.mean()
        elif self.reduction == "sum":
            sdr = sdr.sum()
        return sdr


class MultiScaleSTFTLoss(nn.Module):
    """Computes the multi-scale STFT loss from [1].

    Parameters
    ----------
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    References
    ----------

    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        window_type: str = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal):
        """Computes multi-scale STFT between an estimate and a reference
        signal.

        Parameters
        ----------
        x : AudioSignal
            Estimate signal
        y : AudioSignal
            Reference signal

        Returns
        -------
        torch.Tensor
            Multi-scale STFT loss.
        """
        loss = 0.0
        for s in self.stft_params:
            x.stft(s.window_length, s.hop_length, s.window_type)
            y.stft(s.window_length, s.hop_length, s.window_type)
            
            # # 주파수 축 계산
            # freqs = librosa.fft_frequencies(sr=x.sample_rate, n_fft=s.window_length)
            # high_freq_index = np.where(freqs >= 15000)[0][0]  # 15kHz 이상 제거

            # # 15kHz 이상을 제외한 magnitude
            # x_mag_filtered = x.magnitude[:high_freq_index, :]
            # y_mag_filtered = y.magnitude[:high_freq_index, :]
            
            x_mag_filtered = x.magnitude
            y_mag_filtered = y.magnitude
        
            loss += self.log_weight * self.loss_fn(
                x_mag_filtered.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mag_filtered.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mag_filtered, y_mag_filtered)
        return loss


class MelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [150, 80],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        n_mels: List[int] = [150, 80],
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        mel_fmin: List[float] = [0.0, 0.0],
        mel_fmax: List[float] = [None, None],
        window_type: str = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal):
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : AudioSignal
            Estimate signal
        y : AudioSignal
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """
        loss = 0.0
        for n_mels, fmin, fmax, s in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            kwargs = {
                "window_length": s.window_length,
                "hop_length": s.hop_length,
                "window_type": s.window_type,
            }
            x_mels = x.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)
            y_mels = y.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)

            loss += self.log_weight * self.loss_fn(
                x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
        return loss


class SATFrequencyLoss(nn.Module):
    """Frequency loss used by the original SAT training code.

    SAT's ``l_f`` sums L1 and L2 losses over log-mel spectrograms computed at
    FFT sizes 2**5 through 2**11.  The original implementation pads by
    ``(n_fft - hop_length) // 2`` and then runs an STFT with ``center=False``.
    """

    def __init__(
        self,
        window_exponents: List[int] = [5, 6, 7, 8, 9, 10, 11],
        n_mel_channels: int = 64,
        sample_rate: int = 24000,
        mel_fmin: float = 0.0,
        mel_fmax: float = None,
        clamp_eps: float = 1e-5,
    ):
        super().__init__()
        self.window_exponents = [int(i) for i in window_exponents]
        self.n_mel_channels = int(n_mel_channels)
        self.sample_rate = int(sample_rate)
        self.mel_fmin = float(mel_fmin)
        self.mel_fmax = mel_fmax
        self.clamp_eps = float(clamp_eps)
        self.l1_loss = nn.L1Loss(reduction="mean")
        self.l2_loss = nn.MSELoss(reduction="mean")

        for exponent in self.window_exponents:
            n_fft = 2 ** int(exponent)
            mel_basis = self._build_mel_basis(
                sample_rate=self.sample_rate,
                n_fft=n_fft,
                n_mels=self.n_mel_channels,
                fmin=self.mel_fmin,
                fmax=self.mel_fmax,
            )
            self.register_buffer(
                f"mel_basis_{n_fft}",
                mel_basis.float(),
                persistent=False,
            )
            self.register_buffer(
                f"window_{n_fft}",
                torch.hann_window(n_fft),
                persistent=False,
            )

    @staticmethod
    def _hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
        frequencies = np.asanyarray(frequencies, dtype=np.float64)
        f_sp = 200.0 / 3
        mels = frequencies / f_sp
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = np.log(6.4) / 27.0
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
        return mels

    @staticmethod
    def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
        mels = np.asanyarray(mels, dtype=np.float64)
        f_sp = 200.0 / 3
        frequencies = f_sp * mels
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = np.log(6.4) / 27.0
        log_t = mels >= min_log_mel
        frequencies[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
        return frequencies

    @classmethod
    def _build_mel_basis(
        cls,
        sample_rate: int,
        n_fft: int,
        n_mels: int,
        fmin: float = 0.0,
        fmax: float = None,
    ) -> torch.Tensor:
        fmax = float(sample_rate) / 2 if fmax is None else float(fmax)
        fft_freqs = np.linspace(0.0, float(sample_rate) / 2, 1 + n_fft // 2)
        min_mel = cls._hz_to_mel(np.array([float(fmin)]))[0]
        max_mel = cls._hz_to_mel(np.array([fmax]))[0]
        mel_f = cls._mel_to_hz(np.linspace(min_mel, max_mel, int(n_mels) + 2))

        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fft_freqs)
        weights = np.zeros((int(n_mels), int(1 + n_fft // 2)), dtype=np.float32)
        for i in range(int(n_mels)):
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]
            weights[i] = np.maximum(0.0, np.minimum(lower, upper))

        enorm = 2.0 / (mel_f[2 : int(n_mels) + 2] - mel_f[: int(n_mels)])
        weights *= enorm[:, np.newaxis]
        return torch.from_numpy(weights)

    @staticmethod
    def _audio_tensor(x):
        if isinstance(x, AudioSignal):
            return x.audio_data
        return x

    @staticmethod
    def _reflect_pad(audio: torch.Tensor, pad: int) -> torch.Tensor:
        if pad <= 0:
            return audio
        if audio.shape[-1] > pad:
            return F.pad(audio, (pad, pad), mode="reflect")
        return F.pad(audio, (pad, pad), mode="replicate")

    def _log_mel(self, audio: torch.Tensor, n_fft: int) -> torch.Tensor:
        hop_length = n_fft // 4
        pad = (n_fft - hop_length) // 2
        audio = self._reflect_pad(audio, pad).squeeze(1)
        window = getattr(self, f"window_{n_fft}").to(device=audio.device, dtype=audio.dtype)
        mel_basis = getattr(self, f"mel_basis_{n_fft}").to(device=audio.device, dtype=audio.dtype)
        stft = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=False,
            return_complex=True,
        )
        power = stft.real.pow(2) + stft.imag.pow(2)
        mel = torch.matmul(mel_basis, power)
        return torch.log10(mel.clamp_min(self.clamp_eps))

    def forward(self, x: AudioSignal, y: AudioSignal):
        x_audio = self._audio_tensor(x)
        y_audio = self._audio_tensor(y)
        loss = x_audio.new_tensor(0.0)
        for exponent in self.window_exponents:
            n_fft = 2 ** int(exponent)
            x_mel = self._log_mel(x_audio, n_fft)
            y_mel = self._log_mel(y_audio, n_fft)
            loss = loss + self.l1_loss(x_mel, y_mel) + self.l2_loss(x_mel, y_mel)
        return loss


class GANLoss(nn.Module):
    """
    Computes a discriminator loss, given a discriminator on
    generated waveforms/spectrograms compared to ground truth
    waveforms/spectrograms. Computes the loss for both the
    discriminator and the generator in separate functions.
    """

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, fake, real):
        d_fake = self.discriminator(fake.audio_data)
        d_real = self.discriminator(real.audio_data)
        return d_fake, d_real

    def discriminator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake.clone().detach(), real)
        
        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2)
            loss_d += torch.mean((1 - x_real[-1]) ** 2)
        return loss_d

    def generator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake, real)
        loss_g = 0
        for x_fake in d_fake:
            loss_g += torch.mean((1 - x_fake[-1]) ** 2)

        loss_feature = 0

        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())
        return loss_g, loss_feature


class SATGANLoss(nn.Module):
    """Hinge GAN and relative feature matching losses used by SAT/AAR."""

    def __init__(self, discriminator, eps: float = 1e-8):
        super().__init__()
        self.discriminator = discriminator
        self.eps = float(eps)

    @staticmethod
    def _audio(x):
        return x.audio_data if isinstance(x, AudioSignal) else x

    def forward(self, fake, real):
        logits_fake, fmap_fake = self.discriminator(self._audio(fake))
        logits_real, fmap_real = self.discriminator(self._audio(real))
        return logits_fake, fmap_fake, logits_real, fmap_real

    def discriminator_loss(self, fake, real):
        fake_audio = self._audio(fake).detach().contiguous()
        real_audio = self._audio(real)
        logits_fake, _ = self.discriminator(fake_audio)
        logits_real, _ = self.discriminator(real_audio)

        loss = real_audio.new_tensor(0.0)
        for real_logit, fake_logit in zip(logits_real, logits_fake):
            loss = loss + torch.mean(F.relu(1 - real_logit))
            loss = loss + torch.mean(F.relu(1 + fake_logit))
        return loss / max(1, len(logits_real))

    def generator_loss(self, fake, real):
        fake_audio = self._audio(fake)
        real_audio = self._audio(real)

        # During the generator update the discriminator is only a fixed
        # perceptual map. Running the real branch first under no_grad avoids
        # legacy weight_norm version bumps between fake forward and backward.
        requires_grad = [p.requires_grad for p in self.discriminator.parameters()]
        for p in self.discriminator.parameters():
            p.requires_grad_(False)
        try:
            with torch.no_grad():
                logits_real, fmap_real = self.discriminator(real_audio)
            logits_fake, fmap_fake = self.discriminator(fake_audio)
        finally:
            for p, value in zip(self.discriminator.parameters(), requires_grad):
                p.requires_grad_(value)

        loss_g = fake_audio.new_tensor(0.0)
        for fake_logit in logits_fake:
            loss_g = loss_g + torch.mean(F.relu(1 - fake_logit)) / max(1, len(logits_fake))
        loss_g = loss_g / max(1, len(fmap_real))

        loss_feature = fake_audio.new_tensor(0.0)
        feature_count = 0
        for real_maps, fake_maps in zip(fmap_real, fmap_fake):
            for real_feature, fake_feature in zip(real_maps, fake_maps):
                denom = torch.mean(torch.abs(real_feature)).clamp_min(self.eps)
                loss_feature = loss_feature + F.l1_loss(real_feature, fake_feature) / denom
                feature_count += 1
        if feature_count > 0:
            loss_feature = loss_feature / feature_count
        return loss_g, loss_feature
    
    
class CELoss(nn.CrossEntropyLoss):
    def __init__(self, reduction='none', **kwargs):
        super().__init__(reduction=reduction, **kwargs)

    def forward(self, logits: torch.Tensor, codes: torch.Tensor):
        b, l, v = logits.size()
        logits = logits.view(-1, v)
        
        labels = torch.cat(codes, dim=1)
        labels = labels.view(-1)
        
        loss = super().forward(logits, labels).view(b, -1)

        return loss
