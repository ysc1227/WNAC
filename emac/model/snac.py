import math
from typing import List, Optional

from torch import nn

from .emac import EMAC


def _scale_factors_from_vq_strides(
    vq_strides: Optional[List[int]],
    scale_factors: Optional[List[float]],
) -> List[float]:
    if scale_factors:
        factors = [float(s) for s in scale_factors]
    else:
        if not vq_strides:
            raise ValueError("SNACCodec requires either vq_strides or scale_factors")
        factors = [1.0 / float(stride) for stride in vq_strides]

    for scale in factors:
        if not 0.0 < float(scale) <= 1.0:
            raise ValueError(f"SNACCodec scale factors must be in (0, 1], got {factors}")
    return factors


class SNACCodec(EMAC):
    """SNAC-44k style codec on the EMAC training/checkpoint stack.

    The defaults match the public SNAC 44 kHz architecture: a 441-sample hop
    from encoder rates [3, 3, 7, 7], four VQ stages with temporal strides
    [8, 4, 2, 1], 4096-entry codebooks, 8-dimensional codes, depthwise
    convolution, decoder noise blocks, and local attention.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 1,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [3, 3, 7, 7],
        latent_dim: int = 1024,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [7, 7, 3, 3],
        codebook_size: int = 4096,
        codebook_dim: int = 8,
        vq_strides: List[int] = [8, 4, 2, 1],
        scale_factors: Optional[List[float]] = None,
        quantizer_dropout: float = 0.0,
        quantizer_dropout_mode: str = "prefix",
        quantizer_pooling: str = "area",
        quantizer_pooling_alpha: float = 0.5,
        quantizer_pooling_power: float = 1.0,
        quantizer_upsample_mode: str = "repeat",
        quantizer_codebook_update: str = "gradient",
        noise: bool = True,
        depthwise: bool = True,
        attn_window_size: int = 32,
    ):
        if channels != 1:
            raise ValueError("SNACCodec currently supports mono audio only, matching the EMAC trainer.")

        normalized_scales = _scale_factors_from_vq_strides(
            vq_strides=vq_strides,
            scale_factors=scale_factors,
        )

        self.channels = channels
        self.vq_strides = vq_strides
        self.scale_factors = normalized_scales

        super().__init__(
            encoder_dim=encoder_dim,
            encoder_rates=encoder_rates,
            latent_dim=latent_dim,
            decoder_dim=decoder_dim,
            decoder_rates=decoder_rates,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
            quantizer_dropout_mode=quantizer_dropout_mode,
            sample_rate=sample_rate,
            scale_factor=normalized_scales,
            phi_kernel=None,
            noise=noise,
            attn_window_size=attn_window_size,
            depthwise=depthwise,
            use_wavescale=False,
            use_seanet=False,
            rvq_frame_size=None,
            quantizer_pooling=quantizer_pooling,
            quantizer_pooling_alpha=quantizer_pooling_alpha,
            quantizer_pooling_power=quantizer_pooling_power,
            quantizer_upsample_mode=quantizer_upsample_mode,
            quantizer_codebook_update=quantizer_codebook_update,
        )

        self.channels = channels
        self.vq_strides = vq_strides
        self.scale_factors = normalized_scales

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        first_stride = int(self.vq_strides[0]) if self.vq_strides else 1
        attn_window = int(self.attn_window_size or 1)
        pad_to = self.hop_length * math.lcm(first_stride, attn_window)
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / pad_to) * pad_to - length
        return nn.functional.pad(audio_data, (0, right_pad))
