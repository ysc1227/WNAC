import math
from typing import List, Optional, Union

import numpy as np
import torch
from audiotools.ml.layers.base import BaseModel
from torch import nn

from emac.nn.sat_quantize import SATOriginalResidualVectorQuantizer
from emac.nn.seanet import SEANetDecoder, SEANetEncoder

from .base import CodecMixin


def _sat_ordered_scales(
    multi_scale: Optional[List[int]],
    scale_order: str,
) -> List[int]:
    if not multi_scale:
        raise ValueError("SATCodec.multi_scale must be a non-empty SAT scale list")
    scales = [int(s) for s in multi_scale]
    order = str(scale_order).lower().replace("-", "_")
    if order in {"upscale", "coarse_to_fine", "sat"}:
        return scales
    if order in {"downscale", "fine_to_coarse"}:
        return list(reversed(scales))
    raise ValueError(
        f"Unknown SATCodec.scale_order='{scale_order}'. "
        "Expected upscale/coarse_to_fine/sat or downscale/fine_to_coarse."
    )


class SATCodec(BaseModel, CodecMixin):
    """Native SAT codec with the original public SAT quantizer flow.

    This class intentionally does not inherit from ``EMAC``.  It follows the
    public SAT implementation:

    ``SEANetEncoder -> ResidualVectorQuantizer -> SEANetDecoder``

    The surrounding return dictionary is adapted only so the existing EMAC
    trainer, checkpointing, and evaluation scripts can consume it.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        ratios: List[int] = [8, 5, 4, 2],
        dimension: int = 1024,
        codebook_size: int = 1024,
        codebook_dim: int = 64,
        multi_scale: List[int] = [1, 2, 4, 6, 9, 12, 16, 20, 25, 31, 37, 43, 50, 58, 66, 75],
        reference_frames: Optional[int] = None,
        scale_factors: Optional[List[float]] = None,
        scale_order: str = "upscale",
        use_wavescale: bool = False,
        phi_kernel: Optional[List[int]] = [9, 9, 9, 9, 9, 9],
        model_norm: str = "weight_norm",
        causal: bool = False,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        lstm: int = 2,
        ema_decay: float = 0.99,
        ema_kmeans_init: bool = True,
        ema_kmeans_iters: int = 50,
        ema_threshold_ema_dead_code: int = 2,
        ema_epsilon: float = 1e-5,
        # Kept only for old config/parser compatibility. They are not used by
        # the original SAT quantizer path.
        quantizer_dropout: float = 0.0,
        quantizer_dropout_mode: str = "prefix",
        quantizer_pooling: str = "area",
        quantizer_pooling_alpha: float = 0.5,
        quantizer_pooling_power: float = 1.0,
        quantizer_upsample_mode: str = "linear",
        quantizer_codebook_update: str = "ema",
        quantizer_decoder_grad_alpha: float = 0.0,
        quantizer_residual_detach: bool = True,
        quantizer_shared_projection: bool = False,
        quantizer_separate_lookup_codebook: bool = False,
        quantizer_lookup_commitment_weight: float = 1.0,
        quantizer_lookup_codebook_weight: float = 1.0,
        quantizer_scale_anneal: bool = False,
        quantizer_scale_anneal_steps: int = 30000,
        quantizer_scale_anneal_start: float = 1.0,
        quantizer_scale_anneal_mode: str = "cosine",
        learned_downsample: bool = False,
        learned_downsample_hidden_dim: int = None,
        learned_downsample_kernel: int = 3,
        learned_downsample_init_scale: float = 1e-3,
        learned_downsample_max_scale: Union[float, list] = 0.2,
        guided_upsample: bool = False,
        guided_upsample_hidden_dim: int = None,
        guided_upsample_kernel: int = 5,
        guided_upsample_detach_guide: bool = True,
        guided_upsample_init_scale: float = 1e-3,
        guided_upsample_after_pivot_only: bool = False,
        guided_upsample_decoder_grad_alpha: float = 0.0,
        learned_upsample: bool = False,
        learned_upsample_mode: str = "patch",
        learned_upsample_hidden_dim: int = None,
        learned_upsample_kernel: int = 3,
        learned_upsample_patch_size: Union[int, list] = 4,
        learned_upsample_init_scale: float = 1e-3,
        learned_upsample_condition_guide: bool = False,
        learned_upsample_detach_guide: bool = True,
        learned_upsample_guide_space: str = "latent_proj",
        learned_upsample_max_scale: Union[float, list] = 1.0,
        learned_upsample_after_pivot_only: bool = False,
        learned_upsample_log_stages: bool = False,
    ):
        super().__init__()
        if channels != 1:
            raise ValueError("SATCodec follows the public SAT mono-audio setup.")
        if use_wavescale:
            raise ValueError("SATCodec is the original SAT coarse-to-fine codec; use EMAC for wavescale.")
        # Older checkpoints/configs may persist normalized scale_factors in
        # metadata.  The original SAT path uses absolute multi_scale frame
        # counts, so scale_factors are accepted for load compatibility and
        # ignored after this point.

        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.ratios = list(ratios)
        self.dimension = int(dimension)
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim)
        self.multi_scale = _sat_ordered_scales(multi_scale, scale_order)
        self.reference_frames = reference_frames
        self.scale_order = scale_order
        self.scale_factors = [float(s) / float(max(self.multi_scale)) for s in self.multi_scale]
        self.model_norm = model_norm
        self.causal = bool(causal)
        self.n_filters = int(n_filters)
        self.n_residual_layers = int(n_residual_layers)
        self.lstm = int(lstm)
        self.phi_kernel = [int(k) for k in phi_kernel]
        self.hop_length = int(np.prod(self.ratios))
        self.attn_window_size = None
        self.rvq_frame_size = None
        self.n_codebooks = len(self.multi_scale)
        # SAT uses absolute per-second scale lengths (ending at 75 frames for
        # 24 kHz / hop 320). DAC-style valid-window chunking disables padding
        # and changes the encoded frame count, which invalidates that geometry.
        self.chunk_with_padding = True

        self.encoder = SEANetEncoder(
            channels=channels,
            dimension=dimension,
            n_filters=n_filters,
            n_residual_layers=n_residual_layers,
            ratios=ratios,
            norm=model_norm,
            causal=causal,
            lstm=lstm,
        )
        self.decoder = SEANetDecoder(
            channels=channels,
            dimension=dimension,
            n_filters=n_filters,
            n_residual_layers=n_residual_layers,
            ratios=ratios,
            norm=model_norm,
            causal=causal,
            lstm=lstm,
        )
        self.quantizer = SATOriginalResidualVectorQuantizer(
            dimension=dimension,
            n_q=len(self.multi_scale),
            bins=codebook_size,
            latent_dim=codebook_dim,
            decay=ema_decay,
            kmeans_init=ema_kmeans_init,
            kmeans_iters=ema_kmeans_iters,
            threshold_ema_dead_code=ema_threshold_ema_dead_code,
            multi_scale=self.multi_scale,
            phi_kernel=self.phi_kernel,
        )
        self.quantizer.rvq_frame_size = self.rvq_frame_size
        self.delay = self.get_delay()

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        return nn.functional.pad(audio_data, (0, right_pad))

    def encode(self, audio_data: torch.Tensor, n_quantizers: int = None):
        z = self.encoder(audio_data)
        quantized, codes, stage_losses = self.quantizer(z, n_quantizers=n_quantizers)
        commitment_loss = [loss.mean() for loss in stage_losses]
        codebook_loss = [stage_losses.new_zeros(()) for _ in commitment_loss]
        return quantized, codes, None, commitment_loss, codebook_loss, stage_losses.new_zeros(())

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
    ):
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z, codes, latents, commitment_loss, codebook_loss, _ = self.encode(
            audio_data, n_quantizers
        )
        audio = self.decode(z)
        return {
            "audio": audio[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }

    def fhat_to_audio(self, fhat):
        if fhat.shape[1] != self.dimension and fhat.shape[-1] == self.codebook_dim:
            fhat = self.quantizer.post_conv(fhat)
        return self.decoder(fhat)

    def idxBl_to_h(self, labels_list):
        return self.quantizer.idxBl_to_var_input(labels_list)

    def audio_to_idxBl(self, x):
        emb = self.encoder(x)
        return self.quantizer.encode(emb)
