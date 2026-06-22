import math
import os
import time
from typing import List
from typing import Union

import numpy as np
import torch
from torch import nn

from audiotools.ml.layers.base import BaseModel
from emac.nn.layers import Decoder, Encoder
from emac.nn.seanet import SEANetDecoder, SEANetEncoder

from .base import CodecMixin
from emac.nn.quantize import WavescaleResidualVectorQuantize


def _emac_probe_enabled():
    return os.getenv("EMAC_INTERNAL_PROBE", "0") == "1"


def _emac_probe(name, x):
    if not _emac_probe_enabled():
        return True
    if not torch.is_tensor(x):
        print(f"[EMAC_PROBE] {name}: non_tensor {type(x).__name__}", flush=True)
        return True
    with torch.no_grad():
        finite = torch.isfinite(x)
        ok = bool(finite.all().item())
        ratio = float(finite.float().mean().item())
        safe = torch.nan_to_num(x.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        print(
            f"[EMAC_PROBE] {name}: ok={ok} finite_ratio={ratio:.6f} "
            f"shape={tuple(x.shape)} mean={float(safe.mean().item()):.6g} "
            f"absmax={float(safe.abs().max().item()):.6g}",
            flush=True,
        )
        return ok


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)


class EMAC(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
        quantizer_dropout_mode: str = "prefix",
        sample_rate: int = 44100,
        scale_factor: list[int] = None,
        phi_kernel: list[int] = None,
        noise: bool = False,
        attn_window_size: int = None,
        depthwise: bool = False,
        use_wavescale: bool = False,
        use_seanet: bool = False,
        seanet_n_filters: int = 32,
        seanet_n_residual_layers: int = 1,
        seanet_lstm: int = 2,
        seanet_causal: bool = True,
        seanet_norm: str = "weight_norm",
        rvq_frame_size: int = None,
        quantizer_pooling: str = "area",
        quantizer_pooling_alpha: float = 0.5,
        quantizer_pooling_power: float = 1.0,
        quantizer_loss_target: str = "full",
        guided_upsample: bool = False,
        guided_upsample_hidden_dim: int = None,
        guided_upsample_kernel: int = 5,
        guided_upsample_detach_guide: bool = True,
        guided_upsample_init_scale: float = 1e-3,
        guided_upsample_after_pivot_only: bool = False,
        guided_upsample_decoder_grad_alpha: float = 0.0,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate
        if scale_factor is None or len(scale_factor) == 0:
            raise ValueError("EMAC.scale_factor must be a non-empty list")
        if not 0.0 <= float(quantizer_dropout) <= 1.0:
            raise ValueError(f"EMAC.quantizer_dropout must be in [0, 1], got {quantizer_dropout}")

        self.quantizer_dropout = float(quantizer_dropout)
        self.noise = noise
        self.attn_window_size = attn_window_size
        self.use_wavescale = use_wavescale
        self.use_seanet = use_seanet
        self.seanet_n_filters = seanet_n_filters
        self.seanet_n_residual_layers = seanet_n_residual_layers
        self.seanet_lstm = seanet_lstm
        self.seanet_causal = seanet_causal
        self.seanet_norm = seanet_norm
        self.rvq_frame_size = None if rvq_frame_size is None else int(rvq_frame_size)
        self.quantizer_pooling = quantizer_pooling
        self.quantizer_pooling_alpha = float(quantizer_pooling_alpha)
        self.quantizer_pooling_power = float(quantizer_pooling_power)
        self.quantizer_loss_target = str(quantizer_loss_target)
        self.quantizer_dropout_mode = quantizer_dropout_mode
        self.guided_upsample = bool(guided_upsample)
        self.guided_upsample_hidden_dim = guided_upsample_hidden_dim
        self.guided_upsample_kernel = int(guided_upsample_kernel)
        self.guided_upsample_detach_guide = bool(guided_upsample_detach_guide)
        self.guided_upsample_init_scale = float(guided_upsample_init_scale)
        self.guided_upsample_after_pivot_only = bool(guided_upsample_after_pivot_only)
        self.guided_upsample_decoder_grad_alpha = float(guided_upsample_decoder_grad_alpha)
        
        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim
        self.hop_length = np.prod(encoder_rates)

        if use_seanet:
            if list(encoder_rates) != list(decoder_rates):
                raise ValueError(
                    "SEANet backbone expects encoder_rates and decoder_rates to be the same "
                    f"ratio list, got encoder_rates={encoder_rates}, decoder_rates={decoder_rates}"
                )
            if attn_window_size is not None:
                print("**WARNING**: attn_window_size is ignored when use_seanet=True", flush=True)
            if noise:
                print("**WARNING**: noise decoder option is ignored when use_seanet=True", flush=True)

            self.encoder = SEANetEncoder(
                channels=1,
                dimension=latent_dim,
                n_filters=seanet_n_filters,
                n_residual_layers=seanet_n_residual_layers,
                ratios=encoder_rates,
                norm=seanet_norm,
                causal=seanet_causal,
                lstm=seanet_lstm,
            )
            self.decoder = SEANetDecoder(
                channels=1,
                dimension=latent_dim,
                n_filters=seanet_n_filters,
                n_residual_layers=seanet_n_residual_layers,
                ratios=decoder_rates,
                norm=seanet_norm,
                causal=seanet_causal,
                lstm=seanet_lstm,
            )
        else:
            self.encoder = Encoder(
                d_model=encoder_dim, 
                rates=encoder_rates,
                depthwise=depthwise,
                attn_window_size=attn_window_size
            )
            self.decoder = Decoder(
                latent_dim=latent_dim,
                d_model=decoder_dim,
                rates=decoder_rates,
                noise=noise,
                depthwise=depthwise,
                attn_window_size=attn_window_size
            )

        self.n_codebooks = len(scale_factor) * 2 - 1 if use_wavescale else len(scale_factor)
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.scale_factor = scale_factor
        self.phi_kernel = phi_kernel

        self.quantizer = WavescaleResidualVectorQuantize(
            input_dim=latent_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            scale_factors=scale_factor,
            phi_kernel=phi_kernel,
            quantizer_dropout=quantizer_dropout,
            quantizer_dropout_mode=quantizer_dropout_mode,
            use_wavescale=use_wavescale,
            quantizer_pooling=quantizer_pooling,
            quantizer_pooling_alpha=quantizer_pooling_alpha,
            quantizer_pooling_power=quantizer_pooling_power,
            quantizer_loss_target=quantizer_loss_target,
            guided_upsample=guided_upsample,
            guided_upsample_hidden_dim=guided_upsample_hidden_dim,
            guided_upsample_kernel=guided_upsample_kernel,
            guided_upsample_detach_guide=guided_upsample_detach_guide,
            guided_upsample_init_scale=guided_upsample_init_scale,
            guided_upsample_after_pivot_only=guided_upsample_after_pivot_only,
            guided_upsample_decoder_grad_alpha=guided_upsample_decoder_grad_alpha,
        )
        self.quantizer.rvq_frame_size = self.rvq_frame_size

        self.sample_rate = sample_rate
        self.apply(init_weights)

        self.delay = self.get_delay()

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate
        
        length = audio_data.shape[-1]
        pad_to = self.hop_length * self.attn_window_size if self.attn_window_size is not None else self.hop_length
        right_pad = math.ceil(length / pad_to) * pad_to - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        
        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
        n_quantizers: int = None
    ):
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        n_quantizers : int, optional
            Number of quantizers to use, by default None
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
        """
        # start = time.time()
        _emac_probe("encode.input", audio_data)
        z = self.encoder(audio_data)
        _emac_probe("encode.encoder_z", z)
        z, codes, latents, commitment_loss, codebook_loss, en_loss = self.quantizer(
            z, n_quantizers=n_quantizers
        )
        _emac_probe("encode.quantized_z", z)
        _emac_probe("encode.latents", latents)
        if isinstance(commitment_loss, (list, tuple)):
            for i, v in enumerate(commitment_loss):
                _emac_probe(f"encode.commitment_loss[{i}]", v)
        if isinstance(codebook_loss, (list, tuple)):
            for i, v in enumerate(codebook_loss):
                _emac_probe(f"encode.codebook_loss[{i}]", v)
        # end = time.time()
        # print(end-start)
        return z, codes, latents, commitment_loss, codebook_loss, en_loss

    def decode(self, z: torch.Tensor):
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input
        length : int, optional
            Number of samples in output audio, by default None

        Returns
        -------
        dict
            A dictionary with the following keys:
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        _emac_probe("decode.input_z", z)
        out = self.decoder(z)
        _emac_probe("decode.decoder_out", out)
        return out

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        length = audio_data.shape[-1]
        
        _emac_probe("forward.input_audio", audio_data)
        audio_data = self.preprocess(audio_data, sample_rate)
        _emac_probe("forward.preprocessed_audio", audio_data)
        z, codes, latents, commitment_loss, codebook_loss, en_loss = self.encode(
            audio_data, n_quantizers
        )

        x = self.decode(z)
        _emac_probe("forward.sliced_audio", x[..., :length])

        return {
            "audio": x[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
            "vq/aux_loss": en_loss
        }
        
    def fhat_to_audio(self, fhat):
        return self.decoder(fhat)


if __name__ == "__main__":
    import numpy as np
    from functools import partial

    model = EMAC(
        codebook_dim=64,
        sample_rate=44100,
        encoder_rates = [8, 5, 4, 2],
        decoder_rates = [8, 5, 4, 2],
        scale_factor = [0.0490, 0.1213, 0.2254, 0.3669, 0.5424, 0.7527, 1.0000],
        phi_kernel = [9, 9, 9, 9, 9, 9],
        use_wavescale = True,
        use_seanet=True
    ).cuda()

    for n, m in model.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])
        fn = lambda o, p: o + f" {p/1e6:<.3f}M params."
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
    print(model)
    print("Total # of params: ", sum([np.prod(p.size()) for p in model.parameters()]))

    length = int(44100 * 0.92)
    x = torch.randn(1, 1, length).to(model.device)
    
    x.requires_grad_(True)
    x.retain_grad()

    # Make a forward pass
    out = model(x)["audio"]
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)

    # Create gradient variable
    grad = torch.zeros_like(out)
    grad[:, :, grad.shape[-1] // 2] = 1

    # Make a backward pass
    out.backward(grad)

    # Check non-zero values
    gradmap = x.grad.squeeze(0)
    gradmap = (gradmap != 0).sum(0)  # sum across features
    rf = (gradmap != 0).sum()

    print(f"Receptive field: {rf.item()}")

    # x = AudioSignal(torch.randn(1, 1, 44100 * 60), 44100)
    # model.decompress(model.compress(x, verbose=True), verbose=True)
