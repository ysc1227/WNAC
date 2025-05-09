import math
from typing import List
from typing import Union

import numpy as np
import torch
from audiotools.ml import BaseModel
from torch import nn

from .base import CodecMixin
from wnac.nn.quantize import ResidualVectorQuantize, MultiscaleResidualVectorQuantize, VscaleResidualVectorQuantize, WavescaleResidualVectorQuantize
from wnac.nn.layers import Encoder, Decoder


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class WNAC(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
        scale_factor: list[int] = None,
        phi_kernel: list[int] = None,
        wavescale: bool = False,
        noise: bool = False,
        depthwise: bool = False,
        attn_window_size: int = None,
        codebook_warmup: int = 250,
        pooling: str = 'interp',
        vscale: bool = False
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate
        self.noise = noise
        self.depthwise = depthwise
        self.attn_window_size = attn_window_size
        self.pooling = pooling
        self.vscale = vscale

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        
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

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.scale_factor = scale_factor
        self.phi_kernel = phi_kernel
        self.wavescale = wavescale
        self.codebook_warmup = codebook_warmup
        
        if scale_factor is None:
            self.quantizer = ResidualVectorQuantize(
                input_dim=latent_dim,
                n_codebooks=n_codebooks,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                quantizer_dropout=quantizer_dropout,
            )
        elif not wavescale:
            self.quantizer = MultiscaleResidualVectorQuantize(
                input_dim=latent_dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                scale_factors=scale_factor,
                phi_kernel=phi_kernel,
                quantizer_dropout=quantizer_dropout,
                pooling=pooling
            )
        elif not vscale:
            self.quantizer = WavescaleResidualVectorQuantize(
                input_dim=latent_dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                scale_factors=scale_factor,
                phi_kernel=phi_kernel,
                quantizer_dropout=quantizer_dropout,
                max_init=codebook_warmup,
                pooling=pooling
            )
        else:
            self.quantizer = VscaleResidualVectorQuantize(
                input_dim=latent_dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                scale_factors=scale_factor,
                phi_kernel=phi_kernel,
                quantizer_dropout=quantizer_dropout,
                max_init=codebook_warmup,
                pooling=pooling
            )
            
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
        n_quantizers: int = None,
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
        z = self.encoder(audio_data)
        z, codes, latents, commitment_loss, codebook_loss, wave_loss = self.quantizer(
            z, n_quantizers
        )

        return z, codes, latents, commitment_loss, codebook_loss, wave_loss

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
        out = self.decoder(z)
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
        
        audio_data = self.preprocess(audio_data, sample_rate)
        z, codes, latents, commitment_loss, codebook_loss, aux_loss = self.encode(
            audio_data, n_quantizers
        )

        x = self.decode(z)

        return {
            "audio": x[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
            "vq/aux_loss": aux_loss
        }
    
    def init_vq(self, dataloader):
        self.quantizer.init_vq(dataloader)


if __name__ == "__main__":
    import numpy as np
    from functools import partial
    from torch.autograd import gradcheck

    model = WNAC(
        codebook_dim=64,
        scale_factor=[0.03125, 0.05, 0.08, 0.12, 0.16, 0.21, 0.27, 0.33, 0.41, 0.49, 0.57, 0.67, 0.77, 0.88, 1],
        phi_kernel=[9, 9, 9, 9, 9, 9],
        wavescale=True,
        use_vscale=True,
        depthwise=True,
        noise=True,
        attn_window_size=32,
        sample_rate=44100,
        codebook_warmup=250
    ).to("cpu")

    for n, m in model.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])
        fn = lambda o, p: o + f" {p/1e6:<.3f}M params."
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
    print(model)
    print("Total # of params: ", sum([np.prod(p.size()) for p in model.parameters()]))

    length = 44100
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