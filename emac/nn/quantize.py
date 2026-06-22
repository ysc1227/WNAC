from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from emac.nn.layers import WNConv1d
import time

def cosine_l2_mix(a, b, l2_weight=0.1, eps=1e-8):
    a_n = a / (a.norm(dim=1, keepdim=True) + eps)
    b_n = b / (b.norm(dim=1, keepdim=True) + eps)
    cos = 1 - (a_n * b_n).sum(dim=1).mean(dim=1)      # [B]
    l2  = ((a - b)**2).mean(dim=[1,2])                # [B]
    return cos + l2_weight * l2


class GuidedUpsample(nn.Module):
    """Guide low-rate code embeddings with the previously reconstructed latent."""

    def __init__(
        self,
        input_dim: int,
        codebook_dim: int,
        hidden_dim: Optional[int] = None,
        kernel_size: int = 5,
        detach_guide: bool = True,
        init_scale: float = 1e-3,
    ):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"guided_upsample_kernel must be a positive odd integer, got {kernel_size}")

        hidden_dim = int(hidden_dim or max(32, codebook_dim * 2))
        padding = kernel_size // 2
        self.detach_guide = bool(detach_guide)
        self.guide_proj = nn.Conv1d(input_dim, codebook_dim, kernel_size=1)
        self.net = nn.Sequential(
            nn.Conv1d(codebook_dim * 2, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden_dim, codebook_dim, kernel_size=kernel_size, padding=padding),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x: torch.Tensor, size: int, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        base = F.interpolate(x, size=int(size), mode="linear").contiguous() if x.shape[-1] != int(size) else x.contiguous()
        if guide is None:
            return base
        if guide.shape[-1] != int(size):
            guide = F.interpolate(guide, size=int(size), mode="linear")
        if self.detach_guide:
            guide = guide.detach()

        guide = self.guide_proj(guide)
        delta = self.net(torch.cat([base, guide], dim=1))
        return base + torch.tanh(self.residual_scale) * delta

class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int, stride: int = 1):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.stride = stride

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """
        if self.stride > 1:
            z = torch.nn.functional.avg_pool1d(z, self.stride, self.stride)
            
        z_e = self.in_proj(z) # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)
        
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        if self.stride > 1:
            z_q = z_q.repeat_interleave(self.stride, dim=-1)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices

class MultiscaleVectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        pooling: str = "area",
        pooling_alpha: float = 0.5,
        pooling_power: float = 1.0,
        pooling_eps: float = 1e-6,
        loss_target: str = "full",
        guided_upsample: bool = False,
        guided_upsample_hidden_dim: Optional[int] = None,
        guided_upsample_kernel: int = 5,
        guided_upsample_detach_guide: bool = True,
        guided_upsample_init_scale: float = 1e-3,
        guided_upsample_decoder_grad_alpha: float = 0.0,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.pooling = self._normalize_pooling_name(pooling)
        self.pooling_alpha = float(pooling_alpha)
        self.pooling_power = float(pooling_power)
        self.pooling_eps = float(pooling_eps)
        self.loss_target = self._normalize_loss_target(loss_target)
        self.use_guided_upsample = bool(guided_upsample)
        self.guided_upsample_decoder_grad_alpha = float(guided_upsample_decoder_grad_alpha)

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.guided_upsampler = (
            GuidedUpsample(
                input_dim=input_dim,
                codebook_dim=codebook_dim,
                hidden_dim=guided_upsample_hidden_dim,
                kernel_size=guided_upsample_kernel,
                detach_guide=guided_upsample_detach_guide,
                init_scale=guided_upsample_init_scale,
            )
            if guided_upsample
            else None
        )

    @staticmethod
    def _normalize_pooling_name(pooling: str) -> str:
        pooling = str(pooling).lower().replace("-", "_")
        aliases = {
            "default": "area",
            "mean": "area",
            "avg": "area",
            "weighted": "norm_weighted",
            "norm": "norm_weighted",
            "norm_mix": "area_norm_mix",
            "mix": "area_norm_mix",
            "weighted_mix": "area_norm_mix",
        }
        pooling = aliases.get(pooling, pooling)
        valid = {"area", "norm_weighted", "area_norm_mix"}
        if pooling not in valid:
            raise ValueError(f"Unknown quantizer pooling mode '{pooling}'. Expected one of {sorted(valid)}.")
        return pooling

    @staticmethod
    def _normalize_loss_target(loss_target: str) -> str:
        loss_target = str(loss_target).lower().replace("-", "_")
        aliases = {
            "default": "full",
            "fullrate": "full",
            "full_rate": "full",
            "scale": "lowpass",
            "scale_aware": "lowpass",
            "low_pass": "lowpass",
            "lowpass_target": "lowpass",
            "band": "bandpass",
            "band_pass": "bandpass",
            "band_target": "bandpass",
            "residual": "residual_band",
            "residualband": "residual_band",
            "transition_band": "residual_band",
        }
        loss_target = aliases.get(loss_target, loss_target)
        valid = {"full", "lowpass", "bandpass", "residual_band"}
        if loss_target not in valid:
            raise ValueError(f"Unknown quantizer loss target '{loss_target}'. Expected one of {sorted(valid)}.")
        return loss_target

    def _area_pool(self, x: torch.Tensor, scale: Optional[int]) -> torch.Tensor:
        if scale is None or int(scale) == x.shape[-1]:
            return x
        return F.interpolate(x, size=int(scale), mode="area")

    def _norm_weighted_pool(self, x: torch.Tensor, scale: Optional[int]) -> torch.Tensor:
        if scale is None or int(scale) == x.shape[-1]:
            return x
        # Detach weights to keep this as a stable content-adaptive pooling rule
        # rather than an incentive for the encoder to inflate latent norms.
        weights = x.detach().norm(dim=1, keepdim=True)
        if self.pooling_power != 1.0:
            weights = weights.clamp_min(self.pooling_eps).pow(self.pooling_power)
        weights = weights.clamp_min(self.pooling_eps)
        numerator = F.interpolate(x * weights, size=int(scale), mode="area")
        denominator = F.interpolate(weights, size=int(scale), mode="area").clamp_min(self.pooling_eps)
        return numerator / denominator

    def downsample_latents(self, x: torch.Tensor, scale: Optional[int]) -> torch.Tensor:
        if self.pooling == "area":
            return self._area_pool(x, scale)
        if self.pooling == "norm_weighted":
            return self._norm_weighted_pool(x, scale)
        mean = self._area_pool(x, scale)
        weighted = self._norm_weighted_pool(x, scale)
        alpha = min(1.0, max(0.0, self.pooling_alpha))
        return mean.lerp(weighted, alpha)

    def upsample_code(
        self,
        x: torch.Tensor,
        size: int,
        guide: Optional[torch.Tensor] = None,
        use_guide: bool = True,
    ) -> torch.Tensor:
        if self.guided_upsampler is None or guide is None or not use_guide or not self.use_guided_upsample:
            return F.interpolate(x, size=int(size), mode="linear").contiguous() if x.shape[-1] != int(size) else x.contiguous()
        return self.guided_upsampler(x, size=int(size), guide=guide)

    def lowpass_latents(self, z_e: torch.Tensor, scale: Optional[int], inter_z: Optional[torch.Tensor] = None) -> torch.Tensor:
        if scale is None or int(scale) >= z_e.shape[-1]:
            return z_e
        z_low = inter_z if inter_z is not None and int(inter_z.shape[-1]) == int(scale) else self.downsample_latents(z_e, scale)
        return F.interpolate(z_low, size=z_e.shape[-1], mode="linear").contiguous()

    def loss_target_latents(
        self,
        z_e: torch.Tensor,
        inter_z: torch.Tensor,
        scale: Optional[int],
        floor_scale: Optional[int] = None,
        previous_scale: Optional[int] = None,
    ) -> torch.Tensor:
        if self.loss_target == "full" or scale is None:
            return z_e
        current = self.lowpass_latents(z_e, scale, inter_z)
        if self.loss_target == "lowpass":
            return current
        if self.loss_target == "bandpass":
            if floor_scale is None:
                return current
            return current - self.lowpass_latents(z_e, floor_scale)
        if self.loss_target == "residual_band":
            if previous_scale is not None and int(scale) > int(previous_scale):
                return current - self.lowpass_latents(z_e, previous_scale)
            return current
        raise RuntimeError(f"Unhandled quantizer loss target: {self.loss_target}")
        
    def forward(
        self,
        z,
        scale=None,
        conv=None,
        guide: Optional[torch.Tensor] = None,
        loss_floor_scale: Optional[int] = None,
        loss_previous_scale: Optional[int] = None,
    ):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """
        H = z.shape[-1]
        z_e = self.in_proj(z)
        
        inter_z = self.downsample_latents(z_e, scale)
        z_q_small, indices = self.decode_latents(inter_z)
        
        use_guide = scale is not None and int(scale) != H
        z_q_up = self.upsample_code(z_q_small, H, guide=guide, use_guide=use_guide) if scale != None else z_q_small.contiguous()
        z_q = conv(z_q_up) if conv != None else z_q_up
        
        loss_target = self.loss_target_latents(z_e, inter_z, scale, loss_floor_scale, loss_previous_scale)
        commitment_loss = F.mse_loss(loss_target, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, loss_target.detach(), reduction="none").mean([1, 2])

        z_q_ste = z_e + (z_q - z_e).detach()
        if self.guided_upsample_decoder_grad_alpha != 0.0:
            z_q_ste = z_q_ste + self.guided_upsample_decoder_grad_alpha * (z_q - z_q.detach())

        z_q = self.out_proj(z_q_ste)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        quantize = self.embed_code(embed_id).transpose(1, 2)
        return quantize
    

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices

class EnhancedMultiscaleVectorQuantize(nn.Module):
    """
    Reworked VQ with learnable pre/post filtering around interpolation.
    - scale: 목표 길이(T_i). None이면 길이 변경 없음.
    - pooling:
        * 'lpf_interp' (권장): Pre-LPF -> interpolate(size=scale) -> VQ -> interpolate(size=H) -> Post-LPF
        * 'interp': 기존과 동일한 순수 보간 경로
        * 'avg': 기존 avg_pool/repeat 방식(정수 비율만 안전)
    """
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)

        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        
    # ---------- helpers ----------    
    def _resize_1d(self, x, target_len) -> torch.Tensor:  
        if target_len is None:
            return x
        if target_len < x.shape[-1]:
            return F.interpolate(x, size=target_len, mode='area')
        else:
            return F.interpolate(x, size=target_len, mode='linear', align_corners=False)

    # ---------- main ----------
    def forward(self, z, z_full, scale=None, conv=None):
        """
        z: (B, D_in, T)
        returns:
          z_q_out: (B, D_in, T)
          commitment_loss: (B,)
          codebook_loss:   (B,)
          indices: (B, T_scaled)  # VQ가 일어난 길이
          z_e: (B, D_code, T)     # pre-quant latents (원길이, out_proj 전)
        """
        z_e = self.in_proj(z)
        z_e_full = self.in_proj(z_full)
        
        z_e_low = self._resize_1d(z_e, scale)
        z_q_low, indices = self.decode_latents(z_e_low)  # same length as inter_z
        z_q_full, _ = self.decode_latents(z_e_full)

        z_q_low = conv(z_q_low) if conv is not None else z_q_low
        z_q_full = conv(z_q_full) if conv is not None else z_q_full

        commitment_low = F.mse_loss(z_e_low, z_q_low.detach(), reduction='none').mean([1,2])
        codebook_low = F.mse_loss(z_q_low, z_e_low.detach(), reduction='none').mean([1,2])

        commitment_full = F.mse_loss(z_e_full, z_q_full.detach(), reduction='none').mean([1,2])
        codebook_full = F.mse_loss(z_q_full, z_e_full.detach(), reduction='none').mean([1,2])

        z_q_low = z_e_low + (z_q_low - z_e_low).detach()
        z_q_full = z_e_full + (z_q_full - z_e_full).detach()
        
        # print(commitment_loss.mean())

        return z_q_low, z_q_full, commitment_low, codebook_low, commitment_full, codebook_full, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id, scale=None, conv=None):
        """(선택적) 코드 인덱스를 양자화 벡터로 복호화하고 길이를 scale로 맞춘 뒤 conv."""
        quantize = self.embed_code(embed_id).transpose(1, 2)  # (B, D_code, Tq)
        quantize = self._resize_1d(quantize, scale) if scale is not None else quantize
        
        quantize = conv(quantize) if conv is not None else quantize
        return quantize

    def decode_latents(self, latents):
        """
        latents: (B, D_code, Tq)  # 이미 스케일된 길이
        returns:
          z_q: (B, D_code, Tq)    # 같은 길이로 양자화 복원
          indices: (B, Tq)
        """
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices

    def fuse(self, z_q, scale = None, enforcer = None, target = None):
        target = F.interpolate(target, size=scale, mode='area').detach()
        z_q_en = enforcer.resample(z_q.detach(), scale, guide=target) if enforcer is not None else F.interpolate(z_q, scale, mode='linear', align_corners=False).contiguous()
        z_q_interp = F.interpolate(z_q, size=scale, mode='linear', align_corners=False).contiguous()
        
        if target is not None:
            en_loss = F.mse_loss(z_q_en, target, reduction='none').mean([1, 2])
            z_q_en_st = z_q_interp + (z_q_en - z_q_interp).detach()
            
            return self.out_proj(z_q_en_st), z_q_en, en_loss
        return self.out_proj(z_q_en_st), z_q_en, None
    
class MultiscaleResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        scale_factors: List[int] = None
    ):
        super().__init__()

        self.n_codebooks = len(scale_factors)
        self.scale_factors = scale_factors
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim, stride)
                for stride in scale_factors
            ]
        )
        
        self.ar_input_seq = get_stage_order(scale_factors=self.scale_factors)
    
    def forward(self, z):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
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
        """
        
        z_q = 0
        residual = z
        commitment_loss = []
        codebook_loss = []

        codebook_indices = []
        latents = []
        z_ps = {}

        for i, quantizer in enumerate(self.quantizers):
            
            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual)

            z_q = z_q + z_q_i
            residual = residual - z_q_i

            scale = z_e_i.shape[-1]

            if scale in z_ps:
                z_ps[scale].append(z_q)
            else: z_ps[scale] = [z_q]

            # Sum losses
            commitment_loss.append(commitment_loss_i.mean()) 
            codebook_loss.append(codebook_loss_i.mean())

            codebook_indices.append(indices_i)
            latents.append(z_e_i)
        
        aux_loss = 0.0
        for _, val in z_ps.items():
            if len(val) > 1:
                aux_loss += F.mse_loss(val[1], val[0], reduction="none").mean([1, 2]).mean()

        return z_q, codebook_indices, latents, commitment_loss, codebook_loss, aux_loss

    def from_codes(self, codes: list[torch.Tensor], depth: str = 'full'):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T] or N size list[Tensor[B x T]]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_q = len(self.scale_factors)
        
        T = max([c.shape[-1] for c in codes])
        
        target = codes[:1 if depth == 'early' else (len(codes) // 2 if depth == 'mid' else len(codes))]
        
        for i, (code, quantizer) in enumerate(zip(target, self.quantizers)):
            z_p_i = quantizer.decode_code(code)
            z_p.append(z_p_i)

            z_q_i = quantizer.out_proj(self.quant_resi[i/(n_q-1)](F.interpolate(z_p_i, T, mode='linear'))) 
            z_q = z_q + z_q_i

        return z_q, z_p, codes

    def get_aar_input(self, codes: list[torch.Tensor], use_offset: bool = False):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T] or N size list[Tensor[B x T]]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        out = []
        
        T = max([c.shape[-1] for c in codes])

        if use_offset:
            for i, code in enumerate(codes[:-1]):
                z_q_i = self.quantizers[i].out_proj(F.interpolate(self.quantizers[i].decode_code(code), T, mode='linear')) 
                z_q = z_q + z_q_i
            
                if i >= self.offset:
                    out.append(F.interpolate(z_q, int(self.scale_factors[i+1] * T), mode='area').transpose(1, 2))
        else:
            for k, i in enumerate(self.ar_input_seq[:-1]):
                z_q_i = self.quantizers[i].out_proj(F.interpolate(self.quantizers[i].decode_code(codes[i]), T, mode='linear')) 
                z_q = z_q + z_q_i
            
                out.append(F.interpolate(z_q, int(self.scale_factors[self.ar_input_seq[k+1]] * T), mode='area').transpose(1, 2))
        return out

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T] N size list[Tensor[B x T]]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []

        T = latents.shape[-1]
        residual = latents
        
        for i, quantizer in enumerate(self.quantizers):
            scale = int(self.scale_factors[i] * T)
            scale += 1 if scale == 0 else 0

            z_e = quantizer.in_proj(residual)
        
            inter_z = F.interpolate(z_e, size=scale, mode='area') if scale is not None else z_e
            z_q_i, indices_i = quantizer.decode_latents(inter_z)
            z_q_i_up = F.interpolate(z_q_i, size=T, mode='linear').contiguous() if scale != None else z_q.contiguous()
            
            z_q_i = self.quant_resi[i/float(self.n_codebooks)](z_q_i) if self.quant_resi is not None else z_q_i
            z_q_i_up = self.quant_resi[i/float(self.n_codebooks)](z_q_i_up) if self.quant_resi is not None else z_q_i_up

            z_q_i = quantizer.out_proj(z_q_i)
            z_q_i_up = quantizer.out_proj(z_q_i_up)
            
            z_q = z_q + z_q_i_up
            residual = residual - z_q_i_up

            z_p.append(z_q_i)
            codes.append(indices_i)

        return z_q, z_p, codes
    
    def embedding(self, x: torch.Tensor, i: int, use_offset: bool = True):
        if use_offset:
            i = i + self.offset
        else:
            i = self.ar_input_seq[i]
        out = self.quantizers[i].decode_code(x)
        out = self.quantizers[i].out_proj(out).permute(0, 2, 1)
        return out
    
    def get_next_autoregressive_input(self, si: int, f_hat: torch.Tensor, h_BChw: torch.Tensor, H: int, use_offset: bool = True) -> Tuple[Optional[torch.Tensor], torch.Tensor]: # only used in VAR inference
        SN = len(self.scale_factors)
        next_s = 0
        if use_offset:
            si = si + self.offset
            next_s = si+1
        else:
            next_s = self.ar_input_seq[si+1] if si != SN-1 else 0
            si = self.ar_input_seq[si]
        
        h_BChw = self.quantizers[si].in_proj(h_BChw)
        if si != SN-1:
            h = self.quant_resi[si/(SN-1)](F.interpolate(h_BChw, size=H, mode='linear')) if self.quant_resi is not None else F.interpolate(h_BChw, size=H, mode='linear')
            h = self.quantizers[si].out_proj(h)
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=int(self.scale_factors[next_s] * H), mode='area')
        else:
            h = self.quant_resi[si/(SN-1)](h_BChw) if self.quant_resi is not None else h_BChw
            h = self.quantizers[si].out_proj(h)
            f_hat.add_(h)
            return f_hat, f_hat


class WavescaleResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        scale_factors: list[int] = [0.01, 0.03, 0.05, 0.08, 0.12, 0.16, 0.21, 0.27, 0.33, 0.41, 0.49, 0.57, 0.67, 0.77, 0.88, 1],
        phi_kernel = None,
        quantizer_dropout: float = 0.5,
        quantizer_dropout_mode: str = "prefix",
        use_wavescale: bool = False,
        quantizer_pooling: str = "area",
        quantizer_pooling_alpha: float = 0.5,
        quantizer_pooling_power: float = 1.0,
        quantizer_loss_target: str = "full",
        guided_upsample: bool = False,
        guided_upsample_hidden_dim: Optional[int] = None,
        guided_upsample_kernel: int = 5,
        guided_upsample_detach_guide: bool = True,
        guided_upsample_init_scale: float = 1e-3,
        guided_upsample_after_pivot_only: bool = False,
        guided_upsample_decoder_grad_alpha: float = 0.0,
    ):
        super().__init__()

        if scale_factors is None or len(scale_factors) == 0:
            raise ValueError("scale_factors must be a non-empty list")
        if not 0.0 <= float(quantizer_dropout) <= 1.0:
            raise ValueError(f"quantizer_dropout must be in [0, 1], got {quantizer_dropout}")

        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.use_wavescale = use_wavescale
        self.quantizer_dropout_mode = self._normalize_quantizer_dropout_mode(quantizer_dropout_mode)
        self.quantizer_pooling = quantizer_pooling
        self.quantizer_pooling_alpha = float(quantizer_pooling_alpha)
        self.quantizer_pooling_power = float(quantizer_pooling_power)
        self.quantizer_loss_target = str(quantizer_loss_target)
        self.guided_upsample = bool(guided_upsample)
        self.guided_upsample_hidden_dim = guided_upsample_hidden_dim
        self.guided_upsample_kernel = int(guided_upsample_kernel)
        self.guided_upsample_detach_guide = bool(guided_upsample_detach_guide)
        self.guided_upsample_init_scale = float(guided_upsample_init_scale)
        self.guided_upsample_after_pivot_only = bool(guided_upsample_after_pivot_only)
        self.guided_upsample_decoder_grad_alpha = float(guided_upsample_decoder_grad_alpha)
        self.scale_factors, self.n_codebooks = self._compute_wavescale(scale_factors) if use_wavescale else (scale_factors, len(scale_factors))
        self.offset = len(scale_factors) - 1
        self.loss_floor_scale_factors = self._compute_loss_floor_scale_factors(self.scale_factors)
        self.quantizers = nn.ModuleList(
            [
                MultiscaleVectorQuantize(
                    input_dim,
                    codebook_size,
                    codebook_dim,
                    pooling=quantizer_pooling,
                    pooling_alpha=quantizer_pooling_alpha,
                    pooling_power=quantizer_pooling_power,
                    loss_target=quantizer_loss_target,
                    guided_upsample=guided_upsample,
                    guided_upsample_hidden_dim=guided_upsample_hidden_dim,
                    guided_upsample_kernel=guided_upsample_kernel,
                    guided_upsample_detach_guide=guided_upsample_detach_guide,
                    guided_upsample_init_scale=guided_upsample_init_scale,
                    guided_upsample_decoder_grad_alpha=guided_upsample_decoder_grad_alpha,
                )
                for _ in range(self.n_codebooks)
            ]
        )
        self.configure_guided_upsample_stages()
        if phi_kernel is not None:
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([Phi(codebook_dim, 0.5, ks=ks) for ks in phi_kernel]))
        else:
            self.quant_resi = None
        self.quantizer_dropout = quantizer_dropout

        self.ar_input_seq = get_stage_order(scale_factors=self.scale_factors)
        self.ar_stage_blocks = get_stage_blocks(scale_factors=self.scale_factors, order=self.ar_input_seq)

    @staticmethod
    def _compute_loss_floor_scale_factors(scale_factors):
        unique_scales = sorted({float(scale) for scale in scale_factors})
        floor_by_scale = {}
        for idx, scale in enumerate(unique_scales):
            floor_by_scale[scale] = unique_scales[idx - 1] if idx > 0 else None
        return [floor_by_scale[float(scale)] for scale in scale_factors]

    @staticmethod
    def _scale_factor_to_length(scale_factor: Optional[float], length: int) -> Optional[int]:
        if scale_factor is None:
            return None
        scale = int(float(scale_factor) * int(length))
        return scale + 1 if scale == 0 else scale

    @staticmethod
    def _normalize_quantizer_dropout_mode(mode: str) -> str:
        mode = str(mode).lower().replace("-", "_")
        aliases = {
            "default": "prefix",
            "dac": "prefix",
            "rvq": "prefix",
            "shell": "symmetric_shell",
            "mirrored": "symmetric_shell",
            "mirrored_shell": "symmetric_shell",
            "wavescale": "symmetric_shell",
            "wavescale_shell": "symmetric_shell",
        }
        mode = aliases.get(mode, mode)
        valid = {"prefix", "symmetric_shell"}
        if mode not in valid:
            raise ValueError(f"Unknown quantizer_dropout_mode '{mode}'. Expected one of {sorted(valid)}.")
        return mode

    def _training_dropout_stage_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        mask = torch.ones((batch_size, self.n_codebooks), dtype=torch.bool, device=device)
        n_dropout = int(batch_size * self.quantizer_dropout)
        if n_dropout <= 0:
            return mask

        stage_idx = torch.arange(self.n_codebooks, device=device).view(1, -1)
        if self.quantizer_dropout_mode == "prefix":
            n_quantizers = torch.randint(1, self.n_codebooks + 1, (n_dropout,), device=device)
            mask[:n_dropout] = stage_idx < n_quantizers.view(-1, 1)
            return mask

        if not self.use_wavescale or self.n_codebooks != 2 * self.offset + 1:
            raise ValueError("symmetric_shell quantizer dropout requires an odd Wavescale schedule.")
        n_shells = torch.randint(1, self.offset + 2, (n_dropout,), device=device)
        mask[:n_dropout] = (stage_idx < n_shells.view(-1, 1)) | (
            stage_idx >= (self.n_codebooks - n_shells).view(-1, 1)
        )
        return mask

    def _apply_quant_resi(self, stage_idx: int, h: torch.Tensor) -> torch.Tensor:
        """Apply the stage-specific Phi residual conv, if configured.

        Keep Phi selection consistent across train-time quantization,
        reconstruction from codes, and AAR decode helpers.  The selector expects
        a value in [0, 1], so use n_codebooks - 1 as the denominator such that
        the first and last stages map to the first and last Phi buckets.
        """
        if self.quant_resi is None:
            return h
        denom = max(1, self.n_codebooks - 1)
        return self.quant_resi[stage_idx / denom](h)

    def configure_guided_upsample_stages(self):
        """Enable guided upsampling only for stages that actually use it.

        DDP treats trainable parameters that never participate in the forward
        graph as unused.  The first stage has no previous cumulative guide, and
        full-scale stages do not interpolate, so their guided modules must stay
        frozen even if they exist in the state dict for checkpoint compatibility.
        """
        enabled = bool(getattr(self, "guided_upsample", False))
        after_pivot_only = bool(getattr(self, "guided_upsample_after_pivot_only", False))
        self.guided_upsample_stage_mask = [
            enabled
            and stage_idx > 0
            and (not after_pivot_only or stage_idx > self.offset)
            and float(scale) < 1.0
            for stage_idx, scale in enumerate(self.scale_factors)
        ]
        for active, quantizer in zip(self.guided_upsample_stage_mask, self.quantizers):
            if hasattr(quantizer, "use_guided_upsample"):
                quantizer.use_guided_upsample = bool(active)
            upsampler = getattr(quantizer, "guided_upsampler", None)
            if upsampler is not None:
                for param in upsampler.parameters():
                    param.requires_grad_(bool(active))
        return self.guided_upsample_stage_mask
        
    def forward(self, z, n_quantizers: int = None):
        # start = time.time()     
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
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
        """
        T = z.shape[-1]
        z_q = 0
        
        residual = z
        commitment_loss = []
        codebook_loss = []

        codebook_indices = []
        latents = []
        z_ps = {}
        
        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        elif not self.training:
            n_quantizers = int(n_quantizers)
            if not 1 <= n_quantizers <= self.n_codebooks:
                raise ValueError(f"n_quantizers must be in [1, {self.n_codebooks}], got {n_quantizers}")
        if self.training:
            stage_mask = self._training_dropout_stage_mask(z.shape[0], z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break
            scale = int(self.scale_factors[i] * T)
            scale += 1 if scale == 0 else 0
            loss_floor_scale = self._scale_factor_to_length(self.loss_floor_scale_factors[i], T)
            loss_previous_scale = (
                self._scale_factor_to_length(self.scale_factors[i - 1], T) if i > 0 else None
            )
            guide = z_q if torch.is_tensor(z_q) else None

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual,
                scale,
                (lambda h, stage_idx=i: self._apply_quant_resi(stage_idx, h)) if self.quant_resi is not None else None,
                guide=guide,
                loss_floor_scale=loss_floor_scale,
                loss_previous_scale=loss_previous_scale,
            )
            
            # Create mask to apply quantizer dropout
            if self.training:
                mask = stage_mask[:, i].to(dtype=z_q_i.dtype)
            else:
                mask = torch.ones((z.shape[0],), dtype=z_q_i.dtype, device=z.device)

            z_q = z_q + z_q_i * mask[:, None, None]
            
            if scale in z_ps:
                z_ps[scale].append(z_q)
            else: z_ps[scale] = [z_q]
            
            residual = residual - z_q_i * mask[:, None, None]

            # Sum losses over active quantizers only, matching DAC's quantizer
            # dropout behavior.  Dropped stages do not contribute to the final
            # reconstruction for that sample, so their VQ losses should not
            # contribute either.
            commitment_loss.append((commitment_loss_i * mask).mean())
            codebook_loss.append((codebook_loss_i * mask).mean())

            codebook_indices.append(indices_i)
            latents.append(z_e_i)
        
        aux_loss = 0.0
        if self.use_wavescale:
            for _, val in z_ps.items():
                if len(val) > 1:
                    aux_loss += F.mse_loss(val[1], val[0], reduction="none").mean([1, 2]).mean()
        
        # end = time.time()

        # print(end-start)
        return z_q, codebook_indices, latents, commitment_loss, codebook_loss, aux_loss

    def from_codes(self, codes: list[torch.Tensor], depth='full'):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T] or N size list[Tensor[B x T]]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_q = len(self.scale_factors)
        
        T = max([c.shape[-1] for c in codes])
        
        target = codes[:1 if depth == 'early' else (len(codes) // 2 if depth == 'mid' else len(codes))]
        
        for i, (code, quantizer) in enumerate(zip(target, self.quantizers)):
            z_p_i = quantizer.decode_code(code)
            z_p.append(z_p_i)

            guide = z_q if torch.is_tensor(z_q) else None
            z_p_i_up = quantizer.upsample_code(z_p_i, T, guide=guide, use_guide=z_p_i.shape[-1] != T)
            z_q_i = quantizer.out_proj(self._apply_quant_resi(i, z_p_i_up))
            z_q = z_q + z_q_i

        return z_q, z_p, codes

    def get_aar_input(self, codes: list[torch.Tensor], use_offset: bool = False, use_scalewise: bool = True, use_blockwise: bool = True):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T] or N size list[Tensor[B x T]]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        out = []
        
        T = max([c.shape[-1] for c in codes])

        def stage_to_full(stage_idx: int) -> torch.Tensor:
            z_q_i = F.interpolate(self.quantizers[stage_idx].decode_code(codes[stage_idx]), T, mode='linear')
            return self.quantizers[stage_idx].out_proj(self._apply_quant_resi(stage_idx, z_q_i))

        if use_scalewise and use_blockwise:
            # Blockwise same-scale AAR:
            #   [pivot] -> [same-scale pair] -> [same-scale pair] -> ...
            # All stages in the next same-scale block receive the same previous-block
            # reconstruction context, rather than seeing earlier stages from their own
            # block.  Example: [2] -> [1, 3] -> [0, 4].
            for block_idx, block in enumerate(self.ar_stage_blocks[:-1]):
                for i in block:
                    z_q = z_q + stage_to_full(i)

                for next_i in self.ar_stage_blocks[block_idx + 1]:
                    out.append(F.interpolate(z_q, int(self.scale_factors[next_i] * T), mode='area').transpose(1, 2))
        elif use_scalewise:
            # Reordered sequential AAR:
            #   predict stages in effective-scale order (ar_input_seq), but each
            #   stage sees all previously predicted stages, including same-scale
            #   siblings that came earlier in the order.
            for k, i in enumerate(self.ar_input_seq[:-1]):
                z_q = z_q + stage_to_full(i)
                next_i = self.ar_input_seq[k + 1]
                out.append(F.interpolate(z_q, int(self.scale_factors[next_i] * T), mode='area').transpose(1, 2))
        else:
            for i, code in enumerate(codes[:-1]):
                z_q = z_q + stage_to_full(i)
                
                if not use_offset or i >= self.offset:
                    out.append(F.interpolate(z_q, int(self.scale_factors[i+1] * T), mode='area').transpose(1, 2))
        return out

    def get_aar_target_codes(self, codes: list[torch.Tensor], use_offset: bool = False, use_scalewise: bool = True):
        """Return target code tensors in the same order as AAR logits."""
        if use_scalewise:
            return [codes[i] for i in self.ar_input_seq]
        return codes[self.offset:] if use_offset else codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T] N size list[Tensor[B x T]]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []

        T = latents.shape[-1]
        residual = latents
        
        for i, quantizer in enumerate(self.quantizers):
            scale = int(self.scale_factors[i] * T)
            scale += 1 if scale == 0 else 0

            z_e = quantizer.in_proj(residual)
        
            inter_z = quantizer.downsample_latents(z_e, scale)
            z_q_i, indices_i = quantizer.decode_latents(inter_z)
            guide = z_q if torch.is_tensor(z_q) else None
            z_q_i_up = quantizer.upsample_code(z_q_i, T, guide=guide, use_guide=scale is not None and int(scale) != T) if scale != None else z_q_i.contiguous()
            
            z_q_i = self._apply_quant_resi(i, z_q_i)
            z_q_i_up = self._apply_quant_resi(i, z_q_i_up)

            z_q_i = quantizer.out_proj(z_q_i)
            z_q_i_up = quantizer.out_proj(z_q_i_up)
            
            z_q = z_q + z_q_i_up
            residual = residual - z_q_i_up

            z_p.append(z_q_i)
            codes.append(indices_i)

        return z_q, z_p, codes
    
    def embedding(self, x: torch.Tensor, i: int, use_offset: bool = False, use_scalewise: bool = True):
        if use_scalewise:
            i = self.ar_input_seq[i]
        else:
            i = i + (self.offset if use_offset else 0)
        out = self.quantizers[i].decode_code(x)
        out = self.quantizers[i].out_proj(out).permute(0, 2, 1)
        return out

    def decode_stage_to_full(self, x: torch.Tensor, stage_pos: int, H: int, use_offset: bool = False, use_scalewise: bool = True) -> torch.Tensor:
        """Decode AAR stage-position codes to a full-length VAE contribution."""
        if use_scalewise:
            codebook_idx = self.ar_input_seq[stage_pos]
        else:
            codebook_idx = stage_pos + (self.offset if use_offset else 0)

        h = self.quantizers[codebook_idx].decode_code(x)
        h = F.interpolate(h, size=H, mode="linear")
        return self.quantizers[codebook_idx].out_proj(self._apply_quant_resi(codebook_idx, h))
    
    def get_next_autoregressive_input(self, si: int, f_hat: torch.Tensor, h_BChw: torch.Tensor, H: int, use_offset: bool = False, use_scalewise: bool = True) -> Tuple[Optional[torch.Tensor], torch.Tensor]: # only used in VAR inference
        SN = len(self.scale_factors)
        
        stage_pos = si  # 0..SN-1 (순서상의 스테이지 인덱스)
        
        if use_scalewise:
            codebook_idx = self.ar_input_seq[stage_pos]
            next_codebook = self.ar_input_seq[stage_pos + 1] if stage_pos != SN - 1 else None
            is_last = (stage_pos == SN - 1)   # ★ 여기 중요: stage_pos로 판단
        else: 
            codebook_idx = stage_pos + (self.offset if use_offset else 0)
            next_codebook = codebook_idx + 1
            is_last = (codebook_idx == SN - 1)
        
        h_BChw = self.quantizers[codebook_idx].in_proj(h_BChw)
        
        if not is_last:
            h = F.interpolate(h_BChw, size=H, mode="linear")
            h = self._apply_quant_resi(codebook_idx, h)
            h = self.quantizers[codebook_idx].out_proj(h)
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=int(self.scale_factors[next_codebook] * H), mode="area")
        else:
            h = self._apply_quant_resi(codebook_idx, h_BChw)
            h = self.quantizers[codebook_idx].out_proj(h)
            f_hat.add_(h)
            return f_hat, f_hat
        
    def _compute_wavescale(self, scale_factors):
        return scale_factors[::-1] + scale_factors[1:], len(scale_factors) * 2 - 1
    
    
class Phi(nn.Conv1d):
    def __init__(self, embed_dim, quant_resi, ks):
        padding = (ks // 2)  # Adjust padding based on kernel size and dilation
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=padding)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw):
        return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)
    
    
class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


def get_stage_order(scale_factors):
    """
    scale_factors: 예) [3, 2, 1, 2, 3]
    반환: 스케일 오름차순, 같은 스케일 내에서는 왼→오 인덱스 순서
          예) [2, 1, 3, 0, 4]
    """
    unique_scales = sorted(set(scale_factors))  # [1, 2, 3]
    order = []

    for s in unique_scales:
        for i, v in enumerate(scale_factors):
            if v == s:
                order.append(i)

    return order


def get_stage_blocks(scale_factors, order=None):
    """
    Group stage indices by effective temporal scale in coarse-to-fine order.

    For a wavescale schedule like [1.0, 0.5, 0.1, 0.5, 1.0], this returns
    [[2], [1, 3], [0, 4]].  The flattened order is identical to
    get_stage_order(), but same-scale stages are explicitly represented as a
    block so AAR can predict them from the same previous-scale context.
    """
    if order is None:
        order = get_stage_order(scale_factors)

    blocks = []
    current_scale = None
    current_block = []
    for idx in order:
        scale = scale_factors[idx]
        if current_scale is None or scale == current_scale:
            current_block.append(idx)
        else:
            blocks.append(current_block)
            current_block = [idx]
        current_scale = scale

    if current_block:
        blocks.append(current_block)
    return blocks

if __name__ == "__main__":
    codebook_dim = 64
    T = 96

    emac = WavescaleResidualVectorQuantize(
        scale_factors=[0.03, 0.05, 0.08, 0.12, 0.16, 0.21, 0.27, 0.33, 0.41, 0.49, 0.57, 0.67, 0.77, 0.88, 1],
        input_dim=64 * (2 ** len([2, 4, 8, 8])),
        codebook_size=1024,
        codebook_dim=codebook_dim,        
        phi_kernel=[9, 9, 9, 9, 9, 9],
    ).cuda()
    
    x = torch.randn(16, 1024, T, requires_grad=True).cuda()
    y = emac(x)

    print("----------------------------------------------")
