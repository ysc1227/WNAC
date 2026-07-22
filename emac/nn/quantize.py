import math
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


class LearnedPatchUpsample(nn.Module):
    """Generate a short full-rate latent patch from each low-rate code token.

    This keeps the ordinary linear interpolation path as the initialization and
    learns only a small residual.  The residual path lets a coarse token emit a
    local latent trajectory instead of being painted across time as a single
    vector.
    """

    def __init__(
        self,
        input_dim: int,
        codebook_dim: int,
        hidden_dim: Optional[int] = None,
        kernel_size: int = 3,
        patch_size: int = 4,
        init_scale: float = 1e-3,
        condition_guide: bool = False,
        detach_guide: bool = True,
        max_scale: float = 1.0,
        guide_is_codebook_dim: bool = False,
    ):
        super().__init__()
        kernel_size = int(kernel_size)
        patch_size = int(patch_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"learned_upsample_kernel must be a positive odd integer, got {kernel_size}")
        if patch_size <= 1:
            raise ValueError(f"learned_upsample_patch_size must be > 1, got {patch_size}")

        hidden_dim = int(hidden_dim or max(32, codebook_dim * 4))
        padding = kernel_size // 2
        self.codebook_dim = int(codebook_dim)
        self.patch_size = patch_size
        self.condition_guide = bool(condition_guide)
        self.detach_guide = bool(detach_guide)
        self.guide_is_codebook_dim = bool(guide_is_codebook_dim)
        self.max_scale = float(max_scale)
        if self.max_scale <= 0:
            raise ValueError(f"learned_upsample_max_scale must be positive, got {max_scale}")
        if self.condition_guide and not self.guide_is_codebook_dim:
            self.guide_proj = nn.Conv1d(input_dim, codebook_dim, kernel_size=1)
        self.net = nn.Sequential(
            nn.Conv1d(
                codebook_dim * (2 if self.condition_guide else 1),
                hidden_dim,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden_dim, codebook_dim * patch_size, kernel_size=kernel_size, padding=padding),
        )
        init_scale = min(abs(float(init_scale)), self.max_scale * 0.999)
        raw_init = np.arctanh(init_scale / self.max_scale)
        self.residual_scale = nn.Parameter(torch.tensor(float(raw_init)))

    def residual_scale_value(self) -> torch.Tensor:
        return self.max_scale * torch.tanh(self.residual_scale)

    def forward(self, x: torch.Tensor, size: int, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        size = int(size)
        if x.shape[-1] == size:
            return x.contiguous()

        base = F.interpolate(x, size=size, mode="linear").contiguous()
        net_input = x
        if self.condition_guide:
            if guide is None:
                guide_features = torch.zeros_like(x)
            else:
                if self.detach_guide:
                    guide = guide.detach()
                guide_features = guide if self.guide_is_codebook_dim else self.guide_proj(guide)
                if guide_features.shape[-1] != x.shape[-1]:
                    mode = "area" if guide_features.shape[-1] > x.shape[-1] else "linear"
                    guide_features = F.interpolate(guide_features, size=x.shape[-1], mode=mode).contiguous()
            net_input = torch.cat([x, guide_features], dim=1)

        patch = self.net(net_input)
        b, _, t = patch.shape
        patch = patch.view(b, self.codebook_dim, self.patch_size, t)
        patch = patch.permute(0, 1, 3, 2).reshape(b, self.codebook_dim, t * self.patch_size)
        if patch.shape[-1] != size:
            patch = F.interpolate(patch, size=size, mode="linear").contiguous()
        return base + self.residual_scale_value() * patch


class LearnedFractionalUpsample(nn.Module):
    """Generate full-rate residuals at fractional low-rate positions."""

    def __init__(
        self,
        input_dim: int,
        codebook_dim: int,
        hidden_dim: Optional[int] = None,
        kernel_size: int = 3,
        init_scale: float = 1e-3,
        condition_guide: bool = False,
        detach_guide: bool = True,
        max_scale: float = 1.0,
        guide_is_codebook_dim: bool = False,
    ):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"learned_upsample_kernel must be a positive odd integer, got {kernel_size}")

        hidden_dim = int(hidden_dim or max(32, codebook_dim * 4))
        padding = kernel_size // 2
        self.codebook_dim = int(codebook_dim)
        self.condition_guide = bool(condition_guide)
        self.detach_guide = bool(detach_guide)
        self.guide_is_codebook_dim = bool(guide_is_codebook_dim)
        self.max_scale = float(max_scale)
        if self.max_scale <= 0:
            raise ValueError(f"learned_upsample_max_scale must be positive, got {max_scale}")
        if self.condition_guide and not self.guide_is_codebook_dim:
            self.guide_proj = nn.Conv1d(input_dim, codebook_dim, kernel_size=1)

        in_channels = codebook_dim * 3 + 1 + (codebook_dim if self.condition_guide else 0)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden_dim, codebook_dim, kernel_size=kernel_size, padding=padding),
        )
        init_scale = min(abs(float(init_scale)), self.max_scale * 0.999)
        raw_init = np.arctanh(init_scale / self.max_scale)
        self.residual_scale = nn.Parameter(torch.tensor(float(raw_init)))

    def residual_scale_value(self) -> torch.Tensor:
        return self.max_scale * torch.tanh(self.residual_scale)

    @staticmethod
    def _fractional_neighbors(x: torch.Tensor, size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, t = x.shape
        pos = (torch.arange(size, device=x.device, dtype=x.dtype) + 0.5) * (t / float(size)) - 0.5
        pos = pos.clamp(0, max(0, t - 1))
        left_idx = pos.floor().long()
        right_idx = (left_idx + 1).clamp(max=t - 1)
        frac = (pos - left_idx.to(dtype=x.dtype)).view(1, 1, size)
        left = x.gather(2, left_idx.view(1, 1, size).expand(b, c, size))
        right = x.gather(2, right_idx.view(1, 1, size).expand(b, c, size))
        return left, right, frac

    def forward(self, x: torch.Tensor, size: int, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        size = int(size)
        if x.shape[-1] == size:
            return x.contiguous()

        left, right, frac = self._fractional_neighbors(x, size)
        base = left.lerp(right, frac).contiguous()
        features = [left, right, base, frac.expand(x.shape[0], 1, size)]
        if self.condition_guide:
            if guide is None:
                guide_features = torch.zeros_like(base)
            else:
                if self.detach_guide:
                    guide = guide.detach()
                guide_features = guide if self.guide_is_codebook_dim else self.guide_proj(guide)
                if guide_features.shape[-1] != size:
                    mode = "area" if guide_features.shape[-1] > size else "linear"
                    guide_features = F.interpolate(guide_features, size=size, mode=mode).contiguous()
            features.append(guide_features)
        delta = self.net(torch.cat(features, dim=1))
        return base + self.residual_scale_value() * delta


class LearnedResidualDownsample(nn.Module):
    """Learn a small analysis residual before low-rate vector quantization."""

    def __init__(
        self,
        codebook_dim: int,
        hidden_dim: Optional[int] = None,
        kernel_size: int = 3,
        init_scale: float = 1e-3,
        max_scale: float = 0.2,
    ):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"learned_downsample_kernel must be a positive odd integer, got {kernel_size}")
        self.max_scale = float(max_scale)
        if self.max_scale <= 0:
            raise ValueError(f"learned_downsample_max_scale must be positive, got {max_scale}")
        hidden_dim = int(hidden_dim or max(32, int(codebook_dim) * 4))
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(codebook_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden_dim, codebook_dim, kernel_size=kernel_size, padding=padding),
        )
        init_scale = min(abs(float(init_scale)), self.max_scale * 0.999)
        raw_init = np.arctanh(init_scale / self.max_scale)
        self.residual_scale = nn.Parameter(torch.tensor(float(raw_init)))

    def residual_scale_value(self) -> torch.Tensor:
        return self.max_scale * torch.tanh(self.residual_scale)

    def forward(self, x: torch.Tensor, size: int, base: torch.Tensor) -> torch.Tensor:
        size = int(size)
        if x.shape[-1] == size:
            return x.contiguous()
        residual = self.net(x)
        residual = F.interpolate(residual, size=size, mode="area").contiguous()
        return base + self.residual_scale_value() * residual


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


def _sample_rows(samples: torch.Tensor, n: int) -> torch.Tensor:
    n_samples = int(samples.shape[0])
    if n_samples <= 0:
        raise ValueError("Cannot sample from an empty tensor")
    if n_samples >= n:
        indices = torch.randperm(n_samples, device=samples.device)[:n]
    else:
        indices = torch.randint(0, n_samples, (n,), device=samples.device)
    return samples[indices]


def _kmeans(samples: torch.Tensor, n_clusters: int, n_iters: int) -> tuple[torch.Tensor, torch.Tensor]:
    samples = samples.detach()
    means = _sample_rows(samples, n_clusters).clone()
    bins = samples.new_zeros(n_clusters)
    for _ in range(max(1, int(n_iters))):
        dists = (
            samples.pow(2).sum(dim=1, keepdim=True)
            - 2 * samples @ means.t()
            + means.pow(2).sum(dim=1, keepdim=True).t()
        )
        buckets = dists.argmin(dim=1)
        bins = torch.bincount(buckets, minlength=n_clusters).to(dtype=samples.dtype)
        sums = samples.new_zeros(n_clusters, samples.shape[1])
        sums.index_add_(0, buckets, samples)
        means = torch.where((bins > 0).unsqueeze(1), sums / bins.clamp_min(1).unsqueeze(1), means)
    return means, bins


def _all_reduce_sum_(tensor: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor


def _broadcast_from_rank0_(tensor: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(tensor, src=0)
    return tensor


class ExponentialMovingAverageCodebook(nn.Module):
    """Euclidean codebook updated from assignment statistics instead of gradients."""

    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim)
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.kmeans_init = bool(kmeans_init)
        self.kmeans_iters = int(kmeans_iters)
        self.threshold_ema_dead_code = int(threshold_ema_dead_code)

        weight = torch.empty(self.codebook_size, self.codebook_dim)
        nn.init.kaiming_uniform_(weight)
        if self.kmeans_init:
            weight.zero_()
        self.register_buffer("weight", weight)
        self.register_buffer("cluster_size", torch.zeros(self.codebook_size))
        self.register_buffer("embed_avg", weight.clone())
        self.register_buffer("inited", torch.tensor(not self.kmeans_init, dtype=torch.bool))

    @torch.no_grad()
    def init_embed_(self, samples: torch.Tensor):
        if bool(self.inited.item()):
            return
        samples = samples.detach().float()
        if samples.numel() == 0:
            return
        embed, bins = _kmeans(samples, self.codebook_size, self.kmeans_iters)
        self.weight.copy_(embed.to(dtype=self.weight.dtype))
        self.embed_avg.copy_(embed.to(dtype=self.embed_avg.dtype))
        self.cluster_size.copy_(bins.to(dtype=self.cluster_size.dtype))
        self.inited.fill_(True)
        _broadcast_from_rank0_(self.weight)
        _broadcast_from_rank0_(self.embed_avg)
        _broadcast_from_rank0_(self.cluster_size)
        _broadcast_from_rank0_(self.inited)

    def distances(self, samples: torch.Tensor) -> torch.Tensor:
        return (
            samples.pow(2).sum(dim=1, keepdim=True)
            - 2 * samples @ self.weight.t()
            + self.weight.pow(2).sum(dim=1, keepdim=True).t()
        )

    @torch.no_grad()
    def ema_update_(self, samples: torch.Tensor, indices: torch.Tensor):
        if not self.training:
            return
        samples = samples.detach().float()
        indices = indices.detach().reshape(-1)
        if samples.numel() == 0:
            return

        counts = torch.bincount(indices, minlength=self.codebook_size).to(dtype=samples.dtype)
        sums = samples.new_zeros(self.codebook_size, self.codebook_dim)
        sums.index_add_(0, indices, samples)
        _all_reduce_sum_(counts)
        _all_reduce_sum_(sums)

        self.cluster_size.mul_(self.decay).add_(counts.to(self.cluster_size.dtype), alpha=1.0 - self.decay)
        self.embed_avg.mul_(self.decay).add_(sums.to(self.embed_avg.dtype), alpha=1.0 - self.decay)

        n = self.cluster_size.sum()
        smoothed = (
            (self.cluster_size + self.epsilon)
            / (n + self.codebook_size * self.epsilon).clamp_min(self.epsilon)
            * n.clamp_min(self.epsilon)
        )
        self.weight.copy_(self.embed_avg / smoothed.unsqueeze(1).clamp_min(self.epsilon))

        if self.threshold_ema_dead_code > 0:
            expired = self.cluster_size < self.threshold_ema_dead_code
            if bool(expired.any().item()):
                replacements = _sample_rows(samples, self.codebook_size).to(dtype=self.weight.dtype)
                self.weight[expired] = replacements[expired]
                self.embed_avg[expired] = replacements[expired] * smoothed[expired].unsqueeze(1)


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
        upsample_mode: str = "linear",
        guided_upsample: bool = False,
        guided_upsample_hidden_dim: Optional[int] = None,
        guided_upsample_kernel: int = 5,
        guided_upsample_detach_guide: bool = True,
        guided_upsample_init_scale: float = 1e-3,
        decoder_grad_alpha: float = 0.0,
        guided_upsample_decoder_grad_alpha: float = 0.0,
        separate_lookup_codebook: bool = False,
        lookup_commitment_weight: float = 1.0,
        lookup_codebook_weight: float = 1.0,
        learned_downsample: bool = False,
        learned_downsample_hidden_dim: Optional[int] = None,
        learned_downsample_kernel: int = 3,
        learned_downsample_init_scale: float = 1e-3,
        learned_downsample_max_scale: float = 0.2,
        learned_upsample: bool = False,
        learned_upsample_mode: str = "patch",
        learned_upsample_hidden_dim: Optional[int] = None,
        learned_upsample_kernel: int = 3,
        learned_upsample_patch_size: Union[int, list] = 4,
        learned_upsample_init_scale: float = 1e-3,
        learned_upsample_condition_guide: bool = False,
        learned_upsample_detach_guide: bool = True,
        learned_upsample_guide_space: str = "latent_proj",
        learned_upsample_max_scale: Union[float, list] = 1.0,
        use_projection: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.use_projection = bool(use_projection)
        self.separate_lookup_codebook = bool(separate_lookup_codebook)
        self.lookup_commitment_weight = float(lookup_commitment_weight)
        self.lookup_codebook_weight = float(lookup_codebook_weight)
        self.pooling = self._normalize_pooling_name(pooling)
        self.pooling_alpha = float(pooling_alpha)
        self.pooling_power = float(pooling_power)
        self.pooling_eps = float(pooling_eps)
        self.upsample_mode = self._normalize_upsample_mode(upsample_mode)
        self.use_guided_upsample = bool(guided_upsample)
        self.use_learned_downsample = bool(learned_downsample)
        self.use_learned_upsample = bool(learned_upsample)
        self.learned_upsample_mode = self._normalize_learned_upsample_mode(learned_upsample_mode)
        self.learned_upsample_detach_guide = bool(learned_upsample_detach_guide)
        self.learned_upsample_guide_space = self._normalize_learned_upsample_guide_space(
            learned_upsample_guide_space
        )
        self.decoder_grad_alpha = (
            float(decoder_grad_alpha)
            if float(decoder_grad_alpha) != 0.0
            else float(guided_upsample_decoder_grad_alpha)
        )
        self.guided_upsample_decoder_grad_alpha = float(guided_upsample_decoder_grad_alpha)

        if self.use_projection:
            self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
            self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        else:
            if int(input_dim) != int(codebook_dim):
                raise ValueError(
                    "use_projection=False requires input_dim == codebook_dim, "
                    f"got input_dim={input_dim}, codebook_dim={codebook_dim}"
                )
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()

        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        if self.separate_lookup_codebook:
            self.lookup_codebook = nn.Embedding(codebook_size, codebook_dim)
            with torch.no_grad():
                self.lookup_codebook.weight.copy_(self.codebook.weight)
        self.learned_downsampler = (
            LearnedResidualDownsample(
                codebook_dim=codebook_dim,
                hidden_dim=learned_downsample_hidden_dim,
                kernel_size=learned_downsample_kernel,
                init_scale=learned_downsample_init_scale,
                max_scale=learned_downsample_max_scale,
            )
            if learned_downsample
            else None
        )
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
        learned_upsampler_cls = LearnedFractionalUpsample if self.learned_upsample_mode == "fractional" else LearnedPatchUpsample
        learned_upsampler_kwargs = dict(
            input_dim=input_dim,
            codebook_dim=codebook_dim,
            hidden_dim=learned_upsample_hidden_dim,
            kernel_size=learned_upsample_kernel,
            init_scale=learned_upsample_init_scale,
            condition_guide=learned_upsample_condition_guide,
            detach_guide=learned_upsample_detach_guide,
            max_scale=learned_upsample_max_scale,
            guide_is_codebook_dim=self.learned_upsample_guide_space == "current_in_proj",
        )
        if self.learned_upsample_mode == "patch":
            learned_upsampler_kwargs["patch_size"] = learned_upsample_patch_size
        self.learned_upsampler = (
            learned_upsampler_cls(
                **learned_upsampler_kwargs,
            )
            if learned_upsample
            else None
        )
        self.last_upsampler_metrics = {}
        self.last_downsampler_metrics = {}

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
    def _normalize_upsample_mode(mode: str) -> str:
        mode = str(mode).lower().replace("-", "_")
        aliases = {
            "default": "linear",
            "interp": "linear",
            "interpolate": "linear",
            "nearest_neighbor": "nearest",
            "repeat_interleave": "repeat",
            "snac": "repeat",
        }
        mode = aliases.get(mode, mode)
        valid = {"linear", "nearest", "repeat"}
        if mode not in valid:
            raise ValueError(f"Unknown quantizer upsample mode '{mode}'. Expected one of {sorted(valid)}.")
        return mode

    @staticmethod
    def _normalize_learned_upsample_mode(mode: str) -> str:
        mode = str(mode).lower().replace("-", "_")
        aliases = {
            "default": "patch",
            "patch_size": "patch",
            "patches": "patch",
            "continuous": "fractional",
            "frac": "fractional",
            "position": "fractional",
            "positioned": "fractional",
        }
        mode = aliases.get(mode, mode)
        valid = {"patch", "fractional"}
        if mode not in valid:
            raise ValueError(f"Unknown learned_upsample_mode '{mode}'. Expected one of {sorted(valid)}.")
        return mode

    @staticmethod
    def _normalize_learned_upsample_guide_space(guide_space: str) -> str:
        guide_space = str(guide_space).lower().replace("-", "_")
        aliases = {
            "default": "latent_proj",
            "latent": "latent_proj",
            "guide_proj": "latent_proj",
            "proj": "latent_proj",
            "in_proj": "current_in_proj",
            "current": "current_in_proj",
            "current_stage": "current_in_proj",
            "current_stage_in_proj": "current_in_proj",
        }
        guide_space = aliases.get(guide_space, guide_space)
        valid = {"latent_proj", "current_in_proj"}
        if guide_space not in valid:
            raise ValueError(
                f"Unknown learned_upsample_guide_space '{guide_space}'. Expected one of {sorted(valid)}."
            )
        return guide_space

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
        base = self.downsample_latents_base(x, scale)
        if (
            self.learned_downsampler is not None
            and self.use_learned_downsample
            and scale is not None
            and int(scale) != x.shape[-1]
        ):
            return self.learned_downsampler(x, int(scale), base=base)
        return base

    def downsample_latents_base(self, x: torch.Tensor, scale: Optional[int]) -> torch.Tensor:
        if self.pooling == "area":
            return self._area_pool(x, scale)
        if self.pooling == "norm_weighted":
            return self._norm_weighted_pool(x, scale)
        mean = self._area_pool(x, scale)
        weighted = self._norm_weighted_pool(x, scale)
        alpha = min(1.0, max(0.0, self.pooling_alpha))
        return mean.lerp(weighted, alpha)

    @torch.no_grad()
    def _record_downsampler_metrics(self, z_e: torch.Tensor, inter_z: torch.Tensor, base: torch.Tensor):
        self.last_downsampler_metrics = {}
        downsampler = getattr(self, "learned_downsampler", None)
        if downsampler is None or not getattr(self, "use_learned_downsample", False):
            return
        if inter_z.shape[-1] == z_e.shape[-1]:
            return
        residual = inter_z.detach() - base.detach()
        self.last_downsampler_metrics["residual_rel"] = (
            residual.pow(2).mean().sqrt() / base.detach().pow(2).mean().sqrt().clamp_min(1e-8)
        )
        self.last_downsampler_metrics["residual_scale"] = downsampler.residual_scale_value().detach().abs()

    def lookup_codebook_weight_tensor(self) -> torch.Tensor:
        lookup_codebook = getattr(self, "lookup_codebook", None)
        return self.codebook.weight if lookup_codebook is None else lookup_codebook.weight

    def codebook_distances(self, encodings: torch.Tensor) -> torch.Tensor:
        codebook = self.lookup_codebook_weight_tensor()
        encodings = F.normalize(encodings, dim=1)
        codebook = F.normalize(codebook, dim=1)
        return (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )

    def upsample_code(
        self,
        x: torch.Tensor,
        size: int,
        guide: Optional[torch.Tensor] = None,
        use_guide: bool = True,
    ) -> torch.Tensor:
        size = int(size)
        if x.shape[-1] == size:
            return x.contiguous()
        if self.learned_upsampler is not None and self.use_learned_upsample:
            return self.learned_upsampler(x, size=size, guide=guide)
        if self.guided_upsampler is not None and guide is not None and use_guide and self.use_guided_upsample:
            return self.guided_upsampler(x, size=size, guide=guide)
        if self.upsample_mode == "repeat" and size >= x.shape[-1]:
            repeats = max(1, math.ceil(size / x.shape[-1]))
            return x.repeat_interleave(repeats, dim=-1)[..., :size].contiguous()
        if self.upsample_mode == "nearest":
            return F.interpolate(x, size=size, mode="nearest").contiguous()
        return F.interpolate(x, size=size, mode="linear").contiguous()

    @torch.no_grad()
    def _record_upsampler_metrics(
        self,
        z_q_small: torch.Tensor,
        z_q_up: torch.Tensor,
        z_e: torch.Tensor,
        conv=None,
        guide: Optional[torch.Tensor] = None,
    ):
        self.last_upsampler_metrics = {}
        upsampler = getattr(self, "learned_upsampler", None)
        if upsampler is None or not getattr(self, "use_learned_upsample", False):
            return
        if z_q_small.shape[-1] == z_e.shape[-1]:
            return

        eps = 1e-8
        interp = F.interpolate(z_q_small.detach(), size=z_e.shape[-1], mode="linear").contiguous()
        learned = z_q_up.detach()
        residual = learned - interp
        self.last_upsampler_metrics["residual_rel"] = (
            residual.pow(2).mean().sqrt() / interp.pow(2).mean().sqrt().clamp_min(eps)
        )

        zero_guide = None if guide is None else torch.zeros_like(guide)
        no_guide = upsampler(z_q_small.detach(), size=z_e.shape[-1], guide=zero_guide).detach()
        self.last_upsampler_metrics["guide_sensitivity"] = (
            (learned - no_guide).pow(2).mean().sqrt()
            / learned.pow(2).mean().sqrt().clamp_min(eps)
        )

        interp_for_loss = conv(interp) if conv is not None else interp
        learned_for_loss = conv(learned) if conv is not None else learned
        target = z_e.detach()
        interp_mse = F.mse_loss(interp_for_loss, target)
        learned_mse = F.mse_loss(learned_for_loss, target)
        self.last_upsampler_metrics["interp_gain"] = (
            (interp_mse - learned_mse) / interp_mse.clamp_min(eps)
        )
        self.last_upsampler_metrics["residual_scale"] = upsampler.residual_scale_value().detach().abs()

    def forward(
        self,
        z,
        scale=None,
        conv=None,
        guide: Optional[torch.Tensor] = None,
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

        inter_z_base = self.downsample_latents_base(z_e, scale)
        if (
            self.learned_downsampler is not None
            and self.use_learned_downsample
            and scale is not None
            and int(scale) != z_e.shape[-1]
        ):
            inter_z = self.learned_downsampler(z_e, int(scale), base=inter_z_base)
        else:
            inter_z = inter_z_base
        self._record_downsampler_metrics(z_e, inter_z, inter_z_base)
        z_q_small, indices = self.decode_latents(inter_z)
        
        use_guide = scale is not None and int(scale) != H
        guide_for_upsample = guide
        if (
            guide is not None
            and self.learned_upsampler is not None
            and self.use_learned_upsample
            and self.learned_upsample_guide_space == "current_in_proj"
        ):
            guide_for_upsample = self.in_proj(guide)
        z_q_up = self.upsample_code(z_q_small, H, guide=guide_for_upsample, use_guide=use_guide) if scale != None else z_q_small.contiguous()
        z_q = conv(z_q_up) if conv != None else z_q_up
        self._record_upsampler_metrics(z_q_small, z_q_up, z_e, conv=conv, guide=guide_for_upsample)
        
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
        self.last_lookup_loss = None
        if self.separate_lookup_codebook:
            z_lookup_small = self.decode_lookup_code(indices)
            lookup_commitment_loss = F.mse_loss(
                inter_z,
                z_lookup_small.detach(),
                reduction="none",
            ).mean([1, 2])
            lookup_codebook_loss = F.mse_loss(
                z_lookup_small,
                inter_z.detach(),
                reduction="none",
            ).mean([1, 2])
            commitment_loss = (
                commitment_loss
                + self.lookup_commitment_weight * lookup_commitment_loss
            )
            codebook_loss = (
                codebook_loss
                + self.lookup_codebook_weight * lookup_codebook_loss
            )
            self.last_lookup_loss = lookup_codebook_loss.detach().mean()

        z_q_ste = z_e + (z_q - z_e).detach()
        if self.decoder_grad_alpha != 0.0:
            z_q_ste = z_q_ste + self.decoder_grad_alpha * (z_q - z_q.detach())

        z_q = self.out_proj(z_q_ste)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        quantize = self.embed_code(embed_id).transpose(1, 2)
        return quantize

    def embed_lookup_code(self, embed_id):
        return F.embedding(embed_id, self.lookup_codebook_weight_tensor())

    def decode_lookup_code(self, embed_id):
        return self.embed_lookup_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        dist = self.codebook_distances(encodings)
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class EMAMultiscaleVectorQuantize(MultiscaleVectorQuantize):
    """Multiscale VQ stage with an EMA-updated Euclidean codebook.

    The forward contract intentionally matches ``MultiscaleVectorQuantize`` so
    it can be swapped per-stage by ``WavescaleResidualVectorQuantize``.
    """

    def __init__(
        self,
        *args,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
        ema_kmeans_init: bool = True,
        ema_kmeans_iters: int = 50,
        ema_threshold_ema_dead_code: int = 2,
        **kwargs,
    ):
        if kwargs.get("separate_lookup_codebook", False):
            raise ValueError("EMA codebook update does not support separate_lookup_codebook")
        super().__init__(*args, **kwargs)
        self.codebook_update = "ema"
        self.ema_decay = float(ema_decay)
        self.ema_epsilon = float(ema_epsilon)
        self.ema_kmeans_init = bool(ema_kmeans_init)
        self.ema_kmeans_iters = int(ema_kmeans_iters)
        self.ema_threshold_ema_dead_code = int(ema_threshold_ema_dead_code)
        self.codebook = ExponentialMovingAverageCodebook(
            codebook_size=self.codebook_size,
            codebook_dim=self.codebook_dim,
            decay=self.ema_decay,
            epsilon=self.ema_epsilon,
            kmeans_init=self.ema_kmeans_init,
            kmeans_iters=self.ema_kmeans_iters,
            threshold_ema_dead_code=self.ema_threshold_ema_dead_code,
        )

    def codebook_distances(self, encodings: torch.Tensor) -> torch.Tensor:
        return self.codebook.distances(encodings)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        self.codebook.init_embed_(encodings)
        dist = self.codebook_distances(encodings)
        indices = rearrange(dist.argmin(dim=1), "(b t) -> b t", b=latents.size(0))
        if self.training:
            self.codebook.ema_update_(encodings, indices)
        z_q = self.decode_code(indices)
        return z_q, indices

    def forward(self, *args, **kwargs):
        z_q, commitment_loss, codebook_loss, indices, z_e = super().forward(*args, **kwargs)
        # EMA updates codebook entries directly from assignment statistics.
        # Keep only the learnable post-code regression path, e.g. Phi residual
        # convolutions or learned upsamplers; if no such path exists this term
        # has no gradients and is safely zeroed for cleaner logging.
        if not codebook_loss.requires_grad:
            codebook_loss = torch.zeros_like(codebook_loss)
        return z_q, commitment_loss, codebook_loss, indices, z_e


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

        for i, quantizer in enumerate(self.quantizers):
            
            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual)

            z_q = z_q + z_q_i
            residual = residual - z_q_i

            # Sum losses
            commitment_loss.append(commitment_loss_i.mean()) 
            codebook_loss.append(codebook_loss_i.mean())

            codebook_indices.append(indices_i)
            latents.append(z_e_i)
        
        return z_q, codebook_indices, latents, commitment_loss, codebook_loss, z.new_tensor(0.0)

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
        quantizer_upsample_mode: str = "linear",
        quantizer_codebook_update: str = "gradient",
        quantizer_decoder_grad_alpha: float = 0.0,
        quantizer_residual_detach: bool = False,
        quantizer_shared_projection: bool = False,
        quantizer_separate_lookup_codebook: bool = False,
        quantizer_lookup_commitment_weight: float = 1.0,
        quantizer_lookup_codebook_weight: float = 1.0,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
        ema_kmeans_init: bool = True,
        ema_kmeans_iters: int = 50,
        ema_threshold_ema_dead_code: int = 2,
        quantizer_scale_anneal: bool = False,
        quantizer_scale_anneal_steps: int = 30000,
        quantizer_scale_anneal_start: float = 1.0,
        quantizer_scale_anneal_mode: str = "cosine",
        learned_downsample: bool = False,
        learned_downsample_hidden_dim: Optional[int] = None,
        learned_downsample_kernel: int = 3,
        learned_downsample_init_scale: float = 1e-3,
        learned_downsample_max_scale: Union[float, list] = 0.2,
        guided_upsample: bool = False,
        guided_upsample_hidden_dim: Optional[int] = None,
        guided_upsample_kernel: int = 5,
        guided_upsample_detach_guide: bool = True,
        guided_upsample_init_scale: float = 1e-3,
        guided_upsample_after_pivot_only: bool = False,
        guided_upsample_decoder_grad_alpha: float = 0.0,
        learned_upsample: bool = False,
        learned_upsample_mode: str = "patch",
        learned_upsample_hidden_dim: Optional[int] = None,
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
        self.quantizer_upsample_mode = quantizer_upsample_mode
        self.quantizer_codebook_update = self._normalize_codebook_update(quantizer_codebook_update)
        self.quantizer_decoder_grad_alpha = float(quantizer_decoder_grad_alpha)
        self.quantizer_residual_detach = bool(quantizer_residual_detach)
        self.quantizer_shared_projection = bool(quantizer_shared_projection)
        self.quantizer_separate_lookup_codebook = bool(quantizer_separate_lookup_codebook)
        self.quantizer_lookup_commitment_weight = float(quantizer_lookup_commitment_weight)
        self.quantizer_lookup_codebook_weight = float(quantizer_lookup_codebook_weight)
        self.ema_decay = float(ema_decay)
        self.ema_epsilon = float(ema_epsilon)
        self.ema_kmeans_init = bool(ema_kmeans_init)
        self.ema_kmeans_iters = int(ema_kmeans_iters)
        self.ema_threshold_ema_dead_code = int(ema_threshold_ema_dead_code)
        self.quantizer_scale_anneal = bool(quantizer_scale_anneal)
        self.quantizer_scale_anneal_steps = max(1, int(quantizer_scale_anneal_steps))
        self.quantizer_scale_anneal_start = float(quantizer_scale_anneal_start)
        self.quantizer_scale_anneal_mode = self._normalize_scale_anneal_mode(quantizer_scale_anneal_mode)
        self.learned_downsample = bool(learned_downsample)
        self.learned_downsample_hidden_dim = learned_downsample_hidden_dim
        self.learned_downsample_kernel = int(learned_downsample_kernel)
        self.learned_downsample_init_scale = float(learned_downsample_init_scale)
        self.learned_downsample_max_scale = learned_downsample_max_scale
        self.guided_upsample = bool(guided_upsample)
        self.guided_upsample_hidden_dim = guided_upsample_hidden_dim
        self.guided_upsample_kernel = int(guided_upsample_kernel)
        self.guided_upsample_detach_guide = bool(guided_upsample_detach_guide)
        self.guided_upsample_init_scale = float(guided_upsample_init_scale)
        self.guided_upsample_after_pivot_only = bool(guided_upsample_after_pivot_only)
        self.guided_upsample_decoder_grad_alpha = float(guided_upsample_decoder_grad_alpha)
        self.learned_upsample = bool(learned_upsample)
        self.learned_upsample_mode = learned_upsample_mode
        self.learned_upsample_hidden_dim = learned_upsample_hidden_dim
        self.learned_upsample_kernel = int(learned_upsample_kernel)
        self.learned_upsample_patch_size = learned_upsample_patch_size
        self.learned_upsample_init_scale = float(learned_upsample_init_scale)
        self.learned_upsample_condition_guide = bool(learned_upsample_condition_guide)
        self.learned_upsample_detach_guide = bool(learned_upsample_detach_guide)
        self.learned_upsample_guide_space = learned_upsample_guide_space
        self.learned_upsample_max_scale = learned_upsample_max_scale
        self.learned_upsample_after_pivot_only = bool(learned_upsample_after_pivot_only)
        self.learned_upsample_log_stages = bool(learned_upsample_log_stages)
        self.scale_factors, self.n_codebooks = self._compute_wavescale(scale_factors) if use_wavescale else (scale_factors, len(scale_factors))
        self.offset = len(scale_factors) - 1
        if self.quantizer_shared_projection:
            if isinstance(codebook_dim, (list, tuple)):
                raise ValueError("quantizer_shared_projection=True requires a single integer codebook_dim")
            projected_dim = int(codebook_dim)
            self.shared_project_in = (
                nn.Conv1d(input_dim, projected_dim, kernel_size=1)
                if int(input_dim) != projected_dim
                else nn.Identity()
            )
            self.shared_project_out = (
                nn.Conv1d(projected_dim, input_dim, kernel_size=1)
                if int(input_dim) != projected_dim
                else nn.Identity()
            )
            stage_input_dim = projected_dim
            stage_codebook_dim = projected_dim
            stage_use_projection = False
            phi_dim = projected_dim
        else:
            self.shared_project_in = nn.Identity()
            self.shared_project_out = nn.Identity()
            stage_input_dim = input_dim
            stage_codebook_dim = codebook_dim
            stage_use_projection = True
            phi_dim = codebook_dim
        learned_patch_sizes = self._stage_values(
            learned_upsample_patch_size,
            name="learned_upsample_patch_size",
            use_wavescale=use_wavescale,
            base_length=len(scale_factors),
        )
        learned_max_scales = self._stage_values(
            learned_upsample_max_scale,
            name="learned_upsample_max_scale",
            use_wavescale=use_wavescale,
            base_length=len(scale_factors),
        )
        learned_downsample_max_scales = self._stage_values(
            learned_downsample_max_scale,
            name="learned_downsample_max_scale",
            use_wavescale=use_wavescale,
            base_length=len(scale_factors),
        )
        self.register_buffer("training_step", torch.zeros((), dtype=torch.long), persistent=False)
        self.last_scale_anneal_metrics = {}
        quantizer_cls = (
            EMAMultiscaleVectorQuantize
            if self.quantizer_codebook_update == "ema"
            else MultiscaleVectorQuantize
        )
        if self.quantizer_codebook_update == "ema" and quantizer_separate_lookup_codebook:
            raise ValueError("quantizer_codebook_update='ema' does not support separate lookup codebooks")

        self.quantizers = nn.ModuleList(
            [
                quantizer_cls(
                    stage_input_dim,
                    codebook_size,
                    stage_codebook_dim,
                    pooling=quantizer_pooling,
                    pooling_alpha=quantizer_pooling_alpha,
                    pooling_power=quantizer_pooling_power,
                    upsample_mode=quantizer_upsample_mode,
                    decoder_grad_alpha=quantizer_decoder_grad_alpha,
                    separate_lookup_codebook=quantizer_separate_lookup_codebook,
                    lookup_commitment_weight=quantizer_lookup_commitment_weight,
                    lookup_codebook_weight=quantizer_lookup_codebook_weight,
                    learned_downsample=learned_downsample,
                    learned_downsample_hidden_dim=learned_downsample_hidden_dim,
                    learned_downsample_kernel=learned_downsample_kernel,
                    learned_downsample_init_scale=learned_downsample_init_scale,
                    learned_downsample_max_scale=float(learned_downsample_max_scales[i]),
                    guided_upsample=guided_upsample,
                    guided_upsample_hidden_dim=guided_upsample_hidden_dim,
                    guided_upsample_kernel=guided_upsample_kernel,
                    guided_upsample_detach_guide=guided_upsample_detach_guide,
                    guided_upsample_init_scale=guided_upsample_init_scale,
                    guided_upsample_decoder_grad_alpha=guided_upsample_decoder_grad_alpha,
                    learned_upsample=learned_upsample,
                    learned_upsample_mode=learned_upsample_mode,
                    learned_upsample_hidden_dim=learned_upsample_hidden_dim,
                    learned_upsample_kernel=learned_upsample_kernel,
                    learned_upsample_patch_size=max(2, int(learned_patch_sizes[i])),
                    learned_upsample_init_scale=learned_upsample_init_scale,
                    learned_upsample_condition_guide=learned_upsample_condition_guide,
                    learned_upsample_detach_guide=learned_upsample_detach_guide,
                    learned_upsample_guide_space=learned_upsample_guide_space,
                    learned_upsample_max_scale=float(learned_max_scales[i]),
                    use_projection=stage_use_projection,
                    **(
                        {
                            "ema_decay": self.ema_decay,
                            "ema_epsilon": self.ema_epsilon,
                            "ema_kmeans_init": self.ema_kmeans_init,
                            "ema_kmeans_iters": self.ema_kmeans_iters,
                            "ema_threshold_ema_dead_code": self.ema_threshold_ema_dead_code,
                        }
                        if self.quantizer_codebook_update == "ema"
                        else {}
                    ),
                )
                for i in range(self.n_codebooks)
            ]
        )
        for quantizer, patch_size in zip(self.quantizers, learned_patch_sizes):
            quantizer.learned_upsample_stage_patch_size = int(patch_size)
        self.configure_guided_upsample_stages()
        self.configure_learned_downsample_stages()
        self.configure_learned_upsample_stages()
        if phi_kernel is not None:
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([Phi(phi_dim, 0.5, ks=ks) for ks in phi_kernel]))
        else:
            self.quant_resi = None
        self.quantizer_dropout = quantizer_dropout

        self.ar_input_seq = get_stage_order(scale_factors=self.scale_factors)
        self.ar_stage_blocks = get_stage_blocks(scale_factors=self.scale_factors, order=self.ar_input_seq)

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

    @staticmethod
    def _normalize_codebook_update(mode: str) -> str:
        mode = str(mode).lower().replace("-", "_")
        aliases = {
            "default": "gradient",
            "grad": "gradient",
            "sgd": "gradient",
            "learned": "gradient",
            "exp_moving_average": "ema",
            "moving_average": "ema",
            "ema_codebook": "ema",
        }
        mode = aliases.get(mode, mode)
        valid = {"gradient", "ema"}
        if mode not in valid:
            raise ValueError(f"Unknown quantizer_codebook_update '{mode}'. Expected one of {sorted(valid)}.")
        return mode

    @staticmethod
    def _normalize_scale_anneal_mode(mode: str) -> str:
        mode = str(mode or "cosine").lower().replace("-", "_")
        aliases = {
            "cos": "cosine",
            "linear_ramp": "linear",
            "lin": "linear",
        }
        mode = aliases.get(mode, mode)
        valid = {"linear", "cosine"}
        if mode not in valid:
            raise ValueError(f"Unknown quantizer_scale_anneal_mode '{mode}'. Expected one of {sorted(valid)}.")
        return mode

    def set_training_step(self, step: Optional[int]):
        if step is None:
            step = 0
        self.training_step.fill_(max(0, int(step)))

    def _scale_anneal_alpha(self) -> float:
        if not self.training or not self.quantizer_scale_anneal:
            return 1.0
        progress = float(self.training_step.item()) / float(self.quantizer_scale_anneal_steps)
        progress = min(1.0, max(0.0, progress))
        if self.quantizer_scale_anneal_mode == "cosine":
            return 0.5 - 0.5 * math.cos(math.pi * progress)
        return progress

    def current_scale_factors(self) -> list[float]:
        alpha = self._scale_anneal_alpha()
        start = float(self.quantizer_scale_anneal_start)
        if not self.quantizer_scale_anneal or alpha >= 1.0:
            scales = [float(s) for s in self.scale_factors]
        else:
            scales = [
                float(target) + (start - float(target)) * (1.0 - alpha)
                for target in self.scale_factors
            ]
        if self.training and self.quantizer_scale_anneal:
            device = self.training_step.device
            self.last_scale_anneal_metrics = {
                "scale_anneal/alpha": torch.tensor(alpha, device=device),
                "scale_anneal/min_scale": torch.tensor(min(scales), device=device),
            }
        else:
            self.last_scale_anneal_metrics = {}
        return [min(1.0, max(1e-6, s)) for s in scales]

    @staticmethod
    def _stage_values(value, name: str, use_wavescale: bool, base_length: int):
        if isinstance(value, (list, tuple)):
            values = list(value)
            expected_full = base_length * 2 - 1 if use_wavescale else base_length
            if len(values) == expected_full:
                return values
            if use_wavescale and len(values) == base_length:
                return values[::-1] + values[1:]
            raise ValueError(
                f"{name} must have length {base_length} or {expected_full}, got {len(values)}"
            )
        return [value] * (base_length * 2 - 1 if use_wavescale else base_length)

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

    def configure_learned_downsample_stages(self):
        """Enable learned analysis residuals only on downscaled stages."""
        enabled = bool(getattr(self, "learned_downsample", False))
        self.learned_downsample_stage_mask = [
            enabled and float(scale) < 1.0
            for scale in self.scale_factors
        ]
        for active, quantizer in zip(self.learned_downsample_stage_mask, self.quantizers):
            if hasattr(quantizer, "use_learned_downsample"):
                quantizer.use_learned_downsample = bool(active)
            downsampler = getattr(quantizer, "learned_downsampler", None)
            if downsampler is not None:
                for param in downsampler.parameters():
                    param.requires_grad_(bool(active))
        return self.learned_downsample_stage_mask

    def configure_learned_upsample_stages(self):
        """Enable learned patch synthesis only on selected interpolating stages."""
        enabled = bool(getattr(self, "learned_upsample", False))
        after_pivot_only = bool(getattr(self, "learned_upsample_after_pivot_only", False))
        self.learned_upsample_stage_mask = [
            enabled
            and (not after_pivot_only or stage_idx > self.offset)
            and float(scale) < 1.0
            and int(getattr(quantizer, "learned_upsample_stage_patch_size", 0)) > 1
            for stage_idx, scale in enumerate(self.scale_factors)
            for quantizer in [self.quantizers[stage_idx]]
        ]
        for active, quantizer in zip(self.learned_upsample_stage_mask, self.quantizers):
            if hasattr(quantizer, "use_learned_upsample"):
                quantizer.use_learned_upsample = bool(active)
            upsampler = getattr(quantizer, "learned_upsampler", None)
            if upsampler is not None:
                for param in upsampler.parameters():
                    param.requires_grad_(bool(active))
        return self.learned_upsample_stage_mask

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
        z_in = self.shared_project_in(z)
        T = z_in.shape[-1]
        z_q = 0
        
        residual = z_in
        commitment_loss = []
        codebook_loss = []

        codebook_indices = []
        latents = []
        downsampler_stage_metrics = []
        upsampler_stage_metrics = []
        current_scales = self.current_scale_factors()
        
        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        elif not self.training:
            n_quantizers = int(n_quantizers)
            if not 1 <= n_quantizers <= self.n_codebooks:
                raise ValueError(f"n_quantizers must be in [1, {self.n_codebooks}], got {n_quantizers}")
        if self.training:
            stage_mask = self._training_dropout_stage_mask(z_in.shape[0], z_in.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break
            scale = int(current_scales[i] * T)
            scale += 1 if scale == 0 else 0
            guide = z_q if torch.is_tensor(z_q) else None

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual,
                scale,
                (lambda h, stage_idx=i: self._apply_quant_resi(stage_idx, h)) if self.quant_resi is not None else None,
                guide=guide,
            )
            
            # Create mask to apply quantizer dropout
            if self.training:
                mask = stage_mask[:, i].to(dtype=z_q_i.dtype)
            else:
                mask = torch.ones((z_in.shape[0],), dtype=z_q_i.dtype, device=z_in.device)

            z_q = z_q + z_q_i * mask[:, None, None]
            stage_downsampler_metrics = dict(getattr(quantizer, "last_downsampler_metrics", {}) or {})
            if stage_downsampler_metrics:
                downsampler_stage_metrics.append((i, stage_downsampler_metrics))
            stage_upsampler_metrics = dict(getattr(quantizer, "last_upsampler_metrics", {}) or {})
            if stage_upsampler_metrics:
                with torch.no_grad():
                    contribution = (z_q_i * mask[:, None, None]).detach()
                    cumulative = z_q.detach()
                    stage_upsampler_metrics["contribution_rel"] = (
                        contribution.pow(2).mean().sqrt()
                        / cumulative.pow(2).mean().sqrt().clamp_min(1e-8)
                    )
                upsampler_stage_metrics.append((i, stage_upsampler_metrics))
            
            residual_update = z_q_i.detach() if self.quantizer_residual_detach else z_q_i
            residual = residual - residual_update * mask[:, None, None]

            # Sum losses over active quantizers only, matching DAC's quantizer
            # dropout behavior.  Dropped stages do not contribute to the final
            # reconstruction for that sample, so their VQ losses should not
            # contribute either.
            commitment_loss.append((commitment_loss_i * mask).mean())
            codebook_loss.append((codebook_loss_i * mask).mean())

            codebook_indices.append(indices_i)
            latents.append(z_e_i)
        
        self.last_upsampler_stage_metrics = self._format_upsampler_stage_metrics(upsampler_stage_metrics)
        self.last_downsampler_metrics = self._summarize_named_stage_metrics(
            downsampler_stage_metrics,
            z_in.device,
            prefix="downsampler",
        )
        self.last_upsampler_metrics = self._summarize_upsampler_metrics(
            upsampler_stage_metrics,
            z_in.device,
            log_stages=self.learned_upsample_log_stages,
        )
        
        # end = time.time()

        # print(end-start)
        z_q = self.shared_project_out(z_q)
        return z_q, codebook_indices, latents, commitment_loss, codebook_loss, z.new_tensor(0.0)

    @staticmethod
    def _format_upsampler_stage_metrics(stage_metrics):
        formatted = []
        for stage_idx, metrics in stage_metrics:
            formatted.append(
                {
                    "stage": int(stage_idx),
                    **{
                        name: float(value.detach().float().cpu())
                        for name, value in metrics.items()
                    },
                }
            )
        return formatted

    @staticmethod
    def _summarize_upsampler_metrics(stage_metrics, device, log_stages: bool = False):
        return WavescaleResidualVectorQuantize._summarize_named_stage_metrics(
            stage_metrics,
            device,
            prefix="upsampler",
            log_stages=log_stages,
        )

    @staticmethod
    def _summarize_named_stage_metrics(
        stage_metrics,
        device,
        prefix: str,
        log_stages: bool = False,
        include_max: bool = False,
        include_active_stages: bool = True,
    ):
        if not stage_metrics:
            return {}
        out = {}
        metric_names = sorted({name for _, metrics in stage_metrics for name in metrics.keys()})
        for name in metric_names:
            values = []
            for stage_idx, metrics in stage_metrics:
                if name not in metrics:
                    continue
                value = metrics[name].detach()
                if log_stages:
                    out[f"{prefix}/{name}_s{stage_idx:02d}"] = value
                values.append(value)
            if values:
                stacked = torch.stack(values)
                out[f"{prefix}/{name}_mean"] = stacked.mean()
                if include_max:
                    out[f"{prefix}/{name}_max"] = stacked.max()
        if include_active_stages:
            out[f"{prefix}/active_stages"] = torch.tensor(float(len(stage_metrics)), device=device)
        return out

    def from_codes(
        self,
        codes: list[torch.Tensor],
        depth='full',
    ):
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

        return self.shared_project_out(z_q), z_p, codes

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

        def stage_to_full(stage_idx: int, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
            quantizer = self.quantizers[stage_idx]
            z_q_i = quantizer.decode_code(codes[stage_idx])
            z_q_i = quantizer.upsample_code(z_q_i, T, guide=guide, use_guide=z_q_i.shape[-1] != T)
            return quantizer.out_proj(self._apply_quant_resi(stage_idx, z_q_i))

        if use_scalewise and use_blockwise:
            # Blockwise same-scale AAR:
            #   [pivot] -> [same-scale pair] -> [same-scale pair] -> ...
            # All stages in the next same-scale block receive the same previous-block
            # reconstruction context, rather than seeing earlier stages from their own
            # block.  Example: [2] -> [1, 3] -> [0, 4].
            for block_idx, block in enumerate(self.ar_stage_blocks[:-1]):
                block_guide = z_q if torch.is_tensor(z_q) else None
                for i in block:
                    z_q = z_q + stage_to_full(i, guide=block_guide)

                for next_i in self.ar_stage_blocks[block_idx + 1]:
                    out.append(F.interpolate(z_q, int(self.scale_factors[next_i] * T), mode='area').transpose(1, 2))
        elif use_scalewise:
            # Reordered sequential AAR:
            #   predict stages in effective-scale order (ar_input_seq), but each
            #   stage sees all previously predicted stages, including same-scale
            #   siblings that came earlier in the order.
            for k, i in enumerate(self.ar_input_seq[:-1]):
                guide = z_q if torch.is_tensor(z_q) else None
                z_q = z_q + stage_to_full(i, guide=guide)
                next_i = self.ar_input_seq[k + 1]
                out.append(F.interpolate(z_q, int(self.scale_factors[next_i] * T), mode='area').transpose(1, 2))
        else:
            for i, code in enumerate(codes[:-1]):
                guide = z_q if torch.is_tensor(z_q) else None
                z_q = z_q + stage_to_full(i, guide=guide)
                
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

        z_in = self.shared_project_in(latents)
        T = z_in.shape[-1]
        residual = z_in
        
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

        return self.shared_project_out(z_q), z_p, codes
    
    def embedding(self, x: torch.Tensor, i: int, use_offset: bool = False, use_scalewise: bool = True):
        if use_scalewise:
            i = self.ar_input_seq[i]
        else:
            i = i + (self.offset if use_offset else 0)
        out = self.quantizers[i].decode_code(x)
        out = self.quantizers[i].out_proj(out).permute(0, 2, 1)
        return out

    def decode_stage_to_full(
        self,
        x: torch.Tensor,
        stage_pos: int,
        H: int,
        use_offset: bool = False,
        use_scalewise: bool = True,
        guide: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode AAR stage-position codes to a full-length VAE contribution."""
        if use_scalewise:
            codebook_idx = self.ar_input_seq[stage_pos]
        else:
            codebook_idx = stage_pos + (self.offset if use_offset else 0)

        quantizer = self.quantizers[codebook_idx]
        h = quantizer.decode_code(x)
        h = quantizer.upsample_code(h, size=H, guide=guide, use_guide=h.shape[-1] != H)
        return quantizer.out_proj(self._apply_quant_resi(codebook_idx, h))
    
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
        
        quantizer = self.quantizers[codebook_idx]
        h_BChw = quantizer.in_proj(h_BChw)
        
        if not is_last:
            guide = f_hat if torch.is_tensor(f_hat) else None
            h = quantizer.upsample_code(h_BChw, size=H, guide=guide, use_guide=h_BChw.shape[-1] != H)
            h = self._apply_quant_resi(codebook_idx, h)
            h = quantizer.out_proj(h)
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=int(self.scale_factors[next_codebook] * H), mode="area")
        else:
            h = self._apply_quant_resi(codebook_idx, h_BChw)
            h = quantizer.out_proj(h)
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
