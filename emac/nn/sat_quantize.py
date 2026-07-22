from typing import List, Optional, Tuple

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn
import torch.nn.functional as F


def _default(value, fallback):
    return value if value is not None else fallback


def _ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1.0 - decay))


def _laplace_smoothing(x, n_categories: int, epsilon: float = 1e-3):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def _uniform_init(*shape: int):
    tensor = torch.empty(shape)
    nn.init.kaiming_uniform_(tensor)
    return tensor


def _sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices]


def _kmeans(samples, num_clusters: int, num_iters: int = 10):
    dim, dtype = samples.shape[-1], samples.dtype
    means = _sample_vectors(samples, num_clusters)
    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(means, "c d -> () c d")
        dists = -(diffs ** 2).sum(dim=-1)
        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)
        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]
        means = torch.where(zero_mask[..., None], means, new_means)
    return means, bins


class SATOriginalEuclideanCodebook(nn.Module):
    """Euclidean EMA codebook copied from the public SAT/AAR quantizer."""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = float(decay)
        init_fn = torch.zeros if kmeans_init else _uniform_init
        embed = init_fn(codebook_size, dim)
        self.codebook_size = int(codebook_size)
        self.kmeans_iters = int(kmeans_iters)
        self.epsilon = float(epsilon)
        self.threshold_ema_dead_code = int(threshold_ema_dead_code)
        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return
        embed, cluster_size = _kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], _sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return
        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)

    def preprocess(self, x):
        return rearrange(x, "... d -> (...) d")

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        return dist.max(dim=-1).indices

    @staticmethod
    def postprocess_emb(embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        return F.embedding(embed_ind, self.embed)

    def encode(self, x):
        shape = x.shape
        x = self.preprocess(x)
        embed_ind = self.quantize(x)
        return self.postprocess_emb(embed_ind, shape)

    def decode(self, embed_ind):
        return self.dequantize(embed_ind)

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)
        self.init_embed_(x)
        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)

        if self.training:
            self.expire_codes_(x)
            _ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = x.t() @ embed_onehot
            _ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                _laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind


class SATOriginalVectorQuantization(nn.Module):
    """One SAT multiscale VQ stage, preserving the public implementation."""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.0,
    ):
        super().__init__()
        codebook_dim = _default(codebook_dim, dim)
        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.epsilon = float(epsilon)
        self.commitment_weight = float(commitment_weight)
        self._codebook = SATOriginalEuclideanCodebook(
            dim=codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )
        self.codebook_size = int(codebook_size)

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x, scale=None):
        x = self.project_in(x)
        if scale is not None:
            x = F.interpolate(x.permute(0, 2, 1), size=int(scale), mode="area").permute(0, 2, 1)
        return self._codebook.encode(x)

    def decode(self, embed_ind, H=None, conv=None):
        quantize = self._codebook.decode(embed_ind)
        quantize = (
            F.interpolate(quantize.permute(0, 2, 1), size=int(H), mode="linear")
            if H is not None
            else quantize.permute(0, 2, 1)
        )
        quantize = conv(quantize).permute(0, 2, 1) if conv is not None else quantize.permute(0, 2, 1)
        return self.project_out(quantize)

    def forward(self, x, scale=None, conv=None):
        H = x.shape[1]
        x = self.project_in(x)
        inter_x = (
            F.interpolate(x.permute(0, 2, 1), size=int(scale), mode="area").permute(0, 2, 1)
            if scale is not None
            else x
        )
        quantize, embed_ind = self._codebook(inter_x)
        quantize = (
            F.interpolate(quantize.permute(0, 2, 1), size=H, mode="linear").contiguous()
            if scale is not None
            else quantize.permute(0, 2, 1).contiguous()
        )
        quantize = conv(quantize).permute(0, 2, 1) if conv is not None else quantize.permute(0, 2, 1)

        loss = torch.tensor([0.0], device=x.device, requires_grad=self.training)
        if self.training:
            if self.commitment_weight > 0:
                loss = loss + F.mse_loss(quantize.detach(), x) * self.commitment_weight
            if conv is not None:
                loss = loss + F.mse_loss(quantize, x.detach())
            quantize = x + (quantize - x).detach()

        quantize = self.project_out(quantize)
        return quantize, embed_ind, loss


class SATOriginalPhi(nn.Conv1d):
    def __init__(self, embed_dim, quant_resi, ks):
        padding = ks // 2
        super().__init__(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=ks,
            stride=1,
            padding=padding,
        )
        self.resi_ratio = abs(float(quant_resi))

    def forward(self, h_BChw):
        return h_BChw.mul(1.0 - self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class SATOriginalPhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = (
            np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K)
            if K == 4
            else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)
        )

    def __getitem__(self, at_from_0_to_1: float):
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self) -> str:
        return f"ticks={self.ticks}"


class SATOriginalMultiscaleResidualVectorQuantization(nn.Module):
    """Public SAT/AAR multiscale residual quantizer, adapted only for packaging."""

    def __init__(self, *, scale, phi_kernel, num_quantizers, latent_dim, **kwargs):
        super().__init__()
        self.scale = [int(s) for s in scale]
        self.dim = int(latent_dim)
        self.shared_codebook = False
        self.project_in = (
            nn.Linear(kwargs["dim"], latent_dim) if kwargs["dim"] != latent_dim else nn.Identity()
        )
        self.project_out = (
            nn.Linear(latent_dim, kwargs["dim"]) if kwargs["dim"] != latent_dim else nn.Identity()
        )
        kwargs["dim"] = latent_dim
        self.quant_resi = SATOriginalPhiPartiallyShared(
            nn.ModuleList([SATOriginalPhi(latent_dim, 0.5, ks=ks) for ks in phi_kernel])
        )
        self.layers = nn.ModuleList(
            [SATOriginalVectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def _stage_scale(self, idx: int, H: int):
        return self.scale[idx] if int(self.scale[idx]) != int(H) else None

    def _stage_phi(self, idx: int, n_q: int):
        return self.quant_resi[idx / (n_q - 1)] if n_q > 1 else self.quant_resi[0.0]

    def forward(self, x):
        _, _, H = x.shape
        quantized_out = 0.0
        residual = self.project_in(x.permute(0, 2, 1))
        all_losses = []
        all_indices = []
        n_q = len(self.layers)

        for idx, layer in enumerate(self.layers):
            quantized, indices, loss = layer(
                residual,
                self._stage_scale(idx, H),
                self._stage_phi(idx, n_q),
            )
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
            all_losses.append(loss)

        out_losses = torch.stack(all_losses)
        output = self.project_out(quantized_out).permute(0, 2, 1)
        return output, all_indices, out_losses

    def encode(self, x: torch.Tensor):
        _, _, H = x.shape
        residual = self.project_in(x.permute(0, 2, 1))
        all_indices = []
        n_q = len(self.scale)
        for idx, layer in enumerate(self.layers):
            indices = layer.encode(residual, self._stage_scale(idx, H))
            quantized = layer.decode(
                indices,
                H if self._stage_scale(idx, H) is not None else None,
                self._stage_phi(idx, n_q),
            )
            residual = residual - quantized
            all_indices.append(indices)
        return all_indices

    def decode(self, q_indices: List[torch.Tensor]) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices[0].device)
        n_q = len(q_indices)
        target_H = self.scale[-1]
        for idx, indices in enumerate(q_indices):
            layer = self.layers[idx]
            quantized = layer.decode(
                indices,
                target_H if idx != n_q - 1 else None,
                self._stage_phi(idx, len(self.scale)),
            )
            quantized_out = quantized_out + quantized
        return self.project_out(quantized_out).permute(0, 2, 1)

    def decode_each_scale(self, q_indices) -> List[torch.Tensor]:
        quantized_out = torch.tensor(0.0, device=q_indices[0].device)
        outputs = []
        n_q = len(q_indices)
        H = max(self.scale)
        for idx, indices in enumerate(q_indices):
            layer = self.layers[idx]
            quantized = layer.decode(
                indices,
                H if self.scale[idx] != H else None,
                self._stage_phi(idx, n_q),
            )
            quantized_out = quantized_out + quantized
            outputs.append(self.project_out(quantized_out).permute(0, 2, 1))
        return outputs

    def embedding(self, idx_Bl, layer_id):
        return self.layers[layer_id].decode(idx_Bl).permute(0, 2, 1)

    def post_conv(self, fhat):
        return self.project_out(fhat).permute(0, 2, 1)

    def get_next_autoregressive_input(
        self, si: int, f_hat: torch.Tensor, h_BChw: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        H = self.scale[-1]
        SN = len(self.scale)
        if si != SN - 1:
            h = self.quant_resi[si / (SN - 1)](F.interpolate(h_BChw, size=H, mode="linear"))
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=self.scale[si + 1], mode="area")
        h = self.quant_resi[si / (SN - 1)](h_BChw)
        f_hat.add_(h)
        return f_hat, f_hat

    def idx_to_var_input(self, label_list):
        next_scales = []
        B = label_list[0].shape[0]
        C = self.dim
        H = self.scale[-1]
        SN = len(self.scale)
        with torch.autocast(device_type=label_list[0].device.type, enabled=False):
            f_hat = label_list[0].new_zeros(B, C, H, dtype=torch.float32)
            for si in range(SN - 1):
                layer = self.layers[si]
                f_hat.add_(
                    layer.decode(label_list[si], H, self.quant_resi[si / (SN - 1)]).permute(0, 2, 1)
                )
                pn_next = self.scale[si + 1]
                next_scales.append(
                    F.interpolate(f_hat, size=pn_next, mode="area").view(B, C, -1).transpose(1, 2)
                )
        return next_scales


class SATOriginalResidualVectorQuantizer(nn.Module):
    """Wrapper matching the public SAT ResidualVectorQuantizer API."""

    def __init__(
        self,
        dimension: int = 256,
        n_q: int = 8,
        bins: int = 1024,
        latent_dim: int = 32,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        multi_scale=None,
        phi_kernel=None,
    ):
        super().__init__()
        self.n_q = int(n_q)
        self.dimension = int(dimension)
        self.bins = int(bins)
        self.decay = float(decay)
        self.kmeans_init = bool(kmeans_init)
        self.kmeans_iters = int(kmeans_iters)
        self.threshold_ema_dead_code = int(threshold_ema_dead_code)
        self.multi_scale = [int(s) for s in multi_scale]
        self.phi_kernel = [int(k) for k in phi_kernel]
        self.vq = SATOriginalMultiscaleResidualVectorQuantization(
            scale=self.multi_scale,
            phi_kernel=self.phi_kernel,
            dim=self.dimension,
            latent_dim=int(latent_dim),
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
        )
        self.last_downsampler_metrics = {}
        self.last_scale_anneal_metrics = {}
        self.last_upsampler_metrics = {}

    def forward(self, x: torch.Tensor, n_quantizers: int = None):
        if n_quantizers is not None and int(n_quantizers) != self.n_q:
            raise ValueError("SATOriginalResidualVectorQuantizer uses all SAT stages.")
        quantized, codes, stage_losses = self.vq(x)
        return quantized, codes, stage_losses

    def encode(self, x: torch.Tensor):
        return self.vq.encode(x)

    def decode(self, codes: List[torch.Tensor]) -> torch.Tensor:
        return self.vq.decode(codes)

    def from_codes(self, codes: List[torch.Tensor], depth="full"):
        if isinstance(codes, torch.Tensor):
            codes = [codes[:, i, :] for i in range(codes.shape[1])]
        if depth == "early":
            target = codes[:1]
        elif depth == "mid":
            target = codes[: max(1, len(codes) // 2)]
        else:
            target = codes
        return self.decode(target), None, codes

    def idxBl_to_var_input(self, label_list):
        return self.vq.idx_to_var_input(label_list)

    def post_conv(self, fhat):
        return self.vq.post_conv(fhat)

    def embedding(self, idx_Bl, layer_id):
        return self.vq.embedding(idx_Bl, layer_id)

    def get_next_autoregressive_input(self, si: int, f_hat: torch.Tensor, h_BChw: torch.Tensor):
        return self.vq.get_next_autoregressive_input(si, f_hat, h_BChw)

    def decode_each_scale(self, q_indices) -> List[torch.Tensor]:
        return self.vq.decode_each_scale(q_indices)
