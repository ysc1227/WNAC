from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from wnac.nn.layers import WNConv1d


def augment_data(data, target_size):
    """
    데이터 증강을 통해 데이터 개수를 클러스터 크기 이상으로 확장.
    """
    N, _ = data.size()
    if N >= target_size:
        return data  # 데이터가 이미 충분함

    # 데이터 증강
    while data.size(0) < target_size:
        noise = torch.randn_like(data) * 0.01  # 작은 노이즈 추가
        augmented_data = data + noise
        data = torch.cat([data, augmented_data], dim=0)

    return data[:target_size]  # 목표 크기로 자르기

def kmeans(data, num_clusters, num_iters=10):
    """
    k-means 알고리즘을 PyTorch로 구현. 코드북 크기를 고정하며 빈 클러스터를 처리.
    """
    data = augment_data(data, target_size=num_clusters)
    
    N, _ = data.size()

    # 초기 중심 랜덤 선택
    indices = torch.randperm(N)[:num_clusters]
    centroids = data[indices]

    for _ in range(num_iters):
        # 각 데이터 포인트와 클러스터 중심 간의 거리 계산
        distances = torch.cdist(data, centroids)
        cluster_assignments = distances.argmin(dim=1)

        # 클러스터 중심 업데이트
        new_centroids = []
        for k in range(num_clusters):
            mask = cluster_assignments == k
            if mask.sum().item() == 0:  # 클러스터가 비어있는 경우
                print(f"Cluster {k} is empty. Reinitializing...")
                random_index = torch.randint(0, N, (1,))  # 무작위 데이터 포인트 선택
                new_centroids.append(data[random_index].squeeze(0))
            else:
                # 클러스터 평균 계산
                new_centroids.append(data[mask].mean(dim=0))

        centroids = torch.stack(new_centroids)

    return centroids
    

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

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

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
        
        z_e = self.in_proj(z) # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)
        
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

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

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int, max_init: int = 1000, pooling='interp'):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        
        self.init_num = 0
        self.all_data = None
        self.max_init = max_init
        self.pooling = pooling
        
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        
    def forward(self, z, scale=None, conv=None):
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
        
        if self.pooling == 'interp':
            inter_z = F.interpolate(z_e, size=scale, mode='area') if scale != None else z_e
        else:
            inter_z = torch.nn.functional.avg_pool1d(z_e, int(H / scale), int(H / scale)) if scale != None else z_e  
              
        # z_e : (B x D x T)
        z_q, indices = self.decode_latents(inter_z)
        
        if self.pooling == 'interp':
            z_q = F.interpolate(z_q, size=H, mode='linear').contiguous() if scale != None else z_q.contiguous()
        else:
            z_q = z_q.repeat_interleave(int(H / scale), dim=-1).contiguous() if scale != None else z_q.contiguous()
        
        z_q = conv(z_q) if conv != None else z_q
        
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id, scale=None, conv=None):
        quantize = self.embed_code(embed_id).transpose(1, 2)
        quantize = F.interpolate(quantize, size=scale, mode='linear') if scale != None else quantize
        quantize = conv(quantize) if conv != None else quantize
        return quantize
    

    def decode_latents(self, latents, scale=None, conv=None):
        encodings = rearrange(latents, "b d t -> (b t) d")
        
        if self.training:
            if self.init_num < self.max_init:
                flattened_data = encodings.view(-1, encodings.shape[-1])
                if self.all_data is None:
                    self.all_data = flattened_data
                else:
                    self.all_data = torch.cat([self.all_data, flattened_data], dim=0)
            elif self.init_num == self.max_init:
                centroids = kmeans(self.all_data, num_clusters=self.codebook_size)
                if centroids.size(1) != self.codebook_dim:
                    centroids = torch.nn.functional.adaptive_avg_pool1d(
                        centroids.unsqueeze(0), codebook_dim
                    ).squeeze(0)
                self.codebook.weight.data.copy_(centroids)
                del self.all_data
            self.init_num += 1
        
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
        z_q = self.decode_code(indices, scale=scale, conv=conv)
        return z_q, indices
    
    
class VscaleVectorQuantize(nn.Module):
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

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int, max_init: int = 1000, pooling='interp'):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        
        self.init_num = 0
        self.all_data = None
        self.max_init = max_init
        self.pooling = pooling
        
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        
    def forward(self, z, scale=None, conv=None):
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
        
        if self.pooling == 'interp':
            inter_z = F.interpolate(z_e, size=scale, mode='area') if scale != None else z_e
        else:
            inter_z = torch.nn.functional.avg_pool1d(z_e, int(H / scale), int(H / scale)) if scale != None else z_e  
              
        # z_e : (B x D x T)
        z_q, indices = self.decode_latents(inter_z)
        
        if self.pooling == 'interp':
            z_q = F.interpolate(z_q, size=H, mode='linear').contiguous() if scale != None else z_q.contiguous()
        else:
            z_q = z_q.repeat_interleave(int(H / scale), dim=-1).contiguous() if scale != None else z_q.contiguous()
        
        z_q = conv(z_q) if conv != None else z_q
        
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id, scale=None, conv=None):
        quantize = self.embed_code(embed_id).transpose(1, 2)
        quantize = F.interpolate(quantize, size=scale, mode='linear') if scale != None else quantize
        quantize = conv(quantize) if conv != None else quantize
        return quantize
    

    def decode_latents(self, latents, scale=None, conv=None):
        encodings = rearrange(latents, "b d t -> (b t) d")
        
        if self.training:
            if self.init_num < self.max_init:
                flattened_data = encodings.view(-1, encodings.shape[-1])
                if self.all_data is None:
                    self.all_data = flattened_data
                else:
                    self.all_data = torch.cat([self.all_data, flattened_data], dim=0)
            elif self.init_num == self.max_init:
                centroids = kmeans(self.all_data, num_clusters=self.codebook_size)
                if centroids.size(1) != self.codebook_dim:
                    centroids = torch.nn.functional.adaptive_avg_pool1d(
                        centroids.unsqueeze(0), codebook_dim
                    ).squeeze(0)
                self.codebook.weight.data.copy_(centroids)
                del self.all_data
            self.init_num += 1
        
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
        z_q = self.decode_code(indices, scale=scale, conv=conv)
        return z_q, indices
    

class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0
    ):
        super().__init__()

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim)
                for _ in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int = None):
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
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break
            
            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual
            )

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss, 0

    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T]
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
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
            0
        ]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)
    

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
        scale_factors: list[int] = [0.01, 0.03, 0.05, 0.08, 0.12, 0.16, 0.21, 0.27, 0.33, 0.41, 0.49, 0.57, 0.67, 0.77, 0.88, 1],
        phi_kernel = [9, 9, 9, 9, 9],
        quantizer_dropout: float = 0.5,
        pooling = 'interp'
    ):
        super().__init__()

        self.n_codebooks = len(scale_factors)
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.scale_factors = scale_factors

        self.quantizers = nn.ModuleList(
            [
                MultiscaleVectorQuantize(input_dim, codebook_size, codebook_dim, pooling=pooling)
                for _ in range(self.n_codebooks)
            ]
        )
        
        if phi_kernel is not None:
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([Phi(codebook_dim, 0.5, ks=ks) for ks in phi_kernel]))
        else:
            self.quant_resi = None
        self.quantizer_dropout = quantizer_dropout
    
    def forward(self, z, n_quantizers: int = None):
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
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        for i, quantizer in enumerate(self.quantizers):
            scale = int(self.scale_factors[i] * T)
            scale += 1 if scale == 0 else 0
            
            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual, scale, self.quant_resi[i/float(self.n_codebooks)] if self.quant_resi is not None else None
            )

            z_q = z_q + z_q_i
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i).mean()
            codebook_loss += (codebook_loss_i).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        return z_q, codebook_indices, torch.cat(latents, dim=1), commitment_loss, codebook_loss, 0

    def from_codes(self, codes: list[torch.Tensor], scale_wise=False):
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
        z_scale_wise = []
        n_q = len(self.scale_factors)
        
        T = max([c.shape[-1] for c in codes])
        
        for i, (code, quantizer) in enumerate(zip(codes, self.quantizers)):
            z_p_i = quantizer.decode_code(code, T, self.quant_resi[i/(n_q-1)] if self.quant_resi else None)
            z_p.append(z_p_i)

            z_q_i = quantizer.out_proj(z_p_i) 
            z_q = z_q + z_q_i
            if scale_wise:
                z_scale_wise.append(z_q)
                
        if scale_wise:
            return z_scale_wise
        
        return z_q, z_p, codes

    def from_latents(self, latents: list[torch.Tensor]):
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
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
            0
        ]
        n_q = len(self.scale_factors)
        
        T = latents[-1].shape[-1]
        
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            latent = F.interpolate(latents[:, j:k, :], size=int(self.scale_factors[i] * T), mode='area')
            z_p_i, codes_i = self.quantizers[i].decode_latents(latent, scale=int(self.scale_factors[-1] * T), conv=self.quant_resi[i/(n_q-1)])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), codes


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
        max_init: int = 250,
        pooling = 'interp'
    ):
        super().__init__()

        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.scale_factors, self.n_codebooks = self._compute_wavescale(scale_factors)

        self.quantizers = nn.ModuleList(
            [
                MultiscaleVectorQuantize(input_dim, codebook_size, codebook_dim, max_init=max_init, pooling=pooling)
                for _ in range(self.n_codebooks)
            ]
        )
        if phi_kernel is not None:
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([Phi(codebook_dim, 0.5, ks=ks) for ks in phi_kernel]))
        else:
            self.quant_resi = None
        self.quantizer_dropout = quantizer_dropout
        
    def forward(self, z, n_quantizers: int = None):
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
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []
        z_ps = {}
        
        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break
            scale = int(self.scale_factors[i] * T)
            scale += 1 if scale == 0 else 0

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual, scale, self.quant_resi[i/float(self.n_codebooks)] if self.quant_resi is not None else None
            )
            
            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )

            z_q = z_q + z_q_i * mask[:, None, None]
            
            if scale in z_ps:
                z_ps[scale].append(z_q)
            else: z_ps[scale] = [z_q]
            
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i).mean()
            codebook_loss += (codebook_loss_i).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        aux_loss = 0.0
        for _, val in z_ps.items():
            if len(val) > 1:
                aux_loss += F.mse_loss(val[1], val[0], reduction="none").mean([1, 2]).mean()
        
        return z_q, codebook_indices, latents, commitment_loss, codebook_loss, aux_loss

    def from_codes(self, codes: list[torch.Tensor], interpolate=True, scale_wise=False):
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
        z_scale_wise = []
        n_q = len(self.scale_factors)
        T = max([c.shape[-1] for c in codes])
        
        for i, (code, quantizer) in enumerate(zip(codes, self.quantizers)):
            z_p_i = quantizer.decode_code(code, T if interpolate else None, self.quant_resi[i/(n_q-1)] if self.quant_resi is not None else None)
            z_p.append(z_p_i.permute(0, 2, 1))

            if interpolate:
                z_q_i = quantizer.out_proj(z_p_i)
                z_q = z_q + z_q_i
            if scale_wise:
                z_scale_wise.append(z_q)
                
        if scale_wise:
            return z_scale_wise
        return z_q, z_p, codes

    def from_latents(self, latents: list[torch.Tensor]):
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
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
            0
        ]
        n_q = len(self.scale_factors)
        T = latents[-1].shape[-1]
        
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            latent = F.interpolate(latents[:, j:k, :], size=int(self.scale_factors[i] * T), mode='area')
            z_p_i, codes_i = self.quantizers[i].decode_latents(latent, scale=int(self.scale_factors[-1] * T), conv=self.quant_resi[i/(n_q-1)])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), codes
    
    def embedding(self, idx_Bl, layer_id):
        return self.quantizers[layer_id].decode_code(idx_Bl).permute(0, 2, 1)
    
    def get_next_autoregressive_input(self, si: int, f_hat: torch.Tensor, h_BChw: torch.Tensor, max_scale_size: int) -> Tuple[Optional[torch.Tensor], torch.Tensor]: # only used in VAR inference
        SN = len(self.scale_factors)

        if si != SN-1:
            h = self.quant_resi[si/(SN-1)](F.interpolate(h_BChw, size=max_scale_size, mode='linear'))     # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=int(self.scale_factors[si+1] * max_scale_size), mode='area')
        else:
            h = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat
    
    def _compute_wavescale(self, scale_factors):
        return scale_factors[::-1] + scale_factors[1:], len(scale_factors) * 2 - 1


class VscaleResidualVectorQuantize(nn.Module):
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
        max_init: int = 250,
        pooling = 'interp'
    ):
        super().__init__()

        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.scale_factors, self.n_codebooks = scale_factors, len(scale_factors)

        self.quantizers = nn.ModuleList(
            [
                MultiscaleVectorQuantize(input_dim, codebook_size, codebook_dim, max_init=max_init, pooling=pooling)
                for _ in range(self.n_codebooks)
            ]
        )
        if phi_kernel is not None:
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([Phi(codebook_dim, 0.5, ks=ks) for ks in phi_kernel]))
        else:
            self.quant_resi = None
        self.quantizer_dropout = quantizer_dropout
        
    def forward(self, z, n_quantizers: int = None):
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
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """
        T = z.shape[-1]
        z_q = 0
        
        commitment_loss = 0
        codebook_loss = 0
        aux_loss = 0
        
        codebook_indices = []
        latents = []
        
        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)
            
        l = 0.9
        residual = z
        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break
            scale = int(self.scale_factors[i] * T)
            d_scale = int(self.scale_factors[::-1][i] * T)
            scale += 1 if scale == 0 else 0
            d_scale += 1 if scale == 0 else 0
            
            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual, scale, self.quant_resi[i/float(self.n_codebooks)] if self.quant_resi is not None else None
            )
            
            d_z_q_i, d_commitment_loss_i, d_codebook_loss_i, d_indices_i, d_z_e_i = quantizer(
                residual, None, self.quant_resi[i/float(self.n_codebooks)] if self.quant_resi is not None else None
            )
            
            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            
            residual = residual - z_q

            aux_loss += F.mse_loss(z_q, d_z_q_i, reduction="none").mean([1, 2]).mean()
                
            # Sum losses
            commitment_loss +=  commitment_loss_i.mean() + d_commitment_loss_i.mean()
            codebook_loss +=  codebook_loss_i.mean() + d_codebook_loss_i.mean()

            codebook_indices.append(indices_i)
            codebook_indices.append(d_indices_i)
            latents.append(z_e_i)
            latents.append(d_z_e_i)
        
        return z_q, codebook_indices, latents, commitment_loss, codebook_loss, aux_loss

    def from_codes(self, codes: list[torch.Tensor], interpolate=True):
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
        T = codes[-1].shape[-1]
        
        for i, (code, quantizer) in enumerate(zip(codes, self.quantizers)):
            z_p_i = quantizer.decode_code(code, int(self.scale_factors[-1] * T) if interpolate else None, self.quant_resi[i/(n_q-1)] if self.quant_resi is not None else None)
            z_p.append(z_p_i.permute(0, 2, 1))

            if interpolate:
                z_q_i = quantizer.out_proj(z_p_i)
                z_q = z_q + z_q_i
        
        return z_q, z_p, codes

    def from_latents(self, latents: list[torch.Tensor]):
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
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
            0
        ]
        n_q = len(self.scale_factors)
        T = latents[-1].shape[-1]
        
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            latent = F.interpolate(latents[:, j:k, :], size=int(self.scale_factors[i] * T), mode='area')
            z_p_i, codes_i = self.quantizers[i].decode_latents(latent, scale=int(self.scale_factors[-1] * T), conv=self.quant_resi[i/(n_q-1)])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), codes
    
    def embedding(self, idx_Bl, layer_id):
        return self.quantizers[layer_id].decode_code(idx_Bl).permute(0, 2, 1)
    
    def get_next_autoregressive_input(self, si: int, f_hat: torch.Tensor, h_BChw: torch.Tensor, max_scale_size: int) -> Tuple[Optional[torch.Tensor], torch.Tensor]: # only used in VAR inference
        SN = len(self.scale_factors)

        if si != SN-1:
            h = self.quant_resi[si/(SN-1)](F.interpolate(h_BChw, size=max_scale_size, mode='linear'))     # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=int(self.scale_factors[si+1] * max_scale_size), mode='area')
        else:
            h = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat
    
    def _compute_wavescale(self, scale_factors):
        return scale_factors[::-1] + scale_factors[1:], len(scale_factors) * 2 - 1
    

class ImprovedAttentionUpsampler(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int = None, num_heads: int = 4, max_length: int = 500):
        super().__init__()
        if embed_dim is None:
            embed_dim = in_channels
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # 입력 projection을 위한 1x1 Conv
        self.input_proj = nn.Conv1d(in_channels, embed_dim, kernel_size=1)
        # 조금 더 복잡한 query 생성 네트워크: 1x1 Conv + ReLU + 1x1 Conv
        self.query_generator = nn.Sequential(
            nn.Conv1d(in_channels, embed_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        )
        
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        
        self.fixed_query = nn.Parameter(torch.randn(max_length, embed_dim))
        # scale_value에 대해 별도의 변환층을 추가
        self.scale_transform = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.out_proj = nn.Conv1d(embed_dim, in_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, target_length: int, scale_value: float) -> torch.Tensor:
        B = x.shape[0]
        # 보간된 입력: 여기서는 단순 보간을 통해 target_length로 맞춤
        x_interp = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        
        # 동적 query 생성 (B, embed_dim, L_target)
        dynamic_query = self.query_generator(x_interp)
        
        fixed_query = self.fixed_query[:target_length, :].unsqueeze(1).repeat(1, B, 1)
        dynamic_query = dynamic_query.permute(2, 0, 1)
        residual_proj = self.input_proj(x_interp).permute(2, 0, 1)
        
        # scale_value를 tensor 변환 후 transformation 적용
        raw_scale = torch.tensor([[scale_value]], device=x.device, dtype=torch.float)
        scale_emb = self.scale_transform(raw_scale)  # (1, embed_dim)
        scale_emb = scale_emb.unsqueeze(0).repeat(target_length, B, 1)
        
        final_query = fixed_query + dynamic_query + residual_proj + scale_emb
        
        x_proj = self.input_proj(x).permute(2, 0, 1)
        attn_output, _ = self.attn(final_query, x_proj, x_proj)
        attn_output = attn_output.permute(1, 2, 0)
        output = self.out_proj(attn_output)
        
        return output

    

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


if __name__ == "__main__":
    codebook_dim = 64
    T = 96
    
    from torch.autograd import gradcheck
    module = VscaleResidualVectorQuantize(
        input_dim=4,
        codebook_size=4,
        codebook_dim=codebook_dim,
        scale_factors=[0.01, 0.03, 0.05, 0.08, 0.12, 0.16, 0.21, 0.27, 0.33, 0.41, 0.49, 0.57, 0.67, 0.77, 0.88, 1],
        phi_kernel=[9, 9, 9, 9, 9],
        max_init=2
    ).cuda()
    input = torch.randn(1, 4, T, requires_grad=True).cuda()
    test = gradcheck(lambda x: module(x)[0], input)
    print(test)
    
    mrvq = VscaleResidualVectorQuantize(
        input_dim=64 * (2 ** len([2, 4, 8, 8])),
        codebook_size=1024,
        codebook_dim=codebook_dim,
        scale_factors=[0.01, 0.03, 0.05, 0.08, 0.12, 0.16, 0.21, 0.27, 0.33, 0.41, 0.49, 0.57, 0.67, 0.77, 0.88, 1],
        phi_kernel=[9, 9, 9, 9, 9]
    )

    x = torch.randn(16, 1024, T)
    xs = []
    xss = torch.randn(16, 72, T)
    
    scale_factors = [0.01, 0.03, 0.05, 0.08, 0.12, 0.16, 0.21, 0.27, 0.33, 0.41, 0.49, 0.57, 0.67, 0.77, 0.88, 1]
    
    for i in scale_factors:
        xs.append(torch.randint(low=0, high=codebook_dim-1, size=(16, int(i * T)), dtype=torch.long))

    my = mrvq(x)

    # k = mrvq.from_codes(xs)
    # y = mrvq.from_latents(xss)
    
    for i in my:
        if isinstance(i, int):
            print(i)
            continue
        try:
            print(i.shape)
        except:
            print([j.shape for j in i])
    print("----------------------------------------------")
    # for i in k:
    #     if isinstance(i, int):
    #         print(i)
    #         continue
    #     try:
    #         print(i.shape)
    #     except:
    #         print([j.shape for j in i])
    # print("----------------------------------------------")
    # for i in y:
    #     if isinstance(i, int):
    #         print(i)
    #         continue
    #     try:
    #         print(i.shape)
    #     except:
    #         print([j.shape for j in i])
            