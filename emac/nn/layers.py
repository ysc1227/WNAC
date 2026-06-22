import math
import os

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from .attention import LocalMHA


def _layer_probe_line(prefix, idx, layer, x):
    with torch.no_grad():
        finite = torch.isfinite(x)
        ok = bool(finite.all().item())
        safe = torch.nan_to_num(x.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        print(
            f"[{prefix}] layer={idx} type={layer.__class__.__name__} "
            f"ok={ok} finite_ratio={float(finite.float().mean().item()):.6f} "
            f"shape={tuple(x.shape)} mean={float(safe.mean().item()):.6g} "
            f"absmax={float(safe.abs().max().item()):.6g}",
            flush=True,
        )
    return ok

    
class Encoder(nn.Module):
    def __init__(
        self,
        d_model=64,
        rates=[2, 4, 8, 8],
        depthwise=False,
        attn_window_size=None,
    ):
        super().__init__()
        layers = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in rates:
            d_model *= 2
            groups = d_model // 2 if depthwise else 1
            layers += [EncoderBlock(output_dim=d_model, stride=stride, groups=groups)]
            
        if attn_window_size is not None:
            layers += [LocalMHA(dim=d_model, window_size=attn_window_size)]
            
        groups = d_model if depthwise else 1
        layers += [
            Snake1d(d_model),
            WNConv1d(d_model, d_model, kernel_size=3, padding=1, groups=groups),
        ]
        
        self.block = nn.Sequential(*layers)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        d_model,
        rates,
        noise=False,
        depthwise=False,
        attn_window_size=None,
        d_out=1,
    ):
        super().__init__()
        if depthwise:
            layers = [
                WNConv1d(latent_dim, latent_dim, kernel_size=7, padding=3, groups=latent_dim),
                WNConv1d(latent_dim, d_model, kernel_size=1),
            ]
        else:
            layers = [WNConv1d(latent_dim, d_model, kernel_size=7, padding=3)]

        if attn_window_size is not None:
            layers += [LocalMHA(dim=d_model, window_size=attn_window_size)]

        for i, stride in enumerate(rates):
            input_dim = d_model // 2**i
            output_dim = d_model // 2 ** (i + 1)
            groups = output_dim if depthwise else 1
            layers += [DecoderBlock(input_dim, output_dim, stride, noise, groups=groups)]

        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if os.getenv("EMAC_DECODER_LAYER_PROBE", "0") == "1":
            for i, layer in enumerate(self.model):
                x = layer(x)
                with torch.no_grad():
                    finite = torch.isfinite(x)
                    ok = bool(finite.all().item())
                    safe = torch.nan_to_num(x.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
                    print(
                        f"[DECODER_LAYER_PROBE] layer={i} type={layer.__class__.__name__} "
                        f"ok={ok} finite_ratio={float(finite.float().mean().item()):.6f} "
                        f"shape={tuple(x.shape)} mean={float(safe.mean().item()):.6g} "
                        f"absmax={float(safe.abs().max().item()):.6g}",
                        flush=True,
                    )
                if not ok and os.getenv("EMAC_DECODER_LAYER_PROBE_STOP", "1") == "1":
                    return x
            return x
        x = self.model(x)
        return x


class ResidualUnit(nn.Module):
    def __init__(self, dim=16, dilation=1, kernel=7, groups=1):
        super().__init__()
        pad = ((kernel - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=kernel, dilation=dilation, padding=pad, groups=groups),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        if os.getenv("EMAC_RESIDUAL_UNIT_PROBE", "0") == "1":
            y = x
            for i, layer in enumerate(self.block):
                y = layer(y)
                ok = _layer_probe_line("RESIDUAL_UNIT_PROBE", i, layer, y)
                if not ok and os.getenv("EMAC_RESIDUAL_UNIT_PROBE_STOP", "1") == "1":
                    return y
        else:
            y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        out = x + y
        if os.getenv("EMAC_RESIDUAL_UNIT_PROBE", "0") == "1":
            _layer_probe_line("RESIDUAL_UNIT_PROBE", "skip_add", self, out)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, output_dim=16, input_dim=None, stride=1, groups=1):
        super().__init__()
        input_dim = input_dim or output_dim // 2
        self.block = nn.Sequential(
            ResidualUnit(input_dim, dilation=1, groups=groups),
            ResidualUnit(input_dim, dilation=3, groups=groups),
            ResidualUnit(input_dim, dilation=9, groups=groups),
            Snake1d(input_dim),
            WNConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class NoiseBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = WNConv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, T = x.shape
        noise = torch.randn((B, 1, T), device=x.device, dtype=x.dtype)
        h = self.linear(x)
        n = noise * h
        x = x + n
        return x


class DecoderBlock(nn.Module):
    def __init__(self, input_dim=16, output_dim=8, stride=1, noise=False, groups=1):
        super().__init__()
        layers = [
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
        ]
        if noise:
            layers.append(NoiseBlock(output_dim))
        layers.extend(
            [
                ResidualUnit(output_dim, dilation=1, groups=groups),
                ResidualUnit(output_dim, dilation=3, groups=groups),
                ResidualUnit(output_dim, dilation=9, groups=groups),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if os.getenv("EMAC_DECODER_BLOCK_PROBE", "0") == "1":
            for i, layer in enumerate(self.block):
                x = layer(x)
                ok = _layer_probe_line("DECODER_BLOCK_PROBE", i, layer, x)
                if not ok and os.getenv("EMAC_DECODER_BLOCK_PROBE_STOP", "1") == "1":
                    return x
            return x
        return self.block(x)

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        if os.getenv("EMAC_SNAKE_EAGER", "0") == "1":
            shape = x.shape
            y = x.reshape(shape[0], shape[1], -1)
            alpha = self.alpha
            y = y + (alpha + 1e-9).reciprocal() * torch.sin(alpha * y).pow(2)
            return y.reshape(shape)
        return snake(x, self.alpha)