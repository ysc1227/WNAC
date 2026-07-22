import math
from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from audiotools.ml.layers.base import BaseModel

from emac.nn.basic_var import AdaLNSABlock, SABlock
from emac.nn.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from emac.nn.loss import CELoss
from emac.nn.quantize import WavescaleResidualVectorQuantize
from . import EMAC


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class AAR(BaseModel):
    @staticmethod
    def _metadata_get(container, keys):
        """Best-effort lookup for checkpoint/config metadata keys."""
        if container is None:
            return None
        if isinstance(keys, str):
            keys = (keys,)
        if isinstance(container, dict):
            for key in keys:
                if key in container:
                    return container[key]
            for nested_key in ("kwargs", "args", "config", "metadata"):
                value = AAR._metadata_get(container.get(nested_key), keys)
                if value is not None:
                    return value
        return None

    @staticmethod
    def infer_rvq_frame_size(vae_local: EMAC, fallback: int = 80) -> int:
        """Infer the latent/RVQ frame window from an EMAC/RVQ checkpoint when possible.

        Priority:
        1. explicit runtime/checkpoint attributes on the VAE or quantizer;
        2. metadata kwargs/config fields such as ``frame_size`` or ``AAR.frame_size``;
        3. duration metadata converted through ``sample_rate / hop_length``;
        4. caller-provided fallback.

        RVQ weights themselves do not encode a time-window length, so the fallback is
        kept for checkpoints that were saved without this metadata.
        """
        quant = getattr(vae_local, "quantizer", None)
        attr_names = (
            "rvq_frame_size",
            "frame_size",
            "latent_frame_size",
            "aar_frame_size",
        )
        for obj in (quant, vae_local):
            for name in attr_names:
                value = getattr(obj, name, None)
                if value is not None:
                    return int(value)

        metadata_keys = (
            "rvq_frame_size",
            "frame_size",
            "latent_frame_size",
            "aar_frame_size",
            "AAR.frame_size",
        )
        for obj in (quant, vae_local):
            value = AAR._metadata_get(getattr(obj, "metadata", None), metadata_keys)
            if value is not None:
                return int(value)

        duration_keys = (
            "duration",
            "train/AudioDataset.duration",
            "AudioDataset.duration",
            "segment_duration",
            "rvq_duration",
        )
        for obj in (quant, vae_local):
            duration = AAR._metadata_get(getattr(obj, "metadata", None), duration_keys)
            if duration is not None:
                sample_rate = int(getattr(vae_local, "sample_rate", 1))
                hop_length = int(getattr(vae_local, "hop_length", 1))
                return int(math.ceil(float(duration) * sample_rate / max(1, hop_length)))

        return int(fallback)

    def __init__(
        self, vae_local: EMAC,
        input_dim: int = 512, norm_eps=1e-6, aln=1, aln_gamma_init=1e-3, shared_aln=False, cond_drop_rate=0.1,
        depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=None,
        layer_scale=-1., tau=4, cos_attn=False,
        flash_if_available=True, fused_if_available=True,
        frame_size="auto",
        use_offset=False,
        use_scale_order=True,
        use_blockwise=True,
        use_scalewise: Optional[bool] = None,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.latent_dim, vae_local.quantizer.codebook_size # vae_local.quantizer.bins
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        self.using_aln, self.aln_init, self.aln_gamma_init, self.layer_scale = aln >= 0, aln, aln_gamma_init, layer_scale
        if self.using_aln and layer_scale != -1:
            print(f'**WARNING**: using AdaLNSABlock with {aln=:g}, {aln_gamma_init=:g}; the arg {layer_scale=:g} will be IGNORED because only SABlock cares about layer_scale', flush=True)
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.frame_size = self.infer_rvq_frame_size(vae_local, fallback=80) if frame_size is None or frame_size == "auto" else int(frame_size)
        setattr(vae_local, "rvq_frame_size", self.frame_size)
        setattr(vae_local.quantizer, "rvq_frame_size", self.frame_size)
        self.use_offset = use_offset
        if use_scalewise is not None:
            print("**WARNING**: AAR.use_scalewise is deprecated; use AAR.use_scale_order instead.", flush=True)
            use_scale_order = use_scalewise
        self.use_scale_order = bool(use_scale_order)
        self.use_scalewise = self.use_scale_order  # backward-compatible alias for older scripts/checkpoints
        self.use_blockwise = use_blockwise
        # Keep AAR token lengths identical to the RVQ quantizer.  The quantizer
        # clamps any non-empty scale to at least one token (see
        # WavescaleResidualVectorQuantize.forward: ``scale += 1 if scale == 0``).
        # Without the same clamp here, tiny scales such as 0.01 * frame_size can
        # become length 0 in AAR while the target code tensor has length 1,
        # causing logits/labels mismatches during training.
        scales = [max(1, int(sf * self.frame_size)) for sf in vae_local.scale_factor[::-1] + vae_local.scale_factor[1:]] if vae_local.use_wavescale else [max(1, int(sf * self.frame_size)) for sf in vae_local.scale_factor]
        self.scale_factor = sorted(scales) if self.use_scale_order else scales
        self.L = sum(self.scale_factor)
        self.first_l = self.scale_factor[0]
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.scale_factor):
            self.begin_ends.append((cur, cur+pn))
            cur += pn
        
        self.num_stages_minus_1 = len(self.scale_factor) - 1
        self.rng = torch.Generator(device='cuda')
        
        # 1. input (word) embedding
        quant: WavescaleResidualVectorQuantize = vae_local.quantizer
        self.vae_proxy: Tuple[EMAC] = (vae_local,)
        self.vae_quant_proxy: Tuple[WavescaleResidualVectorQuantize] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding
        self.class_emb = MLP(in_features=input_dim, hidden_features=self.C, out_features=self.D)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(1, input_dim) / input_dim ** 0.5))
        init_std = math.sqrt(1 / self.C / 3)
        # self.num_classes = num_classes
        # self.selecting_idx = torch.full((1, num_classes), fill_value=1/num_classes, dtype=torch.float32, device=dist.get_device())
        # self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        # nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.scale_factor):
            pe = torch.empty(1, pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.scale_factor), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln and self.using_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        drop_path_rate = 0.1 * depth / 24 if drop_path_rate is None else drop_path_rate
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSABlock(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                tau=tau, cos_attn=cos_attn,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            ) if self.using_aln else SABlock(
                layer_scale=layer_scale,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                tau=tau, cos_attn=cos_attn,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        if self.blocks[-1].fused_add_norm_fn is not None:
            self.gamma2_last = nn.Parameter(self.layer_scale * torch.ones(embed_dim), requires_grad=True) if self.layer_scale >= 0 else 1
        else:
            self.gamma2_last = None
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [vGPT config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        lvl_ids = torch.cat([torch.full((pn,), i) for i, pn in enumerate(self.scale_factor)])
        if self.use_scale_order and self.use_blockwise and hasattr(quant, "ar_stage_blocks"):
            stage_to_pos = {stage_idx: pos for pos, stage_idx in enumerate(quant.ar_input_seq)}
            block_ids_by_pos = torch.empty(len(self.scale_factor), dtype=torch.long)
            for block_idx, block in enumerate(quant.ar_stage_blocks):
                for stage_idx in block:
                    block_ids_by_pos[stage_to_pos[stage_idx]] = block_idx
            mask_ids = torch.cat([torch.full((pn,), int(block_ids_by_pos[i])) for i, pn in enumerate(self.scale_factor)])
        else:
            mask_ids = lvl_ids

        d: torch.Tensor = mask_ids.view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = lvl_ids.view(1, self.L).contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        if self.using_aln:
            self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
            self.head = nn.Linear(self.C, self.V)
        else:
            self.head_nm = MultiInpIdentity()
            self.head = nn.Sequential(norm_layer(self.C), nn.Linear(self.C, self.V))
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # is h_and_residual, so fused_add_norm must be used, so self.gamma2_last is not None
            h = resi + self.gamma2_last * self.blocks[-1].drop_path(h)
        else:   # is h, so fused_add_norm is not used, and self.gamma2_last is None
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    def _ar_position_blocks(self):
        """Return AAR stage-position blocks, e.g. [[0], [1, 2], [3, 4]]."""
        quant = self.vae_quant_proxy[0]
        if self.use_scale_order and self.use_blockwise and hasattr(quant, "ar_stage_blocks"):
            stage_to_pos = {stage_idx: pos for pos, stage_idx in enumerate(quant.ar_input_seq)}
            return [[stage_to_pos[stage_idx] for stage_idx in block] for block in quant.ar_stage_blocks]
        return [[i] for i in range(len(self.scale_factor))]

    def _add_stage_to_fhat(self, stage_pos: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> torch.Tensor:
        """Decode one predicted AAR stage embedding and add it to f_hat."""
        quant = self.vae_quant_proxy[0]
        f_hat.add_(quant.decode_stage_to_full(h_BChw, stage_pos, self.frame_size, self.use_offset, self.use_scale_order))
        return f_hat

    def _decode_fhat_to_audio(self, f_hat: torch.Tensor) -> torch.Tensor:
        """Decode f_hat, padding latent length for decoder local-attention windows.

        EMAC.preprocess() pads input audio to ``hop_length * attn_window_size`` so
        encoded latent lengths are divisible by ``attn_window_size``.  AAR can be
        configured with an arbitrary/inferred ``frame_size``; when it is not
        divisible by the decoder's local-attention window, LocalMHA fails while
        rearranging ``T`` into windows.  Pad only for decoding; callers that need
        an exact duration already crop the waveform afterwards.
        """
        vae = self.vae_proxy[0]
        attn_window_size = getattr(vae, "attn_window_size", None)
        if attn_window_size is not None:
            attn_window_size = int(attn_window_size)
            if attn_window_size > 1:
                T = f_hat.shape[-1]
                right_pad = math.ceil(T / attn_window_size) * attn_window_size - T
                if right_pad > 0:
                    f_hat = F.pad(f_hat, (0, right_pad))
        return vae.fhat_to_audio(f_hat)

    def _chunk_position_encoding(self, label_B: torch.Tensor, chunk_idx: int, num_chunks: int) -> torch.Tensor:
        """Return a deterministic sinusoidal chunk-position encoding in CLAP/condition space.

        AAR checkpoints are trained with a fixed RVQ frame window, so this helper is
        intentionally parameter-free: it can be used with existing checkpoints without
        changing their state_dict.  The encoding is added before ``class_emb`` so the
        current chunk receives explicit absolute progress information.
        """
        dim = label_B.shape[-1]
        if dim <= 0:
            return torch.zeros_like(label_B)

        denom = max(1, num_chunks - 1)
        pos = torch.tensor(float(chunk_idx) / denom, device=label_B.device, dtype=label_B.float().dtype)
        half = (dim + 1) // 2
        freq = torch.exp(
            torch.arange(half, device=label_B.device, dtype=label_B.float().dtype)
            * (-math.log(10000.0) / max(1, half - 1))
        )
        pe = torch.empty(dim, device=label_B.device, dtype=label_B.float().dtype)
        pe[0::2] = torch.sin(pos * freq[: pe[0::2].numel()])
        pe[1::2] = torch.cos(pos * freq[: pe[1::2].numel()])
        return pe.view(1, dim).expand(label_B.shape[0], -1).to(dtype=label_B.dtype)

    def _compose_chunk_condition(
        self,
        label_B: torch.Tensor,
        prev_chunk_condition_BD: Optional[torch.Tensor],
        chunk_idx: int,
        num_chunks: int,
        prev_condition_weight: float,
        position_weight: float,
    ) -> torch.Tensor:
        """Mix global condition, previous-chunk condition, and current chunk position."""
        cond = label_B
        if prev_chunk_condition_BD is not None and prev_condition_weight > 0:
            if prev_chunk_condition_BD.shape != label_B.shape:
                raise ValueError(
                    "prev_chunk_condition_BD must match label_B shape, got "
                    f"{tuple(prev_chunk_condition_BD.shape)} vs {tuple(label_B.shape)}"
                )
            cond = (1.0 - prev_condition_weight) * cond + prev_condition_weight * prev_chunk_condition_BD
        if position_weight > 0:
            cond = cond + position_weight * self._chunk_position_encoding(label_B, chunk_idx, num_chunks)
        return cond
    
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: torch.Tensor,
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.selecting_idx, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        sos = cond_BD = self.class_emb(torch.cat((label_B, self.uncond_embedding.expand_as(label_B)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC        
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l] 
        
        f_hat = sos.new_zeros(B, self.Cvae, self.scale_factor[-1])

        pos_blocks = self._ar_position_blocks()
        pos_starts = [0]
        for pn in self.scale_factor:
            pos_starts.append(pos_starts[-1] + pn)

        for b in self.blocks: b.attn.kv_caching(True)
        for block_i, pos_block in enumerate(pos_blocks):
            ratio = max(pos_block) / self.num_stages_minus_1
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            SABlock.forward
            x = next_token_map
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if more_smooth:
                raise NotImplementedError("more_smooth=True is not implemented for Wavescale codebook wrappers")

            cur = 0
            for stage_pos in pos_block:
                pn = self.scale_factor[stage_pos]
                seg_idx_Bl = idx_Bl[:, cur:cur + pn]
                f_hat = self._add_stage_to_fhat(stage_pos, f_hat, seg_idx_Bl)
                cur += pn

            if block_i != len(pos_blocks) - 1:   # prepare shared previous-block context for next same-scale block
                next_pos_block = pos_blocks[block_i + 1]
                next_maps = []
                for next_stage_pos in next_pos_block:
                    next_maps.append(F.interpolate(f_hat, size=self.scale_factor[next_stage_pos], mode="area").view(B, self.Cvae, -1).transpose(1, 2))
                next_token_map = torch.cat(next_maps, dim=1)
                next_start = pos_starts[next_pos_block[0]]
                next_end = pos_starts[next_pos_block[-1] + 1]
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, next_start:next_end]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False)
        return self._decode_fhat_to_audio(f_hat)   # de-normalize, from [-1, 1] to [0, 1]

    @torch.no_grad()
    def autoregressive_infer_cfg_chunked(
        self,
        B: int,
        label_B: torch.Tensor,
        target_frames: Optional[int] = None,
        target_samples: Optional[int] = None,
        g_seed: Optional[int] = None,
        cfg=1.5,
        top_k=0,
        top_p=0.0,
        more_smooth=False,
        prev_condition_weight: float = 0.5,
        position_weight: float = 0.1,
        condition_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Generate longer-than-frame audio by rolling AAR over RVQ-frame chunks.

        If the requested target length fits in one RVQ frame, this falls back to
        ``autoregressive_infer_cfg``.  Otherwise, each chunk is generated with a
        condition composed of:

        1. the original/global condition ``label_B``;
        2. the previous chunk condition (either from ``condition_fn`` or the previous
           composed condition); and
        3. deterministic sinusoidal position information for the current chunk.

        ``target_frames`` is in latent/RVQ frames.  ``target_samples`` is converted
        with ``vae.hop_length`` and the returned waveform is cropped to exactly that
        sample length when provided.
        """
        if target_frames is None and target_samples is None:
            return self.autoregressive_infer_cfg(
                B=B, label_B=label_B, g_seed=g_seed, cfg=cfg,
                top_k=top_k, top_p=top_p, more_smooth=more_smooth,
            )

        if target_frames is None:
            hop_length = int(getattr(self.vae_proxy[0], "hop_length", 1))
            target_frames = int(math.ceil(int(target_samples) / max(1, hop_length)))
        target_frames = int(target_frames)
        if target_frames <= 0:
            raise ValueError(f"target_frames must be positive, got {target_frames}")

        if target_frames <= self.frame_size:
            wav = self.autoregressive_infer_cfg(
                B=B, label_B=label_B, g_seed=g_seed, cfg=cfg,
                top_k=top_k, top_p=top_p, more_smooth=more_smooth,
            )
            return wav[..., :target_samples] if target_samples is not None else wav

        num_chunks = int(math.ceil(target_frames / self.frame_size))
        chunk_wavs = []
        prev_cond = None
        for chunk_idx in range(num_chunks):
            chunk_label_B = self._compose_chunk_condition(
                label_B=label_B,
                prev_chunk_condition_BD=prev_cond,
                chunk_idx=chunk_idx,
                num_chunks=num_chunks,
                prev_condition_weight=prev_condition_weight,
                position_weight=position_weight,
            )
            chunk_seed = None if g_seed is None else int(g_seed) + chunk_idx
            chunk_wav = self.autoregressive_infer_cfg(
                B=B,
                label_B=chunk_label_B,
                g_seed=chunk_seed,
                cfg=cfg,
                top_k=top_k,
                top_p=top_p,
                more_smooth=more_smooth,
            )
            remaining_frames = target_frames - chunk_idx * self.frame_size
            if remaining_frames < self.frame_size:
                hop_length = int(getattr(self.vae_proxy[0], "hop_length", 1))
                chunk_wav = chunk_wav[..., : remaining_frames * hop_length]
            chunk_wavs.append(chunk_wav)

            if condition_fn is not None:
                next_cond = condition_fn(chunk_wav, chunk_idx)
                if not torch.is_tensor(next_cond):
                    raise TypeError("condition_fn must return a torch.Tensor")
                prev_cond = next_cond.to(device=label_B.device, dtype=label_B.dtype)
            else:
                prev_cond = chunk_label_B.detach()

        wav = torch.cat(chunk_wavs, dim=-1)
        return wav[..., :target_samples] if target_samples is not None else wav
    
    @torch.no_grad()
    def autoregressive_infer_teacher_forcing(
        self,
        label_B: torch.Tensor,
        x_BLCv_wo_first_l: torch.Tensor,
        use_argmax: bool = True,
    ) -> torch.Tensor:
        """
        Teacher forcing 모드로 AR prior를 평가하면서 오디오를 복원하는 함수.

        - 입력은 학습 때 forward()에 넣는 것과 동일한 label_B, x_BLCv_wo_first_l.
        - GT latent 시퀀스를 한 번에 넣어서 트랜스포머를 통과시키고
        각 위치의 logits 로부터 토큰을 선택(기본: argmax)한 뒤,
        VAE 경로를 통해 최종 오디오를 복원한다.
        """
        B = x_BLCv_wo_first_l.shape[0]
        device = label_B.device

        # progressive training 이면 전체 길이를 쓰도록 ed 결정 (forward()와 동일한 로직)
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)

        # ---- 1) forward() 와 동일하게 입력 셋업 (cond_drop 없음) ----
        with torch.autocast(device_type=device.type, enabled=False):
            # cond_drop 없이 그냥 class_emb
            cond_BD = self.class_emb(label_B)          # (B, D)
            sos = cond_BD.unsqueeze(1).expand(B, self.first_l, -1) \
                + self.pos_start.expand(B, self.first_l, -1)

            if self.prog_si == 0:
                x_BLC = sos
            else:
                # teacher forcing input 임베딩
                x_BLC = torch.cat(
                    (sos, self.word_embed(x_BLCv_wo_first_l.float())),
                    dim=1,
                )  # (B, L, D)

            # level / position embedding
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) \
                    + self.pos_1LC[:, :ed]

        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]

        # shared adaLN conditioning
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        # mixed precision dtype 맞추기 (forward()와 동일)
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        # ---- 2) 트랜스포머 통과해서 logits 얻기 ----
        for b in self.blocks:
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)

        logits_BLV = self.get_logits(x_BLC.float(), cond_BD)   # (B, L, V)        
        # teacher forcing이지만 “예측 결과”를 보고 싶으니 여기서 토큰 선택
        if use_argmax:
            idx_Bl = logits_BLV.argmax(dim=-1)                 # (B, L)
        else:
            # 필요하면 샘플링 버전도 사용할 수 있음
            probs = torch.softmax(logits_BLV, dim=-1)
            idx_Bl = torch.distributions.Categorical(probs=probs).sample()

        # ---- 3) autoregressive_infer_cfg 와 동일한 VAE 디코딩 경로 ----
        f_hat = logits_BLV.new_zeros(B, self.Cvae, self.scale_factor[-1])

        cur_L = 0
        for si, pn in enumerate(self.scale_factor):
            # 이번 스테이지에서 쓸 토큰 슬라이스
            seg_idx_Bl = idx_Bl[:, cur_L:cur_L + pn]          # (B, pn)

            # 토큰을 full-length VAE contribution으로 복원해서 누적
            f_hat = self._add_stage_to_fhat(si, f_hat, seg_idx_Bl)

            cur_L += pn

        # 최종 오디오 복원
        return self._decode_fhat_to_audio(f_hat)   # [-1,1] -> [0,1]
    
    def forward(self, label_B: torch.Tensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        with torch.autocast(device_type=label_B.device.type, enabled=False):
            label_B = torch.where(torch.rand(B,1, device=label_B.device) < self.cond_drop_rate, self.uncond_embedding, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            
            if self.prog_si == 0: x_BLC = sos
            else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC

        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        SABlock.forward, AdaLNSABlock.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC    # logits BLV, V is vocab_size
    
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSABlock
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):   # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2*C))
    
    def forward(self, x_BLC: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)


class MultiInpIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    

if __name__ == "__main__":
    from transformers import ClapModel, ClapProcessor
    from audiotools import AudioSignal
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cond_processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
    cond_model = ClapModel.from_pretrained("laion/larger_clap_general").to(device)
    cond_model.eval()
    
    B = 16  # batch size
    T = 441000
    
    # 예시용 하이퍼파라미터
    input = AudioSignal(torch.randn(B, 1, T, device=device), sample_rate=44100)
    cond = cond_processor(audio=[data.squeeze(0).cpu().numpy() for data in input.resample(48000).audio_data], return_tensors="pt", sampling_rate=48000)
    
    vae_local = EMAC(
        codebook_dim=8,
        codebook_size=1024,
        sample_rate=44100,
        encoder_rates = [2, 4, 8, 8],
        decoder_rates = [8, 8, 4, 2],
        scale_factor = [0.1, 0.5, 1],
        phi_kernel=[9, 9]
    ).to(device).eval()
    
    model = AAR(
        vae_local=vae_local,
        input_dim=512,
        depth=4,
        embed_dim=256,
        num_heads=4,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        aln=1,
        aln_gamma_init=1e-3,
        shared_aln=False,
        frame_size=938,
        use_offset=False
    ).to(device)
    
    model.init_weights()
    
    print("Cvae:", model.Cvae)
    print("V (vocab size):", model.V)
    print("L (total tokens):", model.L)
    print("first_l:", model.first_l)

    with torch.no_grad():
        _, codes, *_ = vae_local.encode(vae_local.preprocess(input.audio_data, vae_local.sample_rate))
        aar_input = vae_local.quantizer.get_aar_input(codes, model.use_offset, model.use_scale_order, model.use_blockwise)
        conditions = cond_model.get_audio_features(input_features=cond['input_features'].to(device), is_longer=cond['is_longer'].to(device))
    
    print('code shape: ', [c.shape for c in codes])
    aar_input = torch.concat(aar_input, dim=1).to(device)
    
    print('aar input: ', aar_input.shape)
    print('condition: ', conditions.shape)

    model.eval()
    
    loss = CELoss(reduction='none')
    with torch.no_grad():
        logits = model(conditions, aar_input)
        l = loss(logits, vae_local.quantizer.get_aar_target_codes(codes, model.use_offset, model.use_scale_order))

    print("forward ok.")
    print("logits shape:", logits.shape)  # 예상: (B, L, V)
    print("loss: ", l)
    
    out = model.autoregressive_infer_cfg(B=len(conditions), label_B=conditions, cfg=4.0, top_k=900, top_p=0.95, g_seed=42)
    out_tf = model.autoregressive_infer_teacher_forcing(label_B=conditions, x_BLCv_wo_first_l=aar_input)
    
    print(out.shape)
    print(out_tf.shape)
