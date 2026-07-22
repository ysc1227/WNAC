from pathlib import Path
import inspect

import argbind
import torch
from audiotools import ml

import emac
from emac.model.aar import AAR

from .audio_utils import _check_checksum, _linear_overlap_add, _get_checkpoint_url, seed_everything
# from .wandb import CustomWandbTracker
from .lr_control import filter_params, lr_wd_annealing


EMAC = emac.model.EMAC
SATCodec = emac.model.SATCodec
SNACCodec = emac.model.SNACCodec
Accelerator = ml.Accelerator

__MODEL_LATEST_TAGS__ = {
    ("44khz", "8kbps"): "0.0.1",
    ("24khz", "8kbps"): "0.0.4",
    ("16khz", "8kbps"): "0.0.5",
    ("44khz", "16kbps"): "1.0.0",
}

__MODEL_URLS__ = {
    (
        "44khz",
        "0.0.1",
        "8kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.1/weights.pth",
    (
        "24khz",
        "0.0.4",
        "8kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.4/weights_24khz.pth",
    (
        "16khz",
        "0.0.5",
        "8kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.5/weights_16khz.pth",
    (
        "44khz",
        "1.0.0",
        "16kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/1.0.0/weights_44khz_16kbps.pth",
}


@argbind.bind(group="download", positional=True, without_prefix=True)
def download(
    model_type: str = "44khz", model_bitrate: str = "8kbps", tag: str = "latest"
):
    """
    Function that downloads the weights file from URL if a local cache is not found.

    Parameters
    ----------
    model_type : str
        The type of model to download. Must be one of "44khz", "24khz", or "16khz". Defaults to "44khz".
    model_bitrate: str
        Bitrate of the model. Must be one of "8kbps", or "16kbps". Defaults to "8kbps".
        Only 44khz model supports 16kbps.
    tag : str
        The tag of the model to download. Defaults to "latest".

    Returns
    -------
    Path
        Directory path required to load model via audiotools.
    """
    model_type = model_type.lower()
    tag = tag.lower()

    assert model_type in [
        "44khz",
        "24khz",
        "16khz",
    ], "model_type must be one of '44khz', '24khz', or '16khz'"

    assert model_bitrate in [
        "8kbps",
        "16kbps",
    ], "model_bitrate must be one of '8kbps', or '16kbps'"

    if tag == "latest":
        tag = __MODEL_LATEST_TAGS__[(model_type, model_bitrate)]

    download_link = __MODEL_URLS__.get((model_type, tag, model_bitrate), None)

    if download_link is None:
        raise ValueError(
            f"Could not find model with tag {tag} and model type {model_type}"
        )

    local_path = (
        Path.home()
        / ".cache"
        / "descript"
        / "wnac"
        / f"weights_{model_type}_{model_bitrate}_{tag}.pth"
    )
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the model
        import requests

        response = requests.get(download_link)

        if response.status_code != 200:
            raise ValueError(
                f"Could not download model. Received response code {response.status_code}"
            )
        local_path.write_bytes(response.content)

    return local_path


def load_model(
    model_type: str = "44khz",
    model_bitrate: str = "8kbps",
    tag: str = "latest",
    load_path: str = None,
):
    if not load_path:
        load_path = download(
            model_type=model_type, model_bitrate=model_bitrate, tag=tag
        )

    def _load_state_dict_checkpoint(model_cls, kwargs, state_dict, metadata=None):
        valid_args = set(inspect.signature(model_cls.__init__).parameters.keys()) - {"self"}
        fixed_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        generator = model_cls(**fixed_kwargs)
        generator.load_state_dict(state_dict, strict=True)
        if metadata is not None:
            metadata = dict(metadata)
            metadata["kwargs"] = fixed_kwargs
            generator.metadata = metadata
        return generator

    def _infer_depthwise_from_state_dict(state_dict):
        # The first decoder convolution is depthwise-only when depthwise=True.
        weight = state_dict.get("decoder.model.0.weight_v")
        if weight is None:
            weight = state_dict.get("decoder.model.0.weight")
        if weight is not None and getattr(weight, "ndim", 0) == 3:
            return int(weight.shape[1]) == 1
        return None

    # Some early Wavescale checkpoints were saved without `use_wavescale` in
    # metadata kwargs even though their state_dict contains the expanded
    # U-shaped quantizer stack.  EMAC.load() then reconstructs a non-wavescale
    # model with only len(scale_factor) quantizers and silently drops the extra
    # codebooks.  Detect this from the checkpoint structure and reconstruct the
    # model with use_wavescale=True before loading weights.
    try:
        ckpt = torch.load(load_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt and "metadata" in ckpt:
            metadata = ckpt.get("metadata", {}) or {}
            kwargs = dict(metadata.get("kwargs", {}) or {})
            scale_factor = kwargs.get("scale_factor", None)
            state_dict = ckpt["state_dict"]
            state_keys = set(state_dict.keys())
            inferred_depthwise = _infer_depthwise_from_state_dict(state_dict)
            metadata_depthwise = kwargs.get("depthwise", None)
            if inferred_depthwise is not None:
                kwargs["depthwise"] = bool(inferred_depthwise)

            if "multi_scale" in kwargs or any(k.startswith("quantizer.vq.layers.") for k in state_keys):
                return _load_state_dict_checkpoint(SATCodec, kwargs, state_dict, metadata)

            if "vq_strides" in kwargs:
                return _load_state_dict_checkpoint(SNACCodec, kwargs, state_dict, metadata)

            quantizer_ids = sorted({
                int(k.split("quantizer.quantizers.")[1].split(".")[0])
                for k in state_dict.keys()
                if "quantizer.quantizers." in k and "codebook.weight" in k
            })
            # Legacy DAC/RVQ checkpoints stored `n_codebooks` and left
            # `scale_factor=None`.  In the current EMAC constructor,
            # `scale_factor` defines the number of RVQ codebooks.  A DAC-style
            # full-rate RVQ stack is equivalent to one scale factor of 1.0 per
            # codebook, which matches the checkpoint key layout:
            #   quantizer.quantizers.{0..n_codebooks-1}.*
            if scale_factor is None:
                n_codebooks = kwargs.get("n_codebooks", None) or len(quantizer_ids)
                if n_codebooks:
                    valid_args = set(inspect.signature(EMAC.__init__).parameters.keys()) - {"self"}
                    fixed_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
                    fixed_kwargs["scale_factor"] = [1.0] * int(n_codebooks)
                    fixed_kwargs["use_wavescale"] = bool(kwargs.get("use_wavescale", kwargs.get("wavescale", False)))
                    generator = EMAC(**fixed_kwargs)
                    generator.load_state_dict(state_dict, strict=True)
                    metadata = dict(metadata)
                    metadata["kwargs"] = fixed_kwargs
                    generator.metadata = metadata
                    return generator

            expected_wavescale = len(scale_factor) * 2 - 1 if scale_factor is not None else None
            metadata_wavescale = bool(kwargs.get("use_wavescale", False))
            looks_like_wavescale = expected_wavescale is not None and len(quantizer_ids) == expected_wavescale and expected_wavescale != len(scale_factor)

            if looks_like_wavescale and not metadata_wavescale:
                valid_args = set(inspect.signature(EMAC.__init__).parameters.keys()) - {"self"}
                fixed_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
                fixed_kwargs["use_wavescale"] = True
                generator = EMAC(**fixed_kwargs)
                generator.load_state_dict(state_dict, strict=True)
                metadata = dict(metadata)
                metadata["kwargs"] = fixed_kwargs
                generator.metadata = metadata
                return generator

            if inferred_depthwise is not None and metadata_depthwise != inferred_depthwise:
                return _load_state_dict_checkpoint(EMAC, kwargs, state_dict, metadata)
    except Exception as exc:
        print(f"[load_model] Falling back to EMAC.load({load_path!r}) after checkpoint inspection failed: {exc}", flush=True)

    generator = EMAC.load(load_path)
    return generator
