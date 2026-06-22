from contextlib import contextmanager

import torch


class ModelEMA:
    """
    Exponential Moving Average of model parameters.
    - ema decay: e.g. 0.999 ~ 0.9999
    - keeps shadow params on the same device/dtype as model params by default
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}

        # model is expected to be the "unwrapped" model
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = p.detach().clone()
                continue
            # ensure same dtype/device
            shadow = self.shadow[name]
            if shadow.device != p.device or shadow.dtype != p.dtype:
                shadow = shadow.to(device=p.device, dtype=p.dtype)
                self.shadow[name] = shadow
            self.shadow[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    @contextmanager
    def apply_to(self, model: torch.nn.Module):
        """Temporarily swap model params to EMA params."""
        self.backup = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                self.backup[name] = p.detach().clone()
                p.copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    p.copy_(self.backup[name])
            self.backup = {}

    def state_dict(self):
        # keep tensors as-is (device/dtype)
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd):
        self.decay = float(sd.get("decay", self.decay))
        shadow = sd.get("shadow", {})
        # be permissive
        self.shadow = {k: v.clone() for k, v in shadow.items()}
