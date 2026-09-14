"""FP32 EMA of original trainable FSDP shards, with reversible inference swaps."""

from contextlib import contextmanager

import torch

from utils.distributed import canonical_name


def decay_change_preserves_history(state, decay, warmup):
    """An unsaturated warmup has not used either decay cap yet."""
    updates = state["num_updates"]
    return (
        state["warmup"]
        and warmup
        and isinstance(updates, int)
        and updates >= 0
        and 0 < state["decay"] < 1
        and 0 < decay < 1
        and (updates == 0 or (1 + updates) / (10 + updates) <= min(state["decay"], decay))
    )


class ShardedEMA:
    """Average only trainable local shards, without gathering full parameters.

    The recipe fixes decay at 0.995. Each successful optimizer update advances
    EMA once; validation temporarily installs averages and restores raw weights.
    """

    def __init__(self, model, decay, device="cpu", warmup=True):
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be between zero and one")
        self.decay, self.warmup = float(decay), bool(warmup)
        self.num_updates, self.swapped = 0, False
        self.weights = {
            name: value.detach().to(device=device, dtype=torch.float32, copy=True)
            for name, value in self.parameters(model).items()
        }
        if not self.weights:
            raise ValueError("EMA requires trainable parameters")

    @staticmethod
    def parameters(model):
        return {
            canonical_name(name): value for name, value in model.named_parameters() if value.requires_grad
        }

    @classmethod
    def from_config(cls, model, cfg):
        if not cfg.ema_weight:
            return None
        device = "cpu" if cfg.get("ema_cpu_offload", True) else next(model.parameters()).device
        return cls(model, cfg.ema_weight, device, cfg.get("ema_warmup", True))

    def checked_parameters(self, model):
        parameters = self.parameters(model)
        if parameters.keys() != self.weights.keys():
            raise ValueError("EMA trainable parameter scope changed")
        for name, value in parameters.items():
            if value.shape != self.weights[name].shape:
                raise ValueError(f"EMA requires local FSDP shards outside forward/backward: {name}")
        return parameters

    @property
    def current_decay(self):
        return (
            min(self.decay, (1 + self.num_updates) / (10 + self.num_updates)) if self.warmup else self.decay
        )

    @torch.no_grad()
    def update(self, model):
        if self.swapped:
            raise RuntimeError("Cannot update EMA while averaged parameters are installed")
        parameters = self.checked_parameters(model)
        self.num_updates += 1
        values = [
            value.detach().to(device=self.weights[name].device, dtype=torch.float32)
            for name, value in parameters.items()
        ]
        torch._foreach_lerp_(list(self.weights.values()), values, 1 - self.current_decay)

    def state_dict(self):
        return dict(
            decay=self.decay,
            warmup=self.warmup,
            num_updates=self.num_updates,
            weights={name: value.cpu() for name, value in self.weights.items()},
        )

    @torch.no_grad()
    def load_state_dict(self, state, *, allow_schedule_change=False):
        if not 0 < state["decay"] < 1 or not isinstance(state["warmup"], bool):
            raise ValueError("Invalid saved EMA schedule")
        if not allow_schedule_change and (
            state["warmup"] != self.warmup
            or (
                state["decay"] != self.decay
                and not decay_change_preserves_history(state, self.decay, self.warmup)
            )
        ):
            raise ValueError("EMA schedule differs from the checkpoint")
        if state["weights"].keys() != self.weights.keys():
            raise ValueError("EMA checkpoint parameter scope changed")
        if not isinstance(state["num_updates"], int) or state["num_updates"] < 0:
            raise ValueError("Invalid EMA update count")
        for name, value in state["weights"].items():
            if value.shape != self.weights[name].shape or value.dtype != torch.float32:
                raise ValueError(f"EMA checkpoint shard differs for {name}")
        for name, value in state["weights"].items():
            self.weights[name].copy_(value)
        self.num_updates = state["num_updates"]

    @contextmanager
    def average_parameters(self, model):
        if self.swapped:
            raise RuntimeError("EMA parameter swaps cannot be nested")
        parameters = self.checked_parameters(model)
        # FSDP may still be reading pinned CPU shards in asynchronous H2D
        # copies after forward returns. Finish those reads before changing them.
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        backup = {
            name: value.detach().to(device=self.weights[name].device, copy=True)
            for name, value in parameters.items()
        }
        self.swapped = True
        try:
            with torch.no_grad():
                for name, value in parameters.items():
                    value.copy_(self.weights[name])
            yield
        finally:
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
            with torch.no_grad():
                for name, value in self.checked_parameters(model).items():
                    value.copy_(backup[name])
            self.swapped = False


def inference_weight_kind(cfg, validation=False):
    value = cfg.get("validation_weights" if validation else "inference_weights", "auto")
    return ("ema" if cfg.ema_weight else "raw") if value == "auto" else value
