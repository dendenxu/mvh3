"""AdamW with FP32 master weights for updates to BF16 pretrained parameters."""

import torch


class MasterAdamW:
    """Optimizer state adds no parameters to the model or its state-dict topology."""

    def __init__(self, named_parameters, lr=1e-6, betas=(0.9, 0.999), weight_decay=0.01):
        items = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
        if not items:
            raise ValueError("No trainable parameters")
        self.names = [name for name, _ in items]
        self.parameters = [parameter for _, parameter in items]
        self.masters = [parameter.detach().float().clone() for parameter in self.parameters]
        self.optimizer = torch.optim.AdamW(self.masters, lr=lr, betas=betas, weight_decay=weight_decay, foreach=False)

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, max_grad_norm=10.0):
        norm2 = 0.0
        for name, parameter, master in zip(self.names, self.parameters, self.masters):
            if parameter.grad is None:
                raise ValueError(f"Missing gradient for {name}")
            master.grad = parameter.grad.detach().float()
            norm2 += master.grad.square().sum().item()
        norm = norm2 ** 0.5
        if not torch.isfinite(torch.tensor(norm)):
            raise FloatingPointError("Nonfinite gradient norm; no optimizer update was applied")
        scale = min(1.0, max_grad_norm / (norm + 1e-12))
        for master in self.masters:
            master.grad.mul_(scale)
        self.optimizer.step()
        for parameter, master in zip(self.parameters, self.masters):
            parameter.copy_(master)
        return norm

    def state_dict(self):
        return {"names": self.names, "masters": self.masters, "adamw": self.optimizer.state_dict()}

    @torch.no_grad()
    def load_state_dict(self, state):
        if state["names"] != self.names or len(state["masters"]) != len(self.masters):
            raise ValueError("Optimizer parameter identity differs across the stage transition")
        for saved, master, parameter in zip(state["masters"], self.masters, self.parameters):
            if saved.shape != master.shape:
                raise ValueError("Optimizer master tensor shape mismatch")
            master.copy_(saved)
            parameter.copy_(master)
        self.optimizer.load_state_dict(state["adamw"])
