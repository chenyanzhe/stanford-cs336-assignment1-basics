import math
from collections.abc import Callable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-2,
        eps: float = 1e-8,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0.0 or betas[0] >= 1.0 or betas[1] < 0.0 or betas[1] >= 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")

        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]  # Get the b1, b2 to update the moment estimates.
            weight_decay = group["weight_decay"]  # Get the weight decay rate.
            eps = group["eps"]  # Get the epsilon.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p
                if "t" not in state:
                    state["t"] = 1
                if "m" not in state:
                    state["m"] = torch.zeros_like(p.grad)
                if "v" not in state:
                    state["v"] = torch.zeros_like(p.grad)

                lr = group["lr"]  # Get the learning rate.
                lr_t = (
                    lr * math.sqrt(1.0 - b2 ** state["t"]) / (1.0 - b1 ** state["t"])
                )  # Compute the adjusted learning rate for iteration t.
                p -= lr * weight_decay * p  # Apply weight decay.
                state["m"].mul_(b1).add_(p.grad, alpha=1 - b1)  # Update the first moment estimate.
                state["v"].mul_(b2).add_(p.grad**2, alpha=1 - b2)  # Update the second moment estimate.
                p -= lr_t * state["m"] / (torch.sqrt(state["v"]) + eps)  # Apply the moment adjusted weight updates.
                state["t"] += 1  # Increment iteration number.

        return loss
