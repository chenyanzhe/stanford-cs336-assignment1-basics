import math
from collections.abc import Callable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, betas: tuple[float, float], weight_decay: float, eps: float):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "b1": betas[0], "b2": betas[1], "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            b1 = group["b1"]
            b2 = group["b2"]  # Get the b1, b2 to update the moment estimates.
            weight_decay = group["weight_decay"]  # Get the weight decay rate.
            eps = group["eps"]  # Get the epsilon.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                if "t" not in state:
                    state["t"] = 1
                if "m" not in state:
                    state["m"] = torch.zeros_like(grad)
                if "v" not in state:
                    state["v"] = torch.zeros_like(grad)

                t = state.get("t")  # Get iteration number from the state.
                m = state.get("m")  # Get the first moment estimate.
                v = state.get("v")  # Get the second moment estimate.
                lr = group["lr"]  # Get the learning rate.
                lr_t = (
                    lr * math.sqrt(1.0 - b2**t) / (1.0 - b1**t)
                )  # Compute the adjusted learning rate for iteration t.
                p.data -= lr * weight_decay * p.data  # Apply weight decay.
                m = b1 * m + (1 - b1) * grad  # Update the first moment estimate.
                v = b2 * v + (1 - b2) * (grad**2)  # Update the second moment estimate.
                p.data -= lr_t * m / (torch.sqrt(v) + eps)  # Apply the moment adjusted weight updates.
                state["t"] = t + 1  # Increment iteration number.
                state["m"] = m  # Save the first moment estimation for iteration t.
                state["v"] = v  # Save the second moment estimation for iteration t.

        return loss
