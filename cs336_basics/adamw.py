import math
from collections.abc import Callable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, betas: tuple[float, float], weight_decay: float, eps: float):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "b1": betas[0], "b2": betas[1], "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

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
                t = state.get("t", 1)  # Get iteration number from the state, or 1.
                lr = group["lr"]  # Get the learning rate.
                lr_t = (
                    lr * math.sqrt(1.0 - b2**t) / (1.0 - b1**t)
                )  # Compute the adjusted learning rate for iteration t.
                p.data -= lr * weight_decay * p.data  # Apply weight decay.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                m = state.get("m", 0)  # Get the first moment estimate, or 0.
                m = b1 * m + (1 - b1) * grad  # Update the first moment estimate.
                v = state.get("v", 0)  # Get the second moment estimate, or 0.
                v = b2 * v + (1 - b2) * (grad**2)  # Update the second moment estimate.
                p.data -= lr_t * m / (torch.sqrt(v) + eps)  # Apply the moment adjusted weight updates.
                self.state[p]["t"] = t + 1  # Increment iteration number.
                self.state[p]["m"] = m  # Save the first moment estimation for iteration t.
                self.state[p]["v"] = v  # Save the second moment estimation for iteration t.

        return loss
