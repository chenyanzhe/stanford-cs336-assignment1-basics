from collections.abc import Iterable

import torch


@torch.no_grad()
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    squared_l2_norm = 0.0
    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return

    for g in grads:
        squared_l2_norm += torch.sum(torch.square(g))
    l2_norm = torch.sqrt(squared_l2_norm)
    if l2_norm > max_l2_norm:
        scale = max_l2_norm / (l2_norm + eps)
        for g in grads:
            g.mul_(scale)
