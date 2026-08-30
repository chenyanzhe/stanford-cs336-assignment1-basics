import math

import torch
from einops import einsum

from cs336_basics.softmax import softmax


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... n d_k, ... m d_k -> ... n m") / math.sqrt(d_k)
    scores.masked_fill_(~mask, float("-inf"))
    scores = softmax(scores, dim=-1)
    return einsum(scores, V, "... n m, ... m d_v -> ... n d_v")
