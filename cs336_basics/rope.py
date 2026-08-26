import torch
import einops
from torch import nn


def rotate_half(x):
    x = einops.rearrange(x, "... (d pair) -> ... d pair", pair=2)
    x = x[..., [1, 0]]
    x[..., 0] *= -1
    return einops.rearrange(x, "... d pair -> ... (d pair)", pair=2)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        i = torch.arange(max_seq_len, device=device)
        k = torch.arange(start=1, end=(d_k // 2 + 1), device=device)
        theta_ik = torch.outer(i, torch.pow(torch.full((d_k // 2,), theta), (2 - 2 * k) / d_k))
        cos = torch.cos(theta_ik)
        sin = torch.sin(theta_ik)

        self.register_buffer("cos", torch.repeat_interleave(cos, repeats=2, dim=-1), persistent=False)
        self.register_buffer("sin", torch.repeat_interleave(sin, repeats=2, dim=-1), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        return x * self.cos[token_positions] + rotate_half(x) * self.sin[token_positions]
