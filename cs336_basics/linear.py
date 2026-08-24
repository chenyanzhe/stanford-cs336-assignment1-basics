import torch
import math
from torch import nn
from einops import einsum


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(weight, std=std, a=-3 * std, b=3 * std)
        self.W = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
