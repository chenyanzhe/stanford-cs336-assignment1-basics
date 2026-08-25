import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.eps = eps
        gain = torch.empty(d_model, device=device, dtype=dtype)
        torch.nn.init.constant_(gain, 1)
        self.W = nn.Parameter(gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms_x = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        result = x / rms_x * self.W

        return result.to(in_dtype)
