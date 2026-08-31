import torch
from torch import nn

from cs336_basics.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        out = in_features + self.attn(self.ln1(in_features))
        return out + self.ffn(self.ln2(out))
