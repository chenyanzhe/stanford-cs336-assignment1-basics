import einops
import torch
from torch import nn

from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float | None = None,
        token_positions: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.L_q = Linear(d_model, d_model)
        self.L_k = Linear(d_model, d_model)
        self.L_v = Linear(d_model, d_model)
        self.L_o = Linear(d_model, d_model)

        self.use_rope = False
        if theta is not None:
            self.use_rope = True
            self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
            if token_positions is not None:
                self.token_positions = token_positions
            else:
                self.token_positions = torch.arange(max_seq_len)

        self.mask = torch.triu(torch.full((max_seq_len, max_seq_len), True))
        self.mask = einops.rearrange(self.mask, "i j -> j i")
        self.mask.unsqueeze(0).unsqueeze(0)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        Q = self.L_q(in_features)
        K = self.L_k(in_features)
        V = self.L_v(in_features)
        Q_h = einops.rearrange(Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        K_h = einops.rearrange(K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        V_h = einops.rearrange(V, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)

        if self.use_rope:
            Q_h = self.rope(Q_h, self.token_positions)
            K_h = self.rope(K_h, self.token_positions)

        attention_h = scaled_dot_product_attention(Q_h, K_h, V_h, self.mask)
        attention = einops.rearrange(attention_h, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")
        return self.L_o(attention)
