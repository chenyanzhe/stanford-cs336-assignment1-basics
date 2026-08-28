# scores = einsum(Q, K "... n d_k, ... m d_k -> ... n m")
# scores[~mask] = -inf
# scores @ V
