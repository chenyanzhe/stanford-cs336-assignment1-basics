def num_parameters(
    vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int
) -> int:
    token_embeddings = vocab_size * d_model
    # A transformer block has:
    # - 2 RMSNorm
    # - 1 FFN (SwiGLU)
    # - 1 Multi-Head Attention
    rms = 2 * d_model
    ffn = 3 * d_model * d_ff
    mha = 4 * d_model**2
    final_rms = d_model
    lm_head = d_model * vocab_size
    total_params = token_embeddings + (rms + ffn + mha) * num_layers + final_rms + lm_head
    return total_params


def num_activations(
    batch_size: int, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int
) -> int:
    # A transformer block:
    rms = 2 * batch_size * context_length * d_model * 2
    qkv_proj = batch_size * context_length * d_model * 3
    qk_multi = batch_size * num_heads * context_length**2
    softmax = batch_size * num_heads * context_length**2
    weighted_values = batch_size * context_length * d_model
    output_proj = batch_size * context_length * d_model
    ffn_w1 = batch_size * context_length * d_ff
    ffn_w2 = batch_size * context_length * d_model
    ffn_gate = batch_size * context_length * d_ff
    ffn_product = batch_size * context_length * d_ff
    ffn_w3 = batch_size * context_length * d_ff

    final_rms = batch_size * context_length * d_model
    output_embedding = batch_size * context_length * vocab_size
    cross_entropy = batch_size * context_length

    total_activations = (
        (
            rms
            + qkv_proj
            + qk_multi
            + softmax
            + weighted_values
            + output_proj
            + ffn_w1
            + ffn_w2
            + ffn_gate
            + ffn_product
            + ffn_w3
        )
        * num_layers
        + final_rms
        + output_embedding
        + cross_entropy
    )
    return total_activations


def total_memory(
    batch_size: int, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int
) -> int:
    P = num_parameters(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
    )
    A = num_activations(
        batch_size=batch_size,
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
    )
    return (P + A + P + 2 * P) * 4


if __name__ == "__main__":
    memory_budget = 80e9
    for bs in range(1, 100):
        M = total_memory(
            batch_size=bs, vocab_size=50257, context_length=1024, num_layers=48, d_model=1600, num_heads=25, d_ff=4288
        )
        if M < memory_budget:
            print(f"Required memory for batch size {bs}: {M} => within budget")
        else:
            print(f"Required memory for batch size {bs}: {M} => exceed budget")
            break
