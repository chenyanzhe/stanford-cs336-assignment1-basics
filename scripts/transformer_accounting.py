def count_model_trainable_parameters(
    vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int
):
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
    print(f"Our model will have {total_params} trainable parameters:")
    print(f"  - embeddings: {token_embeddings} ", f"({token_embeddings / total_params * 100:.2f}%)")
    print(f"  - rms: {rms * num_layers} ", f"({rms * num_layers / total_params * 100:.2f}%)")
    print(f"  - ffn: {ffn * num_layers} ", f"({ffn * num_layers / total_params * 100:.2f}%)")
    print(f"  - mha: {mha * num_layers} ", f"({mha * num_layers / total_params * 100:.2f}%)")
    print(f"  - final rms: {final_rms} ", f"({final_rms / total_params:.2f}%)")
    print(f"  - lm_head: {lm_head} ", f"({lm_head / total_params:.2f}%)")


def count_flops(vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int):
    embeddings = 0
    mha = 2 * (
        context_length * d_model * d_model * 4  # Q, K, V, O
        + context_length * context_length * d_model  # Q @ K^T
        + context_length * context_length * d_model  # attn @ V
    )
    ffn = 2 * (
        context_length * d_model * d_ff * 3  # w1, w2, w3
    )
    rms = 2 * (
        context_length * d_model * 2  # ln1, ln2
    )
    final_rms = 2 * context_length * d_model
    lm_head = 2 * context_length * d_model * vocab_size
    total_flops = embeddings + (mha + ffn + rms) * num_layers + final_rms + lm_head
    print(f"The matrix multiplies require in total {total_flops} FLOPs:")
    print(f"  - embeddings: {embeddings} ", f"({embeddings / total_flops * 100:.2f}%)")
    print(f"  - rms: {rms * num_layers} ", f"({rms * num_layers / total_flops * 100:.2f}%)")
    print(f"  - ffn: {ffn * num_layers} ", f"({ffn * num_layers / total_flops * 100:.2f}%)")
    print(f"  - mha: {mha * num_layers} ", f"({mha * num_layers / total_flops * 100:.2f}%)")
    print(f"  - final rms: {final_rms} ", f"({final_rms / total_flops:.2f}%)")
    print(f"  - lm_head: {lm_head} ", f"({lm_head / total_flops:.2f}%)")


if __name__ == "__main__":
    count_model_trainable_parameters(
        vocab_size=50257, context_length=1024, num_layers=48, d_model=1600, num_heads=25, d_ff=4288
    )
    count_flops(vocab_size=50257, context_length=1024, num_layers=48, d_model=1600, num_heads=25, d_ff=4288)
