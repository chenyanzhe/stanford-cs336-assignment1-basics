import torch
from torch.distributions import Categorical

from cs336_basics.softmax import softmax
from cs336_basics.transformer_lm import TransformerLM


def sample_next_token(
    logits: torch.Tensor,  # Raw logits for the next token
    temperature: float = 1.0,  # Sampling temperature
    top_p: float = 1.0,  # Top-p (nucleus) sampling probability
):
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)

    if temperature != 1.0:
        logits = logits / temperature
    probs = softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cumulative_probs = torch.roll(cumulative_probs, shifts=1, dims=-1)
        cumulative_probs[..., 0] = 0.0
        mask = cumulative_probs < top_p
        sorted_probs.masked_fill_(~mask, 0.0)
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        next_token = Categorical(sorted_probs).sample()
        return torch.gather(sorted_indices, dim=-1, index=next_token.unsqueeze(-1)).squeeze(-1)
    return Categorical(probs).sample()


def generate(
    model: TransformerLM,
    prompt_tokens: torch.Tensor,  # Initial prompt tokens
    max_new_tokens: int = 50,  # Maximum number of new tokens to generate
    temperature: float = 1.0,  # Sampling temperature
    top_p: float = 1.0,  # Top-p (nucleus) sampling probability
    eos_token_id: int | None = None,  # End-of-sequence token ID
) -> torch.Tensor:
    if prompt_tokens.ndim == 1:
        prompt_tokens = prompt_tokens.unsqueeze(0)
    context_length = model.context_length
    generated_tokens = prompt_tokens.clone()
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated_tokens[:, -context_length:])[:, -1, :]
            next_token = sample_next_token(logits, temperature=temperature, top_p=top_p)
            generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(-1)], dim=-1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
    return generated_tokens
