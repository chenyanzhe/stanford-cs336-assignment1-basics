import torch


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max = torch.amax(inputs, dim=-1, keepdim=True)  # Float[Tensor, " batch_size 1"]
    exp = torch.exp(torch.sub(inputs, max))  # Float[Tensor, " batch_size vocab_size"]
    log_sum_exp = max + torch.log(torch.sum(exp, dim=-1, keepdim=True))  # Float[Tensor, " batch_size 1"]
    return torch.mean(log_sum_exp - torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1)))
