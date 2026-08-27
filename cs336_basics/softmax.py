import torch

def softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    # Substract the largest entry over dim to avoid numerical stability issues.
    in_features = torch.sub(in_features, torch.amax(in_features, dim=dim, keepdim=True))
    exp_in_features = torch.exp(in_features)
    return torch.div(exp_in_features, torch.sum(exp_in_features, dim=dim, keepdim=True))
