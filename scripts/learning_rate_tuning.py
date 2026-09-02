import torch

from cs336_basics.sgd import SGD


def training_loop(lr: float):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)

    print(f"Training with learning rate: {lr}")
    for t in range(10):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        print(f"{loss.cpu().item():.5f}")
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.


if __name__ == "__main__":
    learning_rates = [1e1, 1e2, 1e3]
    for lr in learning_rates:
        training_loop(lr)
