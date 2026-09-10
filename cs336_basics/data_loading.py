import numpy as np
import numpy.random as npr
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset_length = len(dataset)
    start_indices = npr.randint(0, dataset_length - context_length, size=batch_size)
    sample = np.stack([dataset[i : i + context_length + 1] for i in start_indices])
    sample = torch.tensor(sample, dtype=torch.long).to(device)
    return sample[:, :-1], sample[:, 1:]
