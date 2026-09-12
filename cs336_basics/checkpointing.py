import os
import typing

import torch


def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.IO[bytes]
):
    ckpt = {"model_state": model.state_dict(), "opt_state": optimizer.state_dict(), "it": iteration}
    torch.save(ckpt, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    ckpt = torch.load(src)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["opt_state"])
    return ckpt["it"]
