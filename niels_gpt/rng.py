"""Random number generation and seeding utilities."""

import random
import torch


def set_seed(seed: int) -> None:
    """
    sets seeds for:
      - python random
      - torch (torch.manual_seed)
    must not import numpy.
    """
    random.seed(seed)
    torch.manual_seed(seed)


def make_generator(seed: int, device: str | None = None) -> torch.Generator:
    """
    returns a torch.Generator seeded with `seed`.
    if device is provided, construct generator for that device when supported;
    if not supported, fall back to cpu generator.
    """
    if device is not None:
        try:
            gen = torch.Generator(device=device)
        except RuntimeError:
            gen = torch.Generator(device="cpu")
    else:
        gen = torch.Generator()

    gen.manual_seed(seed)
    return gen
