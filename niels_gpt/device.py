"""Device detection for torch."""

import torch


def get_device() -> str:
    """
    returns:
      - "mps" if torch.backends.mps.is_available() and is_built()
      - else "cpu"
    never returns "cuda"
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"
