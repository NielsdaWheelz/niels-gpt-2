"""Device detection for torch."""

import torch


def get_device() -> str:
    """
    Auto-detect best available device.

    Returns:
      - "cuda" if CUDA is available
      - "mps" if MPS (Apple Silicon) is available
      - "cpu" otherwise
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"
