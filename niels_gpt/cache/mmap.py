"""Small helpers for mmap-based cache reading."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def mmap_tokens(path: str) -> np.memmap:
    """Memory-map a uint16 tokens.bin file."""
    return np.memmap(Path(path), dtype=np.uint16, mode="r")


def load_idx(path: str) -> np.ndarray:
    """Load idx.npy and validate dtype."""
    arr = np.load(Path(path))
    if arr.dtype != np.int64:
        raise ValueError(f"idx array must be int64, got {arr.dtype}")
    return arr

