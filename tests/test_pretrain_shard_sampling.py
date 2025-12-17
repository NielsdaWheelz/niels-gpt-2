"""Validate shard-aware sampling in PretrainTokenStreamDataset."""

import tempfile
from pathlib import Path

import numpy as np
import torch

from niels_gpt.cache.pretrain_dataset import PretrainTokenStreamDataset


def test_windows_never_cross_shards_and_stay_in_bounds():
    """
    Create two shards with disjoint token ranges.
    Ensure sampled windows only contain tokens from one shard
    and start indices are within valid bounds.
    """
    shard1_tokens = list(range(10, 110))  # 100 tokens
    shard2_tokens = list(range(1000, 1130))  # 130 tokens
    T = 16

    with tempfile.TemporaryDirectory() as tmpdir:
        shard_dir = Path(tmpdir)

        for idx, tokens in enumerate([shard1_tokens, shard2_tokens]):
            with open(shard_dir / f"shard_{idx:05d}.bin", "wb") as f:
                arr = np.asarray(tokens, dtype="<u2")
                f.write(arr.tobytes())

        ds = PretrainTokenStreamDataset(str(shard_dir), T=T, device="cpu")
        gen = torch.Generator().manual_seed(123)

        for _ in range(32):
            x, y = ds.get_batch(B=1, generator=gen)
            seq = torch.cat([x.flatten(), y.flatten()[-1:]])  # reconstruct window
            unique = seq.unique()

            # Determine which shard range we fell into
            in_shard1 = torch.all((seq >= min(shard1_tokens)) & (seq <= max(shard1_tokens)))
            in_shard2 = torch.all((seq >= min(shard2_tokens)) & (seq <= max(shard2_tokens)))
            assert in_shard1 != in_shard2, "window must belong to exactly one shard"

            # All windows must be length T+1 internally
            assert seq.numel() == T + 1

