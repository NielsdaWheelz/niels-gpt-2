"""Test pretrain window sampling invariants."""

import tempfile
from pathlib import Path

import numpy as np
import torch

from niels_gpt.cache.pretrain_dataset import PretrainTokenStreamDataset


def test_pretrain_window_sampling():
    """
    Test that pretrain dataset sampling satisfies:
    - y[:, :-1] == x[:, 1:] always
    - All token IDs are within vocab size
    """
    # Create a small test shard
    vocab_size = 1000
    shard_tokens = list(range(100, 600))  # 500 tokens

    with tempfile.TemporaryDirectory() as tmpdir:
        shard_dir = Path(tmpdir)
        shard_path = shard_dir / "shard_00000.bin"

        # Write shard (uint16 little-endian)
        with open(shard_path, "wb") as f:
            for token in shard_tokens:
                f.write(token.to_bytes(2, byteorder="little"))

        # Create dataset
        T = 10  # Sequence length
        dataset = PretrainTokenStreamDataset(
            str(shard_dir),
            T=T,
            device="cpu",
        )

        # Sample a batch
        gen = torch.Generator().manual_seed(42)
        B = 4
        x, y = dataset.get_batch(B=B, generator=gen)

        # Check shapes
        assert x.shape == (B, T), f"x shape should be ({B}, {T}), got {x.shape}"
        assert y.shape == (B, T), f"y shape should be ({B}, {T}), got {y.shape}"

        # Check dtype
        assert x.dtype == torch.long, f"x dtype should be torch.long, got {x.dtype}"
        assert y.dtype == torch.long, f"y dtype should be torch.long, got {y.dtype}"

        # Check invariant: y[:, :-1] == x[:, 1:]
        assert torch.all(y[:, :-1] == x[:, 1:]), (
            "Invariant violated: y[:, :-1] should equal x[:, 1:]"
        )

        # Check that all token IDs are valid (within vocab)
        assert torch.all(x >= 0), "All token IDs should be non-negative"
        assert torch.all(x < vocab_size), f"All token IDs should be < {vocab_size}"
        assert torch.all(y >= 0), "All token IDs should be non-negative"
        assert torch.all(y < vocab_size), f"All token IDs should be < {vocab_size}"

        # Check that tokens are actually from our shard
        all_tokens = torch.cat([x.flatten(), y.flatten()])
        unique_tokens = torch.unique(all_tokens).tolist()
        for token in unique_tokens:
            assert token in shard_tokens, (
                f"Token {token} not in original shard"
            )

    print("✓ Pretrain window sampling test passed")


if __name__ == "__main__":
    test_pretrain_window_sampling()
