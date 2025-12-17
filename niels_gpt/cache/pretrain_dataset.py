"""Memory-mapped pretrain dataset for random window sampling."""

from pathlib import Path

import numpy as np
import torch


class PretrainTokenStreamDataset:
    """
    Memory-mapped pretrain dataset that samples random windows from sharded token files.

    Shards are uint16 little-endian token files.
    Does not load all tokens into RAM.
    """

    def __init__(self, shards_dir: str, *, T: int, device: str):
        """
        Initialize dataset from shard directory.

        Args:
            shards_dir: Directory containing shard_*.bin files
            T: Sequence length (context window size)
            device: Device to place tensors on (e.g., "cpu", "cuda", "mps")
        """
        self.shards_dir = Path(shards_dir)
        self.T = T
        self.device = device

        shard_files = sorted(self.shards_dir.glob("shard_*.bin"))
        if not shard_files:
            raise ValueError(f"No shard files found in {shards_dir}")

        self.shards: list[np.memmap] = []
        self.shard_lengths: list[int] = []
        for shard_path in shard_files:
            shard = np.memmap(shard_path, dtype=np.uint16, mode="r")
            if len(shard) >= T + 1:
                self.shards.append(shard)
                self.shard_lengths.append(len(shard))

        if not self.shards:
            raise ValueError("No shard has enough tokens for the requested window length")

        self.total_tokens = sum(self.shard_lengths)
        if self.total_tokens < T + 1:
            raise ValueError(
                f"Not enough tokens across shards; need at least {T + 1}, got {self.total_tokens}"
            )

        lengths_tensor = torch.tensor(self.shard_lengths, dtype=torch.double)
        self._shard_probs = lengths_tensor / lengths_tensor.sum()

    def _sample_shard_index(self, generator: torch.Generator) -> int:
        if len(self.shards) == 1:
            return 0
        idx = torch.multinomial(self._shard_probs, 1, generator=generator).item()
        return int(idx)

    def get_batch(
        self, *, B: int, generator: torch.Generator
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        """
        Sample a batch of random windows.

        Args:
            B: Batch size
            generator: torch.Generator for random sampling

        Returns:
            x, y tuple of shape (B, T) on device, dtype int64
            Invariant: y[:, :-1] == x[:, 1:]
        """
        x = torch.zeros((B, self.T), dtype=torch.long, device=self.device)
        y = torch.zeros((B, self.T), dtype=torch.long, device=self.device)

        for i in range(B):
            shard_idx = self._sample_shard_index(generator)
            shard = self.shards[shard_idx]
            shard_len = self.shard_lengths[shard_idx]

            max_start = shard_len - self.T - 1
            start_idx = (
                0
                if max_start <= 0
                else torch.randint(max_start + 1, (1,), generator=generator).item()
            )
            window = shard[start_idx : start_idx + self.T + 1]
            window_tensor = torch.from_numpy(np.asarray(window, dtype=np.int64)).to(
                self.device
            )

            x[i] = window_tensor[:-1]
            y[i] = window_tensor[1:]

        return x, y
