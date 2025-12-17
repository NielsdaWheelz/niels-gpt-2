"""Memory-mapped SFT dataset with assistant-only loss masking."""

from pathlib import Path

import numpy as np
import torch

from .meta import read_meta


class SFTExampleDataset:
    """
    Memory-mapped SFT dataset.

    Reads concatenated tokens and example offsets.
    Returns (x, y, y_masked) where y_masked has -100 for non-assistant tokens.
    """

    def __init__(
        self,
        tokens_path: str,
        idx_path: str,
        *,
        T: int,
        device: str,
        eot_id: int,
        asst_id: int | None = None,
    ):
        """
        Initialize SFT dataset.

        Args:
            tokens_path: Path to tokens.bin (uint16 concatenated tokens)
            idx_path: Path to idx.npy (int64 offsets array)
            T: Sequence length (context window size)
            device: Device to place tensors on
            eot_id: End-of-turn token ID
            asst_id: Assistant role token ID (needed for masking)
        """
        self.tokens_path = Path(tokens_path)
        self.idx_path = Path(idx_path)
        self.T = T
        self.device = device
        self.eot_id = eot_id

        meta_path = self.tokens_path.parent / "meta.json"
        meta_asst = None
        if meta_path.exists():
            meta = read_meta(str(meta_path))
            meta_asst = meta.get("special_token_ids", {}).get("asst")
        self.asst_id = asst_id if asst_id is not None else meta_asst
        if self.asst_id is None:
            raise ValueError("asst_id is required to compute assistant masking")

        self.tokens = np.memmap(self.tokens_path, dtype=np.uint16, mode="r")
        self.offsets = np.load(self.idx_path)
        self.num_examples = len(self.offsets)
        if self.num_examples == 0:
            raise ValueError(f"No examples found in {idx_path}")

    def get_batch(
        self, *, B: int, generator: torch.Generator
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """
        Sample a batch of examples with assistant-only masking.

        Returns (x, y, y_masked) of shape (B, T) on the requested device.
        """
        x = torch.zeros((B, self.T), dtype=torch.long, device=self.device)
        y = torch.zeros((B, self.T), dtype=torch.long, device=self.device)
        y_masked = torch.full((B, self.T), -100, dtype=torch.long, device=self.device)

        for i in range(B):
            idx = torch.randint(self.num_examples, (1,), generator=generator).item()
            start = int(self.offsets[idx])
            end = int(self.offsets[idx + 1]) if idx + 1 < self.num_examples else len(
                self.tokens
            )

            example_tokens = self.tokens[start:end]

            if len(example_tokens) >= self.T + 1:
                seq = torch.from_numpy(
                    np.asarray(example_tokens[: self.T + 1], dtype=np.int64)
                )
            else:
                seq = torch.full((self.T + 1,), self.eot_id, dtype=torch.int64)
                seq[: len(example_tokens)] = torch.from_numpy(
                    np.asarray(example_tokens, dtype=np.int64)
                )

            seq = seq.to(self.device)
            x[i] = seq[:-1]
            y[i] = seq[1:]

            mask = self._assistant_mask(seq, self.asst_id, self.eot_id)
            y_masked[i] = torch.where(mask, y[i], torch.full_like(y[i], -100))

        return x, y, y_masked

    @staticmethod
    def _assistant_mask(
        seq: torch.Tensor, asst_id: int, eot_id: int
    ) -> torch.Tensor:
        """
        Compute assistant-only mask for a sequence of length T+1.

        Returns a bool tensor of length T corresponding to y positions.
        """
        mask = torch.zeros(seq.shape[0] - 1, dtype=torch.bool, device=seq.device)
        in_assistant = False
        for i in range(seq.shape[0] - 1):
            prev_token = int(seq[i].item())
            next_token = int(seq[i + 1].item())
            if prev_token == asst_id:
                in_assistant = True
            if in_assistant:
                mask[i] = True
            if in_assistant and next_token == eot_id:
                in_assistant = False
        return mask
