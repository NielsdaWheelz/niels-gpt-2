"""Memory-mapped SFT dataset with assistant-only loss masking."""

from pathlib import Path

import numpy as np
import torch

from .meta import read_meta


class SFTExampleDataset:
    """
    Memory-mapped SFT dataset.

    Reads concatenated input_ids, labels, and example offsets.
    Returns (x, y, y_masked) where y_masked comes from cached labels when available.
    """

    def __init__(
        self,
        tokens_path: str,
        idx_path: str,
        labels_path: str | None = None,
        *,
        T: int,
        device: str,
        eot_id: int,
        asst_id: int | None = None,
        assistant_only_loss: bool = True,
        include_eot_in_loss: bool = False,
    ):
        """
        Initialize SFT dataset.

        Args:
            tokens_path: Path to *_input_ids.bin (uint16 concatenated tokens)
            idx_path: Path to idx.npy (int64 offsets array)
            labels_path: Optional path to *_labels.bin (int32 masked targets aligned to tokens)
            T: Sequence length (context window size)
            device: Device to place tensors on
            eot_id: End-of-turn token ID
            asst_id: Assistant role token ID (needed for masking fallback)
            assistant_only_loss: If True, only assistant spans contribute to loss when labels are absent
            include_eot_in_loss: If True, include the EOT token in assistant loss when labels are absent
        """
        self.tokens_path = Path(tokens_path)
        self.labels_path = Path(labels_path) if labels_path is not None else None
        self.idx_path = Path(idx_path)
        self.T = T
        self.device = device
        self.eot_id = eot_id
        self.assistant_only_loss = assistant_only_loss
        self.include_eot_in_loss = include_eot_in_loss

        meta_path = self.tokens_path.parent / "meta.json"
        meta_asst = None
        if meta_path.exists():
            meta = read_meta(str(meta_path))
            meta_asst = meta.get("special_token_ids", {}).get("asst")
        self.asst_id = asst_id if asst_id is not None else meta_asst
        if self.asst_id is None:
            raise ValueError("asst_id is required to compute assistant masking")

        self.tokens = np.memmap(self.tokens_path, dtype=np.uint16, mode="r")
        self.labels = (
            np.memmap(self.labels_path, dtype=np.int32, mode="r") if self.labels_path and self.labels_path.exists() else None
        )
        if self.labels is not None and len(self.labels) != len(self.tokens):
            raise ValueError(
                f"labels length ({len(self.labels)}) must match tokens length ({len(self.tokens)})"
            )
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
            example_labels = self.labels[start:end] if self.labels is not None else None

            if len(example_tokens) >= self.T + 1:
                seq_tokens_np = np.asarray(example_tokens[: self.T + 1], dtype=np.int64)
                seq_labels_np = (
                    np.asarray(example_labels[: self.T + 1], dtype=np.int64)
                    if example_labels is not None
                    else None
                )
            else:
                seq_tokens_np = np.full((self.T + 1,), self.eot_id, dtype=np.int64)
                seq_tokens_np[: len(example_tokens)] = np.asarray(example_tokens, dtype=np.int64)
                if example_labels is not None:
                    seq_labels_np = np.full((self.T + 1,), -100, dtype=np.int64)
                    seq_labels_np[: len(example_labels)] = np.asarray(example_labels, dtype=np.int64)
                else:
                    seq_labels_np = None

            seq = torch.from_numpy(seq_tokens_np).to(self.device)
            x[i] = seq[:-1]
            y[i] = seq[1:]

            if self.labels is not None:
                labels_seq = (
                    torch.from_numpy(seq_labels_np).to(self.device) if seq_labels_np is not None else None
                )
                if labels_seq is None:
                    y_masked[i] = torch.full_like(y[i], -100)
                else:
                    y_masked[i] = labels_seq[1:]
                    if not self.assistant_only_loss:
                        y_masked[i] = y[i]
            else:
                if self.assistant_only_loss:
                    mask = self._assistant_mask(
                        seq, self.asst_id, self.eot_id, include_eot=self.include_eot_in_loss
                    )
                    y_masked[i] = torch.where(mask, y[i], torch.full_like(y[i], -100))
                else:
                    y_masked[i] = y[i]

        return x, y, y_masked

    @staticmethod
    def _assistant_mask(
        seq: torch.Tensor, asst_id: int, eot_id: int, *, include_eot: bool
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
                if include_eot or next_token != eot_id:
                    mask[i] = True
            if in_assistant and next_token == eot_id:
                in_assistant = False
        return mask
