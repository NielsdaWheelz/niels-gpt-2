from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sentencepiece as spm
import torch

SPECIAL_TOKENS = ("<|sys|>", "<|usr|>", "<|asst|>", "<|eot|>")


@dataclass(frozen=True)
class TokenizerPaths:
    model_path: str  # .../spm.model
    meta_path: str  # .../tokenizer_meta.json


class SentencePieceTokenizer:
    """
    SentencePiece-based tokenizer with special tokens for chat.

    Special tokens: <|sys|>, <|usr|>, <|asst|>, <|eot|>
    Each special token MUST encode to exactly one token ID.
    """

    def __init__(self, model_path: str):
        """
        Load a trained SentencePiece model.

        Args:
            model_path: Path to the .model file

        Raises:
            ValueError: If any special token is not in vocabulary
        """
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(model_path)

        # Validate that each special token exists in vocabulary
        # Note: We insert special tokens by ID, not by encoding strings
        for token in SPECIAL_TOKENS:
            piece_id = self._sp.piece_to_id(token)
            if piece_id == self._sp.unk_id():
                raise ValueError(
                    f"Special token '{token}' not found in vocabulary. "
                    f"The tokenizer model may not have been trained with control symbols."
                )

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size including special tokens."""
        return self._sp.GetPieceSize()

    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text string

        Returns:
            List of token IDs (Python ints)
        """
        return self._sp.EncodeAsIds(text)

    def encode_torch(self, text: str, *, device: str | None = None) -> torch.LongTensor:
        """
        Encode text to token IDs as a PyTorch tensor.

        Args:
            text: Input text string
            device: Device to place tensor on (e.g., "cpu", "cuda", "mps")

        Returns:
            1D tensor of token IDs with shape (n,) and dtype int64
        """
        ids = self.encode(text)
        tensor = torch.tensor(ids, dtype=torch.long)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def decode(self, ids: Sequence[int] | torch.Tensor) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: Sequence of token IDs or PyTorch tensor

        Returns:
            Decoded text string
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._sp.DecodeIds(list(ids))

    def special_token_ids(self) -> dict[str, int]:
        """
        Get the token IDs for all special tokens.

        Returns:
            Dict mapping special token names to their IDs
            Keys: "sys", "usr", "asst", "eot"
        """
        result = {}
        for token in SPECIAL_TOKENS:
            # Get ID from vocabulary (validated in __init__)
            piece_id = self._sp.piece_to_id(token)
            # Extract the role name from the token (e.g., "<|sys|>" -> "sys")
            key = token.strip("<|>")
            result[key] = piece_id
        return result


def load_tokenizer(model_path: str) -> SentencePieceTokenizer:
    """
    Load a trained SentencePiece tokenizer.

    Args:
        model_path: Path to the .model file

    Returns:
        Initialized SentencePieceTokenizer instance
    """
    return SentencePieceTokenizer(model_path)
