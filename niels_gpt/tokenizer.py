from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import sentencepiece as spm
import torch

from niels_gpt.paths import REPO_ROOT

SPECIAL_TOKENS = ("<|sys|>", "<|usr|>", "<|asst|>", "<|eot|>")
# Default tokenizer location (from PR-01 artifacts)
DEFAULT_TOKENIZER_PATH = REPO_ROOT / "artifacts" / "tokenizer" / "spm.model"

_DEFAULT_TOKENIZER: "SentencePieceTokenizer | None" = None


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

    def __init__(self, model_path: str, *, special_tokens: tuple[str, ...] | list[str] | None = None):
        """
        Load a trained SentencePiece model.

        Args:
            model_path: Path to the .model file

        Raises:
            ValueError: If any special token is not in vocabulary
        """
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(model_path)
        self.model_path = str(Path(model_path).resolve())
        self._special_tokens = tuple(special_tokens) if special_tokens is not None else SPECIAL_TOKENS

        # Validate special tokens are single-piece, stable, and decode correctly
        self._validate_special_tokens()

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
        for token in self._special_tokens:
            # Get ID from vocabulary (validated in __init__)
            piece_id = self._sp.piece_to_id(token)
            # Extract the role name from the token (e.g., "<|sys|>" -> "sys")
            key = token.strip("<|>")
            result[key] = piece_id
        return result

    @property
    def special_tokens(self) -> tuple[str, ...]:
        return self._special_tokens

    def _validate_special_tokens(self) -> None:
        """
        Enforce single-piece, id-stable special tokens.

        Raises:
            ValueError if any special token:
            - is missing from the vocab
            - encodes to !=1 piece
            - encodes to a different id than piece_to_id
            - decodes incorrectly
        """
        vocab_size = self._sp.GetPieceSize()
        for token in self._special_tokens:
            piece_id = self._sp.piece_to_id(token)
            if piece_id == self._sp.unk_id() or piece_id < 0 or piece_id >= vocab_size:
                raise ValueError(
                    f"special token '{token}' missing or out of range in vocab (id={piece_id}, vocab={vocab_size})"
                )

            encoded = self._sp.EncodeAsIds(token)
            if len(encoded) != 1 or encoded[0] != piece_id:
                raise ValueError(
                    f"special token '{token}' must encode to exactly one piece id; "
                    f"got encoded={encoded}, piece_id={piece_id}"
                )

            decoded = self._sp.DecodeIds([piece_id])
            if decoded != token:
                raise ValueError(
                    f"special token '{token}' decode mismatch: piece id {piece_id} -> '{decoded}'"
                )


def load_tokenizer(model_path: str) -> SentencePieceTokenizer:
    """
    Load a trained SentencePiece tokenizer.

    Args:
        model_path: Path to the .model file

    Returns:
        Initialized SentencePieceTokenizer instance
    """
    return SentencePieceTokenizer(model_path)


def _get_default_tokenizer() -> SentencePieceTokenizer:
    """Lazily load the default tokenizer from artifacts."""
    global _DEFAULT_TOKENIZER
    if _DEFAULT_TOKENIZER is None:
        if not DEFAULT_TOKENIZER_PATH.exists():
            raise FileNotFoundError(
                f"Default tokenizer not found at {DEFAULT_TOKENIZER_PATH}. "
                "Please train or provide tokenizer artifacts."
            )
        _DEFAULT_TOKENIZER = load_tokenizer(str(DEFAULT_TOKENIZER_PATH))
    return _DEFAULT_TOKENIZER


def get_default_tokenizer() -> SentencePieceTokenizer:
    """Public accessor for the default SentencePiece tokenizer."""
    return _get_default_tokenizer()


def encode(text: str, *, device: str | None = None) -> torch.LongTensor:
    """Encode using the default SentencePiece tokenizer."""
    tok = _get_default_tokenizer()
    return tok.encode_torch(text, device=device)


def decode(ids: Sequence[int] | torch.Tensor) -> str:
    """Decode using the default SentencePiece tokenizer."""
    tok = _get_default_tokenizer()
    return tok.decode(ids)
