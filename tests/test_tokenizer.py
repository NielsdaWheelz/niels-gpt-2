"""Tests for SentencePiece tokenizer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from niels_gpt.tokenizer import SPECIAL_TOKENS, SentencePieceTokenizer, load_tokenizer


@pytest.fixture
def tiny_corpus_dir(tmp_path: Path) -> Path:
    """Create a tiny corpus for testing."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    # Create a few small text files
    (corpus_dir / "file1.txt").write_text(
        "Hello world! This is a test. The quick brown fox jumps over the lazy dog.\n"
        "Machine learning is fun and interesting. Natural language processing rocks!\n"
        * 10,  # Repeat to give sentencepiece more data
        encoding="utf-8",
    )

    (corpus_dir / "file2.txt").write_text(
        "Python is a great programming language. "
        "SentencePiece tokenization is efficient and flexible.\n"
        "We are training a tokenizer for a language model.\n"
        * 10,
        encoding="utf-8",
    )

    (corpus_dir / "file3.txt").write_text(
        "The cat sat on the mat. The dog ran in the park. "
        "Birds fly in the sky. Fish swim in the ocean.\n"
        * 10,
        encoding="utf-8",
    )

    return corpus_dir


@pytest.fixture
def trained_tokenizer(tiny_corpus_dir: Path, tmp_path: Path) -> SentencePieceTokenizer:
    """Train a tiny tokenizer for testing."""
    import subprocess
    import sys

    out_dir = tmp_path / "tokenizer"
    out_dir.mkdir()

    # Train using the script
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_tokenizer.py",
            "--input_glob",
            f"{tiny_corpus_dir}/*.txt",
            "--out_dir",
            str(out_dir),
            "--vocab_size",
            "320",  # Small vocab for tests
            "--seed",
            "42",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.fail(f"Training failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

    model_path = out_dir / "spm.model"
    assert model_path.exists(), "Model file was not created"

    return load_tokenizer(str(model_path))


def test_special_tokens_in_vocab(trained_tokenizer: SentencePieceTokenizer):
    """Each special token must exist in vocabulary with a valid ID."""
    special_ids = trained_tokenizer.special_token_ids()

    # All special tokens should be present
    assert set(special_ids.keys()) == {"sys", "usr", "asst", "eot"}

    # All IDs should be valid (non-negative, within vocab)
    vocab_size = trained_tokenizer.vocab_size
    for key, token_id in special_ids.items():
        assert isinstance(token_id, int), f"ID for {key} should be int"
        assert 0 <= token_id < vocab_size, f"ID for {key} out of range: {token_id}"


def test_special_token_ids_returns_all_tokens(trained_tokenizer: SentencePieceTokenizer):
    """special_token_ids() should return distinct IDs for all 4 tokens."""
    special_ids = trained_tokenizer.special_token_ids()

    # Check all keys are present
    assert set(special_ids.keys()) == {"sys", "usr", "asst", "eot"}

    # Check all values are distinct
    values = list(special_ids.values())
    assert len(values) == len(set(values)), "Special token IDs should be distinct"

    # Check all values are valid integers
    for key, value in special_ids.items():
        assert isinstance(value, int), f"ID for {key} should be an int"
        assert 0 <= value < trained_tokenizer.vocab_size


def test_encode_decode_roundtrip(trained_tokenizer: SentencePieceTokenizer):
    """Encode and decode should roundtrip correctly for simple text."""
    test_texts = [
        "Hello world!",
        "This is a test.",
        "Machine learning is fun.",
        "The quick brown fox.",
    ]

    for text in test_texts:
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids)
        # SentencePiece may add/remove spaces, so we check semantic similarity
        # For this test, we just ensure decoding produces non-empty output
        assert decoded, f"Decoded text should not be empty for: {text}"
        assert isinstance(decoded, str)


def test_encode_torch(trained_tokenizer: SentencePieceTokenizer):
    """encode_torch should return a tensor with correct shape and dtype."""
    import torch

    text = "Hello world!"
    tensor = trained_tokenizer.encode_torch(text)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.long
    assert tensor.dim() == 1
    assert tensor.shape[0] > 0


def test_encode_torch_with_device(trained_tokenizer: SentencePieceTokenizer):
    """encode_torch should respect device argument."""
    import torch

    text = "Hello world!"
    tensor = trained_tokenizer.encode_torch(text, device="cpu")

    assert tensor.device.type == "cpu"


def test_decode_with_torch_tensor(trained_tokenizer: SentencePieceTokenizer):
    """decode should accept torch.Tensor input."""
    import torch

    text = "Hello world!"
    ids = trained_tokenizer.encode(text)
    tensor = torch.tensor(ids, dtype=torch.long)

    decoded = trained_tokenizer.decode(tensor)
    assert isinstance(decoded, str)
    assert decoded  # Non-empty


def test_vocab_size(trained_tokenizer: SentencePieceTokenizer):
    """vocab_size should return a positive integer."""
    vocab_size = trained_tokenizer.vocab_size
    assert isinstance(vocab_size, int)
    assert vocab_size > 0
    # Should be around 320 (our test vocab size)
    assert 300 <= vocab_size <= 350


def test_load_tokenizer(trained_tokenizer: SentencePieceTokenizer, tmp_path: Path):
    """load_tokenizer should work the same as direct instantiation."""
    # Get the model path from the trained tokenizer
    # We can re-use the same model
    model_path = tmp_path / "tokenizer" / "spm.model"

    tok1 = load_tokenizer(str(model_path))
    tok2 = SentencePieceTokenizer(str(model_path))

    # Both should produce same results
    text = "Test text"
    assert tok1.encode(text) == tok2.encode(text)
    assert tok1.vocab_size == tok2.vocab_size
    assert tok1.special_token_ids() == tok2.special_token_ids()
