"""Tests for chat template functionality."""

from __future__ import annotations

from pathlib import Path

import pytest

from niels_gpt.chat_template import Message, extract_assistant_reply, format_chat, format_prompt
from niels_gpt.tokenizer import SentencePieceTokenizer, load_tokenizer


@pytest.fixture
def tiny_corpus_dir(tmp_path: Path) -> Path:
    """Create a tiny corpus for testing."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    # Create a few small text files
    (corpus_dir / "file1.txt").write_text(
        "Hello world! This is a test. The quick brown fox jumps over the lazy dog.\n"
        "Machine learning is fun and interesting. Natural language processing rocks!\n"
        * 10,
        encoding="utf-8",
    )

    (corpus_dir / "file2.txt").write_text(
        "Python is a great programming language. "
        "SentencePiece tokenization is efficient and flexible.\n"
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

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_tokenizer.py",
            "--input_glob",
            f"{tiny_corpus_dir}/*.txt",
            "--out_dir",
            str(out_dir),
            "--vocab_size",
            "300",  # Small vocab for tests
            "--seed",
            "42",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.fail(f"Training failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

    model_path = out_dir / "spm.model"
    return load_tokenizer(str(model_path))


def test_format_chat_contains_eot_after_each_message(trained_tokenizer: SentencePieceTokenizer):
    """format_chat should include the EOT sentinel after every message."""
    messages: list[Message] = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    result = format_chat(trained_tokenizer, messages)
    special = trained_tokenizer.special_token_ids()

    # Should have 3 eot tokens (one after each message)
    eot_count = result.count(special["eot"])
    assert eot_count == 3, f"Expected 3 eot tokens, got {eot_count}"


def test_format_prompt_ends_with_asst_no_eot(trained_tokenizer: SentencePieceTokenizer):
    """format_prompt should end with the assistant sentinel and no trailing EOT."""
    messages: list[Message] = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
    ]

    result = format_prompt(trained_tokenizer, messages)
    special = trained_tokenizer.special_token_ids()

    # Last token should be assistant sentinel
    assert result[-1] == special["asst"], "Last token should be assistant sentinel"

    # Second-to-last should be EOT (from the user message)
    assert result[-2] == special["eot"], "Second-to-last should be EOT from user message"


def test_format_prompt_structure(trained_tokenizer: SentencePieceTokenizer):
    """format_prompt should have proper structure with all messages as completed turns."""
    messages: list[Message] = [
        {"role": "user", "content": "Test"},
    ]

    result = format_prompt(trained_tokenizer, messages)
    special = trained_tokenizer.special_token_ids()

    # Should start with user sentinel
    assert result[0] == special["usr"]

    # Should end with assistant sentinel
    assert result[-1] == special["asst"]

    # Should contain exactly one EOT (from the user message)
    eot_count = result.count(special["eot"])
    assert eot_count == 1


def test_extract_assistant_reply_basic(trained_tokenizer: SentencePieceTokenizer):
    """extract_assistant_reply should extract tokens between assistant and EOT sentinels."""
    special = trained_tokenizer.special_token_ids()

    # Create a synthetic sequence: usr content EOT asst hello EOT junk
    user_content = trained_tokenizer.encode("Hi")
    asst_content = trained_tokenizer.encode("Hello")
    junk = trained_tokenizer.encode("extra")

    generated = (
        [special["usr"]]
        + user_content
        + [special["eot"]]
        + [special["asst"]]
        + asst_content
        + [special["eot"]]
        + junk
    )

    result = extract_assistant_reply(trained_tokenizer, generated)

    # Should extract only the assistant content
    assert result == asst_content


def test_extract_assistant_reply_no_eot(trained_tokenizer: SentencePieceTokenizer):
    """extract_assistant_reply should work when EOT is missing (incomplete generation)."""
    special = trained_tokenizer.special_token_ids()

    # Create a synthetic sequence: usr content EOT asst hello (no eot)
    user_content = trained_tokenizer.encode("Hi")
    asst_content = trained_tokenizer.encode("Hello")

    generated = [special["usr"]] + user_content + [special["eot"]] + [special["asst"]] + asst_content

    result = extract_assistant_reply(trained_tokenizer, generated)

    # Should extract all tokens after assistant sentinel
    assert result == asst_content


def test_extract_assistant_reply_multiple_asst_uses_last(trained_tokenizer: SentencePieceTokenizer):
    """extract_assistant_reply should use the LAST assistant token."""
    special = trained_tokenizer.special_token_ids()

    # First assistant turn
    first = trained_tokenizer.encode("first reply")
    # Second assistant turn
    second = trained_tokenizer.encode("second reply")

    generated = (
        [special["asst"]]
        + first
        + [special["eot"]]
        + [special["usr"]]
        + trained_tokenizer.encode("user message")
        + [special["eot"]]
        + [special["asst"]]
        + second
        + [special["eot"]]
    )

    result = extract_assistant_reply(trained_tokenizer, generated)

    # Should extract only the second reply
    assert result == second


def test_extract_assistant_reply_empty_returns_empty(trained_tokenizer: SentencePieceTokenizer):
    """extract_assistant_reply should return empty list if no assistant token found."""
    special = trained_tokenizer.special_token_ids()

    # Sequence with no assistant token
    generated = [special["usr"]] + trained_tokenizer.encode("Hi") + [special["eot"]]

    result = extract_assistant_reply(trained_tokenizer, generated)
    assert result == []


def test_format_chat_system_user_assistant(trained_tokenizer: SentencePieceTokenizer):
    """format_chat should handle all three role types correctly."""
    messages: list[Message] = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant message"},
    ]

    result = format_chat(trained_tokenizer, messages)
    special = trained_tokenizer.special_token_ids()

    # Decode to verify structure
    decoded = trained_tokenizer.decode(result)

    # Should contain all special tokens
    assert result.count(special["sys"]) == 1
    assert result.count(special["usr"]) == 1
    assert result.count(special["asst"]) == 1
    assert result.count(special["eot"]) == 3


def test_format_prompt_then_extract(trained_tokenizer: SentencePieceTokenizer):
    """Test round-trip: format_prompt -> generate -> extract_assistant_reply."""
    messages: list[Message] = [
        {"role": "user", "content": "What is 2+2?"},
    ]

    prompt = format_prompt(trained_tokenizer, messages)
    special = trained_tokenizer.special_token_ids()

    # Simulate generation: add some tokens and end with EOT
    reply_text = "The answer is 4"
    reply_tokens = trained_tokenizer.encode(reply_text)
    generated = prompt + reply_tokens + [special["eot"]]

    # Extract the reply
    extracted = extract_assistant_reply(trained_tokenizer, generated)

    # Should match what we "generated"
    assert extracted == reply_tokens


def test_format_prompt_empty_continuation_no_eot(trained_tokenizer: SentencePieceTokenizer):
    """format_prompt + encode('') should not immediately produce EOT."""
    messages: list[Message] = [
        {"role": "user", "content": "Hello"},
    ]

    prompt_ids = format_prompt(trained_tokenizer, messages)
    special = trained_tokenizer.special_token_ids()

    # Verify prompt ends with assistant sentinel
    assert prompt_ids[-1] == special["asst"], "Prompt should end with assistant sentinel"

    # Encode empty string (may be [] or some tokens, but should not be [eot])
    empty_tokens = trained_tokenizer.encode("")

    # If empty string encodes to nothing, that's fine
    # If it encodes to something, it must NOT be EOT as the first token
    if empty_tokens:
        assert empty_tokens[0] != special["eot"], (
            "Empty string should not encode to EOT token. "
            "This would cause immediate turn termination."
        )
