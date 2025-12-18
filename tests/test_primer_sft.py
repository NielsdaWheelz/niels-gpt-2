"""Tests for primer SFT dataset (PR-03)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from niels_gpt.data.primer_sft import (
    build_primer_sft_cache,
    load_primer_jsonl,
    render_chat_sft,
    split_primer_examples,
)


class FakeTokenizer:
    """Lightweight fake tokenizer for testing."""

    def __init__(self):
        self.vocab_size = 100
        self.model_path = None
        self._special_ids = {
            "sys": 1,
            "usr": 2,
            "asst": 3,
            "eot": 4,
        }

    def special_token_ids(self):
        return self._special_ids

    def encode(self, text: str) -> list[int]:
        """Simple encoding: map each char to an id offset by 10."""
        if not text:
            return []
        return [10 + ord(c) % 80 for c in text]


def test_render_chat_sft_basic():
    """Test basic rendering with system, user, assistant messages."""
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]

    input_ids, labels = render_chat_sft(tokenizer, messages, add_final_eot=True)

    # Check lengths match
    assert len(input_ids) == len(labels)

    # Verify structure: should have 3 role tokens + 3 eot tokens + content
    special_ids = tokenizer.special_token_ids()
    sys_id = special_ids["sys"]
    usr_id = special_ids["usr"]
    asst_id = special_ids["asst"]
    eot_id = special_ids["eot"]

    # First token should be sys
    assert input_ids[0] == sys_id
    assert labels[0] == -100

    # Find assistant section
    asst_idx = input_ids.index(asst_id)
    assert labels[asst_idx] == -100  # role token is masked

    # Check that assistant content and eot are labeled
    # Content starts after asst_id
    asst_content_start = asst_idx + 1
    asst_eot_idx = None
    for i in range(asst_content_start, len(input_ids)):
        if input_ids[i] == eot_id:
            asst_eot_idx = i
            break

    assert asst_eot_idx is not None
    # Assistant content tokens should be labeled
    for i in range(asst_content_start, asst_eot_idx):
        assert labels[i] == input_ids[i], f"assistant content at {i} should be labeled"
    # Assistant eot should be labeled
    assert labels[asst_eot_idx] == eot_id

    # System and user sections should be masked
    for i in range(asst_idx):
        if input_ids[i] not in {sys_id, usr_id, eot_id}:
            # This is content in sys/usr
            assert labels[i] == -100


def test_render_chat_sft_assistant_only_masking():
    """Test that only assistant tokens are labeled."""
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]

    input_ids, labels = render_chat_sft(tokenizer, messages)

    special_ids = tokenizer.special_token_ids()
    usr_id = special_ids["usr"]
    asst_id = special_ids["asst"]
    eot_id = special_ids["eot"]

    # Count labeled tokens (not -100)
    labeled_count = sum(1 for label in labels if label != -100)

    # Should be: assistant content (1 char = 1 token) + assistant eot
    # "A" = 1 token, eot = 1 token -> 2 total
    assert labeled_count == 2

    # Verify user content and user eot are masked
    usr_idx = input_ids.index(usr_id)
    first_eot_idx = input_ids.index(eot_id, usr_idx)
    # user content
    for i in range(usr_idx + 1, first_eot_idx):
        assert labels[i] == -100
    # user eot
    assert labels[first_eot_idx] == -100


def test_render_chat_sft_assistant_eot_labeled():
    """Test that assistant <|eot|> is included in labels."""
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "assistant", "content": "Hello"},
    ]

    input_ids, labels = render_chat_sft(tokenizer, messages)

    eot_id = tokenizer.special_token_ids()["eot"]

    # Last token should be eot
    assert input_ids[-1] == eot_id
    # And it should be labeled
    assert labels[-1] == eot_id


def test_render_chat_sft_no_assistant_raises():
    """Test that messages without assistant raise ValueError."""
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "User"},
    ]

    with pytest.raises(ValueError, match="at least one assistant"):
        render_chat_sft(tokenizer, messages)


def test_render_chat_sft_empty_messages_raises():
    """Test that empty messages list raises ValueError."""
    tokenizer = FakeTokenizer()

    with pytest.raises(ValueError, match="non-empty"):
        render_chat_sft(tokenizer, [])


def test_render_chat_sft_invalid_role_raises():
    """Test that invalid role raises ValueError."""
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "assistant", "content": "valid"},
        {"role": "invalid", "content": "test"},
    ]

    with pytest.raises(ValueError, match="invalid role"):
        render_chat_sft(tokenizer, messages)


def test_split_primer_examples_deterministic():
    """Test that split is deterministic with same seed."""
    examples = [{"messages": [{"role": "assistant", "content": f"msg{i}"}]} for i in range(10)]

    train1, val1 = split_primer_examples(examples, val_frac=0.2, seed=42)
    train2, val2 = split_primer_examples(examples, val_frac=0.2, seed=42)

    # Should be identical
    assert train1 == train2
    assert val1 == val2

    # Should be disjoint
    train_contents = {msg["messages"][0]["content"] for msg in train1}
    val_contents = {msg["messages"][0]["content"] for msg in val1}
    assert len(train_contents & val_contents) == 0


def test_split_primer_examples_different_seed():
    """Test that different seeds produce different splits."""
    examples = [{"messages": [{"role": "assistant", "content": f"msg{i}"}]} for i in range(10)]

    train1, val1 = split_primer_examples(examples, val_frac=0.2, seed=42)
    train2, val2 = split_primer_examples(examples, val_frac=0.2, seed=999)

    # Should be different (highly likely with 10 examples)
    assert train1 != train2 or val1 != val2


def test_load_primer_jsonl_valid():
    """Test loading valid primer.jsonl file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"messages":[{"role":"system","content":"sys"},{"role":"user","content":"usr"},{"role":"assistant","content":"asst"}]}\n')
        f.write('{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n')
        temp_path = f.name

    try:
        examples = load_primer_jsonl(temp_path)
        assert len(examples) == 2
        assert examples[0]["messages"][0]["role"] == "system"
        assert examples[1]["messages"][0]["role"] == "user"
    finally:
        Path(temp_path).unlink()


def test_load_primer_jsonl_malformed():
    """Test that malformed JSONL raises ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"messages":[{"role":"user","content":"test"}]}\n')
        f.write('not valid json\n')
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="invalid JSON"):
            load_primer_jsonl(temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_primer_jsonl_missing_messages():
    """Test that missing 'messages' key raises ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"data":"test"}\n')
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="missing 'messages'"):
            load_primer_jsonl(temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_primer_jsonl_invalid_role():
    """Test that invalid role raises ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"messages":[{"role":"invalid","content":"test"}]}\n')
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="invalid role"):
            load_primer_jsonl(temp_path)
    finally:
        Path(temp_path).unlink()


def test_build_primer_sft_cache_smoke():
    """Test that cache building creates expected files with correct structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a small primer.jsonl
        jsonl_path = Path(tmpdir) / "primer.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(5):
                example = {
                    "messages": [
                        {"role": "user", "content": f"question {i}"},
                        {"role": "assistant", "content": f"answer {i}"},
                    ]
                }
                f.write(json.dumps(example) + "\n")

        # Build cache
        cache_dir = Path(tmpdir) / "cache"
        tokenizer = FakeTokenizer()

        build_primer_sft_cache(
            str(jsonl_path),
            str(cache_dir),
            tokenizer=tokenizer,
            val_frac=0.2,
            seed=42,
            t_max=512,
        )

        # Check files exist
        assert (cache_dir / "meta.json").exists()
        assert (cache_dir / "train_input_ids.bin").exists()
        assert (cache_dir / "train_labels.bin").exists()
        assert (cache_dir / "train_idx.npy").exists()
        assert (cache_dir / "val_input_ids.bin").exists()
        assert (cache_dir / "val_labels.bin").exists()
        assert (cache_dir / "val_idx.npy").exists()

        # Check meta.json structure
        with open(cache_dir / "meta.json") as f:
            meta = json.load(f)

        assert meta["source"] == "primer"
        assert "special_token_ids" in meta
        assert "sys" in meta["special_token_ids"]
        assert "usr" in meta["special_token_ids"]
        assert "asst" in meta["special_token_ids"]
        assert "eot" in meta["special_token_ids"]
        assert "splits" in meta
        assert "train" in meta["splits"]
        assert "val" in meta["splits"]
        assert "num_examples" in meta["splits"]["train"]
        assert "num_tokens" in meta["splits"]["train"]
        assert "num_label_tokens" in meta["splits"]["train"]
        assert meta["val_frac"] == 0.2
        assert meta["seed"] == 42
        assert meta["T_max"] == 512

        # Check train/val split
        train_examples = meta["splits"]["train"]["num_examples"]
        val_examples = meta["splits"]["val"]["num_examples"]
        assert train_examples + val_examples == 5

        # Check idx files
        train_idx = np.load(cache_dir / "train_idx.npy")
        val_idx = np.load(cache_dir / "val_idx.npy")
        assert len(train_idx) == train_examples
        assert len(val_idx) == val_examples

        # Check that tokens files are non-empty
        train_tokens_size = (cache_dir / "train_input_ids.bin").stat().st_size
        val_tokens_size = (cache_dir / "val_input_ids.bin").stat().st_size
        assert train_tokens_size > 0
        if val_examples > 0:
            assert val_tokens_size > 0


def test_build_primer_sft_cache_truncation():
    """Test that examples are truncated to T_max."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create example with long content
        jsonl_path = Path(tmpdir) / "primer.jsonl"
        with open(jsonl_path, "w") as f:
            example = {
                "messages": [
                    {"role": "user", "content": "q" * 100},
                    {"role": "assistant", "content": "a" * 100},
                ]
            }
            f.write(json.dumps(example) + "\n")

        cache_dir = Path(tmpdir) / "cache"
        tokenizer = FakeTokenizer()

        build_primer_sft_cache(
            str(jsonl_path),
            str(cache_dir),
            tokenizer=tokenizer,
            val_frac=0.0,
            seed=42,
            t_max=50,  # Force truncation
        )

        # Load tokens
        tokens = np.memmap(cache_dir / "train_input_ids.bin", dtype=np.uint16, mode="r")

        # Should be truncated to t_max
        assert len(tokens) <= 50


def test_render_chat_sft_multiple_turns():
    """Test rendering with multiple user/assistant turns."""
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]

    input_ids, labels = render_chat_sft(tokenizer, messages)

    # Count assistant content tokens (should be labeled)
    special_ids = tokenizer.special_token_ids()
    asst_id = special_ids["asst"]
    eot_id = special_ids["eot"]

    # Find all assistant sections
    asst_indices = [i for i, tid in enumerate(input_ids) if tid == asst_id]
    assert len(asst_indices) == 2  # Two assistant turns

    # For each assistant turn, verify content + eot are labeled
    for asst_idx in asst_indices:
        # Find the next eot
        eot_idx = None
        for i in range(asst_idx + 1, len(input_ids)):
            if input_ids[i] == eot_id:
                eot_idx = i
                break
        assert eot_idx is not None

        # Content between asst and eot should be labeled
        for i in range(asst_idx + 1, eot_idx):
            assert labels[i] == input_ids[i]
        # eot should be labeled
        assert labels[eot_idx] == eot_id
