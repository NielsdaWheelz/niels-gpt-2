"""Roundtrip tests for SFT cache writing/reading."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from niels_gpt.cache.build_cache import build_sft_cache
from niels_gpt.cache.meta import read_meta
from niels_gpt.data.primer_sft import render_chat_sft
from niels_gpt.tokenizer import DEFAULT_TOKENIZER_PATH, load_tokenizer


def _expected_splits(num_examples: int, val_frac: float, seed: int) -> tuple[set[int], set[int]]:
    """Compute deterministic train/val index sets using the same rule as the builder."""
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_examples, generator=gen).tolist()
    num_val = 0
    if num_examples > 1:
        num_val = max(1, int(num_examples * val_frac))
    val_indices = set(perm[:num_val])
    train_indices = set(range(num_examples)) - val_indices
    return train_indices, val_indices


def _load_sequences(tokens_path: Path, labels_path: Path, idx_path: Path) -> list[tuple[list[int], list[int]]]:
    tokens = np.fromfile(tokens_path, dtype=np.uint16)
    labels = np.fromfile(labels_path, dtype=np.int32)
    offsets = np.load(idx_path)
    assert offsets.dtype == np.int64
    assert offsets.tolist() == sorted(offsets.tolist())
    if len(offsets):
        assert len(tokens) >= offsets[-1]
        assert len(labels) == len(tokens)

    sequences: list[tuple[list[int], list[int]]] = []
    for i, start in enumerate(offsets.tolist()):
        end = offsets[i + 1] if i + 1 < len(offsets) else len(tokens)
        sequences.append((tokens[start:end].tolist(), labels[start:end].tolist()))
    return sequences


def test_sft_cache_roundtrip():
    tok = load_tokenizer(str(DEFAULT_TOKENIZER_PATH))

    examples = [
        [
            {"role": "system", "content": "you are a helpful assistant."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ],
        [
            {"role": "user", "content": "tell me a joke"},
            {"role": "assistant", "content": "knock knock"},
        ],
        [
            {"role": "user", "content": "what time is it?"},
            {"role": "assistant", "content": "time to write tests"},
        ],
        [
            {"role": "system", "content": "follow the rules"},
            {"role": "assistant", "content": "roger"},
        ],
    ]

    val_frac = 0.25
    seed = 123
    train_indices, val_indices = _expected_splits(len(examples), val_frac, seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "sft"
        build_sft_cache(
            examples,
            str(out_dir),
            tokenizer=tok,
            val_frac=val_frac,
            seed=seed,
        )

        meta = read_meta(str(out_dir / "meta.json"))
        assert meta["token_dtype"] == "uint16-le"
        assert meta["seed"] == seed
        assert meta["train_examples"] == len(train_indices)
        assert meta["val_examples"] == len(val_indices)

        expected_tokens_and_labels = [render_chat_sft(tok, ex, add_final_eot=True) for ex in examples]

        # Train split
        train_sequences = _load_sequences(
            out_dir / "train_input_ids.bin", out_dir / "train_labels.bin", out_dir / "train_idx.npy"
        )
        expected_train = {tuple(expected_tokens_and_labels[i][0]) for i in train_indices}
        assert {tuple(seq[0]) for seq in train_sequences} == expected_train
        # verify labels are stored and aligned
        expected_train_labels = {tuple(expected_tokens_and_labels[i][1]) for i in train_indices}
        assert {tuple(seq[1]) for seq in train_sequences} == expected_train_labels

        # Val split
        val_sequences = _load_sequences(
            out_dir / "val_input_ids.bin", out_dir / "val_labels.bin", out_dir / "val_idx.npy"
        )
        expected_val = {tuple(expected_tokens_and_labels[i][0]) for i in val_indices}
        assert {tuple(seq[0]) for seq in val_sequences} == expected_val
        expected_val_labels = {tuple(expected_tokens_and_labels[i][1]) for i in val_indices}
        assert {tuple(seq[1]) for seq in val_sequences} == expected_val_labels

