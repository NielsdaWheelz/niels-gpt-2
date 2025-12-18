"""Determinism tests for cache builders."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from niels_gpt.cache.build_cache import build_pretrain_cache, build_sft_cache
from niels_gpt.cache.meta import sha256_file
from niels_gpt.tokenizer import DEFAULT_TOKENIZER_PATH, load_tokenizer


def _hash_dir(path: Path) -> dict[str, str]:
    """Return sha256 for every file under path (relative paths)."""
    hashes: dict[str, str] = {}
    for root, _, files in os.walk(path):
        for fname in sorted(files):
            fpath = Path(root) / fname
            rel = fpath.relative_to(path)
            hashes[str(rel)] = sha256_file(str(fpath))
    return hashes


def test_cache_build_determinism():
    tokenizer = load_tokenizer(str(DEFAULT_TOKENIZER_PATH))

    texts = ["alpha", "beta", "gamma", "delta"]
    examples = [
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}],
        [{"role": "user", "content": "ping"}, {"role": "assistant", "content": "pong"}],
        [{"role": "assistant", "content": "only assistant message"}],
    ]

    def build_everything(root: Path):
        pretrain_dir = root / "pretrain"
        sft_dir = root / "sft"
        build_pretrain_cache(
            texts,
            str(pretrain_dir),
            tokenizer=tokenizer,
            max_train_tokens=50,
            max_val_tokens=12,
            shard_bytes=1024,
            seed=7,
            shuffle_buffer=3,
            source_name="synthetic",
            source_config=None,
            streaming=False,
        )
        build_sft_cache(
            examples,
            str(sft_dir),
            tokenizer=tokenizer,
            val_frac=0.5,
            seed=7,
        )

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        build_everything(Path(d1))
        build_everything(Path(d2))

        pretrain_hashes_1 = _hash_dir(Path(d1) / "pretrain")
        pretrain_hashes_2 = _hash_dir(Path(d2) / "pretrain")
        assert pretrain_hashes_1 == pretrain_hashes_2

        sft_hashes_1 = _hash_dir(Path(d1) / "sft")
        sft_hashes_2 = _hash_dir(Path(d2) / "sft")
        assert sft_hashes_1 == sft_hashes_2

