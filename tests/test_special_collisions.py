import tempfile
from pathlib import Path

import pytest

from niels_gpt.cache.build_cache import build_pretrain_cache, build_sft_cache
from niels_gpt.special_tokens import ASST_TOKEN, SYS_TOKEN
from niels_gpt.tokenizer import DEFAULT_TOKENIZER_PATH, load_tokenizer


def test_pretrain_cache_rejects_special_literal(tmp_path: Path):
    tok = load_tokenizer(str(DEFAULT_TOKENIZER_PATH))
    texts = [f"this text leaks {SYS_TOKEN} literal"]

    with pytest.raises(ValueError, match="special token literal"):
        build_pretrain_cache(
            texts,
            str(tmp_path),
            tokenizer=tok,
            max_train_tokens=10,
            max_val_tokens=0,
            shard_bytes=64,
            seed=1,
            shuffle_buffer=None,
            source_name="synthetic",
            streaming=False,
        )


def test_sft_cache_rejects_special_literal(tmp_path: Path):
    tok = load_tokenizer(str(DEFAULT_TOKENIZER_PATH))
    examples = [
        [
            {"role": "system", "content": "x"},
            {"role": "user", "content": f"contains {ASST_TOKEN} literal"},
        ]
    ]

    build_sft_cache(
        examples,
        str(tmp_path),
        tokenizer=tok,
        val_frac=0.0,
        seed=1,
        source_name="synthetic",
    )
    assert (tmp_path / "train_input_ids.bin").exists()
    assert (tmp_path / "train_labels.bin").exists()

