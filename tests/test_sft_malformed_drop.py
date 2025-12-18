"""Malformed SFT examples should be dropped during cache build."""

import tempfile
from pathlib import Path

import pytest

from niels_gpt.cache.build_cache import build_sft_cache
from niels_gpt.tokenizer import get_default_tokenizer
from niels_gpt.special_tokens import ASST_TOKEN


def test_malformed_examples_are_dropped():
    tok = get_default_tokenizer()
    bad_content = f"{ASST_TOKEN} sneaky"

    examples = [
        [{"role": "user", "content": "ok"}, {"role": "assistant", "content": "fine"}],
        [{"role": "user", "content": bad_content}, {"role": "assistant", "content": "nope"}],
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        build_sft_cache(
            examples,
            str(out_dir),
            tokenizer=tok,
            val_frac=0.5,
            seed=123,
        )
        assert (out_dir / "train_input_ids.bin").exists()
        assert (out_dir / "train_labels.bin").exists()