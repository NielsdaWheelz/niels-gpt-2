"""Malformed SFT examples should be dropped during cache build."""

import tempfile
from pathlib import Path

from niels_gpt.cache.build_cache import build_sft_cache
from niels_gpt.cache.meta import read_meta
from niels_gpt.tokenizer import get_default_tokenizer


def test_malformed_examples_are_dropped():
    tok = get_default_tokenizer()
    special = tok.special_token_ids()

    # content containing a special token id should be dropped
    bad_content = "<|asst|> sneaky"

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

        meta = read_meta(str(out_dir / "meta.json"))
        assert meta["dropped_malformed_examples"] == 1
        assert meta["train_examples"] + meta["val_examples"] == 1

