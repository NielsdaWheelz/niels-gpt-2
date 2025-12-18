"""Integration: SFT loader uses cached labels (no recomputation)."""

import tempfile
from pathlib import Path

import numpy as np
import torch

import train.sft as sft
from niels_gpt.cache.build_cache import build_sft_cache
from niels_gpt.data.primer_sft import render_chat_sft


class _FakeTokenizer:
    def __init__(self):
        self.vocab_size = 128
        self.model_path = None
        self._special = {"sys": 1, "usr": 2, "asst": 3, "eot": 4}

    def special_token_ids(self):
        return self._special

    def encode(self, text: str) -> list[int]:
        return [10 + (ord(c) % 16) for c in text]


def test_sft_loader_uses_cached_labels():
    tokenizer = _FakeTokenizer()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_root = Path(tmpdir) / "sft"
        source_dir = cache_root / "primer"
        build_sft_cache(
            [messages],
            str(source_dir),
            tokenizer=tokenizer,
            val_frac=0.0,
            seed=0,
            source_name="primer",
        )

        expected_ids, expected_labels = render_chat_sft(tokenizer, messages, add_final_eot=True)
        expected_shifted = np.array(expected_labels[1:], dtype=np.int64)
        T = len(expected_ids) - 1

        ds = sft._load_sft_source(
            cache_root,
            "primer",
            "train",
            T=T,
            device="cpu",
            assistant_only_loss=True,
            include_eot_in_loss=True,
            expected_tokenizer_sha=None,
            expected_special_token_ids=tokenizer.special_token_ids(),
        )

        gen = torch.Generator().manual_seed(0)
        x, _, y_masked = ds.get_batch(B=1, generator=gen)

        np.testing.assert_array_equal(x[0].cpu().numpy()[:T], np.array(expected_ids[:-1], dtype=np.int64))
        np.testing.assert_array_equal(y_masked[0].cpu().numpy(), expected_shifted[:T])
        assert (y_masked[0] != -100).any(), "labels should include assistant targets"
        assert (y_masked[0][: len(expected_shifted) - 1] == -100).sum() > 0, "non-assistant tokens must stay masked"

