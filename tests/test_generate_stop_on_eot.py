"""Ensure generation halts exactly on eot token."""

import torch
import torch.nn as nn

from niels_gpt.generate import generate_ids
from niels_gpt.tokenizer import get_default_tokenizer


class DummyModel(nn.Module):
    def __init__(self, forced_tokens: list[int], vocab_size: int):
        super().__init__()
        self.forced_tokens = forced_tokens
        self.vocab_size = vocab_size
        self.call_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        logits = torch.zeros(B, T, self.vocab_size)
        if T > 0:
            idx = (
                self.forced_tokens[self.call_count]
                if self.call_count < len(self.forced_tokens)
                else 0
            )
            logits[:, -1, idx] = 100.0
            self.call_count += 1
        return logits


def test_generate_stops_exactly_on_eot():
    tok = get_default_tokenizer()
    eot = tok.special_token_ids()["eot"]

    # include eot midway to ensure truncation occurs before it is returned
    forced = tok.encode("keep this") + [eot] + tok.encode("drop this")
    model = DummyModel(forced, vocab_size=tok.vocab_size)

    prompt_ids = tok.encode_torch("preamble")
    out = generate_ids(
        model,
        prompt_ids,
        max_new_tokens=len(forced) + 2,
        T=128,
        temperature=0,
        top_k=None,
        eot_id=eot,
        device="cpu",
        generator=None,
    )

    decoded = tok.decode(out)
    assert "keep this" in decoded
    assert "drop this" not in decoded

