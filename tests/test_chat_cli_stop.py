"""Generation stop behavior tests (no substring truncation)."""

import torch
import torch.nn as nn

from niels_gpt.generate import generate_ids
from niels_gpt.tokenizer import get_default_tokenizer


class DummyModel(nn.Module):
    """Dummy model that outputs forced sequence of tokens."""

    def __init__(self, forced_tokens: list[int], vocab_size: int = 4096):
        super().__init__()
        self.forced_tokens = forced_tokens
        self.vocab_size = vocab_size
        self.call_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        logits = torch.zeros(B, T, self.vocab_size)
        if T > 0:
            forced_idx = (
                self.forced_tokens[self.call_count]
                if self.call_count < len(self.forced_tokens)
                else 0
            )
            logits[:, -1, forced_idx] = 100.0
            self.call_count += 1
        return logits


def test_generation_stops_on_eot_token():
    tok = get_default_tokenizer()
    eot = tok.special_token_ids()["eot"]

    # force model to emit some ids then eot
    forced = tok.encode("hello world")
    forced.append(eot)
    model = DummyModel(forced, vocab_size=tok.vocab_size)

    prompt_ids = tok.encode_torch("x")
    out = generate_ids(
        model,
        prompt_ids,
        max_new_tokens=10,
        T=64,
        temperature=0,
        top_k=None,
        eot_id=eot,
        device="cpu",
        generator=None,
    )

    # Output should exclude the eot token id
    assert eot not in out.tolist()
    decoded = tok.decode(out)
    assert "hello" in decoded


def test_substring_does_not_truncate_without_eot():
    tok = get_default_tokenizer()
    eot = tok.special_token_ids()["eot"]

    # Force sequence that includes "\nuser: " bytes in text but no eot
    forced = tok.encode("this is fine\nuser: not a stop")
    model = DummyModel(forced, vocab_size=tok.vocab_size)

    prompt_ids = tok.encode_torch("x")
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

    # ensure full forced sequence made it through (no truncation)
    out_forced = out.tolist()[len(prompt_ids) :]
    assert out_forced[: len(forced)] == forced
