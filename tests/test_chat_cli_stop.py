"""Test stop sequence truncation in generate_ids."""

import torch
import torch.nn as nn

from niels_gpt.generate import generate_ids
from niels_gpt.tokenizer import decode, encode


class DummyModel(nn.Module):
    """Dummy model that outputs forced sequence of tokens."""

    def __init__(self, forced_tokens: list[int], vocab_size: int = 256):
        super().__init__()
        self.forced_tokens = forced_tokens
        self.vocab_size = vocab_size
        self.call_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return logits with a single very large value at the forced next token.

        Args:
            x: (B, T) token ids

        Returns:
            logits: (B, T, V) with forced token having very large logit
        """
        B, T = x.shape

        # Create logits: (B, T, V) all zeros
        logits = torch.zeros(B, T, self.vocab_size)

        # Only set logits if T > 0
        if T > 0:
            # For the last position, set the forced token to have very large logit
            if self.call_count < len(self.forced_tokens):
                forced_idx = self.forced_tokens[self.call_count]
                logits[:, -1, forced_idx] = 100.0
                self.call_count += 1
            else:
                # If we've exhausted forced tokens, just return argmax=0
                logits[:, -1, 0] = 100.0

        return logits


def test_stop_sequence_truncation():
    """Stop sequence detection truncates output before the stop sequence."""
    # Create a sequence: "hello\nuser: "
    target_text = "hello\nuser: "
    target_bytes = target_text.encode("utf-8")
    forced_tokens = list(target_bytes)

    # Create dummy model that outputs this sequence
    model = DummyModel(forced_tokens, vocab_size=256)

    # Use a simple prompt (single character)
    prompt_ids = encode("x")

    # Generate with stop sequence
    output_ids = generate_ids(
        model,
        prompt_ids,
        max_new_tokens=100,  # Large enough to generate full sequence
        T=256,
        temperature=0,
        top_k=None,
        stop_sequences=[b"\nuser: "],
        device="cpu",
        generator=None,
    )

    # Decode output
    output_text = decode(output_ids)

    # Should include "hello" but NOT "\nuser: "
    assert "hello" in output_text, f"Expected 'hello' in output, got: {repr(output_text)}"
    assert "\nuser: " not in output_text, f"Expected stop sequence to be truncated, got: {repr(output_text)}"
    assert output_text == "xhello", f"Expected exactly 'xhello', got: {repr(output_text)}"
