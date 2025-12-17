"""Tests for causal self-attention with RoPE and causal masking."""

import pytest
import torch

import niels_gpt.model.rope as rope
from niels_gpt.config import ModelConfig
from niels_gpt.device import get_device
from niels_gpt.model.blocks import CausalSelfAttention


@pytest.fixture
def device():
    """Return available device (mps or cpu, never cuda)."""
    return get_device()


@pytest.fixture
def cfg():
    """Small test config: T=8, C=32, H=4, D=8 (even)."""
    return ModelConfig(V=128, T=8, C=32, H=4, d_ff=128, dropout=0.1)


def test_attention_shape_dtype_device_and_rope_called(cfg, device, monkeypatch):
    """
    Test that CausalSelfAttention:
    - Returns correct shape, dtype, and device
    - Calls rope.apply_rope exactly once with correct shapes
    """
    # Track rope.apply_rope calls
    call_count = 0
    original_apply_rope = rope.apply_rope

    def rope_wrapper(q, k, sin, cos):
        nonlocal call_count
        call_count += 1

        # Verify shapes
        B, H, T, D = q.shape
        assert q.shape == k.shape, f"q and k shapes must match: {q.shape} vs {k.shape}"
        assert q.ndim == 4, f"q must be 4D, got {q.ndim}D"
        assert D % 2 == 0, f"D must be even, got {D}"

        # Call original function
        return original_apply_rope(q, k, sin, cos)

    # Monkeypatch rope.apply_rope
    monkeypatch.setattr(rope, "apply_rope", rope_wrapper)

    # Create module and move to device
    attn = CausalSelfAttention(cfg).to(device)
    attn.eval()  # Disable dropout for deterministic behavior

    # Create input
    x = torch.randn(2, cfg.T, cfg.C, device=device)

    # Forward pass
    y = attn(x)

    # Verify output shape, dtype, device
    assert y.shape == (2, cfg.T, cfg.C), f"Expected shape (2, {cfg.T}, {cfg.C}), got {y.shape}"
    assert y.dtype == x.dtype, f"Output dtype {y.dtype} must match input dtype {x.dtype}"
    assert y.device == x.device, f"Output device {y.device} must match input device {x.device}"

    # Verify rope.apply_rope was called exactly once
    assert call_count == 1, f"rope.apply_rope should be called exactly once, was called {call_count} times"


def test_causal_mask_enforced(cfg, device):
    """
    Test that causal mask prevents attention to future positions:
    - Attention weights above diagonal should be ~0
    - Each row should sum to ~1
    """
    # Create module and move to device
    attn = CausalSelfAttention(cfg).to(device)
    attn.eval()  # Disable dropout for deterministic behavior

    # Create input
    x = torch.randn(2, cfg.T, cfg.C, device=device)

    # Forward pass with return_attn=True
    y, a = attn(x, return_attn=True)

    # Verify attention weights shape
    B = 2
    assert a.shape == (B, cfg.H, cfg.T, cfg.T), (
        f"Expected attention shape (2, {cfg.H}, {cfg.T}, {cfg.T}), got {a.shape}"
    )

    # Create upper triangular mask (above diagonal)
    upper_mask = torch.triu(torch.ones(cfg.T, cfg.T, dtype=torch.bool), diagonal=1)

    # Check that upper triangle (future positions) has near-zero attention
    upper_values = a[..., upper_mask]
    max_upper = upper_values.abs().max().item()
    assert max_upper <= 1e-6, (
        f"Attention to future positions should be ~0, but max value is {max_upper}"
    )

    # Check that each row sums to ~1 (valid probability distribution)
    row_sums = a.sum(dim=-1)  # (B, H, T)
    expected = torch.ones_like(row_sums)
    assert torch.allclose(row_sums, expected, atol=1e-5), (
        f"Attention rows should sum to 1, but got range [{row_sums.min()}, {row_sums.max()}]"
    )


def test_attention_short_sequence(device):
    """
    Sanity test: T < cfg.T should work correctly.

    During generation, sequences start short and grow. The module must handle
    sequences shorter than cfg.T without errors.
    """
    # Large config.T but short input sequence
    cfg = ModelConfig(V=256, T=256, C=256, H=4, d_ff=1024, dropout=0.1)
    attn = CausalSelfAttention(cfg).to(device)
    attn.eval()

    # Short sequence: T=17 << cfg.T=256
    B, T_short, C = 2, 17, 256
    x = torch.randn(B, T_short, C, device=device)

    # Forward pass should work
    y = attn(x)

    # Verify output shape
    assert y.shape == (B, T_short, C), f"Expected shape (2, 17, 256), got {y.shape}"

    # Forward with return_attn should also work
    y2, a = attn(x, return_attn=True)
    assert y2.shape == (B, T_short, C)
    assert a.shape == (B, cfg.H, T_short, T_short), (
        f"Expected attention shape (2, {cfg.H}, 17, 17), got {a.shape}"
    )

    # Verify causal mask still enforced for short sequence
    upper_mask = torch.triu(torch.ones(T_short, T_short, dtype=torch.bool), diagonal=1)
    upper_values = a[..., upper_mask]
    max_upper = upper_values.abs().max().item()
    assert max_upper <= 1e-6, (
        f"Causal mask should work for short sequences, but max future attention is {max_upper}"
    )
