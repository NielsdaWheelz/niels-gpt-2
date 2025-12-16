"""Tests for GPT wrapper: shape, dtype, device, weight tying, and backward pass."""

import pytest
import torch
import torch.nn.functional as F

from niels_gpt.config import ModelConfig
from niels_gpt.device import get_device
from niels_gpt.model.gpt import GPT


def test_gpt_shape_dtype_device():
    """Test that GPT produces correct output shape, dtype, and device."""
    cfg = ModelConfig()
    device = get_device()
    model = GPT(cfg).to(device)

    # Create input with full context length
    x = torch.randint(0, cfg.V, (2, cfg.T), dtype=torch.long, device=device)

    # Forward pass
    logits = model(x)

    # Check shape
    assert logits.shape == (2, cfg.T, cfg.V), (
        f"Expected shape (2, {cfg.T}, {cfg.V}), got {logits.shape}"
    )

    # Check dtype
    assert logits.dtype == torch.float32, f"Expected float32, got {logits.dtype}"

    # Check device
    assert logits.device.type == device, (
        f"Expected device {device}, got {logits.device.type}"
    )


def test_gpt_variable_length_ok():
    """Test that GPT handles variable sequence lengths T_cur < T."""
    cfg = ModelConfig()
    device = get_device()
    model = GPT(cfg).to(device)

    # Create input with shorter sequence length
    T_cur = 17
    x = torch.randint(0, cfg.V, (2, T_cur), dtype=torch.long, device=device)

    # Forward pass
    logits = model(x)

    # Check shape
    assert logits.shape == (2, T_cur, cfg.V), (
        f"Expected shape (2, {T_cur}, {cfg.V}), got {logits.shape}"
    )


def test_gpt_rejects_too_long():
    """Test that GPT raises an error for sequences longer than max context."""
    cfg = ModelConfig()
    device = get_device()
    model = GPT(cfg).to(device)

    # Create input with T+1 tokens (too long)
    x = torch.randint(0, cfg.V, (1, cfg.T + 1), dtype=torch.long, device=device)

    # Should raise AssertionError or ValueError
    with pytest.raises((AssertionError, ValueError)):
        model(x)


def test_gpt_weight_tying():
    """Test that lm_head and tok_emb share the same weight tensor."""
    cfg = ModelConfig()
    model = GPT(cfg)

    # Check weight tying by identity
    assert model.lm_head.weight is model.tok_emb.weight, (
        "lm_head.weight and tok_emb.weight must be the same object (weight tying)"
    )


def test_gpt_backward_smoke_finite_grads():
    """Test that backward pass produces finite gradients."""
    cfg = ModelConfig()
    device = get_device()
    model = GPT(cfg).to(device)

    # Use small sequence length for speed
    B, T_cur = 2, 32
    x = torch.randint(0, cfg.V, (B, T_cur), dtype=torch.long, device=device)
    y = torch.randint(0, cfg.V, (B, T_cur), dtype=torch.long, device=device)

    # Forward pass
    logits = model(x)

    # Compute cross-entropy loss
    loss = F.cross_entropy(logits.view(-1, cfg.V), y.view(-1))

    # Backward pass
    loss.backward()

    # Check that gradients exist and are finite
    assert model.tok_emb.weight.grad is not None, (
        "tok_emb.weight.grad should exist after backward"
    )
    assert torch.isfinite(model.tok_emb.weight.grad).all(), (
        "tok_emb.weight.grad should be finite"
    )
