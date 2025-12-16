"""
Tests for transformer blocks: MLP, Block, and init_weights.

Hard, loud tests that prevent silent shape/dtype/device bugs.
"""

import pytest
import torch
from dataclasses import replace

from niels_gpt.config import ModelConfig
from niels_gpt.model.blocks import MLP, Block, init_weights


def get_device() -> str:
    """Get available device: mps if available, else cpu."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def test_mlp_shape_and_finite_grads():
    """Test MLP returns correct shape and produces finite gradients."""
    cfg = ModelConfig()
    mlp = MLP(cfg)
    mlp.apply(init_weights)

    # Create input
    x = torch.randn(2, cfg.T, cfg.C, dtype=torch.float32)

    # Forward pass
    out = mlp(x)

    # Check shape
    assert out.shape == (2, cfg.T, cfg.C), f"MLP output shape mismatch: {out.shape}"

    # Backward pass
    loss = out.pow(2).mean()
    loss.backward()

    # Check at least one parameter has gradients
    has_grad = any(p.grad is not None for p in mlp.parameters())
    assert has_grad, "No parameters have gradients after backward"

    # Check all gradients are finite
    for name, param in mlp.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradients"


def test_block_forward_backward_and_return_attn():
    """Test Block forward/backward pass and return_attn contract."""
    cfg = ModelConfig()
    block = Block(cfg)
    block.apply(init_weights)

    # Create input
    x = torch.randn(2, cfg.T, cfg.C, dtype=torch.float32)

    # Test eval mode without return_attn
    block.eval()
    out = block(x)
    assert out.shape == (2, cfg.T, cfg.C), f"Block output shape mismatch: {out.shape}"

    # Test with return_attn=True
    block.train()
    result = block(x, return_attn=True)
    assert isinstance(result, tuple), "return_attn=True should return a tuple"
    assert len(result) == 2, "return_attn=True should return (output, attn_probs)"

    y, attn = result

    # Check output shape
    assert y.shape == (2, cfg.T, cfg.C), f"Block output shape mismatch: {y.shape}"

    # Check attention shape (B, H, T, T)
    assert isinstance(attn, torch.Tensor), "attention should be a tensor"
    assert attn.ndim == 4, f"attention should be 4D, got {attn.ndim}D"
    assert attn.shape[-2:] == (cfg.T, cfg.T), f"attention last two dims should be ({cfg.T}, {cfg.T}), got {attn.shape[-2:]}"

    # Backward pass
    loss = y.mean()
    loss.backward()

    # Check all gradients are finite
    for name, param in block.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradients"


def test_dropout_semantics():
    """Test dropout is deterministic in eval mode and stochastic in train mode."""
    # Create config with dropout=0.5 for reliable test
    cfg_drop = replace(ModelConfig(), dropout=0.5)
    block = Block(cfg_drop)
    block.apply(init_weights)

    # Create input
    x = torch.randn(2, cfg_drop.T, cfg_drop.C, dtype=torch.float32)

    # Test eval mode: two forwards should be identical
    block.eval()
    y1 = block(x)
    y2 = block(x)
    torch.testing.assert_close(y1, y2, msg="Eval mode should be deterministic")

    # Test train mode: two forwards should differ without reseeding
    block.train()
    torch.manual_seed(123)
    y1 = block(x)
    y2 = block(x)  # No reseeding between calls

    # More robust checks against flakiness
    assert not torch.allclose(y1, y2), "Train mode should be stochastic (outputs should differ)"
    max_diff = torch.max((y1 - y2).abs()).item()
    assert max_diff > 0, f"Train mode outputs should differ, but max diff is {max_diff}"
    frac_identical = (y1 == y2).float().mean().item()
    assert frac_identical < 0.999, f"Train mode outputs should differ, but {frac_identical:.3%} of elements are identical"


def test_init_weights_sanity():
    """Test init_weights sets LayerNorm and Linear weights correctly."""
    cfg = ModelConfig()
    block = Block(cfg)
    block.apply(init_weights)

    # Check LayerNorm initialization (ln1)
    assert torch.allclose(
        block.ln1.weight, torch.ones_like(block.ln1.weight)
    ), "LayerNorm weight should be initialized to ones"
    assert torch.allclose(
        block.ln1.bias, torch.zeros_like(block.ln1.bias)
    ), "LayerNorm bias should be initialized to zeros"

    # Check Linear initialization has reasonable std (should be ~0.02)
    # Check MLP fc1 as a sample
    weight_std = block.mlp.fc1.weight.std().item()
    assert 0.005 < weight_std < 0.05, f"Linear weight std {weight_std} should be near 0.02"
    assert torch.allclose(
        block.mlp.fc1.bias, torch.zeros_like(block.mlp.fc1.bias)
    ), "Linear bias should be initialized to zeros"


def test_block_mps_if_available():
    """Test Block works on MPS device if available (otherwise skip)."""
    device = get_device()
    if device == "cpu":
        pytest.skip("MPS not available, skipping MPS test")

    cfg = ModelConfig()
    block = Block(cfg).to(device)
    block.apply(init_weights)

    # Create input on MPS
    x = torch.randn(2, cfg.T, cfg.C, dtype=torch.float32, device=device)

    # Forward pass
    block.eval()
    out = block(x)

    # Check shape and device
    assert out.shape == (2, cfg.T, cfg.C), f"MPS output shape mismatch: {out.shape}"
    assert out.device.type == device, f"MPS output device mismatch: {out.device}"

    # Backward pass
    block.train()
    out = block(x)
    loss = out.mean()
    loss.backward()

    # Check gradients are finite
    for name, param in block.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"MPS: {name} has non-finite gradients"
