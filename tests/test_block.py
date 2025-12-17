"""
Tests for transformer blocks: MLP, Block, and init_weights.

Hard, loud tests that prevent silent shape/dtype/device bugs.
"""

import pytest
import torch
from dataclasses import replace

from niels_gpt.config import ModelConfig
from niels_gpt.model.blocks import MLP, Block, RMSNorm, init_weights


def get_device() -> str:
    """Get available device: mps if available, else cpu."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def tiny_cfg() -> ModelConfig:
    return ModelConfig(V=128, T=32, C=64, L=2, H=4, d_ff=256, dropout=0.1)


def test_mlp_shape_and_finite_grads():
    """Test MLP returns correct shape and produces finite gradients."""
    cfg = tiny_cfg()
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
    cfg = tiny_cfg()
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
    cfg_drop = replace(tiny_cfg(), dropout=0.5)
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
    """Test init_weights sets RMSNorm and Linear weights correctly."""
    cfg = tiny_cfg()
    block = Block(cfg)
    block.apply(init_weights)

    # Check RMSNorm initialization (ln1)
    assert torch.allclose(
        block.ln1.weight, torch.ones_like(block.ln1.weight)
    ), "RMSNorm weight should be initialized to ones"
    assert not hasattr(block.ln1, "bias"), "RMSNorm should not have a bias parameter"

    # Check Linear initialization has reasonable std (should be ~0.02)
    weight_std = block.mlp.fc_a.weight.std().item()
    assert 0.005 < weight_std < 0.05, f"Linear weight std {weight_std} should be near 0.02"
    assert torch.allclose(
        block.mlp.fc_a.bias, torch.zeros_like(block.mlp.fc_a.bias)
    ), "Linear bias should be initialized to zeros"
    assert torch.allclose(
        block.mlp.fc_out.bias, torch.zeros_like(block.mlp.fc_out.bias)
    ), "Output Linear bias should be initialized to zeros"


def test_block_mps_if_available():
    """Test Block works on MPS device if available (otherwise skip)."""
    device = get_device()
    if device == "cpu":
        pytest.skip("MPS not available, skipping MPS test")

    cfg = tiny_cfg()
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


def test_rmsnorm_has_no_bias():
    """RMSNorm should expose only a weight parameter."""
    norm = RMSNorm(dim=64, eps=1e-5)
    has_bias_param = any("bias" in name for name, _ in norm.named_parameters())
    assert not has_bias_param, "RMSNorm must not register a bias parameter"
    assert not hasattr(norm, "bias"), "RMSNorm should not carry a bias attribute"


def test_swiglu_projection_shapes():
    """SwiGLU projections should follow (C->d_ff, C->d_ff, d_ff->C) layout."""
    cfg = ModelConfig(V=128, T=8, C=512, L=1, H=8, d_ff=1536, dropout=0.1)
    mlp = MLP(cfg)

    assert mlp.fc_a.weight.shape == (cfg.d_ff, cfg.C)
    assert mlp.fc_g.weight.shape == (cfg.d_ff, cfg.C)
    assert mlp.fc_out.weight.shape == (cfg.C, cfg.d_ff)

    x = torch.randn(2, 3, cfg.C)
    out = mlp(x)
    assert out.shape == (2, 3, cfg.C)
