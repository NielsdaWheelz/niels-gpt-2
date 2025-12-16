"""
Tests for RoPE (Rotary Positional Embeddings) utilities.

Hard, loud tests that prevent silent shape/dtype/device bugs.
"""

import pytest
import torch

from niels_gpt.model.rope import apply_rope, rope_cache


def test_rope_cache_shapes_and_defaults():
    """Test rope_cache returns correct shapes, dtype, and device with defaults."""
    T = 256
    D = 64

    sin, cos = rope_cache(T=T, D=D)

    # Check shapes
    assert sin.shape == (1, 1, T, D // 2), f"sin shape mismatch: {sin.shape}"
    assert cos.shape == (1, 1, T, D // 2), f"cos shape mismatch: {cos.shape}"

    # Check dtype (default float32)
    assert sin.dtype == torch.float32, f"sin dtype mismatch: {sin.dtype}"
    assert cos.dtype == torch.float32, f"cos dtype mismatch: {cos.dtype}"

    # Check device (default cpu)
    assert sin.device.type == "cpu", f"sin device mismatch: {sin.device}"
    assert cos.device.type == "cpu", f"cos device mismatch: {cos.device}"


def test_apply_rope_preserves_shape_dtype_device_cpu():
    """Test apply_rope preserves shape, dtype, device and returns finite values."""
    B, H, Tq, D = 2, 4, 17, 64

    # Create random q and k on CPU with float32
    q = torch.randn(B, H, Tq, D, dtype=torch.float32, device="cpu")
    k = torch.randn(B, H, Tq, D, dtype=torch.float32, device="cpu")

    # Build cache with T=256 (larger than Tq=17)
    sin, cos = rope_cache(T=256, D=D, device="cpu", dtype=torch.float32)

    # Apply RoPE
    q_rot, k_rot = apply_rope(q, k, sin, cos)

    # Check shapes
    assert q_rot.shape == q.shape, f"q_rot shape mismatch: {q_rot.shape} vs {q.shape}"
    assert k_rot.shape == k.shape, f"k_rot shape mismatch: {k_rot.shape} vs {k.shape}"

    # Check dtypes
    assert q_rot.dtype == q.dtype, f"q_rot dtype mismatch: {q_rot.dtype} vs {q.dtype}"
    assert k_rot.dtype == k.dtype, f"k_rot dtype mismatch: {k_rot.dtype} vs {k.dtype}"

    # Check devices
    assert q_rot.device == q.device, f"q_rot device mismatch: {q_rot.device} vs {q.device}"
    assert k_rot.device == k.device, f"k_rot device mismatch: {k_rot.device} vs {k.device}"

    # Check all values are finite
    assert torch.isfinite(q_rot).all(), "q_rot contains non-finite values"
    assert torch.isfinite(k_rot).all(), "k_rot contains non-finite values"


def test_apply_rope_pairwise_norm_preserved_cpu():
    """Test that RoPE preserves per-pair norms (rotation invariant)."""
    B, H, Tq, D = 2, 4, 17, 64

    # Create random q and k
    q = torch.randn(B, H, Tq, D, dtype=torch.float32, device="cpu")
    k = torch.randn(B, H, Tq, D, dtype=torch.float32, device="cpu")

    # Build cache
    sin, cos = rope_cache(T=256, D=D, device="cpu", dtype=torch.float32)

    # Apply RoPE
    q_rot, k_rot = apply_rope(q, k, sin, cos)

    # Reshape to pairs: (B, H, Tq, D) -> (B, H, Tq, D//2, 2)
    q_pairs = q.reshape(B, H, Tq, D // 2, 2)
    q_rot_pairs = q_rot.reshape(B, H, Tq, D // 2, 2)
    k_pairs = k.reshape(B, H, Tq, D // 2, 2)
    k_rot_pairs = k_rot.reshape(B, H, Tq, D // 2, 2)

    # Compute norms: sqrt(a^2 + b^2)
    q_norm_before = torch.norm(q_pairs, dim=-1)
    q_norm_after = torch.norm(q_rot_pairs, dim=-1)
    k_norm_before = torch.norm(k_pairs, dim=-1)
    k_norm_after = torch.norm(k_rot_pairs, dim=-1)

    # Check norms are preserved (rotation is norm-preserving)
    torch.testing.assert_close(
        q_norm_before,
        q_norm_after,
        rtol=1e-5,
        atol=1e-5,
        msg="q pairwise norms not preserved",
    )
    torch.testing.assert_close(
        k_norm_before,
        k_norm_after,
        rtol=1e-5,
        atol=1e-5,
        msg="k pairwise norms not preserved",
    )


def test_apply_rope_cache_slicing():
    """Test that apply_rope works with various query lengths using cache slicing."""
    D = 64
    B, H = 2, 4

    # Build cache with T=256
    sin, cos = rope_cache(T=256, D=D, device="cpu", dtype=torch.float32)

    # Test with different query lengths
    for Tq in [1, 17, 256]:
        q = torch.randn(B, H, Tq, D, dtype=torch.float32, device="cpu")
        k = torch.randn(B, H, Tq, D, dtype=torch.float32, device="cpu")

        # Should not crash
        q_rot, k_rot = apply_rope(q, k, sin, cos)

        # Check shapes are correct
        assert q_rot.shape == (B, H, Tq, D), f"q_rot shape mismatch for Tq={Tq}"
        assert k_rot.shape == (B, H, Tq, D), f"k_rot shape mismatch for Tq={Tq}"


def test_rope_requires_even_D():
    """Test that rope_cache raises an error for odd D."""
    T = 8
    D_odd = 63

    # rope_cache should raise for odd D
    with pytest.raises(AssertionError):
        rope_cache(T=T, D=D_odd)

    # Also test apply_rope rejects odd-D inputs
    D_odd = 63
    B, H, Tq = 2, 4, 17
    q = torch.randn(B, H, Tq, D_odd, dtype=torch.float32, device="cpu")
    k = torch.randn(B, H, Tq, D_odd, dtype=torch.float32, device="cpu")

    # Create a dummy sin/cos (won't match, but test the D check)
    sin = torch.randn(1, 1, Tq, D_odd // 2, dtype=torch.float32, device="cpu")
    cos = torch.randn(1, 1, Tq, D_odd // 2, dtype=torch.float32, device="cpu")

    with pytest.raises(AssertionError):
        apply_rope(q, k, sin, cos)


@pytest.mark.skipif(
    not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
    reason="MPS not available",
)
def test_apply_rope_mps_optional():
    """Optional test for MPS device if available."""
    B, H, Tq, D = 2, 4, 17, 64

    # Create random q and k on MPS
    q = torch.randn(B, H, Tq, D, dtype=torch.float32, device="mps")
    k = torch.randn(B, H, Tq, D, dtype=torch.float32, device="mps")

    # Build cache on MPS
    sin, cos = rope_cache(T=256, D=D, device="mps", dtype=torch.float32)

    # Apply RoPE
    q_rot, k_rot = apply_rope(q, k, sin, cos)

    # Check shapes
    assert q_rot.shape == q.shape, f"q_rot shape mismatch on MPS"
    assert k_rot.shape == k.shape, f"k_rot shape mismatch on MPS"

    # Check dtypes
    assert q_rot.dtype == torch.float32, f"q_rot dtype mismatch on MPS"
    assert k_rot.dtype == torch.float32, f"k_rot dtype mismatch on MPS"

    # Check devices
    assert q_rot.device.type == "mps", f"q_rot device mismatch on MPS"
    assert k_rot.device.type == "mps", f"k_rot device mismatch on MPS"

    # Note: We don't require bit-identical values vs CPU
