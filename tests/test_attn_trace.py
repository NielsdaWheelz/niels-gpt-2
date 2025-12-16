"""Tests for attention tracing functionality in GPT.forward_with_attn_trace."""

import pytest
import torch

from niels_gpt.config import ModelConfig
from niels_gpt.model.gpt import GPT


@pytest.fixture
def small_config() -> ModelConfig:
    """
    Small model config for fast CPU testing.

    Keeps V=256 as required, but uses small dimensions for speed:
    - T=32 (context length)
    - C=64 (model width)
    - L=2 (layers)
    - H=4 (heads)
    - d_ff=256 (feedforward width)
    - dropout=0.1
    """
    return ModelConfig(
        V=256,
        T=32,
        C=64,
        L=2,
        H=4,
        d_ff=256,
        dropout=0.1,
    )


@pytest.fixture
def model(small_config: ModelConfig) -> GPT:
    """Create a small GPT model in eval mode for testing."""
    model = GPT(small_config)
    model.eval()  # Disable dropout for deterministic behavior
    return model


@pytest.fixture
def sample_input() -> torch.LongTensor:
    """
    Create sample token input for testing.

    Returns:
        (B=2, t=17) tensor with random token IDs in [0, 255].
    """
    torch.manual_seed(42)
    B, t = 2, 17
    x = torch.randint(0, 256, (B, t), dtype=torch.int64)
    return x


def test_forward_unchanged_eval_allclose(
    model: GPT, sample_input: torch.LongTensor
) -> None:
    """
    Test that forward_with_attn_trace produces identical logits to forward() in eval mode.

    This ensures the new method doesn't change the model's output behavior.
    """
    x = sample_input

    # Compute logits using normal forward
    with torch.no_grad():
        logits_a = model(x)

    # Compute logits using forward_with_attn_trace
    with torch.no_grad():
        logits_b, trace = model.forward_with_attn_trace(
            x, trace_layer=0, return_full_attn=False
        )

    # Verify shapes match
    assert logits_a.shape == logits_b.shape, (
        f"Shape mismatch: forward={logits_a.shape}, "
        f"forward_with_attn_trace={logits_b.shape}"
    )

    # Verify logits are numerically identical (within floating point tolerance)
    assert torch.allclose(logits_a, logits_b, atol=1e-6, rtol=0), (
        "Logits from forward() and forward_with_attn_trace() do not match. "
        f"Max absolute difference: {(logits_a - logits_b).abs().max().item()}"
    )


def test_trace_shapes_and_sums(model: GPT, sample_input: torch.LongTensor) -> None:
    """
    Test that attention trace has correct shapes and probability distributions.

    Verifies:
    - trace["layer"] matches the requested layer
    - attn_row has shape (B, H, t)
    - attn_full is None when return_full_attn=False
    - attn_row sums to 1.0 along the last dimension (proper probability distribution)
    """
    x = sample_input
    B, t = x.shape
    H = model.cfg.H

    # Trace layer 1 (ensure cfg.L >= 2 in test config)
    with torch.no_grad():
        logits, trace = model.forward_with_attn_trace(
            x, trace_layer=1, return_full_attn=False
        )

    # Check layer field
    assert trace["layer"] == 1, f"Expected layer=1, got {trace['layer']}"

    # Check attn_row shape
    attn_row = trace["attn_row"]
    expected_shape = (B, H, t)
    assert attn_row.shape == expected_shape, (
        f"attn_row shape mismatch: expected {expected_shape}, got {attn_row.shape}"
    )

    # Check attn_full is None when return_full_attn=False
    assert trace["attn_full"] is None, (
        "attn_full should be None when return_full_attn=False"
    )

    # Check that attn_row sums to 1.0 (proper probability distribution)
    attn_row_sums = attn_row.sum(dim=-1)  # (B, H)
    expected_sums = torch.ones_like(attn_row_sums)
    assert torch.allclose(attn_row_sums, expected_sums, atol=1e-4), (
        f"attn_row does not sum to 1.0. "
        f"Min sum: {attn_row_sums.min().item():.6f}, "
        f"Max sum: {attn_row_sums.max().item():.6f}"
    )


def test_full_attn_optional(model: GPT, sample_input: torch.LongTensor) -> None:
    """
    Test that full attention matrix is returned when return_full_attn=True.

    Verifies:
    - attn_full has shape (B, H, t, t)
    - attn_full sums to 1.0 along the last dimension
    - attn_row matches the last row of attn_full (query position t-1)
    """
    x = sample_input
    B, t = x.shape
    H = model.cfg.H

    # Request full attention matrix
    with torch.no_grad():
        logits, trace = model.forward_with_attn_trace(
            x, trace_layer=0, return_full_attn=True
        )

    # Check attn_full shape
    attn_full = trace["attn_full"]
    assert attn_full is not None, "attn_full should not be None when return_full_attn=True"
    expected_shape = (B, H, t, t)
    assert attn_full.shape == expected_shape, (
        f"attn_full shape mismatch: expected {expected_shape}, got {attn_full.shape}"
    )

    # Check that attn_full sums to 1.0 along last dimension
    attn_full_sums = attn_full.sum(dim=-1)  # (B, H, t)
    expected_sums = torch.ones_like(attn_full_sums)
    assert torch.allclose(attn_full_sums, expected_sums, atol=1e-4), (
        f"attn_full does not sum to 1.0. "
        f"Min sum: {attn_full_sums.min().item():.6f}, "
        f"Max sum: {attn_full_sums.max().item():.6f}"
    )

    # Check that attn_row matches the last row of attn_full
    attn_row = trace["attn_row"]
    attn_full_last_row = attn_full[:, :, t - 1, :]  # (B, H, t)
    assert torch.allclose(attn_row, attn_full_last_row, atol=1e-6), (
        "attn_row does not match attn_full[:, :, -1, :]. "
        f"Max absolute difference: {(attn_row - attn_full_last_row).abs().max().item()}"
    )


def test_trace_layer_bounds(model: GPT, sample_input: torch.LongTensor) -> None:
    """
    Test that out-of-bounds trace_layer raises ValueError.

    Verifies:
    - Negative trace_layer raises ValueError
    - trace_layer >= cfg.L raises ValueError
    """
    x = sample_input
    L = model.cfg.L

    # Test negative trace_layer
    with pytest.raises(ValueError, match=r"trace_layer must be in range"):
        with torch.no_grad():
            model.forward_with_attn_trace(x, trace_layer=-1)

    # Test trace_layer >= L
    with pytest.raises(ValueError, match=r"trace_layer must be in range"):
        with torch.no_grad():
            model.forward_with_attn_trace(x, trace_layer=L)

    # Test trace_layer way out of bounds
    with pytest.raises(ValueError, match=r"trace_layer must be in range"):
        with torch.no_grad():
            model.forward_with_attn_trace(x, trace_layer=100)


def test_trace_all_layers(model: GPT, sample_input: torch.LongTensor) -> None:
    """
    Test that tracing works for all valid layer indices.

    This is an additional test to ensure robustness across all layers.
    """
    x = sample_input
    L = model.cfg.L

    # Test tracing each layer
    for layer_idx in range(L):
        with torch.no_grad():
            logits, trace = model.forward_with_attn_trace(
                x, trace_layer=layer_idx, return_full_attn=False
            )

        # Verify trace contains correct layer index
        assert trace["layer"] == layer_idx, (
            f"Expected layer={layer_idx}, got {trace['layer']}"
        )

        # Verify attn_row has correct shape
        B, t = x.shape
        H = model.cfg.H
        expected_shape = (B, H, t)
        assert trace["attn_row"].shape == expected_shape, (
            f"Layer {layer_idx}: attn_row shape mismatch: "
            f"expected {expected_shape}, got {trace['attn_row'].shape}"
        )


def test_different_sequence_lengths(model: GPT) -> None:
    """
    Test that tracing works correctly for different sequence lengths.

    This ensures the method handles variable-length inputs properly.
    """
    torch.manual_seed(43)
    B = 2
    H = model.cfg.H

    # Test with different sequence lengths
    for t in [1, 5, 10, 16, 32]:
        # Skip if t exceeds model's max context length
        if t > model.cfg.T:
            continue

        x = torch.randint(0, 256, (B, t), dtype=torch.int64)

        with torch.no_grad():
            logits, trace = model.forward_with_attn_trace(
                x, trace_layer=0, return_full_attn=True
            )

        # Verify shapes
        assert logits.shape == (B, t, 256), f"Logits shape mismatch for t={t}"
        assert trace["attn_row"].shape == (B, H, t), (
            f"attn_row shape mismatch for t={t}"
        )
        assert trace["attn_full"].shape == (B, H, t, t), (
            f"attn_full shape mismatch for t={t}"
        )

        # Verify probability distributions
        attn_row_sums = trace["attn_row"].sum(dim=-1)
        assert torch.allclose(attn_row_sums, torch.ones_like(attn_row_sums), atol=1e-4), (
            f"attn_row sums incorrect for t={t}"
        )


def test_device_consistency_cpu(model: GPT, sample_input: torch.LongTensor) -> None:
    """
    Test that all returned tensors remain on the same device as the model (CPU).

    Verifies that forward_with_attn_trace does not accidentally move tensors
    between devices (e.g., no hidden .cpu() calls).
    """
    x = sample_input
    model_device = next(model.parameters()).device

    # Ensure model and input are on CPU
    assert model_device.type == "cpu", f"Expected CPU model, got {model_device}"
    assert x.device.type == "cpu", f"Expected CPU input, got {x.device}"

    with torch.no_grad():
        logits, trace = model.forward_with_attn_trace(
            x, trace_layer=0, return_full_attn=True
        )

    # Verify all tensors are on CPU
    assert logits.device.type == "cpu", f"logits on wrong device: {logits.device}"
    assert trace["attn_row"].device.type == "cpu", (
        f"attn_row on wrong device: {trace['attn_row'].device}"
    )
    assert trace["attn_full"].device.type == "cpu", (
        f"attn_full on wrong device: {trace['attn_full'].device}"
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_device_consistency_mps(small_config: ModelConfig) -> None:
    """
    Test that all returned tensors remain on MPS device.

    This test is skipped if MPS is not available.
    Verifies device consistency on Apple Silicon GPU backend.
    """
    torch.manual_seed(44)
    model = GPT(small_config)
    model.eval()
    model = model.to("mps")

    B, t = 2, 17
    x = torch.randint(0, 256, (B, t), dtype=torch.int64, device="mps")

    model_device = next(model.parameters()).device
    assert model_device.type == "mps", f"Expected MPS model, got {model_device}"

    with torch.no_grad():
        logits, trace = model.forward_with_attn_trace(
            x, trace_layer=0, return_full_attn=True
        )

    # Verify all tensors are on MPS
    assert logits.device.type == "mps", f"logits on wrong device: {logits.device}"
    assert trace["attn_row"].device.type == "mps", (
        f"attn_row on wrong device: {trace['attn_row'].device}"
    )
    assert trace["attn_full"].device.type == "mps", (
        f"attn_full on wrong device: {trace['attn_full'].device}"
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_forward_equivalence_mps_relaxed_tolerance(
    small_config: ModelConfig,
) -> None:
    """
    Test forward equivalence on MPS with relaxed tolerance.

    MPS may have slightly larger numerical drift than CPU due to different
    compute kernels and floating-point rounding. This test uses a more
    permissive tolerance (1e-5 instead of 1e-6) to avoid spurious failures.
    """
    torch.manual_seed(45)
    model = GPT(small_config)
    model.eval()
    model = model.to("mps")

    B, t = 2, 17
    x = torch.randint(0, 256, (B, t), dtype=torch.int64, device="mps")

    with torch.no_grad():
        logits_a = model(x)
        logits_b, trace = model.forward_with_attn_trace(
            x, trace_layer=0, return_full_attn=False
        )

    # MPS-aware tolerance: atol=1e-5 instead of 1e-6
    assert torch.allclose(logits_a, logits_b, atol=1e-5, rtol=1e-5), (
        "Logits from forward() and forward_with_attn_trace() do not match on MPS. "
        f"Max absolute difference: {(logits_a - logits_b).abs().max().item()}"
    )
