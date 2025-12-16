"""Tests for generation utilities."""

import torch

from niels_gpt.generate import sample_next_token, top_k_filter


def test_temperature_zero_returns_argmax():
    """temperature=0 returns argmax (deterministic)."""
    # Create logits where argmax is at index 3
    logits = torch.tensor([1.0, 2.0, 1.5, 5.0, 0.5])

    result = sample_next_token(
        logits,
        temperature=0,
        top_k=None,
        generator=None,
    )

    assert result == 3, f"Expected argmax index 3, got {result}"


def test_top_k_one_matches_argmax():
    """top_k=1 matches argmax even when temperature>0."""
    # Create logits where argmax is at index 2
    logits = torch.tensor([0.5, 1.0, 3.5, 2.0])

    # Call multiple times to ensure determinism (with different seeds)
    results = []
    for seed in range(10):
        gen = torch.Generator()
        gen.manual_seed(seed)
        result = sample_next_token(
            logits,
            temperature=1.0,
            top_k=1,
            generator=gen,
        )
        results.append(result)

    # All results should be argmax (index 2)
    assert all(r == 2 for r in results), f"Expected all results to be 2, got {results}"


def test_generator_determinism():
    """Generator with same seed produces same sequence."""
    # Create some logits
    logits = torch.tensor([1.0, 2.0, 1.5, 3.0, 0.5])

    # Generate sequence with first generator
    gen1 = torch.Generator()
    gen1.manual_seed(123)
    sequence1 = []
    for _ in range(20):
        token = sample_next_token(
            logits,
            temperature=1.0,
            top_k=None,
            generator=gen1,
        )
        sequence1.append(token)

    # Generate sequence with second generator (same seed)
    gen2 = torch.Generator()
    gen2.manual_seed(123)
    sequence2 = []
    for _ in range(20):
        token = sample_next_token(
            logits,
            temperature=1.0,
            top_k=None,
            generator=gen2,
        )
        sequence2.append(token)

    assert sequence1 == sequence2, f"Sequences don't match:\n{sequence1}\n{sequence2}"


def test_top_k_filter_masks_correctly():
    """top_k_filter keeps exactly k finite values."""
    # Create logits with known values
    logits = torch.tensor([5.0, 2.0, 8.0, 1.0, 6.0, 3.0])
    k = 3

    filtered = top_k_filter(logits, k)

    # Count finite values (not -inf)
    finite_count = torch.isfinite(filtered).sum().item()

    assert finite_count == k, f"Expected {k} finite values, got {finite_count}"

    # The top-3 values should be at indices 2 (8.0), 4 (6.0), 0 (5.0)
    expected_finite_indices = {0, 2, 4}
    actual_finite_indices = set(torch.where(torch.isfinite(filtered))[0].tolist())

    assert actual_finite_indices == expected_finite_indices, (
        f"Expected finite indices {expected_finite_indices}, got {actual_finite_indices}"
    )

    # Verify that filtered values are still the same (not modified)
    for idx in expected_finite_indices:
        assert filtered[idx] == logits[idx], f"Value at {idx} was modified"


def test_top_k_filter_handles_ties():
    """top_k_filter keeps exactly k values even with ties."""
    # Create logits with ties: [5.0, 5.0, 5.0, 2.0, 1.0]
    # Top-3 should be exactly 3 of the tied values at 5.0
    logits = torch.tensor([5.0, 5.0, 5.0, 2.0, 1.0])
    k = 3

    filtered = top_k_filter(logits, k)

    # Count finite values - should be exactly k, not more
    finite_count = torch.isfinite(filtered).sum().item()
    assert finite_count == k, f"Expected exactly {k} finite values with ties, got {finite_count}"

    # All finite values should be 5.0
    finite_mask = torch.isfinite(filtered)
    finite_values = filtered[finite_mask]
    assert torch.all(finite_values == 5.0), f"Expected all finite values to be 5.0, got {finite_values}"


def test_temperature_validation():
    """Negative temperature should raise ValueError."""
    logits = torch.tensor([1.0, 2.0, 3.0])

    try:
        sample_next_token(logits, temperature=-0.5, top_k=None, generator=None)
        assert False, "Expected ValueError for negative temperature"
    except ValueError as e:
        assert "temperature must be >= 0" in str(e)
