"""Tests for dropout policy cleanup (PR-06).

Validates that:
1. Block-level dropout wrapper is removed
2. MLP has its own output dropout
3. Attention tracing returns pre-dropout normalized probabilities
"""

import torch
import torch.nn as nn

from niels_gpt.config import ModelConfig
from niels_gpt.model.blocks import Block, CausalSelfAttention, MLP


def _tiny_config(dropout: float = 0.1) -> ModelConfig:
    """Create a minimal config for testing."""
    return ModelConfig(
        V=128,
        T=32,
        C=64,
        H=4,
        L=1,
        d_ff=128,
        dropout=dropout,
        rope_theta=10000.0,
    )


class TestStructuralDropoutPolicy:
    """Test 1: structural dropout policy - verify dropout modules are placed correctly."""

    def test_block_has_no_dropout_attribute(self):
        """Block should not have a dropout attribute (or it should be Identity)."""
        cfg = _tiny_config(dropout=0.1)
        block = Block(cfg)

        # Block should NOT have a dropout attribute at all
        assert not hasattr(block, "dropout"), (
            "Block should not have a 'dropout' attribute - "
            "dropout should be inside submodules only"
        )

    def test_attention_has_attn_and_resid_dropout(self):
        """CausalSelfAttention should have both attn_dropout and resid_dropout."""
        cfg = _tiny_config(dropout=0.1)
        attn = CausalSelfAttention(cfg)

        assert hasattr(attn, "attn_dropout"), "CausalSelfAttention missing attn_dropout"
        assert hasattr(attn, "resid_dropout"), "CausalSelfAttention missing resid_dropout"
        assert isinstance(attn.attn_dropout, nn.Dropout), "attn_dropout should be nn.Dropout"
        assert isinstance(attn.resid_dropout, nn.Dropout), "resid_dropout should be nn.Dropout"

    def test_mlp_has_resid_dropout(self):
        """MLP should have resid_dropout applied to its output."""
        cfg = _tiny_config(dropout=0.1)
        mlp = MLP(cfg)

        assert hasattr(mlp, "resid_dropout"), "MLP missing resid_dropout"
        assert isinstance(mlp.resid_dropout, nn.Dropout), "MLP resid_dropout should be nn.Dropout"


class TestDropoutActiveDuringTraining:
    """Test 2: dropout is active in train mode."""

    def test_block_output_varies_with_different_seeds(self):
        """With high dropout, different seeds should produce different outputs in train mode."""
        cfg = _tiny_config(dropout=0.5)
        block = Block(cfg)
        block.train()

        B, T, C = 2, 8, cfg.C
        x = torch.randn(B, T, C)

        # Run with seed 0
        torch.manual_seed(0)
        y0 = block(x.clone())

        # Run with seed 1
        torch.manual_seed(1)
        y1 = block(x.clone())

        # Outputs should differ due to dropout
        assert not torch.allclose(y0, y1), (
            "Block outputs with different seeds should differ in train mode - "
            "dropout may not be active"
        )

    def test_block_output_deterministic_in_eval(self):
        """In eval mode, outputs should be deterministic regardless of seed."""
        cfg = _tiny_config(dropout=0.5)
        block = Block(cfg)
        block.eval()

        B, T, C = 2, 8, cfg.C
        x = torch.randn(B, T, C)

        # Run with seed 0
        torch.manual_seed(0)
        y0 = block(x.clone())

        # Run with seed 1
        torch.manual_seed(1)
        y1 = block(x.clone())

        # Outputs should be identical in eval mode
        assert torch.allclose(y0, y1), (
            "Block outputs with different seeds should be identical in eval mode"
        )


class TestAttentionProbsPreDropout:
    """Test 3: attention probs returned are pre-dropout and normalized."""

    def test_attention_probs_normalized_in_train_mode(self):
        """Returned attention probs should sum to 1 even with aggressive dropout."""
        cfg = _tiny_config(dropout=0.9)  # Aggressive dropout
        attn = CausalSelfAttention(cfg)
        attn.train()

        B, T, C = 2, 8, cfg.C
        x = torch.randn(B, T, C)

        # Run attention with tracing
        torch.manual_seed(42)
        out, attn_probs = attn(x, return_attn=True)

        # Check finite
        assert torch.isfinite(attn_probs).all(), "Attention probs should be finite"

        # Check normalized (sum to 1 along last dim)
        sums = attn_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), (
            f"Attention probs should sum to 1, got sums in range [{sums.min()}, {sums.max()}]"
        )

        # Check values in [0, 1]
        assert (attn_probs >= 0).all(), "Attention probs should be non-negative"
        assert (attn_probs <= 1 + 1e-6).all(), "Attention probs should be <= 1"

    def test_block_attention_probs_normalized_in_train_mode(self):
        """Block's returned attention probs should also be pre-dropout and normalized."""
        cfg = _tiny_config(dropout=0.9)
        block = Block(cfg)
        block.train()

        B, T, C = 2, 8, cfg.C
        x = torch.randn(B, T, C)

        torch.manual_seed(42)
        out, attn_probs = block(x, return_attn=True)

        # Check finite
        assert torch.isfinite(attn_probs).all(), "Attention probs should be finite"

        # Check normalized
        sums = attn_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), (
            f"Attention probs should sum to 1, got sums in range [{sums.min()}, {sums.max()}]"
        )

        # Check values in [0, 1]
        assert (attn_probs >= 0).all(), "Attention probs should be non-negative"
        assert (attn_probs <= 1 + 1e-6).all(), "Attention probs should be <= 1"

    def test_attention_probs_are_not_dropped_values(self):
        """With high dropout, if probs were post-dropout they'd have zeros and >1 values."""
        cfg = _tiny_config(dropout=0.9)
        attn = CausalSelfAttention(cfg)
        attn.train()

        B, T, C = 1, 16, cfg.C
        x = torch.randn(B, T, C)

        # Run many times and check the returned probs are always well-behaved
        for seed in range(10):
            torch.manual_seed(seed)
            _, attn_probs = attn(x, return_attn=True)

            # Pre-dropout probs should never have zeros from dropout masking
            # (they can have near-zeros from softmax, but they should still sum to 1)
            sums = attn_probs.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), (
                f"Seed {seed}: attention probs don't sum to 1, suggesting post-dropout values"
            )

