"""Tests for evaluation utilities."""

import torch

from niels_gpt.config import ModelConfig
from niels_gpt.eval import eval_loss_on_stream
from niels_gpt.model.gpt import GPT


def test_eval_restores_training_mode():
    """Verify that eval_loss_on_stream restores model to training mode after evaluation."""
    # Create small model for testing
    model_cfg = ModelConfig(
        V=256,
        T=64,
        C=128,
        L=2,
        H=4,
        d_ff=512,
        dropout=0.1,
        rope_theta=10000.0,
    )

    device = "cpu"
    model = GPT(model_cfg).to(device)

    # Set model to training mode
    model.train()
    assert model.training, "Model should start in training mode"

    # Create a small dummy byte stream (needs at least T+1 bytes)
    dummy_stream = b"x" * 100

    # Call eval_loss_on_stream
    val_loss = eval_loss_on_stream(
        model,
        stream=dummy_stream,
        B=2,
        T=64,
        device=device,
        eval_steps=2,
        seed=42,
    )

    # Verify model is still in training mode after eval
    assert model.training, "Model should be restored to training mode after eval"
    assert isinstance(val_loss, float), "Should return float loss"


def test_eval_preserves_eval_mode():
    """Verify that eval_loss_on_stream preserves eval mode if model was already in eval mode."""
    # Create small model for testing
    model_cfg = ModelConfig(
        V=256,
        T=64,
        C=128,
        L=2,
        H=4,
        d_ff=512,
        dropout=0.1,
        rope_theta=10000.0,
    )

    device = "cpu"
    model = GPT(model_cfg).to(device)

    # Set model to eval mode
    model.eval()
    assert not model.training, "Model should start in eval mode"

    # Create a small dummy byte stream
    dummy_stream = b"x" * 100

    # Call eval_loss_on_stream
    val_loss = eval_loss_on_stream(
        model,
        stream=dummy_stream,
        B=2,
        T=64,
        device=device,
        eval_steps=2,
        seed=42,
    )

    # Verify model is still in eval mode after eval
    assert not model.training, "Model should remain in eval mode after eval"
    assert isinstance(val_loss, float), "Should return float loss"


def test_eval_restores_mode_on_exception():
    """Verify that eval_loss_on_stream restores mode even if an exception occurs."""
    # Create small model for testing
    model_cfg = ModelConfig(
        V=256,
        T=64,
        C=128,
        L=2,
        H=4,
        d_ff=512,
        dropout=0.1,
        rope_theta=10000.0,
    )

    device = "cpu"
    model = GPT(model_cfg).to(device)

    # Set model to training mode
    model.train()
    assert model.training, "Model should start in training mode"

    # Create a stream that's too short (will cause ValueError in get_batch)
    too_short_stream = b"x"  # Only 1 byte, need at least T+1=65

    # Call eval_loss_on_stream and expect it to raise
    try:
        eval_loss_on_stream(
            model,
            stream=too_short_stream,
            B=2,
            T=64,
            device=device,
            eval_steps=1,
            seed=42,
        )
        assert False, "Should have raised ValueError for short stream"
    except ValueError:
        # Expected exception from get_batch
        pass

    # Verify model mode was restored despite exception
    assert model.training, "Model should be restored to training mode even after exception"
