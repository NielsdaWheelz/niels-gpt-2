"""Tests for checkpoint save/load roundtrip."""

import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from niels_gpt.checkpoint import load_checkpoint, save_checkpoint
from niels_gpt.config import ModelConfig, to_dict
from niels_gpt.model.gpt import GPT


def test_checkpoint_roundtrip():
    """Test that model and optimizer state can be saved and loaded exactly."""
    # Use fixed seed for reproducibility
    torch.manual_seed(42)

    # Create small model config for fast testing
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

    train_cfg = {
        "seed": 42,
        "B": 8,
        "total_steps": 1000,
        "base_lr": 3e-4,
    }

    # Create model and optimizer
    device = "cpu"
    model = GPT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    # Do one training step on synthetic data
    B, T = 8, 64
    x = torch.randint(0, 256, (B, T), dtype=torch.int64, device=device)
    y = torch.randint(0, 256, (B, T), dtype=torch.int64, device=device)

    logits = model(x)
    B_cur, T_cur, V = logits.shape
    logits_flat = logits.view(B_cur * T_cur, V)
    targets_flat = y.view(B_cur * T_cur)
    loss = F.cross_entropy(logits_flat, targets_flat)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save checkpoint to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_checkpoint.pt"

        save_checkpoint(
            str(ckpt_path),
            model_cfg=to_dict(model_cfg),
            train_cfg=train_cfg,
            model=model,
            optimizer=optimizer,
            step=1,
            best_val_loss=2.5,
        )

        # Load checkpoint into fresh model and optimizer
        torch.manual_seed(999)  # Different seed to ensure we're loading, not re-initializing
        model_loaded = GPT(model_cfg).to(device)
        optimizer_loaded = torch.optim.AdamW(
            model_loaded.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1
        )

        ckpt = load_checkpoint(str(ckpt_path), device=device)

        # Verify metadata
        assert ckpt["step"] == 1
        assert ckpt["best_val_loss"] == 2.5
        assert ckpt["model_cfg"] == to_dict(model_cfg)
        assert ckpt["train_cfg"] == train_cfg

        # Load state dicts
        model_loaded.load_state_dict(ckpt["model_state"])
        optimizer_loaded.load_state_dict(ckpt["optimizer_state"])

        # Verify parameters are exactly equal
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), model_loaded.named_parameters()
        ):
            assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
            torch.testing.assert_close(
                param1.cpu(),
                param2.cpu(),
                rtol=0,
                atol=0,
                msg=f"Parameter {name1} not exactly equal after roundtrip",
            )

        # Verify optimizer state dict keys match
        orig_state_keys = set(optimizer.state_dict()["state"].keys())
        loaded_state_keys = set(optimizer_loaded.state_dict()["state"].keys())
        assert orig_state_keys == loaded_state_keys, (
            f"Optimizer state keys mismatch: {orig_state_keys} vs {loaded_state_keys}"
        )


def test_checkpoint_without_optimizer():
    """Test checkpoint save/load when optimizer is None."""
    torch.manual_seed(42)

    model_cfg = ModelConfig(V=256, T=64, C=128, L=2, H=4, d_ff=512, dropout=0.1, rope_theta=10000.0)
    train_cfg = {"seed": 42, "B": 8, "total_steps": 1000}

    device = "cpu"
    model = GPT(model_cfg).to(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_checkpoint_no_opt.pt"

        # Save without optimizer
        save_checkpoint(
            str(ckpt_path),
            model_cfg=to_dict(model_cfg),
            train_cfg=train_cfg,
            model=model,
            optimizer=None,
            step=100,
            best_val_loss=None,
        )

        # Load checkpoint
        ckpt = load_checkpoint(str(ckpt_path), device=device)

        # Verify optimizer_state is None
        assert ckpt["optimizer_state"] is None
        assert ckpt["best_val_loss"] is None
        assert ckpt["step"] == 100


def test_checkpoint_determinism():
    """Test that same model state produces identical checkpoint."""
    torch.manual_seed(42)

    model_cfg = ModelConfig(V=256, T=64, C=128, L=2, H=4, d_ff=512, dropout=0.1, rope_theta=10000.0)
    train_cfg = {"seed": 42}

    device = "cpu"
    model = GPT(model_cfg).to(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path1 = Path(tmpdir) / "ckpt1.pt"
        ckpt_path2 = Path(tmpdir) / "ckpt2.pt"

        # Save same model twice
        save_checkpoint(
            str(ckpt_path1),
            model_cfg=to_dict(model_cfg),
            train_cfg=train_cfg,
            model=model,
            optimizer=None,
            step=50,
            best_val_loss=1.5,
        )

        save_checkpoint(
            str(ckpt_path2),
            model_cfg=to_dict(model_cfg),
            train_cfg=train_cfg,
            model=model,
            optimizer=None,
            step=50,
            best_val_loss=1.5,
        )

        # Load both checkpoints
        ckpt1 = load_checkpoint(str(ckpt_path1), device=device)
        ckpt2 = load_checkpoint(str(ckpt_path2), device=device)

        # Compare model states
        for key in ckpt1["model_state"].keys():
            torch.testing.assert_close(
                ckpt1["model_state"][key],
                ckpt2["model_state"][key],
                rtol=0,
                atol=0,
                msg=f"Model state key {key} differs between saves",
            )
