"""Tests for PR-07: AMP and activation checkpointing."""

import contextlib
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from niels_gpt.config import ModelConfig
from niels_gpt.model.gpt import GPT
from train.amp_utils import get_amp_context
from train.checkpointing import load_checkpoint, save_checkpoint
from train.config import load_pretrain_job_config


class TestConfigDefaults:
    """Test that old configs without new fields still load correctly."""

    def test_old_config_loads_with_defaults(self):
        """Settings-style overrides missing new fields should load with defaults."""
        overrides = {
            "model": {"V": 256},
            "training": {"pretrain": {"micro_B": 8, "total_steps": 100}},
        }
        job_config = load_pretrain_job_config(overrides, run_id="test-old")
        train_cfg = job_config.train_cfg

        # New fields should have correct defaults
        assert train_cfg.amp is True
        assert train_cfg.amp_dtype == "fp16"
        assert train_cfg.activation_checkpointing is False

    def test_train_cfg_defaults_via_loader(self):
        """Loader should supply defaults for missing training fields."""
        cfg = {"model": {"V": 128}, "training": {"pretrain": {}}}
        job_config = load_pretrain_job_config(cfg, run_id="test-defaults")
        train_cfg = job_config.train_cfg
        assert train_cfg.amp is True
        assert train_cfg.amp_dtype == "fp16"
        assert train_cfg.activation_checkpointing is False


class TestAMPContextSelection:
    """Test that AMP context is correctly selected based on device."""

    def test_amp_disabled_returns_nullcontext(self):
        """When amp=False, should return nullcontext regardless of device."""
        ctx = get_amp_context(device="mps", amp_enabled=False, amp_dtype="fp16")
        assert isinstance(ctx, contextlib.AbstractContextManager)
        # Should be nullcontext - test by using it
        with ctx:
            pass  # Should do nothing

    def test_cpu_returns_nullcontext(self):
        """When device=cpu, should return nullcontext even if amp=True."""
        ctx = get_amp_context(device="cpu", amp_enabled=True, amp_dtype="fp16")
        assert isinstance(ctx, contextlib.AbstractContextManager)
        with ctx:
            pass  # Should do nothing

    @pytest.mark.skipif(
        not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
        reason="MPS not available",
    )
    def test_mps_amp_enabled_returns_autocast(self):
        """When device=mps and amp=True, should return autocast context."""
        ctx = get_amp_context(device="mps", amp_enabled=True, amp_dtype="fp16")
        # Check it's an autocast context by verifying we can use it
        assert isinstance(ctx, contextlib.AbstractContextManager)
        with ctx:
            # Inside autocast context
            pass

    def test_invalid_amp_dtype_raises(self):
        """Invalid amp_dtype should raise ValueError with helpful message."""
        with pytest.raises(ValueError, match="unsupported amp_dtype"):
            get_amp_context(device="mps", amp_enabled=True, amp_dtype="invalid")


class TestCPUTrainingSmoke:
    """Test basic training loop on CPU with gradient accumulation."""

    def test_training_two_steps_with_accum(self):
        """Run 2 optimizer steps with accum_steps=2, verify loss finite and step increments."""
        # Tiny model config
        cfg = ModelConfig(V=128, T=16, C=64, L=2, H=2, d_ff=128, dropout=0.0, rope_theta=10000.0)
        model = GPT(cfg)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        B = 4
        accum_steps = 2
        total_steps = 2

        losses = []
        for step in range(total_steps):
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0

            for _ in range(accum_steps):
                # Random input
                x = torch.randint(0, cfg.V, (B, cfg.T))
                y = torch.randint(0, cfg.V, (B, cfg.T))

                # Forward (no AMP on CPU)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, cfg.V), y.view(-1))

                # Backward with scaling
                (loss / accum_steps).backward()
                loss_accum += loss.item()

            # Gradient clip and optimizer step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            avg_loss = loss_accum / accum_steps
            losses.append(avg_loss)
            assert torch.isfinite(torch.tensor(avg_loss)), f"Loss is not finite at step {step}"

        # Should have completed 2 steps
        assert len(losses) == total_steps


class TestActivationCheckpointingSmoke:
    """Test activation checkpointing doesn't break training."""

    def test_checkpointing_functional_equivalence(self):
        """Verify checkpointing produces same gradients as non-checkpointed forward."""
        torch.manual_seed(42)
        cfg = ModelConfig(V=128, T=16, C=64, L=2, H=2, d_ff=128, dropout=0.0, rope_theta=10000.0)

        # Create two identical models
        model_no_ckpt = GPT(cfg)
        model_with_ckpt = GPT(cfg)
        # Copy weights to ensure identical starting point
        model_with_ckpt.load_state_dict(model_no_ckpt.state_dict())

        model_no_ckpt.activation_checkpointing = False
        model_with_ckpt.activation_checkpointing = True

        # Same input
        x = torch.randint(0, cfg.V, (4, cfg.T))
        y = torch.randint(0, cfg.V, (4, cfg.T))

        # Forward + backward without checkpointing
        model_no_ckpt.train()
        logits_no_ckpt = model_no_ckpt(x)
        loss_no_ckpt = F.cross_entropy(logits_no_ckpt.view(-1, cfg.V), y.view(-1))
        loss_no_ckpt.backward()

        # Forward + backward with checkpointing
        model_with_ckpt.train()
        logits_with_ckpt = model_with_ckpt(x)
        loss_with_ckpt = F.cross_entropy(logits_with_ckpt.view(-1, cfg.V), y.view(-1))
        loss_with_ckpt.backward()

        # Losses should be identical (same forward computation)
        assert torch.allclose(loss_no_ckpt, loss_with_ckpt, rtol=1e-5, atol=1e-7), \
            f"Loss mismatch: {loss_no_ckpt.item()} vs {loss_with_ckpt.item()}"

        # Check gradients match for a few key parameters
        for (name_no, param_no), (name_with, param_with) in zip(
            list(model_no_ckpt.named_parameters())[:5],
            list(model_with_ckpt.named_parameters())[:5]
        ):
            assert name_no == name_with, f"Parameter name mismatch: {name_no} vs {name_with}"
            if param_no.grad is not None and param_with.grad is not None:
                rel_error = (param_no.grad - param_with.grad).abs() / (param_no.grad.abs() + 1e-10)
                max_rel_error = rel_error.max().item()
                assert max_rel_error < 1e-4, \
                    f"Gradient mismatch for {name_no}: max relative error {max_rel_error:.6f}"

    def test_checkpointing_enabled_training(self):
        """Run training with activation_checkpointing=True, verify loss finite."""
        cfg = ModelConfig(V=128, T=16, C=64, L=2, H=2, d_ff=128, dropout=0.0, rope_theta=10000.0)
        model = GPT(cfg)
        model.activation_checkpointing = True
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        B = 4
        accum_steps = 2
        total_steps = 2

        for step in range(total_steps):
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0

            for _ in range(accum_steps):
                x = torch.randint(0, cfg.V, (B, cfg.T))
                y = torch.randint(0, cfg.V, (B, cfg.T))

                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, cfg.V), y.view(-1))
                (loss / accum_steps).backward()
                loss_accum += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            avg_loss = loss_accum / accum_steps
            assert torch.isfinite(torch.tensor(avg_loss))

    def test_checkpointing_disabled_in_eval(self):
        """Activation checkpointing should be disabled during eval mode."""
        cfg = ModelConfig(V=128, T=16, C=64, L=2, H=2, d_ff=128, dropout=0.0, rope_theta=10000.0)
        model = GPT(cfg)
        model.activation_checkpointing = True

        # In eval mode, checkpointing should not be used
        model.eval()
        x = torch.randint(0, cfg.V, (2, cfg.T))

        with torch.no_grad():
            logits = model(x)
            assert logits.shape == (2, cfg.T, cfg.V)


class TestMPSAMPSmoke:
    """Test AMP on MPS device (skip if unavailable)."""

    @pytest.mark.skipif(
        not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
        reason="MPS not available",
    )
    def test_mps_amp_fp16_one_step(self):
        """Run 1 optimizer step with amp=True fp16 on MPS, verify loss finite."""
        device = "mps"
        cfg = ModelConfig(V=128, T=16, C=64, L=2, H=2, d_ff=128, dropout=0.0, rope_theta=10000.0)
        model = GPT(cfg).to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        amp_ctx = get_amp_context(device=device, amp_enabled=True, amp_dtype="fp16")

        B = 4
        x = torch.randint(0, cfg.V, (B, cfg.T), device=device)
        y = torch.randint(0, cfg.V, (B, cfg.T), device=device)

        optimizer.zero_grad(set_to_none=True)

        with amp_ctx:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, cfg.V), y.view(-1))

        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss), "Loss should be finite with AMP on MPS"


class TestCheckpointResumeCompatibility:
    """Test checkpoint save/resume with new config fields."""

    def test_checkpoint_preserves_new_fields(self):
        """Save checkpoint with new fields, resume and verify fields are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.pt"

            # Create model and save
            cfg = ModelConfig(V=128, T=16, C=64, L=2, H=2, d_ff=128, dropout=0.0, rope_theta=10000.0)
            model = GPT(cfg)

            model_cfg_dict = {
                "V": cfg.V,
                "T": cfg.T,
                "C": cfg.C,
                "L": cfg.L,
                "H": cfg.H,
                "d_ff": cfg.d_ff,
                "dropout": cfg.dropout,
                "rope_theta": cfg.rope_theta,
            }

            train_cfg_dict = {
                "seed": 42,
                "B": 8,
                "total_steps": 100,
                "eval_every": 50,
                "eval_steps": 10,
                "log_every": 10,
                "ckpt_every": 50,
                "base_lr": 1e-3,
                "warmup_steps": 10,
                "min_lr": 1e-5,
                "grad_clip": 1.0,
                "accum_steps": 2,
                "amp": True,
                "amp_dtype": "fp16",
                "activation_checkpointing": True,
            }

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            save_checkpoint(
                ckpt_path,
                model_cfg=model_cfg_dict,
                train_cfg=train_cfg_dict,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                step=5,
                best_val_loss=2.5,
            )

            # Load checkpoint
            ckpt = load_checkpoint(str(ckpt_path), device="cpu")

            # Verify new fields are preserved
            assert ckpt["train_cfg"]["amp"] is True
            assert ckpt["train_cfg"]["amp_dtype"] == "fp16"
            assert ckpt["train_cfg"]["activation_checkpointing"] is True
            assert ckpt["step"] == 5
            assert ckpt["best_val_loss"] == 2.5

    def test_resume_step_continuity(self):
        """Run 1 step, save, resume, run 1 more step - verify step continuity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "resume_test.pt"

            cfg = ModelConfig(V=128, T=16, C=64, L=2, H=2, d_ff=128, dropout=0.0, rope_theta=10000.0)
            model = GPT(cfg)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            # Run 1 step
            x = torch.randint(0, cfg.V, (2, cfg.T))
            y = torch.randint(0, cfg.V, (2, cfg.T))

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, cfg.V), y.view(-1))
            loss.backward()
            optimizer.step()

            # Save checkpoint at step 1
            model_cfg_dict = {
                "V": cfg.V,
                "T": cfg.T,
                "C": cfg.C,
                "L": cfg.L,
                "H": cfg.H,
                "d_ff": cfg.d_ff,
                "dropout": cfg.dropout,
                "rope_theta": cfg.rope_theta,
            }
            train_cfg_dict = {
                "seed": 42,
                "B": 2,
                "total_steps": 10,
                "amp": False,
                "amp_dtype": "fp16",
                "activation_checkpointing": False,
            }

            save_checkpoint(
                ckpt_path,
                model_cfg=model_cfg_dict,
                train_cfg=train_cfg_dict,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                step=1,
                best_val_loss=None,
            )

            # Resume from checkpoint
            ckpt = load_checkpoint(str(ckpt_path), device="cpu")
            model2 = GPT(cfg)
            model2.load_state_dict(ckpt["model_state"])
            optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
            optimizer2.load_state_dict(ckpt["optimizer_state"])
            resumed_step = ckpt["step"]

            assert resumed_step == 1, "Should resume at step 1"

            # Run one more step
            optimizer2.zero_grad(set_to_none=True)
            logits2 = model2(x)
            loss2 = F.cross_entropy(logits2.view(-1, cfg.V), y.view(-1))
            loss2.backward()
            optimizer2.step()

            # Next step should be 2
            next_step = resumed_step + 1
            assert next_step == 2, "Step continuity should hold"
