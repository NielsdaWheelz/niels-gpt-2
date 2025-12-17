from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from niels_gpt.checkpoint import validate_model_config_match
import niels_gpt.paths as paths


def _build_checkpoint(
    *,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any] | None,
    step: int,
    best_val_loss: float | None,
) -> dict[str, Any]:
    """Construct a checkpoint dict with the required keys."""
    return {
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "step": step,
        "best_val_loss": best_val_loss,
    }


def save_checkpoint(
    path: str | Path,
    *,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any] | None,
    step: int,
    best_val_loss: float | None,
) -> None:
    """Save a checkpoint to disk with the exact spec-required keys."""
    ckpt = _build_checkpoint(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        model_state=model_state,
        optimizer_state=optimizer_state,
        step=step,
        best_val_loss=best_val_loss,
    )
    torch.save(ckpt, str(path))


def load_checkpoint(path: str | Path, *, device: str) -> dict[str, Any]:
    """Load a checkpoint from disk and normalize optional fields."""
    raw = torch.load(str(path), map_location=device, weights_only=False)
    return {
        "model_cfg": raw["model_cfg"],
        "train_cfg": raw["train_cfg"],
        "model_state": raw["model_state"],
        "optimizer_state": raw.get("optimizer_state", None),
        "step": int(raw["step"]),
        "best_val_loss": raw.get("best_val_loss", None),
    }


def assert_model_config_compatible(loaded_cfg: dict[str, Any], current_cfg: dict[str, Any]) -> None:
    """Raise if the loaded model config is incompatible with the current one."""
    validate_model_config_match(loaded_cfg, current_cfg)


@dataclass(frozen=True)
class PhasePaths:
    latest: Path
    best: Path
    step_dir: Path


def phase_paths(phase: str) -> PhasePaths:
    base = paths.CHECKPOINT_DIR / phase
    base.mkdir(parents=True, exist_ok=True)
    return PhasePaths(
        latest=base / "latest.pt",
        best=base / "best.pt",
        step_dir=base,
    )

