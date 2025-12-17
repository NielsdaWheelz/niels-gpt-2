"""AMP (Automatic Mixed Precision) utilities for training."""

from __future__ import annotations

import contextlib
from typing import Any

import torch


def get_amp_context(*, device: str, amp_enabled: bool, amp_dtype: str) -> Any:
    """
    Return autocast context for mixed precision training on MPS.

    Args:
        device: Device string ("cpu", "mps")
        amp_enabled: Whether AMP is enabled in config
        amp_dtype: "fp16" or "bf16"

    Returns:
        torch.autocast context if enabled on MPS, else nullcontext

    Raises:
        ValueError: If amp_dtype is invalid
        RuntimeError: If autocast context creation fails (includes bf16 support check)
    """
    if not amp_enabled or device != "mps":
        return contextlib.nullcontext()

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    if amp_dtype not in dtype_map:
        raise ValueError(f"unsupported amp_dtype={amp_dtype}; expected one of {list(dtype_map.keys())}")

    dtype = dtype_map[amp_dtype]

    # Try to create autocast context - this validates bf16 support early
    try:
        ctx = torch.autocast(device_type="mps", dtype=dtype)
        # Test that the context actually works by entering/exiting once
        with ctx:
            pass
        return ctx
    except Exception as e:
        raise RuntimeError(
            f"failed to create or use autocast context for device={device}, dtype={amp_dtype}: {e}\n"
            f"recommendation: set amp=false or try amp_dtype='fp16' (bf16 may not be supported on your MPS device)"
        ) from e
