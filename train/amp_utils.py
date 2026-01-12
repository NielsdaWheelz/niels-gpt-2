"""AMP (Automatic Mixed Precision) utilities for training."""

from __future__ import annotations

import contextlib
from typing import Any, Literal

import torch


def _cuda_supports_bf16() -> bool:
    """Check if CUDA device supports bf16 (Ampere or newer)."""
    if not torch.cuda.is_available():
        return False
    # bf16 requires compute capability >= 8.0 (Ampere)
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def resolve_amp_settings(
    device: str, amp: bool | Literal["auto"], amp_dtype: str
) -> tuple[bool, str]:
    """
    Resolve AMP settings based on device.

    Args:
        device: Device string ("cpu", "mps", "cuda")
        amp: True, False, or "auto"
        amp_dtype: "fp16", "bf16", or "fp32"

    Returns:
        Tuple of (amp_enabled, amp_dtype) with device-appropriate values.

    When amp="auto":
        - CUDA: enabled with fp16 (bf16 if Ampere+ GPU)
        - MPS: disabled (known stability issues)
        - CPU: disabled (no benefit)
    """
    if amp == "auto":
        if device == "cuda":
            # Enable AMP on CUDA - use bf16 if supported, else fp16
            resolved_dtype = "bf16" if _cuda_supports_bf16() else "fp16"
            return True, resolved_dtype
        else:
            # Disable on MPS (unstable) and CPU (no benefit)
            return False, amp_dtype
    else:
        # Explicit setting - respect user's choice but warn for MPS
        if amp and device == "mps":
            import warnings

            warnings.warn(
                "AMP on MPS can cause training instability (loss divergence). "
                "Consider using amp='auto' or amp=false for MPS.",
                stacklevel=2,
            )
        return amp, amp_dtype


def get_amp_context(
    *, device: str, amp_enabled: bool | Literal["auto"], amp_dtype: str
) -> Any:
    """
    Return autocast context for mixed precision training.

    Args:
        device: Device string ("cpu", "mps", "cuda")
        amp_enabled: True, False, or "auto" (device-aware)
        amp_dtype: "fp16", "bf16", or "fp32"

    Returns:
        torch.autocast context if enabled, else nullcontext

    Raises:
        ValueError: If amp_dtype is invalid
        RuntimeError: If autocast context creation fails
    """
    # Resolve "auto" to actual settings
    amp_on, dtype_str = resolve_amp_settings(device, amp_enabled, amp_dtype)

    if not amp_on or device not in {"mps", "cuda"}:
        return contextlib.nullcontext()

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    if dtype_str not in dtype_map:
        raise ValueError(f"unsupported amp_dtype={dtype_str}; expected one of {list(dtype_map.keys())}")

    dtype = dtype_map[dtype_str]

    # Try to create autocast context - this validates bf16 support early
    try:
        ctx = torch.autocast(device_type=device, dtype=dtype)
        # Test that the context actually works by entering/exiting once
        with ctx:
            pass
        return ctx
    except Exception as e:
        raise RuntimeError(
            f"failed to create or use autocast context for device={device}, dtype={dtype_str}: {e}\n"
            f"recommendation: set amp=false or try amp_dtype='fp16' (bf16 may not be supported on your device)"
        ) from e
