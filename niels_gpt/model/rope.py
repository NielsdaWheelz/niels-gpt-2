"""
Rotary Positional Embedding (RoPE) utilities.

Provides precomputed sin/cos caches and application to query/key tensors.
"""

import torch


def rope_cache(
    T: int,
    D: int,
    *,
    theta: float = 10000.0,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute sin/cos tables for RoPE.

    Args:
        T: Maximum sequence length (number of positions).
        D: Head dimension (must be even).
        theta: Base for frequency computation (default 10000.0).
        device: Target device (default: cpu).
        dtype: Target dtype (default: float32).

    Returns:
        (sin, cos): Each shaped (1, 1, T, D//2).

    Raises:
        AssertionError: If D is odd.
    """
    assert D % 2 == 0, f"D must be even, got {D}"

    # Set defaults
    if device is None:
        device = "cpu"
    if dtype is None:
        dtype = torch.float32

    # Determine compute dtype (MPS doesn't support float64)
    device_obj = torch.device(device) if isinstance(device, str) else device
    compute_dtype = torch.float32 if device_obj.type == "mps" else torch.float64

    # Compute inverse frequencies: theta ** (-2*i / D) for i = 0..(D//2 - 1)
    i = torch.arange(0, D // 2, dtype=compute_dtype, device=device)
    inv_freq = theta ** (-2 * i / D)

    # Compute positions
    positions = torch.arange(T, dtype=compute_dtype, device=device)

    # Compute angles: outer product of positions and inv_freq
    # angles[pos, i] = pos * inv_freq[i]
    angles = torch.outer(positions, inv_freq)  # (T, D//2)

    # Compute sin and cos
    sin = torch.sin(angles)
    cos = torch.cos(angles)

    # Convert to target dtype
    sin = sin.to(dtype)
    cos = cos.to(dtype)

    # Reshape to (1, 1, T, D//2) for broadcasting
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)

    return sin, cos


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor, shape (B, H, Tq, D).
        k: Key tensor, shape (B, H, Tq, D).
        sin: Precomputed sin cache, shape (1, 1, Tcache, D//2).
        cos: Precomputed cos cache, shape (1, 1, Tcache, D//2).

    Returns:
        (q_rot, k_rot): Rotated tensors with same shape/dtype/device as inputs.

    Raises:
        AssertionError: If shapes, dtypes, or devices don't match requirements.
    """
    # Validate shapes
    assert q.shape == k.shape, f"q and k must have same shape, got {q.shape} vs {k.shape}"
    assert q.ndim == 4, f"q must be 4D, got {q.ndim}D"

    B, H, Tq, D = q.shape
    assert D % 2 == 0, f"D must be even, got {D}"

    # Validate sin/cos shapes
    assert sin.shape == cos.shape, f"sin and cos must have same shape"
    assert sin.shape[:2] == (1, 1), f"sin/cos must start with (1, 1), got {sin.shape[:2]}"
    assert sin.shape[-1] == D // 2, f"sin/cos last dim must be D//2={D//2}, got {sin.shape[-1]}"

    Tcache = sin.shape[-2]
    assert Tcache >= Tq, f"cache length {Tcache} must be >= query length {Tq}"

    # Validate dtype and device
    assert sin.dtype == q.dtype, f"sin dtype {sin.dtype} must match q dtype {q.dtype}"
    assert sin.device == q.device, f"sin device {sin.device} must match q device {q.device}"
    assert cos.dtype == q.dtype, f"cos dtype {cos.dtype} must match q dtype {q.dtype}"
    assert cos.device == q.device, f"cos device {cos.device} must match q device {q.device}"

    # Slice sin/cos to match query length
    sin = sin[..., :Tq, :]  # (1, 1, Tq, D//2)
    cos = cos[..., :Tq, :]  # (1, 1, Tq, D//2)

    # Reshape q and k to separate even/odd pairs
    # (B, H, Tq, D) -> (B, H, Tq, D//2, 2)
    q_pairs = q.reshape(B, H, Tq, D // 2, 2)
    k_pairs = k.reshape(B, H, Tq, D // 2, 2)

    # Extract even and odd components
    q_even = q_pairs[..., 0]  # (B, H, Tq, D//2)
    q_odd = q_pairs[..., 1]   # (B, H, Tq, D//2)
    k_even = k_pairs[..., 0]  # (B, H, Tq, D//2)
    k_odd = k_pairs[..., 1]   # (B, H, Tq, D//2)

    # Apply rotation
    # a' = a*cos - b*sin
    # b' = a*sin + b*cos
    q_even_rot = q_even * cos - q_odd * sin
    q_odd_rot = q_even * sin + q_odd * cos
    k_even_rot = k_even * cos - k_odd * sin
    k_odd_rot = k_even * sin + k_odd * cos

    # Stack back to pairs and reshape to original shape
    q_rot = torch.stack([q_even_rot, q_odd_rot], dim=-1).reshape(B, H, Tq, D)
    k_rot = torch.stack([k_even_rot, k_odd_rot], dim=-1).reshape(B, H, Tq, D)

    return q_rot, k_rot
