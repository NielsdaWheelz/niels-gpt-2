pr-05a spec: rope module + tests

goal

add niels_gpt/model/rope.py implementing rotary positional embeddings (rope) cache + application, with hard, loud tests that prevent silent shape/dtype/device bugs.

scope

allowed new/modified files (hard allowlist)
	•	niels_gpt/model/rope.py (new)
	•	tests/test_rope.py (new)
	•	(optional) niels_gpt/model/__init__.py (only if needed to make imports work; no other edits)

explicitly out of scope
	•	attention, blocks, gpt wrapper
	•	training loop, batching, datasets
	•	any refactors in existing modules

public API (must match exactly)

niels_gpt/model/rope.py must export exactly these two functions:

def rope_cache(
    T: int,
    D: int,
    *,
    theta: float = 10000.0,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    returns (sin, cos)
    sin.shape == cos.shape == (1, 1, T, D//2)
    """

def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    q,k: (B, H, Tq, D) with D even
    sin,cos: (1, 1, Tcache, D//2) where Tcache >= Tq
    returns (q_rot, k_rot) with same shape/dtype/device as q,k
    """

hard invariants
	•	D % 2 == 0 is required; otherwise raise AssertionError (or ValueError, but be consistent across both functions).
	•	q.shape == k.shape required.
	•	q.ndim == 4 required.
	•	sin.shape == cos.shape required.
	•	sin.shape[:2] == (1, 1) required.
	•	sin.shape[-1] == D//2 required.
	•	sin.device == q.device and sin.dtype == q.dtype required (same for cos).
	•	sin.shape[-2] >= q.shape[-2] required (cache long enough).
	•	no silent casts or .to(...) inside apply_rope. fail loudly instead.

rope math (must match)

cache definition:
	•	let i = 0..(D//2 - 1)
	•	inv_freq[i] = theta ** (-2*i / D)
	•	positions pos = 0..T-1
	•	angles ang[pos, i] = pos * inv_freq[i]
	•	sin = sin(ang), cos = cos(ang)
	•	return tensors shaped (1, 1, T, D//2) on requested device/dtype
	•	defaults: cpu + float32 when device/dtype are None

apply definition (pairwise rotation, even/odd pairing):
	•	treat last dim D as D//2 pairs (even, odd) via reshape to (..., D//2, 2)
	•	for each pair (a, b) at position p:
	•	a' = a*cos[p] - b*sin[p]
	•	b' = a*sin[p] + b*cos[p]
	•	apply to all pairs (full dim)
	•	apply to q and k only (caller will apply to q/k; pr-05a only provides utility)

cache usage:
	•	if Tq < Tcache, slice sin/cos to [..., :Tq, :] inside apply_rope.

tests (must exist)

create tests/test_rope.py with these tests:
	1.	test_rope_cache_shapes_and_defaults

	•	call rope_cache(T=256, D=64)
	•	assert shapes (1,1,256,32)
	•	assert dtype float32
	•	assert device cpu

	2.	test_apply_rope_preserves_shape_dtype_device_cpu

	•	make random q,k on cpu float32: shape (B=2, H=4, Tq=17, D=64)
	•	build cache with T=256, D=64, then apply
	•	assert output shapes exactly equal input
	•	assert dtype/device match
	•	assert outputs are finite (torch.isfinite)

	3.	test_apply_rope_pairwise_norm_preserved_cpu

	•	using same q/k, compare per-pair norms before/after:
	•	reshape last dim to (..., D//2, 2)
	•	norm = sqrt(a^2 + b^2)
	•	torch.testing.assert_close(norm_before, norm_after, rtol=1e-5, atol=1e-5)
	•	do for q and k

	4.	test_apply_rope_cache_slicing

	•	build cache with T=256
	•	apply to q/k with Tq=1 and Tq=17 and Tq=256
	•	should not crash; shapes correct

	5.	test_rope_requires_even_D

	•	rope_cache(T=8, D=63) must raise
	•	also optionally test apply_rope rejects odd-D inputs

	6.	test_apply_rope_mps_optional (optional test, must be skipped if unavailable)

	•	if torch.backends.mps.is_available() and torch.backends.mps.is_built():
	•	run the shape/dtype/device preservation test on mps
	•	do not require bit-identical values vs cpu

acceptance criteria
	•	pytest -q passes.
	•	rope functions meet all invariants and tests above.
	•	no changes outside allowlist.
