"""Text generation utilities with temperature and top-k sampling."""

from typing import TYPE_CHECKING

import torch

from niels_gpt.config import ModelConfig
from niels_gpt.settings import GenerationSettings
from niels_gpt.tokenizer import get_default_tokenizer

if TYPE_CHECKING:
    from niels_gpt.model.gpt import GPT


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Filter logits to keep only top-k values, setting others to -inf.

    Args:
        logits: (..., V) float tensor
        k: Number of top values to keep, must satisfy 1 <= k <= V

    Returns:
        Same shape as logits, where all but top-k entries are set to -inf.
        Exactly k values remain finite (handles ties correctly).
    """
    # Get indices of top-k values
    # This guarantees exactly k survivors even if there are ties
    _, topk_indices = torch.topk(logits, k, dim=-1)

    # Create result filled with -inf
    filtered = torch.full_like(logits, float('-inf'))

    # Scatter original logits back only at top-k indices
    # For 1D case (most common): filtered[topk_indices] = logits[topk_indices]
    # For general ND case, we need to use scatter
    filtered.scatter_(-1, topk_indices, logits.gather(-1, topk_indices))

    return filtered


def sample_next_token(
    logits_1d: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None = None,
    generator: torch.Generator | None,
) -> int:
    """
    Sample next token from logits.

    Args:
        logits_1d: (V,) float tensor on any device
        temperature: Sampling temperature. If 0, returns argmax (deterministic).
                     Must be >= 0.
        top_k: If not None, filter to top-k before sampling
        generator: Random generator for reproducibility (must be CPU generator)

    Returns:
        Python int in [0..V-1]
    """
    # Validate temperature
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")

    if temperature == 0:
        # Deterministic: return argmax
        return int(logits_1d.argmax().item())

    # Scale by temperature
    scaled = logits_1d / temperature

    # Apply top-k filter if requested
    if top_k is not None:
        scaled = top_k_filter(scaled, top_k)

    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Keep tokens up to and including first index that pushes cumulative over top_p
        cutoff_idx = torch.searchsorted(cumulative, top_p)
        cutoff_idx = int(min(cutoff_idx.item(), sorted_logits.numel()))
        if cutoff_idx < sorted_logits.numel():
            mask_indices = sorted_indices[cutoff_idx:]
            scaled = scaled.clone()
            scaled[mask_indices] = float("-inf")

    # Compute probabilities
    probs = torch.softmax(scaled, dim=-1)

    # Move to CPU for sampling (generator must be on CPU for cross-device compatibility)
    # This is especially important for MPS which has inconsistent generator support
    probs_cpu = probs.cpu()

    # Sample one index
    sampled_idx = torch.multinomial(probs_cpu, num_samples=1, generator=generator)

    return int(sampled_idx.item())


def generate_ids(
    model: torch.nn.Module,
    prompt_ids: torch.LongTensor,
    *,
    max_new_tokens: int,
    T: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float | None,
    eot_id: int,
    banned_token_ids: list[int] | None = None,
    device: str,
    generator: torch.Generator | None = None,
) -> torch.LongTensor:
    """
    Generate token IDs autoregressively.

    Args:
        model: GPT model
        prompt_ids: (n,) int64 on cpu (from tokenizer.encode)
        max_new_tokens: Maximum number of tokens to generate
        T: Context window size (cfg.T)
        temperature: Sampling temperature (0 = greedy)
        top_k: Top-k filtering (None = no filtering)
        eot_id: Token id that terminates generation when produced
        device: Device to run model on ("mps" or "cpu")
        generator: Random generator for reproducibility

    Returns:
        (n + m,) int64 on cpu, where m <= max_new_tokens
        Stops early if eot_id is generated (truncates before it).
    """
    model.eval()

    # Use list for O(1) append instead of O(n) tensor concatenation
    ids_list = prompt_ids.tolist()
    start = len(ids_list)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Convert current ids to tensor for model input
            ids_tensor = torch.tensor(ids_list, dtype=torch.long)

            # Crop context to last T tokens
            ctx = ids_tensor[-T:] if len(ids_tensor) > T else ids_tensor

            # Move to device and add batch dimension
            ctx_device = ctx.to(device)[None, :]  # (1, t)

            # Forward pass
            logits = model(ctx_device)  # (1, t, V)

            # Take last position logits
            logits_last = logits[0, -1].clone()  # (V,)

            if banned_token_ids:
                logits_last[banned_token_ids] = float("-inf")

            if repetition_penalty and repetition_penalty != 1.0:
                unique_tokens = set(ids_list)
                for tok_id in unique_tokens:
                    if 0 <= tok_id < logits_last.shape[0]:
                        logits_last[tok_id] = logits_last[tok_id] / repetition_penalty

            # Sample next token
            next_token = sample_next_token(
                logits_last,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )

            # Append to list (O(1))
            ids_list.append(next_token)

            # Stop on eot_id
            if next_token == eot_id:
                return torch.tensor(ids_list[:-1], dtype=torch.long)

    return torch.tensor(ids_list, dtype=torch.long)


def generate_text(
    model: torch.nn.Module,
    prompt_text: str,
    *,
    cfg: ModelConfig,
    generation: GenerationSettings,
    stop_token_id: int | None = None,
    banned_token_ids: list[int] | None = None,
    device: str,
    generator: torch.Generator | None = None,
) -> str:
    """
    Generate text from a prompt string.

    Args:
        model: GPT model
        prompt_text: Input prompt string
        cfg: Model configuration (for T)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_k: Top-k filtering (None = no filtering)
        device: Device to run model on
        generator: Random generator for reproducibility

    Returns:
        Generated text (prompt + completion)
    """
    tok = get_default_tokenizer()
    prompt_ids = tok.encode_torch(prompt_text)
    stop_id = stop_token_id if stop_token_id is not None else tok.special_token_ids()["eot"]

    # Generate ids
    output_ids = generate_ids(
        model,
        prompt_ids,
        max_new_tokens=generation.max_new_tokens,
        T=cfg.T,
        temperature=generation.temperature,
        top_k=generation.top_k,
        top_p=generation.top_p,
        repetition_penalty=generation.repetition_penalty,
        eot_id=stop_id,
        banned_token_ids=banned_token_ids or generation.banned_token_ids or [],
        device=device,
        generator=generator,
    )

    return tok.decode(output_ids)


@torch.no_grad()
def generate_ids_greedy_full(
    model: "GPT",
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    eot_token_id: int,
) -> list[int]:
    """
    Generate tokens greedily using full forward passes (baseline for testing).

    This function recomputes the full forward pass for each new token. It enforces
    the same hard cap as the cached version for equivalence testing.

    Args:
        model: GPT model.
        prompt_ids: List of prompt token IDs.
        max_new_tokens: Maximum number of tokens to generate.
        eot_token_id: Token ID that ends generation.

    Returns:
        List of token IDs (prompt + generated, including eot if generated).

    Raises:
        ValueError: If prompt + max_new_tokens exceeds T_max.
    """
    model.eval()

    T_max = model.cfg.T
    device = next(model.parameters()).device

    # Hard cap enforcement (same as cached version)
    if len(prompt_ids) + max_new_tokens > T_max:
        raise ValueError(
            f"prompt length {len(prompt_ids)} + max_new_tokens {max_new_tokens} "
            f"= {len(prompt_ids) + max_new_tokens} exceeds T_max {T_max}"
        )

    ids_list = list(prompt_ids)

    for _ in range(max_new_tokens):
        # Check hard cap before generating
        if len(ids_list) >= T_max:
            raise ValueError(f"exceeded T_max={T_max} during generation")

        # Convert to tensor (NO CROPPING - use full sequence for equivalence testing)
        ids_tensor = torch.tensor(ids_list, dtype=torch.long, device=device)

        # Add batch dimension
        ids_batch = ids_tensor.unsqueeze(0)  # (1, t)

        # Forward pass on full sequence
        logits = model(ids_batch)  # (1, t, V)

        # Take last position, greedy decode
        next_token = int(logits[0, -1].argmax().item())

        # Append
        ids_list.append(next_token)

        # Stop on eot (include eot in output)
        if next_token == eot_token_id:
            return ids_list

    return ids_list


@torch.no_grad()
def generate_ids_greedy_cached(
    model: "GPT",
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    eot_token_id: int,
) -> list[int]:
    """
    Generate tokens greedily using KV-cache (efficient version).

    Args:
        model: GPT model.
        prompt_ids: List of prompt token IDs.
        max_new_tokens: Maximum number of tokens to generate.
        eot_token_id: Token ID that ends generation.

    Returns:
        List of token IDs (prompt + generated, including eot if generated).

    Raises:
        ValueError: If prompt + max_new_tokens exceeds T_max.
    """
    from niels_gpt.infer.kv_cache import allocate_kv_cache, decode_step, prefill

    model.eval()

    T_max = model.cfg.T
    device = next(model.parameters()).device
    dtype = torch.float32  # Use fp32 for inference (Option A from spec)

    # Hard cap enforcement at function entry
    if len(prompt_ids) + max_new_tokens > T_max:
        raise ValueError(
            f"prompt length {len(prompt_ids)} + max_new_tokens {max_new_tokens} "
            f"= {len(prompt_ids) + max_new_tokens} exceeds T_max {T_max}"
        )

    # Allocate cache
    L = len(model.blocks)
    H = model.cfg.H
    D = model.cfg.C // model.cfg.H
    cache = allocate_kv_cache(L=L, B=1, H=H, T_max=T_max, D=D, device=device, dtype=dtype)

    # Prefill with prompt
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, t0)
    _, cache, _ = prefill(model, prompt_tensor, cache)

    # Start with prompt
    ids_list = list(prompt_ids)

    # Decode loop
    for _ in range(max_new_tokens):
        # Check hard cap before decode_step
        if cache.t >= T_max:
            raise ValueError(f"exceeded T_max={T_max} during generation")

        # Decode one step
        last_token = torch.tensor([[ids_list[-1]]], dtype=torch.long, device=device)  # (1, 1)
        logits, cache, _ = decode_step(model, last_token, cache)

        # Greedy decode
        next_token = int(logits[0, 0].argmax().item())

        # Append
        ids_list.append(next_token)

        # Stop on eot (include eot in output)
        if next_token == eot_token_id:
            return ids_list

    return ids_list


@torch.no_grad()
def generate_ids_cached(
    model: "GPT",
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    eot_token_id: int,
    temperature: float = 0.9,
    top_k: int | None = 50,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    banned_token_ids: list[int] | None = None,
    generator: torch.Generator | None = None,
    trace_layer: int | None = None,
) -> dict:
    """
    Generate tokens using KV-cache with sampling and optional tracing.

    Args:
        model: GPT model.
        prompt_ids: List of prompt token IDs.
        max_new_tokens: Maximum number of tokens to generate.
        eot_token_id: Token ID that ends generation.
        temperature: Sampling temperature (0 = greedy).
        top_k: Top-k filtering (None = no filtering).
        top_p: Top-p (nucleus) filtering (None = no filtering).
        repetition_penalty: Penalty for repeating tokens (None or 1.0 = no penalty).
        banned_token_ids: Token IDs to ban from generation.
        generator: Random generator for reproducibility (CPU generator).
        trace_layer: Optional layer index to trace attention.

    Returns:
        Dict with:
            - "ids": list[int] full sequence (prompt + generated, including eot if generated)
            - "steps": list[dict] each step contains:
                - "token_id": int
                - "attn_row": list[list[float]] (if trace_layer is set)

    Raises:
        ValueError: If prompt + max_new_tokens exceeds T_max.
    """
    from niels_gpt.infer.kv_cache import allocate_kv_cache, decode_step, prefill

    model.eval()

    T_max = model.cfg.T
    device = next(model.parameters()).device
    dtype = torch.float32  # Use fp32 for inference

    # Hard cap enforcement at function entry
    if len(prompt_ids) + max_new_tokens > T_max:
        raise ValueError(
            f"prompt length {len(prompt_ids)} + max_new_tokens {max_new_tokens} "
            f"= {len(prompt_ids) + max_new_tokens} exceeds T_max {T_max}"
        )

    # Allocate cache
    L = len(model.blocks)
    H = model.cfg.H
    D = model.cfg.C // model.cfg.H
    cache = allocate_kv_cache(L=L, B=1, H=H, T_max=T_max, D=D, device=device, dtype=dtype)

    # Prefill with prompt
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    _, cache, _ = prefill(
        model, prompt_tensor, cache, trace_layer=trace_layer, return_attn_row=False
    )

    # Start with prompt
    ids_list = list(prompt_ids)
    steps = []

    # Use provided generator or create a CPU generator
    if generator is None:
        generator = torch.Generator(device="cpu")

    # Decode loop
    for _ in range(max_new_tokens):
        # Check hard cap before decode_step
        if cache.t >= T_max:
            raise ValueError(f"exceeded T_max={T_max} during generation")

        # Decode one step
        last_token = torch.tensor([[ids_list[-1]]], dtype=torch.long, device=device)
        logits, cache, trace = decode_step(
            model,
            last_token,
            cache,
            trace_layer=trace_layer,
            return_attn_row=(trace_layer is not None),
        )

        # Sample next token
        logits_1d = logits[0, 0].clone()  # (V,)

        # Apply banned tokens
        if banned_token_ids:
            logits_1d[banned_token_ids] = float("-inf")

        # Apply repetition penalty
        if repetition_penalty and repetition_penalty != 1.0:
            unique_tokens = set(ids_list)
            for tok_id in unique_tokens:
                if 0 <= tok_id < logits_1d.shape[0]:
                    logits_1d[tok_id] = logits_1d[tok_id] / repetition_penalty

        next_token = sample_next_token(
            logits_1d,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )

        # Build step info
        step_info = {"token_id": next_token}
        if trace is not None and trace.get("attn_row") is not None:
            attn_row = trace["attn_row"]  # (B, H, t)
            # Convert to list[list[float]]
            step_info["attn_row"] = attn_row[0].cpu().tolist()  # (H, t) -> list[list]

        steps.append(step_info)

        # Append
        ids_list.append(next_token)

        # Stop on eot (include eot in output)
        if next_token == eot_token_id:
            return {"ids": ids_list, "steps": steps}

    return {"ids": ids_list, "steps": steps}
