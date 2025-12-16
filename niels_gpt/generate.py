"""Text generation utilities with temperature and top-k sampling."""

import torch

from niels_gpt.config import ModelConfig
from niels_gpt.tokenizer import decode, encode


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
    temperature: float = 0.9,
    top_k: int | None = 50,
    stop_sequences: list[bytes] | None = None,
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
        stop_sequences: List of byte sequences that stop generation
        device: Device to run model on ("mps" or "cpu")
        generator: Random generator for reproducibility

    Returns:
        (n + m,) int64 on cpu, where m <= max_new_tokens
        Stops early if stop sequence detected, truncating before it.
    """
    model.eval()

    # Use list for O(1) append instead of O(n) tensor concatenation
    ids_list = prompt_ids.tolist()
    start = len(ids_list)

    # Precompute max stop sequence length for efficient checking
    max_stop = 0
    if stop_sequences:
        max_stop = max(len(s) for s in stop_sequences)

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
            logits_last = logits[0, -1]  # (V,)

            # Sample next token
            next_token = sample_next_token(
                logits_last,
                temperature=temperature,
                top_k=top_k,
                generator=generator,
            )

            # Append to list (O(1))
            ids_list.append(next_token)

            # Check for stop sequences in newly generated portion
            if stop_sequences:
                # Convert to bytes for stop sequence checking
                # Include overlap to catch sequences that straddle the window boundary
                # Window size: max_stop + 4 for current + (max_stop - 1) for overlap
                window_start = max(0, len(ids_list) - (max_stop + 4) - (max_stop - 1))
                window_bytes = bytes(ids_list[window_start:])

                # Search starting from the first position that could be in generated portion
                # This ensures we find the earliest stop sequence in the generated region
                search_from = max(0, start - window_start)

                # Check each stop sequence
                for stop_seq in stop_sequences:
                    pos = window_bytes.find(stop_seq, search_from)
                    if pos != -1:
                        # Calculate absolute position in ids
                        abs_pos = window_start + pos
                        # Truncate before the stop sequence
                        ids_list = ids_list[:abs_pos]
                        return torch.tensor(ids_list, dtype=torch.long)

    return torch.tensor(ids_list, dtype=torch.long)


def generate_text(
    model: torch.nn.Module,
    prompt_text: str,
    *,
    cfg: ModelConfig,
    max_new_tokens: int,
    temperature: float = 0.9,
    top_k: int | None = 50,
    stop_sequences: list[bytes] | None = None,
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
        stop_sequences: List of byte sequences that stop generation
        device: Device to run model on
        generator: Random generator for reproducibility

    Returns:
        Generated text (prompt + completion)
    """
    # Encode prompt
    prompt_ids = encode(prompt_text)

    # Generate ids
    output_ids = generate_ids(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        T=cfg.T,
        temperature=temperature,
        top_k=top_k,
        stop_sequences=stop_sequences,
        device=device,
        generator=generator,
    )

    # Decode and return
    return decode(output_ids)
