"""GPT wrapper: decoder-only transformer with token embedding, blocks, and LM head."""

from typing import Optional, TypedDict

import torch
import torch.nn as nn

from niels_gpt.config import ModelConfig
from niels_gpt.model.blocks import Block, init_weights


class AttnTrace(TypedDict):
    """
    Attention trace for a specific layer.

    Fields:
        layer: The layer index that was traced (0 <= layer < L).
        attn_row: Attention probabilities for the last token (query pos = t-1),
                  shape (B, H, t), where t is the current sequence length.
        attn_full: Full attention matrix (B, H, t, t) if requested, else None.
    """

    layer: int
    attn_row: torch.Tensor  # (B, H, t)
    attn_full: Optional[torch.Tensor]  # (B, H, t, t) or None


class GPT(nn.Module):
    """
    Decoder-only transformer for byte-level causal language modeling.

    Architecture:
        tok_emb -> dropout -> L x Block -> ln_f -> lm_head
        (weight tying: lm_head.weight = tok_emb.weight)

    Forward pass:
        x: (B, T_cur) int64 -> logits: (B, T_cur, V) float32
        where T_cur <= cfg.T
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Token embedding
        self.tok_emb = nn.Embedding(cfg.V, cfg.C)

        # Dropout at embeddings
        self.drop = nn.Dropout(cfg.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.L)])

        # Final layer norm
        self.ln_f = nn.LayerNorm(cfg.C, eps=1e-5)

        # LM head (no bias)
        self.lm_head = nn.Linear(cfg.C, cfg.V, bias=False)

        # Weight tying: lm_head shares weights with tok_emb
        self.lm_head.weight = self.tok_emb.weight

        # Initialize all weights
        self.apply(init_weights)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        """
        Forward pass through GPT.

        Args:
            x: Token IDs, shape (B, T_cur), dtype int64, values in [0..V-1].

        Returns:
            logits: shape (B, T_cur, V), dtype float32.

        Raises:
            AssertionError: If x.ndim != 2 or T_cur > cfg.T.
        """
        # Validate input
        assert x.ndim == 2, f"Expected 2D input (B, T), got shape {x.shape}"
        B, T_cur = x.shape
        assert T_cur <= self.cfg.T, (
            f"Sequence length {T_cur} exceeds max context {self.cfg.T}"
        )

        # Token embeddings + dropout
        h = self.tok_emb(x)  # (B, T_cur, C)
        h = self.drop(h)

        # Apply transformer blocks sequentially
        for block in self.blocks:
            h = block(h)  # (B, T_cur, C)

        # Final layer norm
        h = self.ln_f(h)  # (B, T_cur, C)

        # LM head
        logits = self.lm_head(h)  # (B, T_cur, V)

        return logits

    def forward_with_attn_trace(
        self,
        x: torch.LongTensor,
        *,
        trace_layer: int,
        return_full_attn: bool = False,
    ) -> tuple[torch.FloatTensor, AttnTrace]:
        """
        Forward pass with attention trace for a selected layer.

        This method produces identical logits to forward() when model.eval() is used,
        but additionally captures attention probabilities from the specified layer.

        Args:
            x: Token IDs, shape (B, t), dtype int64, values in [0..V-1],
               where t <= cfg.T.
            trace_layer: Which transformer block to trace, must be 0 <= trace_layer < cfg.L.
            return_full_attn: If True, return the full (B, H, t, t) attention matrix.
                              If False, only return the attention row for the last token.

        Returns:
            logits: shape (B, t, V), dtype float32 (same as forward()).
            trace: AttnTrace dict containing:
                - layer: trace_layer value
                - attn_row: (B, H, t) attention probabilities for last token (query pos = t-1)
                - attn_full: (B, H, t, t) full attention matrix if return_full_attn=True, else None

        Raises:
            ValueError: If trace_layer is out of bounds [0, cfg.L).

        Invariants:
            - Logits are numerically identical to forward() when model.eval() is used.
            - Attention probabilities are pre-dropout (from CausalSelfAttention).
            - attn_row.sum(dim=-1) is approximately 1.0 (proper probability distribution).
            - Dropout is disabled in eval mode, ensuring deterministic attention.
            - All returned tensors (logits, attn_row, attn_full) remain on model's device.

        Performance notes:
            - Full (t, t) attention scores and softmax are computed even when
              return_full_attn=False. This is unavoidable - only the *return* is conditional.
            - For streaming (calling per-token), you pay the full attention compute cost
              at every step. At T=256 with small models this is acceptable, but does not
              scale to large contexts or frontier model sizes.
            - No additional memory overhead when return_full_attn=False (attn_probs is
              not cloned, just sliced).
        """
        # Validate trace_layer bounds
        if not (0 <= trace_layer < self.cfg.L):
            raise ValueError(
                f"trace_layer must be in range [0, {self.cfg.L}), got {trace_layer}"
            )

        # Validate input
        assert x.ndim == 2, f"Expected 2D input (B, t), got shape {x.shape}"
        B, t = x.shape
        assert t <= self.cfg.T, (
            f"Sequence length {t} exceeds max context {self.cfg.T}"
        )

        # Token embeddings + dropout
        h = self.tok_emb(x)  # (B, t, C)
        h = self.drop(h)

        # Apply transformer blocks, capturing attention from trace_layer
        attn_probs = None
        for i, block in enumerate(self.blocks):
            if i == trace_layer:
                # Capture attention probabilities from the selected layer
                h, attn_probs = block(h, return_attn=True)  # h: (B, t, C), attn_probs: (B, H, t, t)
            else:
                # Normal forward pass for other layers
                h = block(h)  # (B, t, C)

        # Final layer norm
        h = self.ln_f(h)  # (B, t, C)

        # LM head
        logits = self.lm_head(h)  # (B, t, V)

        # Extract attention row for the last token (query position t-1)
        # attn_probs shape: (B, H, t, t)
        # attn_row = attn_probs[:, :, t-1, :t] -> (B, H, t)
        attn_row = attn_probs[:, :, t - 1, :t]  # (B, H, t)

        # Build trace dict
        trace: AttnTrace = {
            "layer": trace_layer,
            "attn_row": attn_row,
            "attn_full": attn_probs if return_full_attn else None,
        }

        return logits, trace
