"""GPT wrapper: decoder-only transformer with token embedding, blocks, and LM head."""

import torch
import torch.nn as nn

from niels_gpt.config import ModelConfig
from niels_gpt.model.blocks import Block, init_weights


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
