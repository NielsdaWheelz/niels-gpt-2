"""Transformer building blocks: causal self-attention."""

import math

import torch
import torch.nn as nn

import niels_gpt.model.rope as rope
from niels_gpt.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE.

    Computes attention with:
    - Rotary position embeddings (RoPE) applied to queries and keys
    - Causal masking to prevent attending to future positions
    - Dropout on attention probabilities and output projection
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        # Compute head dimension and validate
        assert cfg.C % cfg.H == 0, f"C={cfg.C} must be divisible by H={cfg.H}"
        D = cfg.C // cfg.H
        assert D % 2 == 0, f"Head dim D={D} must be even for RoPE"

        self.cfg = cfg
        self.H = cfg.H
        self.D = D

        # QKV projection (single linear layer for efficiency)
        self.qkv = nn.Linear(cfg.C, 3 * cfg.C, bias=False)

        # Output projection
        self.proj = nn.Linear(cfg.C, cfg.C, bias=False)

        # Dropout layers
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Precompute and register RoPE sin/cos buffers
        sin, cos = rope.rope_cache(
            cfg.T, D, theta=cfg.rope_theta, device="cpu", dtype=torch.float32
        )
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

        # Precompute and register causal mask (lower triangular)
        mask = torch.tril(torch.ones(cfg.T, cfg.T, dtype=torch.bool))
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with causal self-attention.

        Args:
            x: Input tensor, shape (B, T, C).
            return_attn: If True, return (output, attention_weights).

        Returns:
            If return_attn=False: output tensor (B, T, C).
            If return_attn=True: tuple of (output, attention_weights) where
                attention_weights has shape (B, H, T, T).
        """
        B, T, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x)  # (B, T, 3*C)
        qkv = qkv.reshape(B, T, 3, C)  # (B, T, 3, C)
        q, k, v = qkv.unbind(dim=2)  # Each: (B, T, C)

        # Reshape to multi-head format: (B, T, C) -> (B, H, T, D)
        q = q.reshape(B, T, self.H, self.D).transpose(1, 2)  # (B, H, T, D)
        k = k.reshape(B, T, self.H, self.D).transpose(1, 2)  # (B, H, T, D)
        v = v.reshape(B, T, self.H, self.D).transpose(1, 2)  # (B, H, T, D)

        # Apply RoPE to queries and keys
        q, k = rope.apply_rope(q, k, self.rope_sin, self.rope_cos)

        # Compute attention scores: (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.D))

        # Apply causal mask (mask out future positions)
        scores = scores.masked_fill(
            ~self.mask[:, :, :T, :T], torch.finfo(scores.dtype).min
        )

        # Compute attention probabilities
        attn = torch.softmax(scores, dim=-1)  # (B, H, T, T)
        attn = self.attn_dropout(attn)

        # Apply attention to values: (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)
        out = attn @ v  # (B, H, T, D)

        # Merge heads: (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
        out = out.transpose(1, 2).reshape(B, T, C)

        # Output projection and dropout
        out = self.proj(out)
        out = self.resid_dropout(out)

        if return_attn:
            return out, attn
        return out
