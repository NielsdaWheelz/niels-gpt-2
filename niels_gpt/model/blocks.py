"""Transformer building blocks: causal self-attention, MLP, and transformer block."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import niels_gpt.model.rope as rope
from niels_gpt.config import ModelConfig


class RMSNorm(nn.Module):
    """Root-mean-square normalization without bias."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize by root mean square over the last dimension.
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class MLP(nn.Module):
    """
    Feed-forward network with SwiGLU activation.

    Architecture:
        Linear(C, d_ff) -> gate with SiLU
        Linear(C, d_ff) -> value
        out = Linear(d_ff, C)(silu(gate) * value)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc_a = nn.Linear(cfg.C, cfg.d_ff, bias=True)
        self.fc_g = nn.Linear(cfg.C, cfg.d_ff, bias=True)
        self.fc_out = nn.Linear(cfg.d_ff, cfg.C, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.

        Args:
            x: Input tensor, shape (B, T, C).

        Returns:
            Output tensor, shape (B, T, C).
        """
        a = self.fc_a(x)
        g = F.silu(self.fc_g(x))
        h = g * a
        out = self.fc_out(h)
        return out


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

        # Cached dtype-converted versions of rope buffers (for AMP)
        self._rope_sin_fp16 = None
        self._rope_cos_fp16 = None
        self._rope_sin_bf16 = None
        self._rope_cos_bf16 = None

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
        # Get cached dtype-converted rope buffers (lazy init for AMP compatibility)
        if q.dtype == torch.float16:
            if self._rope_sin_fp16 is None:
                self._rope_sin_fp16 = self.rope_sin.to(torch.float16)
                self._rope_cos_fp16 = self.rope_cos.to(torch.float16)
            sin, cos = self._rope_sin_fp16, self._rope_cos_fp16
        elif q.dtype == torch.bfloat16:
            if self._rope_sin_bf16 is None:
                self._rope_sin_bf16 = self.rope_sin.to(torch.bfloat16)
                self._rope_cos_bf16 = self.rope_cos.to(torch.bfloat16)
            sin, cos = self._rope_sin_bf16, self._rope_cos_bf16
        else:
            # float32 or other - use original buffers
            sin, cos = self.rope_sin, self.rope_cos

        q, k = rope.apply_rope(q, k, sin, cos)

        # Compute attention scores: (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.D))

        # Apply causal mask (mask out future positions)
        scores = scores.masked_fill(
            ~self.mask[:, :, :T, :T], torch.finfo(scores.dtype).min
        )

        # Compute attention probabilities
        attn_probs = torch.softmax(scores, dim=-1)  # (B, H, T, T)
        attn_dropped = self.attn_dropout(attn_probs)

        # Apply attention to values: (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)
        out = attn_dropped @ v  # (B, H, T, D)

        # Merge heads: (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
        out = out.transpose(1, 2).reshape(B, T, C)

        # Output projection and dropout
        out = self.proj(out)
        out = self.resid_dropout(out)

        if return_attn:
            return out, attn_probs  # Return pre-dropout probabilities
        return out


class Block(nn.Module):
    """
    Pre-norm transformer block with causal self-attention and SwiGLU MLP.

    Architecture:
        x = x + dropout(attn(ln1(x)))
        x = x + dropout(mlp(ln2(x)))
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.C, eps=1e-5)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.C, eps=1e-5)
        self.mlp = MLP(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor, shape (B, T, C).
            return_attn: If True, return (output, attention_weights).

        Returns:
            If return_attn=False: output tensor (B, T, C).
            If return_attn=True: tuple of (output, attention_weights) where
                attention_weights has shape (B, H, T, T).
        """
        # Attention with residual
        if return_attn:
            attn_out, attn_probs = self.attn(self.ln1(x), return_attn=True)
            x = x + self.dropout(attn_out)
        else:
            x = x + self.dropout(self.attn(self.ln1(x)))

        # MLP with residual
        x = x + self.dropout(self.mlp(self.ln2(x)))

        if return_attn:
            return x, attn_probs
        return x


def init_weights(module: nn.Module) -> None:
    """
    Initialize weights for GPT-style models.

    Applies to:
        - nn.Linear: weight ~ Normal(0, 0.02), bias = 0 (if present)
        - nn.Embedding: weight ~ Normal(0, 0.02)
        - nn.LayerNorm: weight = 1, bias = 0
        - RMSNorm: weight = 1

    Safe to use with model.apply(init_weights).
    """
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, RMSNorm):
        module.weight.data.fill_(1.0)
    elif isinstance(module, nn.LayerNorm):
        module.weight.data.fill_(1.0)
        module.bias.data.zero_()
