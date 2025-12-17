from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn.functional as F


BatchFn = Callable[[], Tuple[torch.Tensor, torch.Tensor]]


@torch.no_grad()
def evaluate_pretrain(
    model,
    *,
    batch_fn: BatchFn,
    eval_batches: int,
) -> float:
    """Average CE loss over eval_batches using (x, y) batches."""
    was_training = model.training
    model.eval()
    try:
        total = 0.0
        for _ in range(eval_batches):
            x, y = batch_fn()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total += float(loss.item())
        return total / max(1, eval_batches)
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def evaluate_sft(
    model,
    *,
    batch_fn: BatchFn,
    eval_batches: int,
    ignore_index: int = -100,
) -> float:
    """Average CE loss over eval_batches using masked targets."""
    was_training = model.training
    model.eval()
    try:
        total = 0.0
        for _ in range(eval_batches):
            x, y_masked = batch_fn()
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_masked.view(-1),
                ignore_index=ignore_index,
            )
            total += float(loss.item())
        return total / max(1, eval_batches)
    finally:
        if was_training:
            model.train()

