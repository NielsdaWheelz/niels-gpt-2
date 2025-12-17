"""Configuration dataclasses."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    V: int
    T: int
    C: int
    L: int
    H: int
    d_ff: int
    dropout: float
    rope_theta: float


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    B: int
    total_steps: int
    eval_every: int
    eval_steps: int
    log_every: int
    ckpt_every: int
    base_lr: float
    warmup_steps: int
    min_lr: float
    grad_clip: float
    accum_steps: int
    amp: bool
    amp_dtype: str
    activation_checkpointing: bool
    optimizer: str
    beta1: float
    beta2: float
    weight_decay: float
    eps: float
    micro_B_eval: int | None
    eval_batches: int | None
    decay_norm_and_bias: bool
    decay_embeddings: bool
    save_best: bool
    best_metric: str | None
    best_metric_weights: dict[str, float] | None


def to_dict(obj: Any) -> dict:
    """dataclass -> plain dict (recursive for dataclasses only)."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj
