"""Configuration dataclasses and JSON utilities."""

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    V: int = 256
    T: int = 256
    C: int = 256
    L: int = 4
    H: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    rope_theta: float = 10000.0


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    B: int = 32
    total_steps_smoke: int = 1000
    total_steps: int = 20000
    eval_every: int = 200
    ckpt_every: int = 1000
    base_lr: float = 3e-4
    warmup_steps: int = 200
    min_lr: float = 3e-5
    grad_clip: float = 1.0
    p_train: dict[str, float] | None = None


def default_p_train() -> dict[str, float]:
    """
    returns p_train with primer weight 0.02 and wiki/roam at 80/20 of remaining:
      {"wiki": 0.784, "roam": 0.196, "primer": 0.020}
    """
    primer_weight = 0.020
    remaining = 1.0 - primer_weight
    wiki_weight = remaining * 0.8
    roam_weight = remaining * 0.2
    return {
        "wiki": wiki_weight,
        "roam": roam_weight,
        "primer": primer_weight,
    }


def to_dict(obj: Any) -> dict:
    """dataclass -> plain dict (recursive for dataclasses only)."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj


def save_json(path: str, data: dict) -> None:
    """write utf-8 json with indent=2 and sorted keys."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)


def load_json(path: str) -> dict:
    """read json and return dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
