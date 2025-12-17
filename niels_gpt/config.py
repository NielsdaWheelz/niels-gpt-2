"""Configuration dataclasses and JSON utilities."""

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    V: int
    T: int = 1024
    C: int = 512
    L: int = 8
    H: int = 8
    d_ff: int = 1536
    dropout: float = 0.1
    rope_theta: float = 10000.0


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    B: int = 32
    total_steps_smoke: int = 1000
    total_steps: int = 20000
    eval_every: int = 200
    eval_steps: int = 100
    log_every: int = 50
    ckpt_every: int = 1000
    base_lr: float = 3e-4
    warmup_steps: int = 200
    min_lr: float = 3e-5
    grad_clip: float = 1.0
    accum_steps: int = 1
    p_train: dict[str, float] | None = None
    amp: bool = True
    amp_dtype: str = "fp16"
    activation_checkpointing: bool = False


def default_p_train() -> dict[str, float]:
    """
    returns p_train with primer weight 0.10 and wiki/roam at 80/20 of remaining:
      {"wiki": 0.72, "roam": 0.18, "primer": 0.10}
    """
    primer_weight = 0.10
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


def load_config_from_json(path: str) -> tuple[ModelConfig, TrainConfig]:
    """
    Load ModelConfig and TrainConfig from a JSON file.

    JSON schema:
        {
          "model": { /* overrides for ModelConfig */ },
          "train": { /* overrides for TrainConfig */ }
        }

    Args:
        path: Path to config JSON file

    Returns:
        (ModelConfig, TrainConfig) with overrides applied

    Raises:
        ValueError: If JSON contains unknown keys in model or train sections
    """
    data = load_json(path)

    model_overrides = data.get("model", {})
    train_overrides = data.get("train", {})

    # Validate model overrides
    model_fields = {f.name for f in ModelConfig.__dataclass_fields__.values()}
    unknown_model = set(model_overrides.keys()) - model_fields
    if unknown_model:
        raise ValueError(f"Unknown keys in model config: {sorted(unknown_model)}")

    # Validate train overrides
    train_fields = {f.name for f in TrainConfig.__dataclass_fields__.values()}
    unknown_train = set(train_overrides.keys()) - train_fields
    if unknown_train:
        raise ValueError(f"Unknown keys in train config: {sorted(unknown_train)}")

    # Create configs with overrides
    # Handle p_train default separately
    if "p_train" not in train_overrides:
        train_overrides = {**train_overrides, "p_train": default_p_train()}

    # accum_steps must be >= 1
    if "accum_steps" in train_overrides and train_overrides["accum_steps"] < 1:
        raise ValueError("accum_steps must be >= 1")

    model_cfg = ModelConfig(**model_overrides)
    train_cfg = TrainConfig(**train_overrides)

    return model_cfg, train_cfg
