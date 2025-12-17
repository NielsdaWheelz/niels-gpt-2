from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

import niels_gpt.paths as paths
from niels_gpt.config import ModelConfig, TrainConfig, default_p_train


# ----------------------------- pydantic schemas ----------------------------- #


class ModelCfgSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    V: int
    T: int = 128
    C: int = 256
    L: int = 4
    H: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    rope_theta: float = 10000.0

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(**self.model_dump())


class TrainCfgSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    B: int = 16
    total_steps: int = 20000
    eval_every: int = 200
    eval_steps: int = 50
    eval_batches: int | None = None
    log_every: int = 50
    ckpt_every: int = 1000
    base_lr: float = 3e-4
    warmup_steps: int = 200
    min_lr: float = 3e-5
    grad_clip: float = 1.0
    accum_steps: int = 1
    p_train: dict[str, float] | None = None
    amp: bool = True
    amp_dtype: Literal["fp16", "bf16"] = "fp16"
    activation_checkpointing: bool = False

    def to_train_config(self) -> TrainConfig:
        p_train_resolved = self.p_train if self.p_train is not None else default_p_train()
        return TrainConfig(
            seed=self.seed,
            B=self.B,
            total_steps=self.total_steps,
            eval_every=self.eval_every,
            eval_steps=self.eval_steps,
            log_every=self.log_every,
            ckpt_every=self.ckpt_every,
            base_lr=self.base_lr,
            warmup_steps=self.warmup_steps,
            min_lr=self.min_lr,
            grad_clip=self.grad_clip,
            accum_steps=self.accum_steps,
            p_train=p_train_resolved,
            amp=self.amp,
            amp_dtype=self.amp_dtype,
            activation_checkpointing=self.activation_checkpointing,
        )

    def resolved_eval_batches(self) -> int:
        return self.eval_batches if self.eval_batches is not None else self.eval_steps


class PretrainConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_cfg: ModelCfgSchema | None = None
    model_cfg_path: str | None = None
    train_cfg: TrainCfgSchema
    sources: dict[str, float] = Field(default_factory=default_p_train)
    val_source: str = "wiki"
    cache_dir: str = str(paths.REPO_ROOT / "data" / "cache" / "streams")


class SFTConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_cfg: ModelCfgSchema | None = None
    model_cfg_path: str | None = None
    train_cfg: TrainCfgSchema
    sft_sources: dict[str, float] = Field(default_factory=lambda: {"dolly": 0.5, "oasst1": 0.5})
    val_source: str = "wiki"
    cache_dir: str = str(paths.REPO_ROOT / "data" / "cache" / "sft")
    streams_cache_dir: str = str(paths.REPO_ROOT / "data" / "cache" / "streams")
    allow_missing_idx: bool = False


class PipelineConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pretrain_config_path: str
    sft_config_path: str


# ----------------------------- resolved configs ----------------------------- #


@dataclass
class PretrainJobConfig:
    model_cfg: ModelConfig
    train_cfg: TrainConfig
    model_cfg_raw: dict[str, Any]
    train_cfg_raw: dict[str, Any]
    sources: dict[str, float]
    val_source: str
    cache_dir: Path
    eval_batches: int


@dataclass
class SFTJobConfig:
    model_cfg: ModelConfig
    train_cfg: TrainConfig
    model_cfg_raw: dict[str, Any]
    train_cfg_raw: dict[str, Any]
    sft_sources: dict[str, float]
    val_source: str
    cache_dir: Path
    streams_cache_dir: Path
    allow_missing_idx: bool
    eval_batches: int


# ----------------------------- loaders ----------------------------- #


def _read_json(path_or_dict: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_dict, dict):
        return path_or_dict
    with open(path_or_dict, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_model_cfg(schema: ModelCfgSchema | None, model_cfg_path: str | None) -> tuple[ModelConfig, dict[str, Any]]:
    if schema is None and model_cfg_path is None:
        raise ValueError("model_cfg is required (or model_cfg_path)")
    if schema is None and model_cfg_path:
        with open(model_cfg_path, "r", encoding="utf-8") as f:
            model_dict = json.load(f)
        schema = ModelCfgSchema(**model_dict)
        raw = model_dict
    else:
        raw = schema.model_dump() if schema else {}
    return schema.to_model_config(), raw


def load_pretrain_job_config(cfg: str | dict[str, Any]) -> PretrainJobConfig:
    data = _read_json(cfg)
    try:
        parsed = PretrainConfigSchema(**data)
    except ValidationError as e:
        raise ValueError(f"invalid pretrain config: {e}") from e

    model_cfg, model_raw = _resolve_model_cfg(parsed.model_cfg, parsed.model_cfg_path)
    train_cfg_schema = parsed.train_cfg
    train_cfg = train_cfg_schema.to_train_config()
    eval_batches = train_cfg_schema.resolved_eval_batches()
    return PretrainJobConfig(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        model_cfg_raw=model_raw,
        train_cfg_raw=train_cfg_schema.model_dump(),
        sources=parsed.sources,
        val_source=parsed.val_source,
        cache_dir=Path(parsed.cache_dir),
        eval_batches=eval_batches,
    )


def load_sft_job_config(cfg: str | dict[str, Any]) -> SFTJobConfig:
    data = _read_json(cfg)
    try:
        parsed = SFTConfigSchema(**data)
    except ValidationError as e:
        raise ValueError(f"invalid sft config: {e}") from e

    model_cfg, model_raw = _resolve_model_cfg(parsed.model_cfg, parsed.model_cfg_path)
    train_cfg_schema = parsed.train_cfg
    train_cfg = train_cfg_schema.to_train_config()
    eval_batches = train_cfg_schema.resolved_eval_batches()

    return SFTJobConfig(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        model_cfg_raw=model_raw,
        train_cfg_raw=train_cfg_schema.model_dump(),
        sft_sources=parsed.sft_sources,
        val_source=parsed.val_source,
        cache_dir=Path(parsed.cache_dir),
        streams_cache_dir=Path(parsed.streams_cache_dir),
        allow_missing_idx=parsed.allow_missing_idx,
        eval_batches=eval_batches,
    )


def load_pipeline_config(cfg: str | dict[str, Any]) -> PipelineConfigSchema:
    data = _read_json(cfg)
    try:
        return PipelineConfigSchema(**data)
    except ValidationError as e:
        raise ValueError(f"invalid pipeline config: {e}") from e

