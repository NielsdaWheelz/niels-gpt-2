from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from niels_gpt.config import ModelConfig, TrainConfig
from niels_gpt.settings import ResolvedSettings, resolve_settings


@dataclass
class PretrainJobConfig:
    resolved: ResolvedSettings
    model_cfg: ModelConfig
    train_cfg: TrainConfig
    model_cfg_raw: dict[str, Any]
    train_cfg_raw: dict[str, Any]
    sources: dict[str, float]
    val_source: str
    cache_dir: Path
    eval_batches: int
    special_token_ids: dict[str, int]
    stop_token_id: int
    banned_token_ids: list[int]
    run_id: str
    resolved_settings_path: Path | None
    tokenizer_sha256: str | None


@dataclass
class SFTJobConfig:
    resolved: ResolvedSettings
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
    sft_format: dict[str, Any]
    special_token_ids: dict[str, int]
    stop_token_id: int
    banned_token_ids: list[int]
    run_id: str
    resolved_settings_path: Path | None
    tokenizer_sha256: str | None


@dataclass
class PipelineConfig:
    pretrain_config_path: str
    sft_config_path: str


# ----------------------------- helpers ----------------------------- #


def _read_json(path_or_dict: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_dict, dict):
        return path_or_dict
    with open(path_or_dict, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_pretrain_job(resolved: ResolvedSettings) -> PretrainJobConfig:
    settings = resolved.settings
    train_phase = settings.training.pretrain
    eval_batches = train_phase.eval_batches if train_phase.eval_batches is not None else train_phase.eval_steps
    return PretrainJobConfig(
        resolved=resolved,
        model_cfg=resolved.model_cfg,
        train_cfg=resolved.train_cfg,
        model_cfg_raw=settings.model.model_dump(),
        train_cfg_raw=train_phase.model_dump(),
        sources=settings.data.mix_pretrain,
        val_source=settings.data.val_pretrain_source,
        cache_dir=Path(settings.data.caches.pretrain_token_cache),
        eval_batches=eval_batches,
        special_token_ids=resolved.special_token_ids,
        stop_token_id=resolved.stop_token_id,
        banned_token_ids=resolved.banned_token_ids,
        run_id=resolved.run_id,
        resolved_settings_path=resolved.resolved_path,
        tokenizer_sha256=resolved.tokenizer_sha256,
    )


def _build_sft_job(resolved: ResolvedSettings) -> SFTJobConfig:
    settings = resolved.settings
    train_phase = settings.training.sft
    eval_batches = train_phase.eval_batches if train_phase.eval_batches is not None else train_phase.eval_steps
    return SFTJobConfig(
        resolved=resolved,
        model_cfg=resolved.model_cfg,
        train_cfg=resolved.train_cfg,
        model_cfg_raw=settings.model.model_dump(),
        train_cfg_raw=train_phase.model_dump(),
        sft_sources=settings.data.mix_sft,
        val_source=settings.data.val_sft_source,
        cache_dir=Path(settings.data.caches.sft_token_cache),
        streams_cache_dir=Path(settings.data.caches.pretrain_token_cache),
        allow_missing_idx=settings.data.allow_missing_idx,
        eval_batches=eval_batches,
        sft_format=settings.sft_format.model_dump(),
        special_token_ids=resolved.special_token_ids,
        stop_token_id=resolved.stop_token_id,
        banned_token_ids=resolved.banned_token_ids,
        run_id=resolved.run_id,
        resolved_settings_path=resolved.resolved_path,
        tokenizer_sha256=resolved.tokenizer_sha256,
    )


# ----------------------------- loaders ----------------------------- #


def load_pretrain_job_config(
    cfg: str | dict[str, Any], *, run_id: str | None = None, write_resolved: bool = True
) -> PretrainJobConfig:
    resolved = resolve_settings(
        phase="pretrain", overrides_path=cfg, run_id=run_id, write_resolved=write_resolved
    )
    return _build_pretrain_job(resolved)


def load_sft_job_config(
    cfg: str | dict[str, Any], *, run_id: str | None = None, write_resolved: bool = True
) -> SFTJobConfig:
    resolved = resolve_settings(phase="sft", overrides_path=cfg, run_id=run_id, write_resolved=write_resolved)
    return _build_sft_job(resolved)


def load_pipeline_config(cfg: str | dict[str, Any]) -> PipelineConfig:
    data = _read_json(cfg)
    try:
        pretrain_path = data["pretrain_config_path"]
        sft_path = data["sft_config_path"]
    except KeyError as exc:  # noqa: PERF203
        raise ValueError("pipeline config must include pretrain_config_path and sft_config_path") from exc
    return PipelineConfig(pretrain_config_path=str(pretrain_path), sft_config_path=str(sft_path))


