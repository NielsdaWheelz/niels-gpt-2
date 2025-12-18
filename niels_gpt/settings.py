from __future__ import annotations

import copy
import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

import niels_gpt.paths as paths
from niels_gpt.config import ModelConfig, TrainConfig
from niels_gpt.special_tokens import SPECIAL_TOKENS


# ----------------------------- helper utilities ----------------------------- #


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge two dicts without mutating inputs."""
    merged = copy.deepcopy(base)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{ts}-{suffix}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


# ----------------------------- tokenizer settings --------------------------- #


class SpecialTokens(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sys: str = SPECIAL_TOKENS[0]
    usr: str = SPECIAL_TOKENS[1]
    asst: str = SPECIAL_TOKENS[2]
    eot: str = SPECIAL_TOKENS[3]

    def as_list(self) -> list[str]:
        return [self.sys, self.usr, self.asst, self.eot]


class TokenizerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokenizer_type: Literal["bpe", "unigram", "byte"] = "bpe"
    model_path: str = str(paths.REPO_ROOT / "artifacts" / "tokenizer" / "v2" / "spm.model")
    vocab_size: int = 16000
    normalization: Literal["none", "nfkc"] = "nfkc"
    byte_fallback: bool = True
    special_tokens: SpecialTokens = SpecialTokens()
    tokenizer_sha256: str | None = "a7ccda70ba310e6571295e6833dd6c340a906c5506cb1391ca0aeb7aba20c762"
    expected_special_token_ids: dict[str, int] | None = {"sys": 3, "usr": 4, "asst": 5, "eot": 6}

    @model_validator(mode="after")
    def _validate_expected_ids(self) -> "TokenizerSettings":
        if self.expected_special_token_ids is not None:
            expected_keys = set(self.expected_special_token_ids.keys())
            required = {"sys", "usr", "asst", "eot"}
            if expected_keys != required:
                raise ValueError(
                    f"expected_special_token_ids must contain exactly {required}, got {sorted(expected_keys)}"
                )
        return self


# ----------------------------- data settings -------------------------------- #


class TokenChunkingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pack_to_T: bool = True
    stride: int = 1


class CacheSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_cache_dir: str = str(paths.REPO_ROOT / "data" / "cache" / "hf")
    pretrain_token_cache: str = str(paths.REPO_ROOT / "data" / "cache" / "pretrain")
    sft_token_cache: str = str(paths.REPO_ROOT / "data" / "cache" / "sft")
    streams_token_cache: str = str(paths.REPO_ROOT / "data" / "cache" / "streams")


class WikitextSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = "wikitext"
    config: str = "wikitext-103-raw-v1"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    streaming: bool = False
    max_examples: int | None = None
    max_bytes: int | None = None
    max_tokens: int | None = None
    chunking: TokenChunkingSettings = TokenChunkingSettings()


class FinewebEduSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = "HuggingFaceFW/fineweb-edu"
    name: str = "CC-MAIN-2024-10"
    split: str = "train"
    streaming: bool = True
    shuffle: bool = True
    max_examples: int | None = None
    max_bytes: int | None = None
    max_tokens: int | None = None
    chunking: TokenChunkingSettings = TokenChunkingSettings()


class GutenbergSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = "nikolina-p/gutenberg_clean_en_splits"
    split: str = "train"
    streaming: bool = True
    shuffle: bool = True
    max_examples: int | None = None
    max_bytes: int | None = None
    max_tokens: int | None = None
    chunking: TokenChunkingSettings = TokenChunkingSettings()


class DollySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = "databricks/databricks-dolly-15k"
    split: str = "train"
    include_system: bool = False
    max_examples: int | None = None


class Oasst1Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = "OpenAssistant/oasst1"
    split: str = "train"
    english_only: bool = True
    max_messages: int = 32
    shuffle_trees: bool = True
    take_trees: int | None = None


class RoamSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_dir: str = str(paths.REPO_ROOT / ".roam-data")
    glob_pattern: str = "**/*.md"
    val_frac: float = 0.1
    split_seed: int = 42
    chunking: TokenChunkingSettings = TokenChunkingSettings()


class PrimerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = str(paths.REPO_ROOT / "data" / "primer.txt")
    delimiter: str = "\n\n<dialogue>\n\n"
    val_frac: float = 0.1
    seed: int = 42


def _default_pretrain_mix() -> dict[str, float]:
    return {"fineweb_edu": 0.70, "wikitext": 0.20, "roam": 0.10}


def _default_sft_mix() -> dict[str, float]:
    return {"primer": 0.10, "oasst1": 0.70, "dolly15k": 0.20}


VALID_PRETRAIN_SOURCES = {"wikitext", "fineweb_edu", "gutenberg", "roam"}
VALID_SFT_SOURCES = {"dolly15k", "oasst1", "primer"}


class DataSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    caches: CacheSettings = CacheSettings()
    chunking: TokenChunkingSettings = TokenChunkingSettings()
    wikitext: WikitextSettings = WikitextSettings()
    fineweb_edu: FinewebEduSettings = FinewebEduSettings()
    gutenberg: GutenbergSettings = GutenbergSettings()
    dolly: DollySettings = DollySettings()
    oasst1: Oasst1Settings = Oasst1Settings()
    roam: RoamSettings = RoamSettings()
    primer: PrimerSettings = PrimerSettings()
    mix_pretrain: dict[str, float] = Field(default_factory=_default_pretrain_mix)
    mix_sft: dict[str, float] = Field(default_factory=_default_sft_mix)
    val_pretrain_source: str = "wikitext"
    val_sft_source: str = "sft"
    allow_missing_idx: bool = False

    @model_validator(mode="after")
    def _validate_mix(self) -> "DataSettings":
        mix_specs = [
            ("mix_pretrain", self.mix_pretrain, VALID_PRETRAIN_SOURCES),
            ("mix_sft", self.mix_sft, VALID_SFT_SOURCES),
        ]
        for name, mix, valid_keys in mix_specs:
            if not mix:
                raise ValueError(f"{name} must be non-empty and use keys from {sorted(valid_keys)}")

            invalid = sorted(set(mix.keys()) - valid_keys)
            if invalid:
                raise ValueError(f"{name} contains invalid keys {invalid}; valid keys are {sorted(valid_keys)}")

            for key, val in mix.items():
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise ValueError(f"{name}[{key}] must be a float in (0, 1], got {val!r}")
                if not (0.0 < float(val) <= 1.0):
                    raise ValueError(f"{name}[{key}] must be in (0, 1], got {val}")

            total = sum(float(v) for v in mix.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"{name} probabilities must sum to 1.0, got {total}")

        # Validate val_sft_source
        if self.val_sft_source not in {"sft", "wikitext"}:
            raise ValueError(f"val_sft_source must be 'sft' or 'wikitext', got {self.val_sft_source!r}")

        return self


# ----------------------------- model settings ------------------------------- #


class AttentionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attn_type: Literal["mha"] = "mha"
    use_kv_cache: bool = True
    kv_cache_dtype: Literal["fp16", "fp32"] = "fp16"


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    V: int = 16000
    T: int = 512
    C: int = 384
    L: int = 8
    H: int = 6
    d_ff: int = 1152
    dropout: float = 0.1
    rope_theta: float = 10000.0
    norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    mlp_type: Literal["swiglu", "gelu"] = "swiglu"
    attention: AttentionSettings = AttentionSettings()


# ----------------------------- training settings ---------------------------- #


class TrainPhaseSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    micro_B: int = 16
    accum_steps: int = 1
    total_steps: int = 17000
    eval_every: int = 200
    eval_steps: int = 50
    eval_batches: int | None = None
    micro_B_eval: int | None = None
    log_every: int = 50
    ckpt_every: int = 1000

    optimizer: Literal["adamw"] = "adamw"
    base_lr: float = 3e-4
    warmup_steps: int = 340
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    decay_norm_and_bias: bool = True  # kept true to preserve existing behavior
    decay_embeddings: bool = True

    grad_clip: float = 1.0
    amp: bool = False
    amp_dtype: Literal["fp16", "bf16"] = "fp16"
    activation_checkpointing: bool = False

    save_best: bool = True
    best_metric: Literal["pretrain_val_loss", "sft_val_loss", "weighted"] = "pretrain_val_loss"
    best_metric_weights: dict[str, float] | None = None

    @model_validator(mode="after")
    def _validate(self) -> "TrainPhaseSettings":
        if self.accum_steps < 1:
            raise ValueError("accum_steps must be >= 1")
        if self.micro_B < 1:
            raise ValueError("micro_B must be >= 1")
        if self.micro_B_eval is not None and self.micro_B_eval < 1:
            raise ValueError("micro_B_eval must be >= 1 when set")
        return self


def _default_pretrain_train() -> TrainPhaseSettings:
    return TrainPhaseSettings()


def _default_sft_train() -> TrainPhaseSettings:
    return TrainPhaseSettings(
        total_steps=6000,
        base_lr=1e-4,
        warmup_steps=120,
        min_lr=1e-5,
        weight_decay=0.05,
        best_metric="sft_val_loss",
    )


class TrainingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pretrain: TrainPhaseSettings = Field(default_factory=_default_pretrain_train)
    sft: TrainPhaseSettings = Field(default_factory=_default_sft_train)


class SFTFormattingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant_only_loss: bool = True
    include_eot_in_loss: bool = False
    pack_sequences: bool = True
    packing_policy: Literal["prefer_boundary_allow_truncate"] = "prefer_boundary_allow_truncate"
    require_eot: bool = True
    allow_truncate_mid_turn: bool = True
    ban_role_tokens_during_generation: bool = True


# ----------------------------- generation settings -------------------------- #


class GenerationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int = 256
    temperature: float = 0.9
    top_k: int | None = 50
    top_p: float | None = None
    repetition_penalty: float | None = None
    stop_token_id: int | None = None
    banned_token_ids: list[int] | None = None


# ----------------------------- benchmark settings --------------------------- #


def _default_benchmark_model_grid() -> list[dict[str, int]]:
    return [
        {"C": 384, "L": 8, "H": 6},
        {"C": 512, "L": 8, "H": 8},
        {"C": 512, "L": 12, "H": 8},
    ]


class BenchmarkSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_T: list[int] = Field(default_factory=lambda: [512, 1024])
    candidate_model_dims: list[dict[str, int]] = Field(default_factory=_default_benchmark_model_grid)
    checkpointing_modes: list[bool] = Field(default_factory=lambda: [False, True])
    timeout_s: float = 20.0
    steps_warmup: int = 2
    steps_measure: int = 10
    max_micro_B: int = 256
    binary_search_max_micro_B: int = 256
    seed: int = 42
    lr: float = 3e-4


# ----------------------------- reproducibility ------------------------------ #


class ReproSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    torch_num_threads: int | None = None
    torch_matmul_precision: str | None = None
    dataset_shuffle_buffer_size: int = 10_000


# ----------------------------- root settings -------------------------------- #


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokenizer: TokenizerSettings = TokenizerSettings()
    data: DataSettings = DataSettings()
    model: ModelSettings = ModelSettings()
    training: TrainingSettings = TrainingSettings()
    sft_format: SFTFormattingSettings = SFTFormattingSettings()
    generation: GenerationSettings = GenerationSettings()
    benchmark: BenchmarkSettings = BenchmarkSettings()
    reproducibility: ReproSettings = ReproSettings()


@dataclass
class ResolvedSettings:
    settings: Settings
    model_cfg: ModelConfig
    train_cfg: TrainConfig
    special_token_ids: dict[str, int]
    stop_token_id: int
    banned_token_ids: list[int]
    phase: Literal["pretrain", "sft", "pipeline"]
    overrides: dict[str, Any]
    overrides_source: str | None
    run_id: str
    resolved_path: Path | None
    tokenizer_sha256: str | None
    tokenizer_meta: dict[str, Any]


def _resolved_payload(resolved: ResolvedSettings) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    return {
        "run_id": resolved.run_id,
        "phase": resolved.phase,
        "generated_at_utc": generated_at,
        "overrides_source": resolved.overrides_source,
        "overrides": resolved.overrides,
        "settings": resolved.settings.model_dump(),
        "model_cfg": asdict(resolved.model_cfg),
        "train_cfg": asdict(resolved.train_cfg),
        "special_token_ids": resolved.special_token_ids,
        "stop_token_id": resolved.stop_token_id,
        "banned_token_ids": resolved.banned_token_ids,
        "tokenizer": {
            "model_path": resolved.settings.tokenizer.model_path,
            "sha256": resolved.tokenizer_sha256,
            "special_tokens": resolved.settings.tokenizer.special_tokens.model_dump(),
            "meta": resolved.tokenizer_meta,
        },
    }


def _dump_resolved_settings(resolved: ResolvedSettings) -> Path:
    run_root = paths.REPO_ROOT / "runs" / resolved.run_id
    path = run_root / "resolved_settings.json"
    _write_json(path, _resolved_payload(resolved))
    return path


# ----------------------------- API functions -------------------------------- #


def default_settings() -> Settings:
    """Return a fresh Settings instance with defaults."""
    return Settings()


def load_overrides(path_or_dict: str | dict | None) -> dict:
    """Load overrides from a path or dict; None -> empty dict."""
    if path_or_dict is None:
        return {}
    if isinstance(path_or_dict, dict):
        return path_or_dict
    return _load_json(path_or_dict)


def _looks_legacy(cfg: dict[str, Any]) -> bool:
    """Detect legacy (model_cfg/train_cfg) style configs."""
    return "model_cfg" in cfg or "train_cfg" in cfg or "model" in cfg and "train" in cfg


def _resolve_special_tokens(tokenizer_settings: TokenizerSettings) -> tuple[dict[str, int], dict[str, Any]]:
    from niels_gpt.tokenizer import SentencePieceTokenizer

    settings_specials = tokenizer_settings.special_tokens.as_list()
    if settings_specials != list(SPECIAL_TOKENS):
        raise ValueError(
            "tokenizer.special_tokens must match the project constants; "
            f"got {settings_specials}, expected {list(SPECIAL_TOKENS)}"
        )

    model_path = Path(tokenizer_settings.model_path)
    actual_sha = _sha256(model_path)
    if tokenizer_settings.tokenizer_sha256 and actual_sha != tokenizer_settings.tokenizer_sha256:
        raise ValueError(
            "tokenizer_sha256 mismatch: "
            f"settings expects {tokenizer_settings.tokenizer_sha256}, actual {actual_sha}"
        )

    tok = SentencePieceTokenizer(
        tokenizer_settings.model_path,
        special_tokens=settings_specials,
        expected_special_token_ids=tokenizer_settings.expected_special_token_ids,
    )
    special_ids = tok.special_token_ids()
    meta = {
        "unk_id": tok._sp.unk_id(),  # type: ignore[attr-defined]
        "bos_id": tok._sp.bos_id(),  # type: ignore[attr-defined]
        "eos_id": tok._sp.eos_id(),  # type: ignore[attr-defined]
        "pad_id": tok._sp.pad_id(),  # type: ignore[attr-defined]
        "expected_special_token_ids": tokenizer_settings.expected_special_token_ids,
    }
    return special_ids, meta


def resolve_settings(
    *,
    phase: Literal["pretrain", "sft"],
    overrides_path: str | dict | None,
    run_id: str | None = None,
    write_resolved: bool = False,
) -> ResolvedSettings:
    """
    Resolve settings for a given phase, applying overrides and legacy translation.
    """
    if phase not in {"pretrain", "sft"}:
        raise ValueError("resolve_settings expects phase in {'pretrain','sft'}")

    overrides_raw = load_overrides(overrides_path)
    if not isinstance(overrides_raw, dict):
        raise ValueError("overrides must be a dict or JSON file path")
    if _looks_legacy(overrides_raw):
        raise ValueError("legacy config format (model_cfg/train_cfg) is no longer supported")
    overrides_source = overrides_path if isinstance(overrides_path, str) else None
    overrides = overrides_raw

    base = default_settings()
    merged_dict = base.model_dump()
    # replace (not merge) for mix dictionaries if provided
    data_over = overrides.get("data", {}) if isinstance(overrides, dict) else {}
    if "mix_pretrain" in data_over:
        merged_dict.setdefault("data", {})["mix_pretrain"] = data_over["mix_pretrain"]
    if "mix_sft" in data_over:
        merged_dict.setdefault("data", {})["mix_sft"] = data_over["mix_sft"]
    merged_dict = _deep_merge(merged_dict, overrides)
    settings = Settings.model_validate(merged_dict)

    # Build model + train configs
    model_cfg = ModelConfig(
        V=settings.model.V,
        T=settings.model.T,
        C=settings.model.C,
        L=settings.model.L,
        H=settings.model.H,
        d_ff=settings.model.d_ff,
        dropout=settings.model.dropout,
        rope_theta=settings.model.rope_theta,
    )

    phase_train = settings.training.pretrain if phase == "pretrain" else settings.training.sft
    betas = phase_train.betas
    train_cfg = TrainConfig(
        seed=phase_train.seed,
        B=phase_train.micro_B,
        total_steps=phase_train.total_steps,
        eval_every=phase_train.eval_every,
        eval_steps=phase_train.eval_steps,
        log_every=phase_train.log_every,
        ckpt_every=phase_train.ckpt_every,
        base_lr=phase_train.base_lr,
        warmup_steps=phase_train.warmup_steps,
        min_lr=phase_train.min_lr,
        grad_clip=phase_train.grad_clip,
        accum_steps=phase_train.accum_steps,
        amp=phase_train.amp,
        amp_dtype=phase_train.amp_dtype,
        activation_checkpointing=phase_train.activation_checkpointing,
        optimizer=phase_train.optimizer,
        beta1=betas[0],
        beta2=betas[1],
        weight_decay=phase_train.weight_decay,
        eps=phase_train.eps,
        micro_B_eval=phase_train.micro_B_eval,
        eval_batches=phase_train.eval_batches,
        decay_norm_and_bias=phase_train.decay_norm_and_bias,
        decay_embeddings=phase_train.decay_embeddings,
        save_best=phase_train.save_best,
        best_metric=phase_train.best_metric,
        best_metric_weights=phase_train.best_metric_weights,
    )

    special_token_ids, tokenizer_meta = _resolve_special_tokens(settings.tokenizer)
    for name, tid in special_token_ids.items():
        if tid >= model_cfg.V:
            raise ValueError(f"special token {name} id {tid} exceeds model vocab size {model_cfg.V}")

    stop_token_id = settings.generation.stop_token_id or special_token_ids["eot"]
    banned_token_ids = settings.generation.banned_token_ids or []
    if settings.sft_format.ban_role_tokens_during_generation and not banned_token_ids:
        banned_token_ids = [
            special_token_ids["sys"],
            special_token_ids["usr"],
            special_token_ids["asst"],
        ]

    resolved = ResolvedSettings(
        settings=settings,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        special_token_ids=special_token_ids,
        stop_token_id=stop_token_id,
        banned_token_ids=banned_token_ids,
        phase=phase,
        overrides=overrides_raw,
        overrides_source=overrides_source,
        run_id=run_id or _now_run_id(),
        resolved_path=None,
        tokenizer_sha256=_sha256(Path(settings.tokenizer.model_path)),
        tokenizer_meta=tokenizer_meta,
    )

    if write_resolved:
        resolved.resolved_path = _dump_resolved_settings(resolved)

    return resolved


