from __future__ import annotations

import json
import shutil
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils

import niels_gpt.paths as paths
from niels_gpt.config import to_dict
from niels_gpt.device import get_device
from niels_gpt.lr_schedule import lr_at_step
from niels_gpt.model.gpt import GPT
from niels_gpt.train.cache_validate import format_data_plan, validate_token_caches

from train.amp_utils import get_amp_context
from train.checkpointing import (
    PhasePaths,
    assert_model_config_compatible,
    load_checkpoint,
    phase_paths,
    save_checkpoint,
)
from train.config import PretrainJobConfig, load_pretrain_job_config
from train.eval import evaluate_pretrain


def _resolve_resume_path(
    resume: str | None, *, no_auto_resume: bool, phase: str, allow_root_fallback: bool = False
) -> Path | None:
    if resume:
        return Path(resume)
    if no_auto_resume:
        return None
    phase_latest = phase_paths(phase).latest
    if phase_latest.exists():
        return phase_latest
    if allow_root_fallback:
        candidate = paths.CHECKPOINT_DIR / "latest.pt"
        if candidate.exists():
            return candidate
    return None


def _load_meta(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_dtype(meta: dict[str, Any]) -> np.dtype:
    if meta.get("token_dtype") == "uint16-le":
        return np.uint16
    return np.uint8


def _parameter_groups(
    model: torch.nn.Module,
    *,
    weight_decay: float,
    decay_norm_and_bias: bool,
    decay_embeddings: bool,
) -> list[dict[str, Any]] | list[torch.nn.Parameter]:
    """Configure parameter groups for AdamW with optional exclusions."""
    decay_params: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_bias = name.endswith("bias")
        is_norm = "norm" in name.lower()
        is_emb = "tok_emb" in name or "embedding" in name

        if (not decay_norm_and_bias and (is_bias or is_norm)) or (not decay_embeddings and is_emb):
            no_decay.append(param)
        else:
            decay_params.append(param)

    groups: list[dict[str, Any]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})

    return groups if groups else list(model.parameters())


class PretrainSource:
    """Memory-mapped token source that samples random windows from shards."""

    def __init__(self, name: str, *, shards: list[np.memmap], shard_lengths: list[int], T: int):
        self.name = name
        self.shards = shards
        self.shard_lengths = shard_lengths
        self.T = T
        self.total_tokens = sum(shard_lengths)

        if self.total_tokens < T + 1:
            raise ValueError(
                f"source {name} too short for T={T}: need >= {T+1}, got {self.total_tokens}"
            )

        # Precompute shard sampling probabilities based on lengths
        lengths_tensor = torch.tensor(shard_lengths, dtype=torch.float32)
        self._shard_probs = lengths_tensor / lengths_tensor.sum()

    def sample(self, *, device: str, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        # Sample which shard to use
        if len(self.shards) == 1:
            shard_idx = 0
        else:
            shard_idx = torch.multinomial(self._shard_probs, 1, generator=generator).item()

        shard = self.shards[shard_idx]
        shard_len = self.shard_lengths[shard_idx]
        max_start = shard_len - (self.T + 1)

        start_idx = (
            0
            if max_start <= 0
            else torch.randint(max_start + 1, (1,), generator=generator).item()
        )

        window = shard[start_idx : start_idx + self.T + 1]
        window_tensor = torch.from_numpy(np.asarray(window, dtype=np.int64)).to(device)
        x = window_tensor[:-1]
        y = window_tensor[1:]
        return x, y


class PretrainMixture:
    """Mixture sampler across multiple sources with fixed probabilities."""

    def __init__(
        self,
        sources: dict[str, PretrainSource],
        probs: dict[str, float],
    ):
        if set(sources.keys()) != set(probs.keys()):
            raise ValueError("sources and probs keys must match exactly")
        total = sum(probs.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"source probabilities must sum to 1.0, got {total}")

        self.names = sorted(sources.keys())
        self.sources = sources
        self.probs = torch.tensor([probs[name] for name in self.names], dtype=torch.float32)

    def get_batch(
        self, *, B: int, device: str, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        choices = torch.multinomial(self.probs, num_samples=B, replacement=True, generator=generator)
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        for idx in choices:
            src = self.sources[self.names[idx.item()]]
            x, y = src.sample(device=device, generator=generator)
            xs.append(x)
            ys.append(y)
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def _expected_paths(cache_dir: Path, source: str, split: str) -> tuple[Path, Path]:
    """Get expected paths for sharded cache format."""
    source_dir = cache_dir / source
    split_dir = source_dir / split
    meta_path = source_dir / "meta.json"
    return split_dir, meta_path


def _load_source_sharded(
    cache_dir: Path,
    source: str,
    split: str,
    *,
    T: int,
    expected_tokenizer_sha: str | None,
    expected_special_token_ids: dict[str, int] | None,
) -> PretrainSource:
    """Load source from sharded cache format: cache_dir/source/{train,val}/shard_*.bin"""
    split_dir, meta_path = _expected_paths(cache_dir, source, split)

    if not meta_path.exists():
        raise FileNotFoundError(f"missing meta file: {meta_path}")

    meta = _load_meta(meta_path)

    # Validate tokenizer hash
    if expected_tokenizer_sha and meta.get("tokenizer_sha256") not in {expected_tokenizer_sha}:
        raise ValueError(
            f"tokenizer hash mismatch for {source}/{split}: cache has {meta.get('tokenizer_sha256')}, "
            f"expected {expected_tokenizer_sha}"
        )

    # Validate special token ids
    if expected_special_token_ids:
        meta_special = meta.get("special_token_ids")
        if meta_special != expected_special_token_ids:
            raise ValueError(
                f"special token ids mismatch for {source}/{split}: cache has {meta_special}, "
                f"expected {expected_special_token_ids}"
            )

    # Load all shards from the split directory
    if not split_dir.exists():
        raise FileNotFoundError(f"missing split directory: {split_dir}")

    shard_files = sorted(split_dir.glob("shard_*.bin"))
    if not shard_files:
        raise FileNotFoundError(f"no shard files found in {split_dir}")

    dtype = _infer_dtype(meta)

    # Keep shards as separate memmaps - don't concatenate to save memory
    shards: list[np.memmap] = []
    shard_lengths: list[int] = []

    for shard_path in shard_files:
        shard = np.memmap(shard_path, dtype=dtype, mode="r")
        # Only include shards with enough tokens
        if len(shard) >= T + 1:
            shards.append(shard)
            shard_lengths.append(len(shard))

    if not shards:
        raise ValueError(f"no shards in {split_dir} have enough tokens (need >= {T + 1})")

    return PretrainSource(source, shards=shards, shard_lengths=shard_lengths, T=T)


def _load_source(
    cache_dir: Path,
    source: str,
    split: str,
    *,
    T: int,
    expected_tokenizer_sha: str | None,
    expected_special_token_ids: dict[str, int] | None,
) -> PretrainSource:
    """Legacy flat cache loader preserved for compatibility/testing."""
    meta_path = cache_dir / f"{source}_{split}.meta.json"
    tokens_path = cache_dir / f"{source}_{split}.bin"

    if not meta_path.exists():
        raise FileNotFoundError(f"missing meta file: {meta_path}")
    if not tokens_path.exists():
        raise FileNotFoundError(f"missing tokens file: {tokens_path}")

    meta = _load_meta(meta_path)

    if expected_tokenizer_sha and meta.get("tokenizer_sha256") not in {expected_tokenizer_sha}:
        raise ValueError(
            f"tokenizer hash mismatch for {source}/{split}: cache has {meta.get('tokenizer_sha256')}, "
            f"expected {expected_tokenizer_sha}"
        )

    if expected_special_token_ids:
        meta_special = meta.get("special_token_ids")
        if meta_special != expected_special_token_ids:
            raise ValueError(
                f"special token ids mismatch for {source}/{split}: cache has {meta_special}, "
                f"expected {expected_special_token_ids}"
            )

    shard = np.memmap(tokens_path, dtype=_infer_dtype(meta), mode="r")
    if len(shard) < T + 1:
        raise ValueError(f"source {source} too short for T={T}: need >= {T + 1}, got {len(shard)}")

    return PretrainSource(source, shards=[shard], shard_lengths=[len(shard)], T=T)


def _load_sources(
    cache_dir: Path,
    source_names: Iterable[str],
    *,
    split: str,
    T: int,
    expected_tokenizer_sha: str | None,
    expected_special_token_ids: dict[str, int] | None,
) -> dict[str, PretrainSource]:
    """Load sources from sharded cache format."""
    missing: list[str] = []
    for src in source_names:
        split_dir, meta_path = _expected_paths(cache_dir, src, split)
        if not meta_path.exists():
            missing.append(str(meta_path))
        if not split_dir.exists():
            missing.append(str(split_dir))

    if missing:
        raise FileNotFoundError(
            "missing cache files:\n  "
            + "\n  ".join(sorted(missing))
            + "\nexpected format: data/cache/pretrain/{source}/meta.json and {source}/{split}/shard_*.bin"
        )

    sources: dict[str, PretrainSource] = {}
    for src in source_names:
        sources[src] = _load_source_sharded(
            cache_dir,
            src,
            split,
            T=T,
            expected_tokenizer_sha=expected_tokenizer_sha,
            expected_special_token_ids=expected_special_token_ids,
        )
    return sources


def run_pretrain(
    config: PretrainJobConfig | dict[str, Any],
    *,
    device: str | None = None,
    resume_path: str | None = None,
    no_auto_resume: bool = False,
) -> dict[str, Any]:
    """
    Run pretrain phase.

    Returns dict with keys: final_step, best_val_loss, latest_path, best_path, last_val_loss.
    """
    job = config if isinstance(config, PretrainJobConfig) else load_pretrain_job_config(config)
    device = device or get_device()
    paths.ensure_dirs()

    def _save_settings_sidecar(target: Path) -> None:
        if job.resolved_settings_path and Path(job.resolved_settings_path).exists():
            sidecar = target.with_name(target.name + ".resolved_settings.json")
            shutil.copy2(job.resolved_settings_path, sidecar)

    model_cfg = job.model_cfg
    train_cfg = job.train_cfg
    eval_batches = job.eval_batches
    repro = job.resolved.settings.reproducibility
    if repro.torch_num_threads is not None:
        try:
            torch.set_num_threads(repro.torch_num_threads)
        except Exception:
            pass
    if repro.torch_matmul_precision:
        try:
            torch.set_float32_matmul_precision(repro.torch_matmul_precision)  # type: ignore[attr-defined]
        except Exception:
            pass

    settings_meta = {
        "run_id": job.run_id,
        "resolved_settings_path": str(job.resolved_settings_path) if job.resolved_settings_path else None,
        "overrides_source": job.resolved.overrides_source,
        "tokenizer_sha256": job.resolved.tokenizer_sha256,
    }

    model_cfg_saved = {**to_dict(model_cfg), "_raw": job.model_cfg_raw}
    train_cfg_saved = {**to_dict(train_cfg), "_raw": job.train_cfg_raw, "_settings_meta": settings_meta}
    ckpt_paths: PhasePaths = phase_paths("pretrain")

    source_probs = job.sources
    if not source_probs:
        raise ValueError("sources must be a non-empty mapping of source -> probability")
    val_source_name = job.val_source
    cache_dir = job.cache_dir

    validate_token_caches(cache_dir, source_probs.keys(), splits=("train", "val"))
    print(format_data_plan("pretrain", cache_dir, source_probs.keys(), splits=("train", "val")))

    train_sources = _load_sources(
        cache_dir,
        source_probs.keys(),
        split="train",
        T=model_cfg.T,
        expected_tokenizer_sha=job.tokenizer_sha256,
        expected_special_token_ids=job.special_token_ids,
    )
    val_sources = _load_sources(
        cache_dir,
        [val_source_name],
        split="val",
        T=model_cfg.T,
        expected_tokenizer_sha=job.tokenizer_sha256,
        expected_special_token_ids=job.special_token_ids,
    )
    val_source = val_sources[val_source_name]

    mixture = PretrainMixture(train_sources, source_probs)

    resume_ckpt = _resolve_resume_path(
        resume_path, no_auto_resume=no_auto_resume, phase="pretrain", allow_root_fallback=False
    )

    model = GPT(model_cfg).to(device)
    model.activation_checkpointing = train_cfg.activation_checkpointing

    param_groups = _parameter_groups(
        model,
        weight_decay=train_cfg.weight_decay,
        decay_norm_and_bias=train_cfg.decay_norm_and_bias,
        decay_embeddings=train_cfg.decay_embeddings,
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=train_cfg.base_lr,
        betas=(train_cfg.beta1, train_cfg.beta2),
        eps=train_cfg.eps,
        weight_decay=0.0 if isinstance(param_groups, list) else train_cfg.weight_decay,
    )

    start_step = 0
    best_val_loss: float | None = None
    last_val_loss: float | None = None

    torch.manual_seed(train_cfg.seed)

    if resume_ckpt:
        ckpt = load_checkpoint(str(resume_ckpt), device=device)
        assert_model_config_compatible(ckpt["model_cfg"], to_dict(model_cfg))
        model.load_state_dict(ckpt["model_state"])
        if ckpt["optimizer_state"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt["step"]
        best_val_loss = ckpt["best_val_loss"]
        print(f"resumed from {resume_ckpt} at step {start_step}")

    # Training configuration logging
    n_params = sum(p.numel() for p in model.parameters())
    amp_status = f"amp={train_cfg.amp} ({train_cfg.amp_dtype})" if train_cfg.amp else "amp=false"
    effective_batch = train_cfg.B * train_cfg.accum_steps
    print(
        f"pretrain: device={device}, {amp_status}, "
        f"activation_checkpointing={train_cfg.activation_checkpointing}, "
        f"micro_B={train_cfg.B}, accum_steps={train_cfg.accum_steps}, effective_batch={effective_batch}, "
        f"params={n_params:,}"
    )

    train_gen = torch.Generator(device="cpu").manual_seed(train_cfg.seed)
    loss_window: deque[float] = deque(maxlen=50)

    latest_path = ckpt_paths.latest
    best_path = ckpt_paths.best

    total_steps = train_cfg.total_steps

    # Get AMP context (nullcontext if disabled or device is cpu)
    amp_ctx = get_amp_context(device=device, amp_enabled=train_cfg.amp, amp_dtype=train_cfg.amp_dtype)

    for step in range(start_step, total_steps):
        lr = lr_at_step(
            step,
            total_steps,
            base_lr=train_cfg.base_lr,
            warmup_steps=train_cfg.warmup_steps,
            min_lr=train_cfg.min_lr,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(train_cfg.accum_steps):
            x, y = mixture.get_batch(B=train_cfg.B, device=device, generator=train_gen)
            with amp_ctx:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            (loss / train_cfg.accum_steps).backward()
            loss_accum += loss.item()

        nn_utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        avg_loss = loss_accum / max(1, train_cfg.accum_steps)
        loss_window.append(avg_loss)

        step_num = step + 1
        if step_num % train_cfg.log_every == 0:
            ma_loss = sum(loss_window) / len(loss_window)
            lv = f"{last_val_loss:.4f}" if last_val_loss is not None else "n/a"
            print(
                f"[pretrain] step {step_num}/{total_steps} | lr {lr:.6f} | "
                f"train_loss {ma_loss:.4f} | last_val {lv}"
            )

        if step_num % train_cfg.eval_every == 0:
            eval_gen = torch.Generator(device="cpu").manual_seed(train_cfg.seed)
            eval_B = train_cfg.micro_B_eval or train_cfg.B

            def val_batch_fn() -> tuple[torch.Tensor, torch.Tensor]:
                xs: list[torch.Tensor] = []
                ys: list[torch.Tensor] = []
                for _ in range(eval_B):
                    x_single, y_single = val_source.sample(device=device, generator=eval_gen)
                    xs.append(x_single)
                    ys.append(y_single)
                return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

            last_val_loss = evaluate_pretrain(
                model, batch_fn=val_batch_fn, eval_batches=eval_batches
            )
            print(f"[pretrain] val_{val_source_name}_loss {last_val_loss:.4f}")
            if train_cfg.save_best and (best_val_loss is None or last_val_loss < best_val_loss):
                best_val_loss = last_val_loss
                save_checkpoint(
                    best_path,
                    model_cfg=model_cfg_saved,
                    train_cfg=train_cfg_saved,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    step=step_num,
                    best_val_loss=best_val_loss,
                )
                shutil.copy2(best_path, paths.CHECKPOINT_DIR / "best.pt")
                _save_settings_sidecar(best_path)
                _save_settings_sidecar(paths.CHECKPOINT_DIR / "best.pt")
                print(f"[pretrain] new best checkpoint at step {step_num} (val_loss={best_val_loss:.4f})")

        if step_num % train_cfg.ckpt_every == 0:
            periodic_path = ckpt_paths.step_dir / f"step_{step_num:07d}.pt"
            save_checkpoint(
                periodic_path,
                model_cfg=model_cfg_saved,
                train_cfg=train_cfg_saved,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                step=step_num,
                best_val_loss=best_val_loss,
            )
            save_checkpoint(
                latest_path,
                model_cfg=model_cfg_saved,
                train_cfg=train_cfg_saved,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                step=step_num,
                best_val_loss=best_val_loss,
            )
            shutil.copy2(latest_path, paths.CHECKPOINT_DIR / "latest.pt")
            _save_settings_sidecar(periodic_path)
            _save_settings_sidecar(latest_path)
            _save_settings_sidecar(paths.CHECKPOINT_DIR / "latest.pt")
    # final latest
    save_checkpoint(
        latest_path,
        model_cfg=model_cfg_saved,
        train_cfg=train_cfg_saved,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        step=total_steps,
        best_val_loss=best_val_loss,
    )
    shutil.copy2(latest_path, paths.CHECKPOINT_DIR / "latest.pt")
    _save_settings_sidecar(latest_path)
    _save_settings_sidecar(paths.CHECKPOINT_DIR / "latest.pt")

    return {
        "final_step": total_steps,
        "best_val_loss": best_val_loss,
        "last_val_loss": last_val_loss,
        "latest_path": latest_path,
        "best_path": best_path,
    }

