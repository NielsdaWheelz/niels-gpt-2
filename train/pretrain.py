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


class PretrainSource:
    """Memory-mapped token source that samples random windows."""

    def __init__(self, name: str, *, tokens: np.memmap, T: int):
        self.name = name
        self.tokens = tokens
        self.T = T
        if len(self.tokens) < T + 1:
            raise ValueError(
                f"source {name} too short for T={T}: need >= {T+1}, got {len(self.tokens)}"
            )
        self._max_start = len(self.tokens) - (T + 1)

    def sample(self, *, device: str, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        start_idx = (
            0
            if self._max_start <= 0
            else torch.randint(self._max_start + 1, (1,), generator=generator).item()
        )
        window = self.tokens[start_idx : start_idx + self.T + 1]
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
    bin_path = cache_dir / f"{source}_{split}.bin"
    meta_path = cache_dir / f"{source}_{split}.meta.json"
    return bin_path, meta_path


def _load_source(cache_dir: Path, source: str, split: str, *, T: int) -> PretrainSource:
    bin_path, meta_path = _expected_paths(cache_dir, source, split)
    meta = _load_meta(meta_path)
    dtype = _infer_dtype(meta)
    tokens = np.memmap(bin_path, dtype=dtype, mode="r")
    return PretrainSource(source, tokens=tokens, T=T)


def _load_sources(
    cache_dir: Path, source_names: Iterable[str], *, split: str, T: int
) -> dict[str, PretrainSource]:
    missing: list[str] = []
    for src in source_names:
        bin_path, meta_path = _expected_paths(cache_dir, src, split)
        if not bin_path.exists():
            missing.append(str(bin_path))
        if not meta_path.exists():
            missing.append(str(meta_path))
    if missing:
        raise FileNotFoundError(
            "missing cache files:\n  "
            + "\n  ".join(sorted(missing))
            + "\nexpected naming: data/cache/streams/{source}_{split}.bin + .meta.json"
        )

    sources: dict[str, PretrainSource] = {}
    for src in source_names:
        sources[src] = _load_source(cache_dir, src, split, T=T)
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

    model_cfg = job.model_cfg
    train_cfg = job.train_cfg
    eval_batches = job.eval_batches
    model_cfg_saved = {**to_dict(model_cfg), "_raw": job.model_cfg_raw}
    train_cfg_saved = {**to_dict(train_cfg), "_raw": job.train_cfg_raw}
    ckpt_paths: PhasePaths = phase_paths("pretrain")

    source_probs = job.sources
    if not source_probs:
        raise ValueError("sources must be a non-empty mapping of source -> probability")
    val_source_name = job.val_source
    cache_dir = job.cache_dir

    train_sources = _load_sources(cache_dir, source_probs.keys(), split="train", T=model_cfg.T)
    val_sources = _load_sources(cache_dir, [val_source_name], split="val", T=model_cfg.T)
    val_source = val_sources[val_source_name]

    mixture = PretrainMixture(train_sources, source_probs)

    resume_ckpt = _resolve_resume_path(
        resume_path, no_auto_resume=no_auto_resume, phase="pretrain", allow_root_fallback=False
    )

    model = GPT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.base_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    start_step = 0
    best_val_loss: float | None = None
    last_val_loss: float | None = None

    if resume_ckpt:
        ckpt = load_checkpoint(str(resume_ckpt), device=device)
        assert_model_config_compatible(ckpt["model_cfg"], to_dict(model_cfg))
        model.load_state_dict(ckpt["model_state"])
        if ckpt["optimizer_state"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt["step"]
        best_val_loss = ckpt["best_val_loss"]
        print(f"resumed from {resume_ckpt} at step {start_step}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"pretrain: device={device}, params={n_params:,}")

    train_gen = torch.Generator(device="cpu").manual_seed(train_cfg.seed)
    loss_window: deque[float] = deque(maxlen=50)

    latest_path = ckpt_paths.latest
    best_path = ckpt_paths.best

    total_steps = train_cfg.total_steps

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

            def val_batch_fn() -> tuple[torch.Tensor, torch.Tensor]:
                xs: list[torch.Tensor] = []
                ys: list[torch.Tensor] = []
                for _ in range(train_cfg.B):
                    x_single, y_single = val_source.sample(device=device, generator=eval_gen)
                    xs.append(x_single)
                    ys.append(y_single)
                return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

            last_val_loss = evaluate_pretrain(
                model, batch_fn=val_batch_fn, eval_batches=eval_batches
            )
            print(f"[pretrain] val_{val_source_name}_loss {last_val_loss:.4f}")
            if best_val_loss is None or last_val_loss < best_val_loss:
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

    return {
        "final_step": total_steps,
        "best_val_loss": best_val_loss,
        "last_val_loss": last_val_loss,
        "latest_path": latest_path,
        "best_path": best_path,
    }

