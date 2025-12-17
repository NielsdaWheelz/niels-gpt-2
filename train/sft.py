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
from niels_gpt.cache.sft_dataset import SFTExampleDataset
from niels_gpt.config import ModelConfig, load_json
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
from train.eval import evaluate_pretrain, evaluate_sft
from train.pretrain import (
    PretrainSource,
    _load_meta,
    _load_train_cfg,
    _resolve_resume_path,
    _load_source as _load_stream_source,
)


def _default_sft_sources() -> dict[str, float]:
    return {"dolly": 0.5, "oasst1": 0.5}


def _load_model_cfg_from_config(cfg: dict[str, Any]) -> tuple[ModelConfig, dict[str, Any]]:
    if "model_cfg" in cfg:
        model_cfg_dict = cfg["model_cfg"]
    elif "model_cfg_path" in cfg:
        model_cfg_dict = load_json(cfg["model_cfg_path"])
    else:
        raise ValueError("config must include model_cfg or model_cfg_path")
    model_cfg = ModelConfig(**model_cfg_dict)
    return model_cfg, model_cfg_dict


def _expected_sft_paths(cache_dir: Path, source: str, split: str) -> tuple[Path, Path, Path]:
    tokens = cache_dir / f"{source}_{split}.bin"
    idx = cache_dir / f"{source}_{split}.idx.npy"
    meta = cache_dir / f"{source}_{split}.meta.json"
    return tokens, idx, meta


def _load_sft_source(
    cache_dir: Path, source: str, split: str, *, T: int, device: str
) -> SFTExampleDataset:
    tokens_path, idx_path, meta_path = _expected_sft_paths(cache_dir, source, split)
    meta = _load_meta(meta_path)
    special_ids = meta.get("special_token_ids", {})
    asst_id = special_ids.get("asst")
    eot_id = special_ids.get("eot")
    if eot_id is None or asst_id is None:
        raise ValueError(f"meta for {source}_{split} missing special_token_ids.asst/eot")
    if not idx_path.exists():
        # fall back to a single-example offset covering the whole tokens file
        tokens_mem = np.memmap(tokens_path, dtype=np.uint16, mode="r")
        offsets = np.asarray([0], dtype=np.int64) if len(tokens_mem) else np.asarray([], dtype=np.int64)
        np.save(idx_path, offsets)
        print(f"[sft] generated missing idx file for {source}_{split} at {idx_path}")
    # ensure files are readable; SFTExampleDataset handles length validation
    return SFTExampleDataset(
        str(tokens_path),
        str(idx_path),
        T=T,
        device=device,
        eot_id=int(eot_id),
        asst_id=int(asst_id),
    )


def _load_sft_sources(
    cache_dir: Path,
    source_names: Iterable[str],
    *,
    split: str,
    T: int,
    device: str,
    allow_missing_idx: bool,
) -> dict[str, SFTExampleDataset]:
    missing: list[str] = []
    for src in source_names:
        tokens, idx, meta = _expected_sft_paths(cache_dir, src, split)
        if not tokens.exists():
            missing.append(str(tokens))
        if not meta.exists():
            missing.append(str(meta))
    if missing:
        raise FileNotFoundError(
            "missing SFT cache files:\n  "
            + "\n  ".join(sorted(missing))
            + "\nexpected naming: data/cache/sft/{source}_{split}.bin + .meta.json (+ optional .idx.npy)"
        )

    datasets: dict[str, SFTExampleDataset] = {}
    for src in source_names:
        tokens, idx, _ = _expected_sft_paths(cache_dir, src, split)
        if not idx.exists() and not allow_missing_idx:
            raise FileNotFoundError(
                f"missing idx file for {src}_{split}: {idx}\n"
                "set allow_missing_idx=true to auto-generate a trivial idx (may harm training)"
            )
        datasets[src] = _load_sft_source(cache_dir, src, split, T=T, device=device)
    return datasets


class SFTMixture:
    """Sample batches from multiple SFT datasets with fixed probs."""

    def __init__(self, datasets: dict[str, SFTExampleDataset], probs: dict[str, float]):
        if set(datasets.keys()) != set(probs.keys()):
            raise ValueError("datasets and probs keys must match exactly")
        total = sum(probs.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"sft source probabilities must sum to 1.0, got {total}")
        self.names = sorted(datasets.keys())
        self.datasets = datasets
        self.probs = torch.tensor([probs[name] for name in self.names], dtype=torch.float32)

    def get_batch(
        self, *, B: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        choices = torch.multinomial(self.probs, num_samples=B, replacement=True, generator=generator)
        xs: list[torch.Tensor] = []
        y_masked_list: list[torch.Tensor] = []
        for idx in choices:
            ds = self.datasets[self.names[idx.item()]]
            x, _, y_masked = ds.get_batch(B=1, generator=generator)
            xs.append(x[0])
            y_masked_list.append(y_masked[0])
        return torch.stack(xs, dim=0), torch.stack(y_masked_list, dim=0)


def _load_wiki_val(cache_dir: Path, *, T: int) -> PretrainSource:
    return _load_stream_source(cache_dir, "wiki", "val", T=T)


def run_sft(
    config: dict[str, Any],
    *,
    device: str | None = None,
    resume_path: str | None = None,
    no_auto_resume: bool = False,
    init_model_path: str | None = None,
) -> dict[str, Any]:
    """
    Run SFT phase. If init_model_path is provided, model weights are loaded from it
    and optimizer state is reset.
    """
    device = device or get_device()
    paths.ensure_dirs()

    model_cfg, model_cfg_dict = _load_model_cfg_from_config(config)
    train_cfg, train_cfg_dict, eval_batches = _load_train_cfg(config.get("train_cfg", {}))
    model_cfg_saved = {**model_cfg_dict, "_raw": config.get("model_cfg", model_cfg_dict)}
    train_cfg_saved = {**train_cfg_dict, "_raw": config.get("train_cfg", {})}
    ckpt_paths: PhasePaths = phase_paths("sft")

    source_probs = config.get("sft_sources", _default_sft_sources())
    if not source_probs:
        raise ValueError("sft_sources must be a non-empty mapping of source -> probability")
    val_source_choice = config.get("val_source", "wiki")
    allow_missing_idx = bool(config.get("allow_missing_idx", False))
    cache_dir = Path(config.get("cache_dir", paths.REPO_ROOT / "data" / "cache" / "sft")).resolve()
    streams_cache_dir = Path(config.get("streams_cache_dir", paths.REPO_ROOT / "data" / "cache" / "streams")).resolve()

    sft_train = _load_sft_sources(
        cache_dir,
        source_probs.keys(),
        split="train",
        T=model_cfg.T,
        device=device,
        allow_missing_idx=allow_missing_idx,
    )
    mixture = SFTMixture(sft_train, source_probs)

    val_sft = None
    wiki_val_source = None
    if val_source_choice == "sft":
        val_sft = _load_sft_sources(
            cache_dir,
            source_probs.keys(),
            split="val",
            T=model_cfg.T,
            device=device,
            allow_missing_idx=allow_missing_idx,
        )
        val_mixture = SFTMixture(val_sft, {k: v / sum(source_probs.values()) for k, v in source_probs.items()})
    else:
        wiki_val_source = _load_wiki_val(streams_cache_dir, T=model_cfg.T)

    resume_ckpt = _resolve_resume_path(resume_path, no_auto_resume=no_auto_resume, phase="sft")

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

    if init_model_path:
        ckpt = load_checkpoint(init_model_path, device=device)
        assert_model_config_compatible(ckpt["model_cfg"], model_cfg_dict)
        model.load_state_dict(ckpt["model_state"])
        best_val_loss = None
        start_step = 0
        print(f"sft init from {init_model_path}")
    elif resume_ckpt:
        ckpt = load_checkpoint(str(resume_ckpt), device=device)
        assert_model_config_compatible(ckpt["model_cfg"], model_cfg_dict)
        model.load_state_dict(ckpt["model_state"])
        if ckpt["optimizer_state"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt["step"]
        best_val_loss = ckpt["best_val_loss"]
        print(f"resumed sft from {resume_ckpt} at step {start_step}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"sft: device={device}, params={n_params:,}")

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
            x, y_masked = mixture.get_batch(B=train_cfg.B, generator=train_gen)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_masked.view(-1),
                ignore_index=-100,
            )
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
                f"[sft] step {step_num}/{total_steps} | lr {lr:.6f} | train_loss {ma_loss:.4f} | last_val {lv}"
            )

        if step_num % train_cfg.eval_every == 0:
            if val_source_choice == "sft":
                eval_gen = torch.Generator(device="cpu").manual_seed(train_cfg.seed)

                def val_batch_fn():
                    return val_mixture.get_batch(B=train_cfg.B, generator=eval_gen)

                last_val_loss = evaluate_sft(
                    model, batch_fn=val_batch_fn, eval_batches=eval_batches
                )
            else:
                eval_gen = torch.Generator(device="cpu").manual_seed(train_cfg.seed)

                def val_batch_fn():
                    xs = []
                    ys = []
                    for _ in range(train_cfg.B):
                        x_single, y_single = wiki_val_source.sample(device=device, generator=eval_gen)
                        xs.append(x_single)
                        ys.append(y_single)
                    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

                last_val_loss = evaluate_pretrain(
                    model, batch_fn=val_batch_fn, eval_batches=eval_batches
                )
            print(f"[sft] val_{val_source_choice}_loss {last_val_loss:.4f}")
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
                print(f"[sft] new best checkpoint at step {step_num} (val_loss={best_val_loss:.4f})")

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

