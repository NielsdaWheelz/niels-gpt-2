#!/usr/bin/env python3
"""
bench_trial.py: Run ONE benchmark trial in isolation.

Usage:
    python tools/bench_trial.py --trial '{"device":"cpu","model":{...},"amp":false,...}'
    python tools/bench_trial.py --trial_json path/to/trial.json

Outputs a single JSON line to stdout on success/OOM.
"""
import argparse
import contextlib
import json
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

from niels_gpt.config import ModelConfig
from niels_gpt.model.gpt import GPT


def get_amp_context(device: str, amp: bool, amp_dtype: str) -> Any:
    """Return autocast context for MPS when amp is enabled, else nullcontext."""
    if not amp or device != "mps":
        return contextlib.nullcontext()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[amp_dtype]
    return torch.autocast(device_type="mps", dtype=dtype)


def synchronize_device(device: str) -> None:
    """Synchronize device before timing to avoid async timing lies."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def get_dtype_summary(model: GPT, device: str) -> dict[str, str]:
    """Extract dtype summary from model parameters."""
    return {
        "embed": str(model.tok_emb.weight.dtype),
        "qk": str(model.blocks[0].attn.qkv.weight.dtype),
        "logits": "computed",
        "loss": "computed",
    }


def get_memory_mb(device: str) -> dict[str, float] | None:
    """Best-effort memory measurement."""
    try:
        if device == "mps":
            if hasattr(torch.mps, "current_allocated_memory"):
                allocated = torch.mps.current_allocated_memory() / (1024**2)
                return {"allocated_mb": round(allocated, 2)}
            if hasattr(torch.mps, "driver_allocated_memory"):
                allocated = torch.mps.driver_allocated_memory() / (1024**2)
                return {"driver_allocated_mb": round(allocated, 2)}
        elif device == "cuda":
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            return {"allocated_mb": round(allocated, 2), "reserved_mb": round(reserved, 2)}
    except Exception:
        pass

    try:
        import psutil
        process = psutil.Process()
        rss = process.memory_info().rss / (1024**2)
        return {"rss_mb": round(rss, 2)}
    except ImportError:
        pass

    return None


def is_oom_error(e: Exception) -> bool:
    """Check if exception is an OOM error."""
    err_str = str(e).lower()
    oom_keywords = ["out of memory", "oom", "mps", "alloc"]
    return any(kw in err_str for kw in oom_keywords)


def run_trial(trial_cfg: dict) -> dict:
    """Run a single benchmark trial and return results as dict."""
    device = trial_cfg["device"]
    model_cfg_dict = trial_cfg["model"]
    amp = trial_cfg["amp"]
    amp_dtype = trial_cfg.get("amp_dtype", "fp16")
    activation_checkpointing = trial_cfg["activation_checkpointing"]
    micro_B = trial_cfg["micro_B"]
    steps_warmup = trial_cfg["steps_warmup"]
    steps_measure = trial_cfg["steps_measure"]
    seed = trial_cfg["seed"]
    lr = trial_cfg.get("lr", 3e-4)

    # Set seed
    torch.manual_seed(seed)

    # Build model config
    model_cfg = ModelConfig(
        V=model_cfg_dict["V"],
        T=model_cfg_dict.get("T", 1024),
        C=model_cfg_dict.get("C", 512),
        L=model_cfg_dict.get("L", 8),
        H=model_cfg_dict.get("H", 8),
        d_ff=model_cfg_dict.get("d_ff", 1536),
        dropout=model_cfg_dict.get("dropout", 0.1),
        rope_theta=model_cfg_dict.get("rope_theta", 10000.0),
    )

    # Instantiate model
    model = GPT(model_cfg).to(device)
    model.activation_checkpointing = activation_checkpointing
    model.train()

    # Count parameters
    params = sum(p.numel() for p in model.parameters())

    # Get dtype summary
    dtype_summary = get_dtype_summary(model, device)

    # Build optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    # Get AMP context
    amp_ctx = get_amp_context(device, amp, amp_dtype)

    # Warmup steps
    for _ in range(steps_warmup):
        x = torch.randint(0, model_cfg.V, (micro_B, model_cfg.T), device=device, dtype=torch.long)
        y = torch.randint(0, model_cfg.V, (micro_B, model_cfg.T), device=device, dtype=torch.long)

        with amp_ctx:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Measured steps
    synchronize_device(device)
    times = []

    for _ in range(steps_measure):
        x = torch.randint(0, model_cfg.V, (micro_B, model_cfg.T), device=device, dtype=torch.long)
        y = torch.randint(0, model_cfg.V, (micro_B, model_cfg.T), device=device, dtype=torch.long)

        t0 = time.perf_counter()

        with amp_ctx:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        synchronize_device(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    # Compute metrics
    sec_per_step = sum(times) / len(times)
    ms_per_step = sec_per_step * 1000
    tokens_per_sec = (micro_B * model_cfg.T) / sec_per_step

    # Memory measurement
    memory = get_memory_mb(device)

    result = {
        "status": "ok",
        "device": device,
        "model": {
            "V": model_cfg.V,
            "T": model_cfg.T,
            "C": model_cfg.C,
            "L": model_cfg.L,
            "H": model_cfg.H,
            "d_ff": model_cfg.d_ff,
            "dropout": model_cfg.dropout,
            "rope_theta": model_cfg.rope_theta,
        },
        "amp": amp,
        "amp_dtype": amp_dtype,
        "activation_checkpointing": activation_checkpointing,
        "micro_B": micro_B,
        "params": params,
        "ms_per_step": round(ms_per_step, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "dtype_summary": dtype_summary,
    }

    if memory is not None:
        result["memory"] = memory

    return result


def main():
    parser = argparse.ArgumentParser(description="Run a single benchmark trial")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trial", type=str, help="Trial config as JSON string")
    group.add_argument("--trial_json", type=str, help="Path to trial config JSON file")

    args = parser.parse_args()

    if args.trial:
        trial_cfg = json.loads(args.trial)
    else:
        with open(args.trial_json, "r") as f:
            trial_cfg = json.load(f)

    try:
        result = run_trial(trial_cfg)
        print(json.dumps(result))
        sys.exit(0)
    except RuntimeError as e:
        if is_oom_error(e):
            result = {
                "status": "oom",
                "device": trial_cfg["device"],
                "model": trial_cfg["model"],
                "amp": trial_cfg["amp"],
                "amp_dtype": trial_cfg.get("amp_dtype", "fp16"),
                "activation_checkpointing": trial_cfg["activation_checkpointing"],
                "micro_B": trial_cfg["micro_B"],
            }
            print(json.dumps(result))
            sys.exit(0)
        else:
            raise


if __name__ == "__main__":
    main()
