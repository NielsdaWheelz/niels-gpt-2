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
from niels_gpt.settings import default_settings


_SETTINGS = default_settings()
_BENCH = _SETTINGS.benchmark
_PRETRAIN = _SETTINGS.training.pretrain


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


def capture_activation_dtypes(model: GPT, x: torch.Tensor, amp_ctx, device: str) -> dict[str, str]:
    """Capture actual activation dtypes during forward pass."""
    dtypes = {}

    # Hook to capture attention intermediate
    attn_qkv_dtype = [None]
    def attn_hook(module, input, output):
        attn_qkv_dtype[0] = output.dtype

    # Hook to capture MLP intermediate
    mlp_intermediate_dtype = [None]
    def mlp_hook(module, input, output):
        mlp_intermediate_dtype[0] = output.dtype

    # Register hooks
    attn_handle = model.blocks[0].attn.qkv.register_forward_hook(attn_hook)
    mlp_handle = model.blocks[0].mlp.fc_a.register_forward_hook(mlp_hook)

    # Run forward pass
    with amp_ctx:
        # Check autocast status for the specific device
        if device == "mps":
            dtypes["autocast_enabled"] = str(torch.is_autocast_enabled("mps"))
        elif device == "cuda":
            dtypes["autocast_enabled"] = str(torch.is_autocast_enabled("cuda"))
        else:
            dtypes["autocast_enabled"] = str(torch.is_autocast_enabled())

        logits = model(x)
        dtypes["logits"] = str(logits.dtype)
        dtypes["attn_qkv"] = str(attn_qkv_dtype[0]) if attn_qkv_dtype[0] is not None else "not_captured"
        dtypes["mlp_intermediate"] = str(mlp_intermediate_dtype[0]) if mlp_intermediate_dtype[0] is not None else "not_captured"

    # Remove hooks
    attn_handle.remove()
    mlp_handle.remove()

    # Add parameter dtype for reference
    dtypes["embed_param"] = str(model.tok_emb.weight.dtype)

    return dtypes


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
    lr = trial_cfg.get("lr", _BENCH.lr)

    repro = _SETTINGS.reproducibility
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

    # Set seed
    torch.manual_seed(seed)

    # Build model config
    required_fields = ("V", "T", "C", "L", "H", "d_ff", "dropout", "rope_theta")
    missing = [f for f in required_fields if f not in model_cfg_dict]
    if missing:
        raise ValueError(f"model config missing required fields: {missing}")

    model_cfg = ModelConfig(
        V=model_cfg_dict["V"],
        T=model_cfg_dict["T"],
        C=model_cfg_dict["C"],
        L=model_cfg_dict["L"],
        H=model_cfg_dict["H"],
        d_ff=model_cfg_dict["d_ff"],
        dropout=model_cfg_dict["dropout"],
        rope_theta=model_cfg_dict["rope_theta"],
    )

    # Instantiate model
    model = GPT(model_cfg).to(device)
    model.activation_checkpointing = activation_checkpointing
    model.train()

    # Count parameters
    params = sum(p.numel() for p in model.parameters())

    # Build optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=_PRETRAIN.betas,
        eps=_PRETRAIN.eps,
        weight_decay=_PRETRAIN.weight_decay,
    )

    # Get AMP context
    amp_ctx = get_amp_context(device, amp, amp_dtype)

    # Capture dtype summary during a forward pass
    x_probe = torch.randint(0, model_cfg.V, (micro_B, model_cfg.T), device=device, dtype=torch.long)
    dtype_summary = capture_activation_dtypes(model, x_probe, amp_ctx, device)

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
