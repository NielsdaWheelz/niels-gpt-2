"""Benchmark AMP and activation checkpointing performance and memory usage.

Run this on your MacBook Air M4 to measure:
- Throughput (tokens/sec) for each config
- Memory usage (if psutil available)
- Training speed impact

Usage:
    python tools/benchmark_amp_checkpointing.py --device mps --steps 50
    python tools/benchmark_amp_checkpointing.py --device cpu --steps 20
"""

import argparse
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from niels_gpt.config import ModelConfig
from niels_gpt.model.gpt import GPT
from train.amp_utils import get_amp_context

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_memory_mb():
    """Get current process memory usage in MB."""
    if not HAS_PSUTIL:
        return None
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def benchmark_config(
    *,
    device: str,
    amp: bool,
    amp_dtype: str,
    activation_checkpointing: bool,
    B: int,
    T: int,
    steps: int,
    cfg: ModelConfig,
):
    """Run benchmark for a specific configuration."""
    print(f"\n{'='*80}")
    amp_desc = f"amp={amp} dtype={amp_dtype}" if amp else "amp=False (fp32)"
    print(f"Config: device={device}, {amp_desc}, activation_checkpointing={activation_checkpointing}")
    print(f"Batch: B={B}, T={T}, steps={steps}")
    print(f"Model: V={cfg.V}, C={cfg.C}, L={cfg.L}, H={cfg.H}")
    print(f"{'='*80}")

    # Create model
    model = GPT(cfg).to(device)
    model.activation_checkpointing = activation_checkpointing
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Get AMP context
    amp_ctx = get_amp_context(device=device, amp_enabled=amp, amp_dtype=amp_dtype)

    # Warmup (not timed)
    print("Warmup...")
    for i in range(3):
        x = torch.randint(0, cfg.V, (B, T), device=device)
        y = torch.randint(0, cfg.V, (B, T), device=device)
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, cfg.V), y.view(-1))
        loss.backward()
        optimizer.step()

        # After first warmup step, inspect actual dtypes
        if i == 0:
            print(f"\nActual dtypes observed in forward pass:")
            print(f"  Input (x):           {x.dtype}")
            # Peek inside model to check intermediate tensors
            model.eval()
            with torch.no_grad():
                with amp_ctx:
                    x_sample = torch.randint(0, cfg.V, (2, T), device=device)
                    emb = model.tok_emb(x_sample)
                    print(f"  Embedding output:    {emb.dtype}")
                    # Forward through first block to check q/k/v
                    h = model.drop(emb)
                    block = model.blocks[0]
                    h_norm = block.ln1(h)
                    qkv = block.attn.qkv(h_norm)
                    B_sample, T_sample, _ = qkv.shape
                    qkv = qkv.reshape(B_sample, T_sample, 3, cfg.C)
                    q, k, v = qkv.unbind(dim=2)
                    print(f"  Query/Key/Value:     {q.dtype}")
                    # Full forward to check logits
                    logits_sample = model(x_sample)
                    print(f"  Logits output:       {logits_sample.dtype}")
                    loss_sample = F.cross_entropy(logits_sample.view(-1, cfg.V), x_sample.view(-1))
                    print(f"  Loss:                {loss_sample.dtype}")
            model.train()

    if device == "mps":
        torch.mps.synchronize()

    # Measure baseline memory
    mem_start = get_memory_mb()
    if mem_start:
        print(f"Memory before benchmark: {mem_start:.1f} MB")

    # Timed benchmark
    print(f"Running {steps} steps...")
    start_time = time.time()

    for step in range(steps):
        x = torch.randint(0, cfg.V, (B, T), device=device)
        y = torch.randint(0, cfg.V, (B, T), device=device)

        optimizer.zero_grad(set_to_none=True)

        with amp_ctx:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, cfg.V), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 10 == 0:
            print(f"  Step {step + 1}/{steps}, loss={loss.item():.4f}")

    if device == "mps":
        torch.mps.synchronize()

    end_time = time.time()
    elapsed = end_time - start_time

    # Measure final memory
    mem_end = get_memory_mb()
    if mem_end:
        print(f"Memory after benchmark: {mem_end:.1f} MB")
        print(f"Memory delta: {mem_end - mem_start:.1f} MB")

    # Calculate metrics
    total_tokens = steps * B * T
    tokens_per_sec = total_tokens / elapsed
    ms_per_step = (elapsed / steps) * 1000

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Time per step: {ms_per_step:.1f}ms")
    print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
    print(f"  Loss (final): {loss.item():.4f}")

    return {
        "elapsed": elapsed,
        "tokens_per_sec": tokens_per_sec,
        "ms_per_step": ms_per_step,
        "memory_mb": mem_end if mem_end else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark AMP and activation checkpointing")
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps", help="Device to use")
    parser.add_argument("--steps", type=int, default=50, help="Number of training steps")
    parser.add_argument("--B", type=int, default=16, help="Batch size")
    parser.add_argument("--T", type=int, default=128, help="Sequence length")
    args = parser.parse_args()

    # Model config (small enough for CPU, but large enough to show differences)
    cfg = ModelConfig(
        V=256,
        T=args.T,
        C=256,
        L=4,
        H=4,
        d_ff=512,
        dropout=0.1,
    )

    configs = [
        # Baseline: no AMP (fp32), no checkpointing
        # Note: amp_dtype is ignored when amp=False, actual dtype will be fp32
        {"amp": False, "amp_dtype": "fp16", "activation_checkpointing": False, "label": "Baseline (fp32)"},
    ]

    if args.device == "mps":
        # Add MPS-specific configs
        configs.extend([
            {"amp": True, "amp_dtype": "fp16", "activation_checkpointing": False, "label": "AMP fp16 only"},
            {"amp": False, "amp_dtype": "fp16", "activation_checkpointing": True, "label": "Checkpointing only (fp32)"},
            {"amp": True, "amp_dtype": "fp16", "activation_checkpointing": True, "label": "AMP fp16 + Checkpointing"},
        ])
    else:
        # CPU: AMP is auto-disabled, just test checkpointing
        configs.append(
            {"amp": False, "amp_dtype": "fp16", "activation_checkpointing": True, "label": "Checkpointing only (CPU fp32)"}
        )

    results = {}
    for config in configs:
        label = config.pop("label")
        try:
            result = benchmark_config(
                device=args.device,
                B=args.B,
                T=args.T,
                steps=args.steps,
                cfg=cfg,
                **config,
            )
            results[label] = result
        except Exception as e:
            print(f"\nERROR in {label}: {e}")
            results[label] = None

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Configuration':<40} {'Tokens/sec':>12} {'ms/step':>10} {'Memory MB':>12}")
    print(f"{'-'*80}")

    baseline = None
    for label, result in results.items():
        if result is None:
            print(f"{label:<40} {'ERROR':>12} {'-':>10} {'-':>12}")
            continue

        tokens_str = f"{result['tokens_per_sec']:.0f}"
        ms_str = f"{result['ms_per_step']:.1f}"
        mem_str = f"{result['memory_mb']:.1f}" if result['memory_mb'] else "N/A"

        print(f"{label:<40} {tokens_str:>12} {ms_str:>10} {mem_str:>12}")

        if baseline is None:
            baseline = result

    print(f"{'-'*80}")

    if baseline:
        print("\nRelative to baseline:")
        for label, result in results.items():
            if result is None or result == baseline:
                continue
            speedup = result['tokens_per_sec'] / baseline['tokens_per_sec']
            slowdown_pct = (1 - speedup) * 100
            print(f"  {label}: {speedup:.2f}x throughput ({slowdown_pct:+.1f}%)")

    print(f"\nNote: Install psutil for memory measurements: pip install psutil")


if __name__ == "__main__":
    main()
