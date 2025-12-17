"""Find max batch size before OOM for production model configs.

Benchmarks the actual v1 target configs to measure real memory savings from
AMP and activation checkpointing.

Usage:
    python tools/benchmark_oom_boundary.py --device mps
"""

import argparse
import gc
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from niels_gpt.config import ModelConfig
from niels_gpt.model.gpt import GPT
from niels_gpt.settings import default_settings
from train.amp_utils import get_amp_context

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("WARNING: psutil not installed, memory measurements will be unavailable")
    print("Install with: pip install psutil")


def get_peak_memory_mb():
    """Get peak memory usage in MB."""
    if not HAS_PSUTIL:
        return None
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def reset_peak_memory():
    """Reset peak memory tracking."""
    if HAS_PSUTIL:
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        # Force a new measurement by triggering GC
        gc.collect()


@dataclass
class BenchResult:
    max_B: int
    tokens_per_sec: float
    peak_memory_mb: float | None
    oom_at_B: int | None  # What B caused OOM


def find_max_batch_size(
    *,
    device: str,
    cfg: ModelConfig,
    T: int,
    amp: bool,
    amp_dtype: str,
    activation_checkpointing: bool,
    optimizer_cfg,
    max_B_search: int = 128,
    steps: int = 5,
) -> BenchResult:
    """Binary search to find max batch size before OOM."""

    print(f"\nSearching for max batch size...")
    print(f"  Config: C={cfg.C}, L={cfg.L}, T={T}")
    print(f"  AMP: {amp} ({amp_dtype if amp else 'N/A'}), Checkpointing: {activation_checkpointing}")

    # Binary search for max B
    low, high = 1, max_B_search
    max_working_B = None
    oom_B = None

    while low <= high:
        mid = (low + high) // 2
        print(f"  Trying B={mid}...", end=" ", flush=True)

        reset_peak_memory()

        try:
            # Try to run a few steps at this batch size
            model = GPT(cfg).to(device)
            model.activation_checkpointing = activation_checkpointing
            model.train()

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=optimizer_cfg.base_lr,
                betas=optimizer_cfg.betas,
                weight_decay=optimizer_cfg.weight_decay,
                eps=optimizer_cfg.eps,
            )
            amp_ctx = get_amp_context(device=device, amp_enabled=amp, amp_dtype=amp_dtype)

            # Run 3 warmup steps to trigger peak memory
            for _ in range(3):
                x = torch.randint(0, cfg.V, (mid, T), device=device)
                y = torch.randint(0, cfg.V, (mid, T), device=device)

                optimizer.zero_grad(set_to_none=True)
                with amp_ctx:
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, cfg.V), y.view(-1))
                loss.backward()
                optimizer.step()

            if device == "mps":
                torch.mps.synchronize()

            print("✓ OK")
            max_working_B = mid
            low = mid + 1

            # Clean up
            del model, optimizer, x, y, logits, loss
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "memory" in str(e).lower():
                print("✗ OOM")
                oom_B = mid
                high = mid - 1
                # Clean up after OOM
                gc.collect()
                if device == "mps":
                    torch.mps.empty_cache()
            else:
                print(f"✗ ERROR: {e}")
                raise

    if max_working_B is None:
        print(f"  FAILED: Even B=1 causes OOM!")
        return BenchResult(max_B=0, tokens_per_sec=0, peak_memory_mb=None, oom_at_B=1)

    print(f"  Max working B: {max_working_B}")

    # Now benchmark at max_working_B
    print(f"  Benchmarking at B={max_working_B} for {steps} steps...")

    reset_peak_memory()
    mem_before = get_peak_memory_mb()

    model = GPT(cfg).to(device)
    model.activation_checkpointing = activation_checkpointing
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_cfg.base_lr,
        betas=optimizer_cfg.betas,
        weight_decay=optimizer_cfg.weight_decay,
        eps=optimizer_cfg.eps,
    )
    amp_ctx = get_amp_context(device=device, amp_enabled=amp, amp_dtype=amp_dtype)

    # Warmup
    for _ in range(2):
        x = torch.randint(0, cfg.V, (max_working_B, T), device=device)
        y = torch.randint(0, cfg.V, (max_working_B, T), device=device)
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, cfg.V), y.view(-1))
        loss.backward()
        optimizer.step()

    if device == "mps":
        torch.mps.synchronize()

    # Timed benchmark
    start = time.time()
    for _ in range(steps):
        x = torch.randint(0, cfg.V, (max_working_B, T), device=device)
        y = torch.randint(0, cfg.V, (max_working_B, T), device=device)
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, cfg.V), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    if device == "mps":
        torch.mps.synchronize()

    elapsed = time.time() - start
    tokens_per_sec = (steps * max_working_B * T) / elapsed

    mem_after = get_peak_memory_mb()
    peak_mem = mem_after if mem_after else None

    print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
    print(f"  Peak memory: {peak_mem:.1f} MB" if peak_mem else "  Peak memory: N/A")

    # Clean up
    del model, optimizer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    return BenchResult(
        max_B=max_working_B,
        tokens_per_sec=tokens_per_sec,
        peak_memory_mb=peak_mem,
        oom_at_B=oom_B,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--max-B-search", type=int, default=128, help="Upper bound for B search")
    parser.add_argument("--steps", type=int, default=5, help="Steps for throughput measurement")
    args = parser.parse_args()

    if not HAS_PSUTIL:
        print("\nERROR: psutil required for this benchmark")
        print("Install with: pip install psutil")
        sys.exit(1)

    # Production model configs to test
    configs = [
        # V1 target configs
        {"C": 384, "L": 8, "label": "v1-small (C=384, L=8)"},
        {"C": 512, "L": 8, "label": "v1-medium (C=512, L=8)"},
        {"C": 512, "L": 12, "label": "v1-large (C=512, L=12)"},
    ]

    # Context lengths to test
    context_lengths = [512, 1024]

    # AMP/checkpointing combinations
    modes = [
        {"amp": False, "amp_dtype": "fp16", "activation_checkpointing": False, "label": "baseline (fp32)"},
        {"amp": True, "amp_dtype": "fp16", "activation_checkpointing": False, "label": "amp_fp16"},
        {"amp": False, "amp_dtype": "fp16", "activation_checkpointing": True, "label": "checkpoint_fp32"},
        {"amp": True, "amp_dtype": "fp16", "activation_checkpointing": True, "label": "amp_fp16+checkpoint"},
    ]

    if args.device == "cpu":
        # CPU doesn't support AMP, filter modes
        modes = [m for m in modes if not m["amp"]]

    print("="*100)
    print("OOM BOUNDARY BENCHMARK - Production Model Configs")
    print("="*100)
    print(f"Device: {args.device}")
    print(f"Max B search range: 1-{args.max_B_search}")
    print(f"Throughput measurement: {args.steps} steps")
    print()

    settings = default_settings()
    model_defaults = settings.model
    train_defaults = settings.training.pretrain

    results = {}

    for cfg_spec in configs:
        for T in context_lengths:
            model_cfg = ModelConfig(
                V=256,
                T=T,
                C=cfg_spec["C"],
                L=cfg_spec["L"],
                H=cfg_spec["C"] // 64,  # Keep head dim ~64
                d_ff=cfg_spec["C"] * 4,  # Standard 4x expansion
                dropout=model_defaults.dropout,
                rope_theta=model_defaults.rope_theta,
            )
            print(f"\n{'='*100}")
            print(f"MODEL: {cfg_spec['label']}, T={T}")
            print(f"{'='*100}")

            for mode in modes:
                mode_label = mode.pop("label")
                key = f"{cfg_spec['label']}_T{T}_{mode_label}"

                try:
                    result = find_max_batch_size(
                        device=args.device,
                        cfg=model_cfg,
                        T=T,
                        optimizer_cfg=train_defaults,
                        max_B_search=args.max_B_search,
                        steps=args.steps,
                        **mode,
                    )
                    results[key] = result
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results[key] = None

                # Re-add label for next iteration
                mode["label"] = mode_label

    # Summary table
    print(f"\n{'='*100}")
    print("SUMMARY: Maximum Batch Size & Throughput by Configuration")
    print(f"{'='*100}")
    print(f"{'Configuration':<50} {'Max B':>8} {'Tokens/sec':>12} {'Peak MB':>10} {'OOM@B':>8}")
    print("-"*100)

    for key, result in results.items():
        if result is None:
            print(f"{key:<50} {'ERROR':>8} {'-':>12} {'-':>10} {'-':>8}")
        else:
            mem_str = f"{result.peak_memory_mb:.0f}" if result.peak_memory_mb else "N/A"
            oom_str = str(result.oom_at_B) if result.oom_at_B else "-"
            print(f"{key:<50} {result.max_B:>8} {result.tokens_per_sec:>12.0f} {mem_str:>10} {oom_str:>8}")

    print("-"*100)
    print()
    print("Legend:")
    print("  Max B: Maximum batch size that doesn't OOM")
    print("  Tokens/sec: Throughput at max B")
    print("  Peak MB: Peak memory usage during training")
    print("  OOM@B: Batch size where OOM occurred (if found)")
    print()


if __name__ == "__main__":
    main()
