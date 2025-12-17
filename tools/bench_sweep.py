#!/usr/bin/env python3
"""
bench_sweep.py: Orchestrate benchmark trials to find optimal configs.

Usage:
    python tools/bench_sweep.py --device auto
    python tools/bench_sweep.py --device mps --out_dir bench/ --timeout_s 20

Runs multiple trials in subprocesses with timeout and OOM detection.
Searches for max safe micro_B per config (doubling + binary search).
Writes bench/results.jsonl and bench/summary.md.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from niels_gpt.settings import default_settings


_SETTINGS = default_settings()
_BENCH = _SETTINGS.benchmark


def get_device(device_arg: str) -> str:
    """Resolve device string."""
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_arg


def run_trial_subprocess(
    trial_cfg: dict,
    timeout_s: float,
) -> dict:
    """
    Run bench_trial.py in a subprocess with timeout.

    Returns the result dict with status in {"ok", "oom", "timeout", "error"}.
    """
    trial_json = json.dumps(trial_cfg)
    cmd = [sys.executable, "tools/bench_trial.py", "--trial", trial_json]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

        if result.returncode == 0:
            # Parse JSON output
            try:
                return json.loads(result.stdout.strip())
            except json.JSONDecodeError:
                return {
                    "status": "error",
                    "message": "Failed to parse trial output",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
        else:
            return {
                "status": "error",
                "message": f"Trial exited with code {result.returncode}",
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "device": trial_cfg["device"],
            "model": trial_cfg["model"],
            "amp": trial_cfg["amp"],
            "amp_dtype": trial_cfg.get("amp_dtype", "fp16"),
            "activation_checkpointing": trial_cfg["activation_checkpointing"],
            "micro_B": trial_cfg["micro_B"],
        }


def find_max_micro_B(
    base_trial_cfg: dict,
    max_micro_B: int,
    timeout_s: float,
    steps_warmup: int,
    steps_measure: int,
    seed: int,
) -> tuple[int, list[dict]]:
    """
    Binary search for max micro_B that succeeds.

    Returns (max_safe_B, all_results).
    """
    all_results = []

    # Start with micro_B = 1
    trial_cfg = {
        **base_trial_cfg,
        "micro_B": 1,
        "steps_warmup": steps_warmup,
        "steps_measure": steps_measure,
        "seed": seed,
        "lr": 3e-4,
    }

    result = run_trial_subprocess(trial_cfg, timeout_s)
    all_results.append(result)

    if result["status"] != "ok":
        # Even micro_B=1 fails
        return 0, all_results

    last_ok = 1

    # Doubling phase: find upper bound
    current_B = 2
    while current_B <= max_micro_B:
        trial_cfg["micro_B"] = current_B
        result = run_trial_subprocess(trial_cfg, timeout_s)
        all_results.append(result)

        if result["status"] == "ok":
            last_ok = current_B
            current_B *= 2
        else:
            # Found failure, now binary search between last_ok and current_B
            break
    else:
        # Never failed in doubling phase
        return last_ok, all_results

    # Binary search phase
    low = last_ok
    high = min(current_B, max_micro_B)

    while low + 1 < high:
        mid = (low + high) // 2
        trial_cfg["micro_B"] = mid
        result = run_trial_subprocess(trial_cfg, timeout_s)
        all_results.append(result)

        if result["status"] == "ok":
            last_ok = mid
            low = mid
        else:
            high = mid

    return last_ok, all_results


def run_sweep(args: argparse.Namespace) -> None:
    """Run the full benchmark sweep."""
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.md"

    # Load grid
    if args.grid == "preset":
        from tools.bench_configs import get_default_grid
        grid = get_default_grid()
    else:
        with open(args.grid, "r") as f:
            grid = json.load(f)

    # Vocabulary size (use settings default)
    V = _SETTINGS.model.V

    all_results = []
    best_configs = []

    print(f"Running benchmark sweep on device: {device}")
    print(f"Grid size: {len(grid)} configs")
    print(f"Timeout: {args.timeout_s}s per trial")
    print()

    for i, cfg in enumerate(grid):
        print(f"[{i+1}/{len(grid)}] Testing config: T={cfg['T']}, C={cfg['C']}, L={cfg['L']}, H={cfg['H']}, ckpt={cfg['activation_checkpointing']}")

        model_cfg = {
            "V": V,
            "T": cfg["T"],
            "C": cfg["C"],
            "L": cfg["L"],
            "H": cfg["H"],
            "d_ff": cfg["d_ff"],
            "dropout": _SETTINGS.model.dropout,
            "rope_theta": _SETTINGS.model.rope_theta,
        }

        base_trial_cfg = {
            "device": device,
            "model": model_cfg,
            "amp": cfg["amp"],
            "amp_dtype": cfg.get("amp_dtype", "fp16"),
            "activation_checkpointing": cfg["activation_checkpointing"],
        }

        # Find max micro_B
        max_B, trial_results = find_max_micro_B(
            base_trial_cfg,
            max_micro_B=args.max_micro_B,
            timeout_s=args.timeout_s,
            steps_warmup=args.steps_warmup,
            steps_measure=args.steps_measure,
            seed=args.seed,
        )

        # Record all trial results
        all_results.extend(trial_results)

        if max_B == 0:
            print(f"  -> Failed at micro_B=1, skipping")
            continue

        print(f"  -> Max micro_B: {max_B}")

        # Test max_B and max_B//2, keep better throughput
        candidates = []

        # Test max_B
        max_B_trial = {
            **base_trial_cfg,
            "micro_B": max_B,
            "steps_warmup": args.steps_warmup,
            "steps_measure": args.steps_measure,
            "seed": args.seed,
            "lr": _BENCH.lr,
        }
        result = run_trial_subprocess(max_B_trial, args.timeout_s)
        all_results.append(result)
        if result["status"] == "ok":
            candidates.append(result)

        # Test max_B // 2 if > 0
        half_B = max(1, max_B // 2)
        if half_B != max_B and half_B > 0:
            half_B_trial = {
                **base_trial_cfg,
                "micro_B": half_B,
                "steps_warmup": args.steps_warmup,
                "steps_measure": args.steps_measure,
                "seed": args.seed,
            "lr": _BENCH.lr,
            }
            result = run_trial_subprocess(half_B_trial, args.timeout_s)
            all_results.append(result)
            if result["status"] == "ok":
                candidates.append(result)

        # Pick best throughput
        if candidates:
            best = max(candidates, key=lambda r: r["tokens_per_sec"])
            best_configs.append(best)
            print(f"  -> Best: micro_B={best['micro_B']}, {best['tokens_per_sec']:.0f} tok/s")
        else:
            print(f"  -> No successful trials at max_B or half_B")

    # Write results.jsonl
    with open(results_path, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")

    print()
    print(f"Wrote {len(all_results)} trial results to {results_path}")

    # Compute rankings
    for cfg in best_configs:
        cfg["tokens_8h"] = cfg["tokens_per_sec"] * 8 * 3600

    # Sort by tokens_8h desc, then params desc
    best_configs.sort(key=lambda c: (-c["tokens_8h"], -c["params"]))

    # Write summary.md
    write_summary(summary_path, best_configs, device)
    print(f"Wrote summary to {summary_path}")


def write_summary(path: Path, configs: list[dict], device: str) -> None:
    """Write summary.md with ranked configs."""
    with open(path, "w") as f:
        f.write("# Benchmark Summary\n\n")
        f.write(f"**Device:** {device}\n\n")

        if not configs:
            f.write("No successful configs.\n")
            return

        f.write("## Top Configurations\n\n")
        f.write("Ranked by estimated 8-hour throughput, then parameter count.\n\n")

        # Table header
        f.write("| Rank | V | T | C | L | H | d_ff | AMP | Ckpt | micro_B | ms/step | tok/s | tok_8h | params |\n")
        f.write("|------|---|---|---|---|---|------|-----|------|---------|---------|-------|--------|--------|\n")

        # Best tokens_8h for 90% threshold
        best_tokens_8h = configs[0]["tokens_8h"]
        threshold_90 = 0.9 * best_tokens_8h

        for rank, cfg in enumerate(configs, 1):
            model = cfg["model"]
            tokens_8h_str = f"{cfg['tokens_8h']:.2e}"

            # Mark configs within 90% of best
            marker = ""
            if cfg["tokens_8h"] >= threshold_90:
                marker = " ⭐"

            f.write(
                f"| {rank}{marker} | "
                f"{model['V']} | {model['T']} | {model['C']} | {model['L']} | {model['H']} | {model['d_ff']} | "
                f"{cfg['amp']} | {cfg['activation_checkpointing']} | "
                f"{cfg['micro_B']} | {cfg['ms_per_step']:.1f} | "
                f"{cfg['tokens_per_sec']:.0f} | {tokens_8h_str} | {cfg['params']} |\n"
            )

        f.write("\n")
        f.write("⭐ = Within 90% of best throughput\n")


def main():
    parser = argparse.ArgumentParser(description="Run benchmark sweep")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"], help="Device to use")
    parser.add_argument("--out_dir", type=str, default="bench/", help="Output directory")
    parser.add_argument("--timeout_s", type=float, default=_BENCH.timeout_s, help="Timeout per trial in seconds")
    parser.add_argument("--steps_warmup", type=int, default=_BENCH.steps_warmup, help="Warmup steps per trial")
    parser.add_argument("--steps_measure", type=int, default=_BENCH.steps_measure, help="Measured steps per trial")
    parser.add_argument("--seed", type=int, default=_BENCH.seed, help="Random seed")
    parser.add_argument("--max_micro_B", type=int, default=_BENCH.max_micro_B, help="Maximum micro_B to try")
    parser.add_argument("--grid", type=str, default="preset", help="Grid config: 'preset' or path to JSON")

    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
