"""Quick OOM test for specific batch sizes."""

import gc
import sys
import time

import torch
import torch.nn.functional as F

from niels_gpt.config import ModelConfig
from niels_gpt.model.gpt import GPT
from train.amp_utils import get_amp_context

def test_batch_size(cfg, T, B, device, amp, amp_dtype, activation_checkpointing):
    """Test if a specific batch size works."""
    print(f"Testing B={B}...", end=" ", flush=True)

    try:
        model = GPT(cfg).to(device)
        model.activation_checkpointing = activation_checkpointing
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        amp_ctx = get_amp_context(device=device, amp_enabled=amp, amp_dtype=amp_dtype)

        # Run 3 steps
        start = time.time()
        for _ in range(3):
            x = torch.randint(0, cfg.V, (B, T), device=device)
            y = torch.randint(0, cfg.V, (B, T), device=device)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, cfg.V), y.view(-1))
            loss.backward()
            optimizer.step()

        if device == "mps":
            torch.mps.synchronize()

        elapsed = time.time() - start
        tokens_per_sec = (3 * B * T) / elapsed

        print(f"✓ OK - {tokens_per_sec:.0f} tokens/sec")

        # Clean up
        del model, optimizer, x, y, logits, loss
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

        return True, tokens_per_sec

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "memory" in str(e).lower():
            print("✗ OOM")
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            return False, 0
        else:
            print(f"✗ ERROR: {e}")
            raise

def main():
    device = "mps"

    # Test configs
    configs = [
        {"C": 384, "L": 8, "label": "v1-small (C=384, L=8)"},
        {"C": 512, "L": 8, "label": "v1-medium (C=512, L=8)"},
    ]

    T_values = [512, 1024]

    # Test batch sizes (powers of 2 + some in-between)
    test_Bs = [8, 16, 24, 32, 48, 64]

    modes = [
        {"amp": False, "amp_dtype": "fp16", "activation_checkpointing": False, "label": "baseline_fp32"},
        {"amp": True, "amp_dtype": "fp16", "activation_checkpointing": False, "label": "amp_fp16"},
        {"amp": False, "amp_dtype": "fp16", "activation_checkpointing": True, "label": "checkpoint_fp32"},
        {"amp": True, "amp_dtype": "fp16", "activation_checkpointing": True, "label": "amp_fp16+checkpoint"},
    ]

    print("="*80)
    print("QUICK OOM BOUNDARY TEST")
    print("="*80)
    print(f"Device: {device}")
    print(f"Testing batch sizes: {test_Bs}")
    print()

    results = {}

    for cfg_spec in configs:
        cfg = ModelConfig(
            V=256,
            C=cfg_spec["C"],
            L=cfg_spec["L"],
            H=cfg_spec["C"] // 64,
            d_ff=cfg_spec["C"] * 4,
            dropout=0.1,
        )

        for T in T_values:
            print(f"\n{'='*80}")
            print(f"MODEL: {cfg_spec['label']}, T={T}")
            print(f"{'='*80}")

            for mode in modes:
                mode_label = mode["label"]
                print(f"\n{mode_label}:")

                max_working_B = None
                max_throughput = 0

                for B in test_Bs:
                    key = f"{cfg_spec['label']}_T{T}_{mode_label}_B{B}"
                    worked, tps = test_batch_size(
                        cfg, T, B, device,
                        mode["amp"], mode["amp_dtype"], mode["activation_checkpointing"]
                    )

                    if worked:
                        max_working_B = B
                        max_throughput = tps
                    else:
                        # Hit OOM, stop testing larger batches
                        break

                results[f"{cfg_spec['label']}_T{T}_{mode_label}"] = {
                    "max_B": max_working_B,
                    "tokens_per_sec": max_throughput,
                }

                if max_working_B:
                    print(f"  → Max working B: {max_working_B} ({max_throughput:.0f} tokens/sec)")
                else:
                    print(f"  → Even B=8 failed!")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Configuration':<60} {'Max B':>8} {'Tokens/sec':>12}")
    print("-"*80)

    for key, result in results.items():
        max_B = result["max_B"] if result["max_B"] else "FAIL"
        tps = f"{result['tokens_per_sec']:.0f}" if result["max_B"] else "-"
        print(f"{key:<60} {str(max_B):>8} {tps:>12}")

    print("-"*80)

if __name__ == "__main__":
    main()
