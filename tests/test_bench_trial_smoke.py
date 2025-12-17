"""
Smoke test for bench_trial.py.

Runs bench_trial.py on CPU with a tiny model and verifies:
- Exit code 0
- Stdout parses as JSON
- status == "ok"
- tokens_per_sec present and > 0
"""
import json
import subprocess
import sys


def test_bench_trial_smoke():
    """Smoke test: run bench_trial.py with tiny config on CPU."""
    trial_cfg = {
        "device": "cpu",
        "model": {
            "V": 256,
            "T": 64,
            "C": 128,
            "L": 2,
            "H": 2,
            "d_ff": 384,
            "dropout": 0.1,
            "rope_theta": 10000.0,
        },
        "amp": False,
        "amp_dtype": "fp16",
        "activation_checkpointing": False,
        "micro_B": 2,
        "steps_warmup": 1,
        "steps_measure": 2,
        "seed": 42,
        "lr": 3e-4,
    }

    trial_json = json.dumps(trial_cfg)
    cmd = [sys.executable, "tools/bench_trial.py", "--trial", trial_json]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Assert exit code 0
    assert result.returncode == 0, f"Trial failed with code {result.returncode}\nstderr: {result.stderr}"

    # Assert stdout parses as JSON
    try:
        output = json.loads(result.stdout.strip())
    except json.JSONDecodeError as e:
        raise AssertionError(f"Failed to parse JSON output: {e}\nstdout: {result.stdout}")

    # Assert status == "ok"
    assert output["status"] == "ok", f"Expected status='ok', got {output['status']}"

    # Assert tokens_per_sec present and > 0
    assert "tokens_per_sec" in output, "Missing tokens_per_sec in output"
    assert output["tokens_per_sec"] > 0, f"Expected tokens_per_sec > 0, got {output['tokens_per_sec']}"

    print(f"✓ Smoke test passed: {output['tokens_per_sec']:.1f} tok/s")
