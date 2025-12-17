"""
Smoke test for bench_sweep.py.

Runs bench_sweep.py with a tiny grid (one config) on CPU and ensures:
- Summary files are created
- results.jsonl exists and is non-empty
- summary.md exists and is non-empty
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def test_bench_sweep_smoke():
    """Smoke test: run bench_sweep.py with minimal grid on CPU."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "bench_test"

        # Create a tiny grid file
        grid = [
            {
                "T": 64,
                "C": 128,
                "L": 2,
                "H": 2,
                "d_ff": 384,
                "amp": False,
                "amp_dtype": "fp16",
                "activation_checkpointing": False,
            }
        ]

        grid_path = Path(tmpdir) / "grid.json"
        with open(grid_path, "w") as f:
            json.dump(grid, f)

        # Run bench_sweep.py
        cmd = [
            sys.executable,
            "tools/bench_sweep.py",
            "--device", "cpu",
            "--out_dir", str(out_dir),
            "--timeout_s", "30",
            "--steps_warmup", "1",
            "--steps_measure", "2",
            "--seed", "42",
            "--max_micro_B", "16",
            "--grid", str(grid_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Assert exit code 0
        assert result.returncode == 0, f"Sweep failed with code {result.returncode}\nstderr: {result.stderr}\nstdout: {result.stdout}"

        # Check results.jsonl exists and is non-empty
        results_path = out_dir / "results.jsonl"
        assert results_path.exists(), f"results.jsonl not found at {results_path}"
        assert results_path.stat().st_size > 0, "results.jsonl is empty"

        # Check summary.md exists and is non-empty
        summary_path = out_dir / "summary.md"
        assert summary_path.exists(), f"summary.md not found at {summary_path}"
        assert summary_path.stat().st_size > 0, "summary.md is empty"

        # Parse at least one result
        with open(results_path, "r") as f:
            lines = f.readlines()
            assert len(lines) > 0, "No results in results.jsonl"

            first_result = json.loads(lines[0])
            assert "status" in first_result, "Missing status in result"

        print(f"✓ Smoke test passed: {len(lines)} trial results written")
