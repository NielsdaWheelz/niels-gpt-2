import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "audit_config_coverage.py"


def _run_audit() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def _write_canary(rel_path: str, content: str) -> Path:
    path = ROOT / rel_path
    path.write_text(content, encoding="utf-8")
    return path


def test_audit_passes_clean_tree():
    result = _run_audit()
    assert result.returncode == 0, f"audit failed on clean tree:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_audit_flags_literal_assignment():
    path = _write_canary("niels_gpt/_audit_canary_literal.py", "base_lr = 1e-3\n")
    try:
        result = _run_audit()
        assert result.returncode != 0, "audit should fail when literal assignment is present"
        assert "base_lr" in result.stdout or "base_lr" in result.stderr
    finally:
        path.unlink(missing_ok=True)


def test_audit_flags_default_argument():
    path = _write_canary("train/_audit_canary_default.py", "def bad_fn(warmup_steps=10):\n    return warmup_steps\n")
    try:
        result = _run_audit()
        assert result.returncode != 0, "audit should fail when default arg is present"
        assert "warmup_steps" in result.stdout or "warmup_steps" in result.stderr
    finally:
        path.unlink(missing_ok=True)

