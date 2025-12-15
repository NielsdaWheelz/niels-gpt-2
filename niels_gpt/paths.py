"""Repository path constants and utilities."""

from pathlib import Path

# resolve repo root from this file's location
REPO_ROOT: Path = Path(__file__).parent.parent.resolve()

ROAM_DIR: Path = REPO_ROOT / ".roam-data"
PRIMER_PATH: Path = REPO_ROOT / "data" / "primer.txt"
CHECKPOINT_DIR: Path = REPO_ROOT / "checkpoints"
CONFIG_DIR: Path = REPO_ROOT / "configs"


def ensure_dirs() -> None:
    """
    create CHECKPOINT_DIR and CONFIG_DIR if missing.
    do not create ROAM_DIR or data/ (those are user-provided).
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
