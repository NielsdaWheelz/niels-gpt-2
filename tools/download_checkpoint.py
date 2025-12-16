#!/usr/bin/env python3
"""Download checkpoint from Hugging Face Hub.

Usage:
    python tools/download_checkpoint.py

Requires: pip install huggingface-hub
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "nnandal/niels-gpt"
FILENAME = "best.pt"
LOCAL_DIR = Path("checkpoints")


def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    dest = LOCAL_DIR / FILENAME
    if dest.exists():
        print(f"✓ Checkpoint already exists: {dest}")
        return

    print(f"→ Downloading {FILENAME} from {REPO_ID}...")
    hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(LOCAL_DIR),
        local_dir_use_symlinks=False,
    )
    print(f"✓ Downloaded checkpoint to {dest}")


if __name__ == "__main__":
    main()
