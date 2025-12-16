#!/usr/bin/env python3
"""Upload checkpoint to Hugging Face Hub.

Usage:
    python tools/upload_checkpoint.py [--checkpoint checkpoints/best.pt]

Requires: pip install huggingface-hub
First run: huggingface-cli login
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "nnandal/niels-gpt"
DEFAULT_CHECKPOINT = "checkpoints/best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload checkpoint to Hugging Face Hub")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(DEFAULT_CHECKPOINT),
        help=f"Path to checkpoint file (default: {DEFAULT_CHECKPOINT})",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    api = HfApi()

    # Create repo if it doesn't exist (model type for .pt files)
    create_repo(REPO_ID, repo_type="model", exist_ok=True)

    # Upload the checkpoint
    print(f"→ Uploading {args.checkpoint} to {REPO_ID}...")
    api.upload_file(
        path_or_fileobj=str(args.checkpoint),
        path_in_repo=args.checkpoint.name,
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"✓ Uploaded to https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
