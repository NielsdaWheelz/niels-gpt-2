"""Metadata handling for cache files."""

import hashlib
import json
from pathlib import Path


def write_meta(path: str, meta: dict) -> None:
    """
    Write metadata dict to JSON file.

    Args:
        path: Path to write meta.json
        meta: Dictionary of metadata
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, sort_keys=True)


def read_meta(path: str) -> dict:
    """
    Read metadata dict from JSON file.

    Args:
        path: Path to meta.json file

    Returns:
        Dictionary of metadata
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        path: Path to file

    Returns:
        Hex string of SHA256 hash
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        # Read in chunks to handle large files
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()
