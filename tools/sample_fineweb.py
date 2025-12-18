#!/usr/bin/env python3
"""
Stream a small deterministic sample from fineweb-edu and write to a local text file.

Example:
  python tools/sample_fineweb.py --out data/fineweb_sample_20mb.txt --bytes 20000000
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable


def _iter_fineweb(dataset: str, name: str, split: str, sample_bytes: int) -> Iterable[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("datasets is required for fineweb sampling") from exc

    ds = load_dataset(dataset, name=name, split=split, streaming=True)
    seen = 0
    for row in ds:
        text = row.get("text", "")
        seen += len(text.encode("utf-8"))
        yield text
        if sample_bytes and seen >= sample_bytes:
            break


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample a small fineweb-edu slice to a local text file")
    parser.add_argument("--out", type=Path, required=True, help="Destination text file")
    parser.add_argument("--bytes", type=int, default=20_000_000, help="Byte budget to stream (default: 20MB)")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu", help="Dataset id")
    parser.add_argument("--name", type=str, default="CC-MAIN-2024-10", help="Dataset config name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Sampling fineweb: dataset={args.dataset} name={args.name} split={args.split} "
        f"bytes={args.bytes} -> {args.out}"
    )
    with open(args.out, "w", encoding="utf-8") as f:
        for text in _iter_fineweb(args.dataset, args.name, args.split, args.bytes):
            f.write(text.replace("\r\n", "\n"))
            f.write("\n")

    digest = _sha256(args.out)
    print(f"wrote {args.out} (sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

