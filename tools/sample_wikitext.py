#!/usr/bin/env python3
"""
Sample a small deterministic slice of wikitext-103 and write to a local text file.

Example:
  python tools/sample_wikitext.py --out data/wikitext_sample_5mb.txt --bytes 5000000
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample wikitext-103 to a local text file")
    parser.add_argument("--out", type=Path, required=True, help="Destination text file")
    parser.add_argument("--bytes", type=int, default=5_000_000, help="Byte budget to write (default: 5MB)")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset id")
    parser.add_argument("--name", type=str, default="wikitext-103-raw-v1", help="Dataset config name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("datasets is required for sampling wikitext") from exc

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Sampling wikitext: dataset={args.dataset} name={args.name} split={args.split} "
        f"bytes={args.bytes} -> {args.out}"
    )

    ds = load_dataset(args.dataset, args.name, split=args.split, streaming=True)
    written_bytes = 0
    docs = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text", "")
            if not text:
                continue
            norm = text.replace("\r\n", "\n")
            encoded = norm.encode("utf-8", errors="replace")
            if args.bytes and written_bytes + len(encoded) > args.bytes:
                # trim to budget
                remaining = args.bytes - written_bytes
                norm = encoded[:remaining].decode("utf-8", errors="replace")
                encoded = norm.encode("utf-8", errors="replace")
            f.write(norm)
            f.write("\n\n")
            written_bytes += len(encoded)
            docs += 1
            if args.bytes and written_bytes >= args.bytes:
                break

    digest = _sha256(args.out)
    print(
        f"wrote {docs} docs, {written_bytes} bytes to {args.out} "
        f"(sha256={digest}, dataset={args.dataset}:{args.name}:{args.split})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

