#!/usr/bin/env python3
"""
Tokenizer quality report: %unk, tokens/char, top tokens, specials frequency.

Examples:
  python tools/tokenizer_report.py --tokenizer artifacts/tokenizer/v2/spm.model --text-file data/primer.txt
  python tools/tokenizer_report.py --tokenizer artifacts/tokenizer/v2/spm.model --wikitext --sample-bytes 5000000
  python tools/tokenizer_report.py --tokenizer artifacts/tokenizer/v2/spm.model --fineweb-bytes 20000000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import sentencepiece as spm

from niels_gpt.special_tokens import SPECIAL_TOKENS


def _iter_wikitext(sample_bytes: int | None) -> Iterable[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("datasets required for --wikitext") from exc

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    seen = 0
    for row in ds:
        text = row.get("text", "")
        seen += len(text.encode("utf-8"))
        yield text
        if sample_bytes is not None and seen >= sample_bytes:
            break


def _iter_fineweb(dataset: str, name: str, split: str, sample_bytes: int) -> Iterable[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("datasets required for --fineweb-bytes") from exc

    ds = load_dataset(dataset, name=name, split=split, streaming=True)
    seen = 0
    for row in ds:
        text = row.get("text", "")
        seen += len(text.encode("utf-8"))
        yield text
        if sample_bytes and seen >= sample_bytes:
            break


def _iter_files(paths: list[Path]) -> Iterable[str]:
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            yield from f.read().splitlines()


def report(tokenizer_path: Path, texts: Iterable[str], label: str) -> None:
    sp = spm.SentencePieceProcessor()
    sp.Load(str(tokenizer_path))
    unk_id = sp.unk_id()
    specials = {name: sp.piece_to_id(tok) for name, tok in {"unk": "<unk>"}.items()}
    for s in SPECIAL_TOKENS:
        specials[s] = sp.piece_to_id(s)

    total_tokens = 0
    total_unk = 0
    total_chars = 0
    freq = Counter()
    specials_seen = Counter()

    for text in texts:
        if not text:
            continue
        ids = sp.EncodeAsIds(text)
        total_tokens += len(ids)
        total_chars += len(text)
        total_unk += sum(1 for i in ids if i == unk_id)
        freq.update(ids)
        for tok in SPECIAL_TOKENS:
            tid = specials.get(tok, -1)
            specials_seen[tok] += sum(1 for i in ids if i == tid)

    if total_tokens == 0:
        print(f"[{label}] no tokens to report")
        return

    pct_unk = (total_unk / total_tokens) * 100
    tpc = total_tokens / max(1, total_chars)
    print(f"[{label}] tokens={total_tokens} chars={total_chars} tokens/char={tpc:.3f} unk%={pct_unk:.4f}")
    print(f"[{label}] specials frequency: " + ", ".join(f"{k}={v}" for k, v in specials_seen.items()))
    top = freq.most_common(20)
    print(f"[{label}] top-20 tokens (id,count): {top}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenizer quality report")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Path to spm.model")
    parser.add_argument("--text-file", action="append", type=Path, help="Extra text files to sample")
    parser.add_argument("--wikitext", action="store_true", help="Include wikitext-103 train sample")
    parser.add_argument("--wikitext-bytes", type=int, default=None, help="Byte cap for wikitext sample")
    parser.add_argument("--fineweb-bytes", type=int, default=0, help="If >0, sample fineweb-edu up to this many bytes")
    parser.add_argument("--fineweb-dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--fineweb-name", type=str, default="CC-MAIN-2024-10")
    parser.add_argument("--fineweb-split", type=str, default="train")
    parser.add_argument("--fineweb-file", type=Path, help="If provided, read fineweb sample from local file (one text per line)")
    args = parser.parse_args()

    iters: list[Iterable[str]] = []
    if args.text_file:
        iters.append(_iter_files(args.text_file))
    if args.wikitext:
        iters.append(_iter_wikitext(args.wikitext_bytes))
    if args.fineweb_file:
        iters.append(_iter_files([args.fineweb_file]))
    if args.fineweb_bytes and args.fineweb_bytes > 0:
        iters.append(
            _iter_fineweb(
                args.fineweb_dataset,
                args.fineweb_name,
                args.fineweb_split,
                args.fineweb_bytes,
            )
        )

    if not iters:
        print("No sources provided; add --text-file and/or --wikitext/--fineweb-bytes")
        return 1

    def chained() -> Iterable[str]:
        for it in iters:
            yield from it

    report(args.tokenizer, chained(), label="combined")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

