#!/usr/bin/env python3
"""
Scan text files for literal occurrences of project special tokens.

Usage:
    python tools/scan_special_collisions.py --roots data/ another_dir/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from niels_gpt.special_tokens import SPECIAL_TOKENS, find_special_collision


def _scan_file(path: Path, specials: Sequence[str]) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - defensive
        return [f"{path}: failed to read ({exc})"]

    collision = find_special_collision(text, specials=specials)
    if collision:
        msg = (
            f"{path}: token={collision['token']} "
            f"byte_offset={collision['byte_offset']} "
            f"char_offset={collision['char_offset']} "
            f"snippet='{collision['snippet']}'"
        )
        return [msg]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan for special token literals.")
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="Directories/files to scan recursively.",
    )
    args = parser.parse_args()

    specials = tuple(SPECIAL_TOKENS)
    problems: list[str] = []

    for root in args.roots:
        root_path = Path(root)
        targets = [root_path]
        if root_path.is_dir():
            targets = list(root_path.rglob("*"))
        for path in targets:
            if not path.is_file():
                continue
            problems.extend(_scan_file(path, specials=specials))

    if problems:
        print("collision(s) found:")
        for msg in problems:
            print(msg)
        return 1

    print("no special token collisions detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

