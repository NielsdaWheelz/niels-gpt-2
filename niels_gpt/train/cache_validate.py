from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CacheRequirement:
    source: str
    split: str  # "train" or "val"
    meta_path: Path
    split_dir: Path


def list_requirements(cache_dir: Path, sources: Iterable[str], *, splits: Iterable[str]) -> list[CacheRequirement]:
    """Pure helper: return required paths for each (source, split)."""
    cache_root = Path(cache_dir)
    requirements: list[CacheRequirement] = []
    for source in sources:
        source_dir = cache_root / source
        meta_path = source_dir / "meta.json"
        for split in splits:
            requirements.append(
                CacheRequirement(
                    source=str(source),
                    split=str(split),
                    meta_path=meta_path,
                    split_dir=source_dir / split,
                )
            )
    return requirements


def validate_token_caches(
    cache_dir: Path, sources: Iterable[str], *, splits: Iterable[str] = ("train", "val")
) -> None:
    """
    raises FileNotFoundError with a multi-line message listing ALL missing paths.
    does not partially succeed: either everything exists or it raises.
    """
    requirements = list_requirements(cache_dir, sources, splits=splits)
    missing: list[str] = []
    seen: set[str] = set()

    for req in requirements:
        for path in (req.meta_path, req.split_dir):
            path_str = str(path)
            if path_str in seen:
                continue
            if not path.exists():
                missing.append(path_str)
                seen.add(path_str)

    if missing:
        message = "missing cache paths:\n  " + "\n  ".join(missing)
        raise FileNotFoundError(message)


def _read_token_count(meta_path: Path) -> str:
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        for key in ("num_tokens", "tokens", "token_count", "total_tokens"):
            if key in meta:
                return str(meta[key])
    except Exception:
        return "?"
    return "?"


def format_data_plan(
    phase: str, cache_dir: Path, sources: Iterable[str], *, splits: Iterable[str] = ("train", "val")
) -> str:
    """Return a short human-readable summary of the cached data."""
    cache_root = Path(cache_dir)
    entries: list[str] = []
    for source in sources:
        source_dir = cache_root / source
        meta_path = source_dir / "meta.json"
        tokens = _read_token_count(meta_path) if meta_path.exists() else "?"
        present_splits = [split for split in splits if (source_dir / split).exists()]
        split_summary = ",".join(present_splits) if present_splits else "none"
        entries.append(f"{source}:tokens={tokens},splits={split_summary}")
    sources_str = "; ".join(entries)
    return f"data plan ({phase}): cache={cache_root} sources=[{sources_str}]"

