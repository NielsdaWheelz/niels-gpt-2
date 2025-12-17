"""Streams: build cached byte streams for each source/split."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from niels_gpt.data import (
    list_roam_paths,
    load_primer_text,
    load_texts,
    load_wikitext,
    split_primer_dialogues,
    split_roam_paths,
)
from niels_gpt.settings import default_settings

BytesSources = Dict[str, bytes]  # keys: "wiki" | "roam" | "primer"


@dataclass(frozen=True)
class StreamBuildConfig:
    roam_root: str = "data/.roam-data/"
    primer_path: str = "data/primer.txt"
    cache_dir: str = "data/cache/streams"
    doc_separator: str = "\n\n"
    roam_val_frac: float = 0.10
    primer_val_frac: float = 0.10
    seed: int = 42
    delimiter: str = "\n\n<dialogue>\n\n"
    force_rebuild: bool = False
    allow_missing_sources: bool = False
    enabled_sources: tuple[str, ...] = ("wiki", "roam", "primer")
    required_sources: tuple[str, ...] = ("wiki", "roam", "primer")


def build_wiki_stream(docs: list[str], *, sep: str) -> bytes:
    """utf-8 encode docs and join with sep."""
    return sep.join(docs).encode("utf-8")


def build_roam_stream(docs: list[str], *, sep: str) -> bytes:
    """utf-8 encode docs and join with sep."""
    return sep.join(docs).encode("utf-8")


def build_primer_stream(text: str) -> bytes:
    """utf-8 encode text exactly as-is."""
    return text.encode("utf-8")


def _get_file_metadata(path: str) -> dict:
    """Get file metadata for cache invalidation."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    stat = p.stat()
    return {
        "path": path,
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def _build_wiki_metadata(docs: list[str], sep: str, split: str) -> dict:
    """Build metadata for wiki cache."""
    return {
        "source": "wiki",
        "split": split,
        "n_docs": len(docs),
        "total_chars": sum(len(doc) for doc in docs),
        "sep": sep,
    }


def _build_roam_metadata(paths: list[str], sep: str, split: str, seed: int, val_frac: float) -> dict:
    """Build metadata for roam cache."""
    file_metas = []
    for path in paths:
        file_metas.append(_get_file_metadata(path))
    return {
        "source": "roam",
        "split": split,
        "sep": sep,
        "files": file_metas,
        "seed": seed,
        "val_frac": val_frac,
    }


def _build_primer_metadata(
    path: str, delimiter: str, split: str, seed: int, val_frac: float
) -> dict:
    """Build metadata for primer cache."""
    try:
        meta = _get_file_metadata(path)
        meta["delimiter"] = delimiter
        meta["source"] = "primer"
        meta["split"] = split
        meta["seed"] = seed
        meta["val_frac"] = val_frac
        return meta
    except FileNotFoundError:
        return {
            "source": "primer",
            "split": split,
            "delimiter": delimiter,
            "path": path,
            "seed": seed,
            "val_frac": val_frac,
        }


def _metadata_matches(cache_meta: dict, current_meta: dict) -> bool:
    """Check if cache metadata matches current metadata."""
    if cache_meta is None or current_meta is None:
        return False

    # Different sources or splits means mismatch
    if cache_meta.get("source") != current_meta.get("source"):
        return False
    if cache_meta.get("split") != current_meta.get("split"):
        return False

    source = cache_meta.get("source")

    if source == "wiki":
        # For wiki, check n_docs, total_chars, sep
        return (
            cache_meta.get("n_docs") == current_meta.get("n_docs")
            and cache_meta.get("total_chars") == current_meta.get("total_chars")
            and cache_meta.get("sep") == current_meta.get("sep")
        )
    elif source == "roam":
        # For roam, check file list metadata, sep, seed, val_frac
        if cache_meta.get("sep") != current_meta.get("sep"):
            return False
        if cache_meta.get("seed") != current_meta.get("seed"):
            return False
        if cache_meta.get("val_frac") != current_meta.get("val_frac"):
            return False
        cache_files = cache_meta.get("files", [])
        current_files = current_meta.get("files", [])
        if len(cache_files) != len(current_files):
            return False
        # Compare each file's metadata
        for cf, curf in zip(cache_files, current_files):
            if (
                cf.get("path") != curf.get("path")
                or cf.get("mtime_ns") != curf.get("mtime_ns")
                or cf.get("size") != curf.get("size")
            ):
                return False
        return True
    elif source == "primer":
        # For primer, check path, mtime_ns, size, delimiter, seed, val_frac
        return (
            cache_meta.get("path") == current_meta.get("path")
            and cache_meta.get("mtime_ns") == current_meta.get("mtime_ns")
            and cache_meta.get("size") == current_meta.get("size")
            and cache_meta.get("delimiter") == current_meta.get("delimiter")
            and cache_meta.get("seed") == current_meta.get("seed")
            and cache_meta.get("val_frac") == current_meta.get("val_frac")
        )

    return False


def _load_from_cache(cache_dir: Path, source: str, split: str) -> Tuple[bytes | None, dict | None]:
    """Load stream and metadata from cache if they exist."""
    bin_path = cache_dir / f"{source}_{split}.bin"
    meta_path = cache_dir / f"{source}_{split}.meta.json"

    if not bin_path.exists() or not meta_path.exists():
        return None, None

    try:
        with open(bin_path, "rb") as f:
            stream_bytes = f.read()
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return stream_bytes, metadata
    except Exception:
        return None, None


def _save_to_cache(
    cache_dir: Path, source: str, split: str, stream_bytes: bytes, metadata: dict
) -> None:
    """Save stream and metadata to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    bin_path = cache_dir / f"{source}_{split}.bin"
    meta_path = cache_dir / f"{source}_{split}.meta.json"

    with open(bin_path, "wb") as f:
        f.write(stream_bytes)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True, ensure_ascii=False)


def build_sources(cfg: StreamBuildConfig) -> Tuple[BytesSources, BytesSources]:
    """
    returns (sources_train, sources_val)

    sources_* keys are subset of {"wiki","roam","primer"}.
    each value is a bytes stream encoded utf-8.

    strictness:
      - enabled_sources controls which sources are attempted
      - required_sources must be a subset of enabled_sources; default ("wiki",)
      - By default (allow_missing_sources=False), required sources must build successfully
      - Set allow_missing_sources=True to skip failed sources gracefully
      - Errors are collected and reported with helpful hints

    caching:
      - if cache exists and metadata matches current inputs, load .bin
      - else rebuild, write .bin and .meta.json
    """
    cache_dir = Path(cfg.cache_dir)
    T = default_settings().model.T

    sources_train: BytesSources = {}
    sources_val: BytesSources = {}
    errors: dict[str, str] = {}

    enabled_sources = set(cfg.enabled_sources)
    required_sources = set(cfg.required_sources)

    missing_required_not_enabled = required_sources - enabled_sources
    if missing_required_not_enabled:
        raise ValueError(
            "required_sources must be a subset of enabled_sources; missing from enabled: "
            f"{sorted(missing_required_not_enabled)}"
        )

    # Build wiki streams
    if "wiki" in enabled_sources:
        try:
            wiki_data = load_wikitext()
            wiki_train_docs = wiki_data["train"]
            wiki_val_docs = wiki_data["val"]

            # Train
            current_meta = _build_wiki_metadata(
                wiki_train_docs, cfg.doc_separator, "train"
            )
            cached_stream, cached_meta = _load_from_cache(cache_dir, "wiki", "train")

            if not cfg.force_rebuild and _metadata_matches(cached_meta, current_meta):
                sources_train["wiki"] = cached_stream
            else:
                stream = build_wiki_stream(wiki_train_docs, sep=cfg.doc_separator)
                _save_to_cache(cache_dir, "wiki", "train", stream, current_meta)
                sources_train["wiki"] = stream

            # Val
            current_meta = _build_wiki_metadata(
                wiki_val_docs, cfg.doc_separator, "val"
            )
            cached_stream, cached_meta = _load_from_cache(cache_dir, "wiki", "val")

            if not cfg.force_rebuild and _metadata_matches(cached_meta, current_meta):
                sources_val["wiki"] = cached_stream
            else:
                stream = build_wiki_stream(wiki_val_docs, sep=cfg.doc_separator)
                _save_to_cache(cache_dir, "wiki", "val", stream, current_meta)
                sources_val["wiki"] = stream

        except Exception as e:
            errors["wiki"] = (
                f"{type(e).__name__}: {str(e)}\n"
                "  Hint: wikitext download requires internet connection. "
                "Run with allow_missing_sources=True or required_sources=() to skip."
            )

    # Build roam streams
    if "roam" in enabled_sources:
        try:
            roam_root = Path(cfg.roam_root)
            if not roam_root.exists():
                raise FileNotFoundError(f"Roam directory not found: {cfg.roam_root}")

            all_paths = list_roam_paths(cfg.roam_root)
            if not all_paths:
                raise ValueError(f"No .md files found in {cfg.roam_root}")

            train_paths, val_paths = split_roam_paths(
                all_paths, val_frac=cfg.roam_val_frac, seed=cfg.seed
            )

            if train_paths:
                train_docs = load_texts(train_paths)
                current_meta = _build_roam_metadata(
                    train_paths,
                    cfg.doc_separator,
                    "train",
                    cfg.seed,
                    cfg.roam_val_frac,
                )
                cached_stream, cached_meta = _load_from_cache(
                    cache_dir, "roam", "train"
                )

                if not cfg.force_rebuild and _metadata_matches(
                    cached_meta, current_meta
                ):
                    sources_train["roam"] = cached_stream
                else:
                    stream = build_roam_stream(train_docs, sep=cfg.doc_separator)
                    _save_to_cache(cache_dir, "roam", "train", stream, current_meta)
                    sources_train["roam"] = stream

            if val_paths:
                val_docs = load_texts(val_paths)
                current_meta = _build_roam_metadata(
                    val_paths, cfg.doc_separator, "val", cfg.seed, cfg.roam_val_frac
                )
                cached_stream, cached_meta = _load_from_cache(cache_dir, "roam", "val")

                if not cfg.force_rebuild and _metadata_matches(
                    cached_meta, current_meta
                ):
                    sources_val["roam"] = cached_stream
                else:
                    stream = build_roam_stream(val_docs, sep=cfg.doc_separator)
                    _save_to_cache(cache_dir, "roam", "val", stream, current_meta)
                    sources_val["roam"] = stream
            elif "roam" in required_sources and cfg.roam_val_frac > 0:
                raise ValueError(
                    f"Roam validation set is empty. Add more markdown files to {cfg.roam_root} "
                    f"or reduce roam_val_frac."
                )
        except Exception as e:
            errors["roam"] = (
                f"{type(e).__name__}: {str(e)}\n"
                f"  Hint: ensure directory exists at {cfg.roam_root} with .md files. "
                "Run with allow_missing_sources=True or remove 'roam' from required_sources to skip."
            )

    # Build primer streams
    if "primer" in enabled_sources:
        try:
            primer_path = Path(cfg.primer_path)
            if not primer_path.exists():
                raise FileNotFoundError(f"Primer file not found: {cfg.primer_path}")

            full_text = load_primer_text(cfg.primer_path)
            train_text, val_text = split_primer_dialogues(
                full_text,
                val_frac=cfg.primer_val_frac,
                seed=cfg.seed,
                delimiter=cfg.delimiter,
            )

            if train_text:
                current_meta = _build_primer_metadata(
                    cfg.primer_path,
                    cfg.delimiter,
                    "train",
                    cfg.seed,
                    cfg.primer_val_frac,
                )
                cached_stream, cached_meta = _load_from_cache(
                    cache_dir, "primer", "train"
                )

                if not cfg.force_rebuild and _metadata_matches(
                    cached_meta, current_meta
                ):
                    sources_train["primer"] = cached_stream
                else:
                    stream = build_primer_stream(train_text)
                    _save_to_cache(cache_dir, "primer", "train", stream, current_meta)
                    sources_train["primer"] = stream

            if val_text:
                current_meta = _build_primer_metadata(
                    cfg.primer_path, cfg.delimiter, "val", cfg.seed, cfg.primer_val_frac
                )
                cached_stream, cached_meta = _load_from_cache(cache_dir, "primer", "val")

                if not cfg.force_rebuild and _metadata_matches(
                    cached_meta, current_meta
                ):
                    sources_val["primer"] = cached_stream
                else:
                    stream = build_primer_stream(val_text)
                    _save_to_cache(cache_dir, "primer", "val", stream, current_meta)
                    sources_val["primer"] = stream
            elif "primer" in required_sources and cfg.primer_val_frac > 0:
                raise ValueError(
                    f"Primer validation set is empty. Add more dialogue blocks to {cfg.primer_path} "
                    f"or reduce primer_val_frac."
                )
        except Exception as e:
            errors["primer"] = (
                f"{type(e).__name__}: {str(e)}\n"
                f"  Hint: ensure file exists at {cfg.primer_path} with dialogue blocks. "
                "Run with allow_missing_sources=True or remove 'primer' from required_sources to skip."
            )

    # Check if required sources were built
    missing_required = []
    for source in required_sources:
        if source not in sources_train or source not in sources_val:
            missing_required.append(source)

    if missing_required and not cfg.allow_missing_sources:
        error_lines = [
            f"Required source(s) failed to build: {', '.join(sorted(missing_required))}"
        ]
        for source in missing_required:
            if source in errors:
                error_lines.append(f"\n{source}:")
                error_lines.append(f"  {errors[source]}")
        raise RuntimeError("\n".join(error_lines))

    # Check that we have at least one source
    if not sources_train:
        raise RuntimeError(
            "No training sources were built successfully.\n"
            f"Errors encountered: {list(errors.keys())}\n"
            "Set allow_missing_sources=True and required_sources=() to proceed anyway, "
            "or fix the errors above."
        )
    if not sources_val:
        raise RuntimeError(
            "No validation sources were built successfully.\n"
            f"Errors encountered: {list(errors.keys())}\n"
            "Set allow_missing_sources=True and required_sources=() to proceed anyway, "
            "or fix the errors above."
        )

    # Length guard: check all streams are >= T+1
    too_short = []
    for split_name, sources in [("train", sources_train), ("val", sources_val)]:
        for source_name, stream in sources.items():
            if len(stream) < T + 1:
                too_short.append(f"{source_name}_{split_name}: {len(stream)} bytes")

    if too_short:
        raise ValueError(
            f"Stream(s) too short for T={T} (need at least {T+1} bytes):\n  "
            + "\n  ".join(too_short)
        )

    return sources_train, sources_val
