#!/usr/bin/env python3
"""
Train a SentencePiece tokenizer from local text files.

Usage:
    python scripts/train_tokenizer.py \
        --input_glob "data/**/*.txt" \
        --input_glob ".roam-data/**/*.md" \
        --out_dir artifacts/tokenizer \
        --vocab_size 16000 \
        --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import sentencepiece as spm

from niels_gpt.tokenizer import SPECIAL_TOKENS

DEFAULT_OUT_DIR = "artifacts/tokenizer"
DEFAULT_VOCAB_SIZE = 16000
DEFAULT_SEED = 42
DEFAULT_MODEL_TYPE = "unigram"


def collect_input_files(globs: list[str], repo_root: Path) -> list[Path]:
    """
    Expand glob patterns and collect all matching files.

    Returns sorted list of paths for deterministic ordering.
    """
    all_files = []
    for pattern in globs:
        # Handle absolute paths by using Path.glob directly
        pattern_path = Path(pattern)
        if pattern_path.is_absolute():
            # For absolute paths, extract parent and pattern
            # Pattern like /tmp/test/*.txt -> parent=/tmp/test, pattern=*.txt
            if '*' in pattern or '?' in pattern or '[' in pattern:
                # Has glob chars - need to find the base directory
                parts = []
                for i, part in enumerate(pattern_path.parts):
                    if any(c in part for c in '*?['):
                        # Found glob pattern, use parent up to here
                        base = Path(*pattern_path.parts[:i]) if i > 0 else Path('/')
                        remaining = str(Path(*pattern_path.parts[i:]))
                        matches = list(base.glob(remaining))
                        all_files.extend(matches)
                        break
                else:
                    # No glob chars, just a path
                    if pattern_path.exists():
                        all_files.append(pattern_path)
            else:
                # No glob chars, just add if exists
                if pattern_path.exists():
                    all_files.append(pattern_path)
        else:
            # Relative pattern - use repo_root
            matches = list(repo_root.glob(pattern))
            all_files.extend(matches)

    # Remove duplicates and sort for determinism
    unique_files = sorted(set(all_files))
    return [f for f in unique_files if f.is_file()]


def compute_inputs_sha256(files: list[Path]) -> str:
    """
    Compute SHA256 hash of concatenated file bytes in deterministic order.

    Args:
        files: Sorted list of file paths

    Returns:
        Hex string of SHA256 hash
    """
    hasher = hashlib.sha256()
    for path in files:
        with open(path, "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()


def prepare_training_text(files: list[Path], temp_file: Path) -> None:
    """
    Read all files, normalize line endings, and write to temp file.

    - Read as UTF-8 with errors="replace"
    - Normalize \\r\\n to \\n
    - Write all text to a single temp file for sentencepiece training
    """
    with open(temp_file, "w", encoding="utf-8") as out:
        for path in files:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
                # Normalize line endings
                text = text.replace("\r\n", "\n")
                out.write(text)
                # Add separator between files
                out.write("\n")


def train_sentencepiece(
    input_file: Path,
    model_prefix: Path,
    vocab_size: int,
    seed: int,
) -> None:
    """
    Train a SentencePiece unigram model.

    Args:
        input_file: Path to concatenated training text
        model_prefix: Output path prefix (will create .model and .vocab)
        vocab_size: Target vocabulary size
        seed: Random seed for training (recorded in metadata but not enforced)
    """
    # Convert special tokens to comma-separated string
    # Use control_symbols - they get reserved IDs and decode to empty
    # We never encode them from text; we insert them by ID in chat templates
    control_symbols = ",".join(SPECIAL_TOKENS)

    spm.SentencePieceTrainer.train(
        input=str(input_file),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        byte_fallback=True,
        control_symbols=control_symbols,
        # Prevent splitting digits
        split_digits=False,
        # Determinism controls (best-effort)
        input_sentence_size=0,  # Use all sentences; no sampling
        shuffle_input_sentence=False,  # No shuffling
        num_threads=1,  # Avoid nondeterministic parallelism
        # Use default normalization (nmt_nfkc) and add_dummy_prefix (True)
    )


def validate_special_tokens(model_path: Path) -> dict[str, int]:
    """
    Validate that each special token exists in vocabulary.

    Note: We insert special tokens by ID (not by encoding strings),
    so we only need to verify they're in the vocab with distinct IDs.

    Args:
        model_path: Path to trained .model file

    Returns:
        Dict mapping token name to ID (e.g., {"sys": 3, "usr": 4, ...})

    Raises:
        ValueError: If any special token is not in vocab or IDs are not distinct
    """
    sp = spm.SentencePieceProcessor()
    sp.Load(str(model_path))

    token_ids = {}
    for token in SPECIAL_TOKENS:
        # Check if token exists in vocab (control symbols use piece_to_id)
        piece_id = sp.piece_to_id(token)
        if piece_id == sp.unk_id():
            raise ValueError(
                f"Special token '{token}' not found in vocabulary. "
                f"Did training fail to reserve control symbols?"
            )
        # Store with simple key (sys, usr, asst, eot)
        key = token.strip("<|>")
        token_ids[key] = piece_id

    # Verify all IDs are distinct
    if len(set(token_ids.values())) != len(SPECIAL_TOKENS):
        raise ValueError(
            f"Special tokens must have distinct IDs, got: {token_ids}"
        )

    return token_ids


def write_metadata(
    meta_path: Path,
    vocab_size: int,
    model_type: str,
    special_tokens: dict[str, int],
    seed: int,
    input_globs: list[str],
    input_files: list[Path],
    inputs_sha256: str,
    repo_root: Path,
) -> None:
    """
    Write tokenizer metadata to JSON file.

    Args:
        meta_path: Output path for metadata JSON
        vocab_size: Vocabulary size
        model_type: Model type (e.g., "unigram")
        special_tokens: Dict mapping token name to ID
        seed: Random seed used
        input_globs: List of glob patterns used
        input_files: List of actual files processed
        inputs_sha256: SHA256 hash of input data
        repo_root: Repository root for computing relative paths
    """
    # Convert paths to repo-relative strings
    relative_files = []
    for f in input_files:
        try:
            rel = f.relative_to(repo_root)
            relative_files.append(str(rel))
        except ValueError:
            # If file is outside repo, use absolute path
            relative_files.append(str(f.absolute()))

    metadata = {
        "vocab_size": vocab_size,
        "model_type": model_type,
        "special_tokens": special_tokens,
        "seed": seed,
        "sentencepiece_version": spm.__version__,
        "input_globs": input_globs,
        "input_files": relative_files,
        "inputs_sha256": inputs_sha256,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece tokenizer from local text files"
    )
    parser.add_argument(
        "--input_glob",
        action="append",
        required=True,
        help="Glob pattern for input files (can be repeated)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help=f"Vocabulary size (default: {DEFAULT_VOCAB_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        choices=["unigram"],
        help=f"Model type (default: {DEFAULT_MODEL_TYPE})",
    )

    args = parser.parse_args()

    # Determine repo root (current working directory)
    repo_root = Path.cwd()

    # Collect input files
    print(f"Collecting files from {len(args.input_glob)} glob pattern(s)...")
    input_files = collect_input_files(args.input_glob, repo_root)

    if not input_files:
        print("ERROR: No input files matched the provided glob patterns", file=sys.stderr)
        print(f"Patterns tried: {args.input_glob}", file=sys.stderr)
        print(f"Current working directory: {repo_root}", file=sys.stderr)

        # Provide helpful suggestions
        roam_dir = repo_root / ".roam-data"
        if roam_dir.exists() and roam_dir.is_dir():
            print(f"Suggestion: Try '.roam-data/**/*.md' (found .roam-data/ directory)", file=sys.stderr)
        else:
            # Show top-level directories to help user
            top_level_dirs = sorted([d.name for d in repo_root.iterdir() if d.is_dir() and not d.name.startswith('.')])[:20]
            if top_level_dirs:
                print(f"Top-level directories: {', '.join(top_level_dirs)}", file=sys.stderr)

        return 1

    print(f"Found {len(input_files)} input file(s)")

    # Compute hash of inputs for provenance
    print("Computing input hash...")
    inputs_sha256 = compute_inputs_sha256(input_files)

    # Prepare output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_prefix = out_dir / "spm"
    model_path = out_dir / "spm.model"
    meta_path = out_dir / "tokenizer_meta.json"

    # Prepare training text
    print("Preparing training text...")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt") as tmp:
        temp_path = Path(tmp.name)

    try:
        prepare_training_text(input_files, temp_path)

        # Train tokenizer
        print(f"Training SentencePiece tokenizer (vocab_size={args.vocab_size}, seed={args.seed})...")
        train_sentencepiece(temp_path, model_prefix, args.vocab_size, args.seed)

        # Validate special tokens
        print("Validating special tokens...")
        special_token_ids = validate_special_tokens(model_path)

        # Write metadata
        print("Writing metadata...")
        write_metadata(
            meta_path,
            args.vocab_size,
            args.model_type,
            special_token_ids,
            args.seed,
            args.input_glob,
            input_files,
            inputs_sha256,
            repo_root,
        )

        print(f"\nTokenizer saved to: {out_dir}")
        print(f"  Model: {model_path}")
        print(f"  Metadata: {meta_path}")
        print(f"\nSpecial token IDs: {special_token_ids}")
        print(f"Input hash: {inputs_sha256}")

        return 0

    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    sys.exit(main())
