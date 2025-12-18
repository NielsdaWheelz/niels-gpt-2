"""CLI for building token caches."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

import torch
from datasets import load_dataset

from niels_gpt.data.dolly import iter_dolly_sft
from niels_gpt.data.fineweb_edu import iter_fineweb_edu
from niels_gpt.data.oasst1 import iter_oasst1_sft
from niels_gpt.data.primer_sft import build_primer_sft_cache
from niels_gpt.data.roam import list_roam_paths, load_texts
from niels_gpt.data.wikitext import iter_wikitext
from niels_gpt.paths import REPO_ROOT, ROAM_DIR
from niels_gpt.tokenizer import DEFAULT_TOKENIZER_PATH, load_tokenizer

from .build_cache import build_pretrain_cache, build_sft_cache
from .formats import DEFAULT_SHARD_BYTES, TOKEN_DTYPE
from .meta import read_meta, sha256_file, write_meta

FINEWEB_DATASET_ID = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "CC-MAIN-2024-10"
DEFAULT_FINEWEB_TRAIN_TOKENS = 200_000_000
DEFAULT_FINEWEB_VAL_TOKENS = 5_000_000
DEFAULT_SFT_VAL_FRAC = 0.10
DEFAULT_SHUFFLE_BUFFER = 10_000


def _ensure_tokenizer(cache_dir: Path):
    tokenizer_model_path = DEFAULT_TOKENIZER_PATH
    tokenizer_meta_path = tokenizer_model_path.with_name("tokenizer_meta.json")
    if not tokenizer_model_path.exists():
        print(f"error: tokenizer not found at {tokenizer_model_path}")
        print("run tokenizer training first (PR-01)")
        sys.exit(1)

    tokenizer_cache_dir = cache_dir / "tokenizer"
    tokenizer_cache_dir.mkdir(parents=True, exist_ok=True)
    cached_model = tokenizer_cache_dir / "spm.model"
    cached_meta = tokenizer_cache_dir / "tokenizer_meta.json"
    if not cached_model.exists():
        shutil.copy2(tokenizer_model_path, cached_model)
    if tokenizer_meta_path.exists() and not cached_meta.exists():
        shutil.copy2(tokenizer_meta_path, cached_meta)

    expected_ids = None
    if tokenizer_meta_path.exists():
        try:
            meta = read_meta(str(tokenizer_meta_path))
            expected_ids = meta.get("special_tokens") or meta.get("special_token_ids")
        except Exception:
            expected_ids = None

    tokenizer = load_tokenizer(str(cached_model), expected_special_token_ids=expected_ids)
    meta = {
        "dataset_name": "tokenizer",
        "dataset_config": None,
        "split_rule": "copy of artifacts/tokenizer/v2",
        "tokenizer_sha256": sha256_file(str(tokenizer_model_path)),
        "vocab_size": tokenizer.vocab_size,
        "special_token_ids": tokenizer.special_token_ids(),
        "token_dtype": TOKEN_DTYPE,
    }
    write_meta(str(tokenizer_cache_dir / "meta.json"), meta)
    return tokenizer


def _split_indices(count: int, *, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    if count == 0:
        return [], []
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(count, generator=gen).tolist()
    num_val = 0
    if count > 1:
        num_val = max(1, int(count * val_frac))
    val_set = set(perm[:num_val])
    train_indices = [i for i in perm if i not in val_set]
    val_indices = [i for i in perm if i in val_set]
    return train_indices, val_indices


def _token_count(texts: list[str], tokenizer) -> int:
    return sum(len(tokenizer.encode(t)) for t in texts)


def _update_meta(meta_path: Path, updates: dict) -> None:
    meta = read_meta(str(meta_path)) if meta_path.exists() else {}
    meta.update(updates)
    write_meta(str(meta_path), meta)


def build_all(
    *,
    cache_dir: str,
    seed: int,
    fineweb_train_tokens: int,
    fineweb_val_tokens: int,
    shard_bytes: int,
    roam_dir: str | None,
) -> None:
    def _shutdown_arrow():
        """Try to tear down pyarrow threadpools proactively to avoid exit hangs."""
        try:
            import pyarrow as pa  # type: ignore

            pool = pa.default_io_pool()
            if hasattr(pool, "shutdown"):
                pool.shutdown(wait=False)
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            print(f"warning: pyarrow shutdown failed: {exc}", file=sys.stderr)

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    tokenizer = _ensure_tokenizer(cache_path)

    pretrain_dir = cache_path / "pretrain"
    pretrain_dir.mkdir(exist_ok=True)

    # fineweb-edu
    print("\n=== building fineweb-edu cache ===")
    fineweb_dir = pretrain_dir / "fineweb_edu"
    fineweb_dir.mkdir(exist_ok=True)
    try:
        ds_info = load_dataset(
            FINEWEB_DATASET_ID,
            name=FINEWEB_CONFIG,
            split="train",
            streaming=True,
        ).info
        dataset_revision = getattr(ds_info, "dataset_revision", None)

        texts = (
            sample.text
            for sample in iter_fineweb_edu(
                name=FINEWEB_CONFIG,
                split="train",
                streaming=True,
                shuffle=True,
                shuffle_buffer_size=DEFAULT_SHUFFLE_BUFFER,
                seed=seed,
            )
        )
        build_pretrain_cache(
            texts,
            str(fineweb_dir),
            tokenizer=tokenizer,
            max_train_tokens=fineweb_train_tokens,
            max_val_tokens=fineweb_val_tokens,
            shard_bytes=shard_bytes,
            seed=seed,
            shuffle_buffer=DEFAULT_SHUFFLE_BUFFER,
            source_name=FINEWEB_DATASET_ID,
            source_config=FINEWEB_CONFIG,
            streaming=True,
        )
        _update_meta(
            fineweb_dir / "meta.json",
            {
                "dataset_name": FINEWEB_DATASET_ID,
                "dataset_config": FINEWEB_CONFIG,
                "split_rule": "hf streaming shuffle buffer 10000; first N tokens to val",
                "streaming": True,
                "dataset_revision": dataset_revision,
                "shuffle_buffer": DEFAULT_SHUFFLE_BUFFER,
                "seed": seed,
                "train_tokens_target": fineweb_train_tokens,
                "val_tokens_target": fineweb_val_tokens,
            },
        )
        meta = read_meta(str(fineweb_dir / "meta.json"))
        print(f"✓ fineweb-edu: {meta['train_tokens']} train, {meta['val_tokens']} val tokens")
    except Exception as exc:
        print(f"✗ fineweb-edu failed: {exc}")

    # roam
    print("\n=== building roam cache ===")
    roam_cache_dir = pretrain_dir / "roam"
    roam_cache_dir.mkdir(exist_ok=True)
    try:
        roam_root = Path(roam_dir) if roam_dir is not None else ROAM_DIR
        paths = list_roam_paths(str(roam_root)) if roam_root.exists() else []
        if not paths:
            _update_meta(
                roam_cache_dir / "meta.json",
                {
                    "dataset_name": "roam",
                    "dataset_config": None,
                    "split_rule": "no files found",
                    "stub": True,
                    "token_dtype": TOKEN_DTYPE,
                    "seed": seed,
                    "train_tokens": 0,
                    "val_tokens": 0,
                    "tokenizer_sha256": sha256_file(tokenizer.model_path),
                    "vocab_size": tokenizer.vocab_size,
                    "special_token_ids": tokenizer.special_token_ids(),
                },
            )
            print("⚠ roam skipped (no files)")
        else:
            train_idx, val_idx = _split_indices(len(paths), val_frac=0.1, seed=seed)
            train_paths = [paths[i] for i in train_idx]
            val_paths = [paths[i] for i in val_idx]
            train_texts = load_texts(train_paths)
            val_texts = load_texts(val_paths)
            val_tokens = _token_count(val_texts, tokenizer)
            train_tokens = _token_count(train_texts, tokenizer)
            build_pretrain_cache(
                list(val_texts) + list(train_texts),
                str(roam_cache_dir),
                tokenizer=tokenizer,
                max_train_tokens=train_tokens,
                max_val_tokens=val_tokens,
                shard_bytes=shard_bytes,
                seed=seed,
                shuffle_buffer=None,
                source_name="roam",
                source_config=None,
                streaming=False,
            )
            _update_meta(
                roam_cache_dir / "meta.json",
                {
                    "dataset_name": "roam",
                    "dataset_config": None,
                    "split_rule": f"by file list, val_frac=0.1, seed={seed}",
                    "num_files": len(paths),
                "streaming": False,
                },
            )
            meta = read_meta(str(roam_cache_dir / "meta.json"))
            print(f"✓ roam: {meta['train_tokens']} train, {meta['val_tokens']} val tokens from {len(paths)} files")
    except Exception as exc:
        print(f"✗ roam failed: {exc}")

    # wikitext
    print("\n=== building wikitext cache ===")
    wikitext_dir = pretrain_dir / "wikitext"
    wikitext_dir.mkdir(exist_ok=True)
    try:
        val_texts = [sample.text for sample in iter_wikitext(config="wikitext-103-raw-v1", split="validation")]
        train_texts = [sample.text for sample in iter_wikitext(config="wikitext-103-raw-v1", split="train")]
        val_tokens = _token_count(val_texts, tokenizer)
        train_tokens = _token_count(train_texts, tokenizer)
        build_pretrain_cache(
            list(val_texts) + list(train_texts),
            str(wikitext_dir),
            tokenizer=tokenizer,
            max_train_tokens=train_tokens,
            max_val_tokens=val_tokens,
            shard_bytes=shard_bytes,
            seed=seed,
            shuffle_buffer=None,
            source_name="wikitext",
            source_config="wikitext-103-raw-v1",
            streaming=False,
        )
        _update_meta(
            wikitext_dir / "meta.json",
            {
                "dataset_name": "wikitext",
                "dataset_config": "wikitext-103-raw-v1",
                "split_rule": "hf validation->val, train->train",
                "streaming": False,
            },
        )
        meta = read_meta(str(wikitext_dir / "meta.json"))
        print(f"✓ wikitext: {meta['train_tokens']} train, {meta['val_tokens']} val tokens")
    except Exception as exc:
        print(f"✗ wikitext failed: {exc}")

    # gutenberg stub
    print("\n=== building gutenberg cache (stub) ===")
    gutenberg_dir = pretrain_dir / "gutenberg"
    (gutenberg_dir / "train").mkdir(parents=True, exist_ok=True)
    (gutenberg_dir / "val").mkdir(parents=True, exist_ok=True)
    _update_meta(
        gutenberg_dir / "meta.json",
        {
            "dataset_name": "gutenberg",
            "dataset_config": None,
            "split_rule": "not built",
            "token_dtype": TOKEN_DTYPE,
            "seed": seed,
            "train_tokens": 0,
            "val_tokens": 0,
            "tokenizer_sha256": sha256_file(tokenizer.model_path),
            "vocab_size": tokenizer.vocab_size,
            "special_token_ids": tokenizer.special_token_ids(),
            "stub": True,
        },
    )
    print("✓ gutenberg stub created")

    # SFT
    sft_dir = cache_path / "sft"
    sft_dir.mkdir(exist_ok=True)

    print("\n=== building dolly15k SFT cache ===")
    dolly_dir = sft_dir / "dolly15k"
    dolly_dir.mkdir(exist_ok=True)
    try:
        examples = (
            [{"role": msg.role, "content": msg.content} for msg in sample.messages]
            for sample in iter_dolly_sft(split="train", seed=seed, shuffle=False)
        )
        build_sft_cache(
            list(examples),
            str(dolly_dir),
            tokenizer=tokenizer,
            val_frac=DEFAULT_SFT_VAL_FRAC,
            seed=seed,
            source_name="databricks/databricks-dolly-15k",
        )
        _update_meta(
            dolly_dir / "meta.json",
            {
                "dataset_name": "databricks/databricks-dolly-15k",
                "dataset_config": None,
                "split_rule": f"torch randperm val_frac={DEFAULT_SFT_VAL_FRAC}, seed={seed}",
            },
        )
        meta = read_meta(str(dolly_dir / "meta.json"))
        print(f"✓ dolly15k: {meta['train_examples']} train, {meta['val_examples']} val examples")
    except Exception as exc:
        print(f"✗ dolly15k failed: {exc}")

    print("\n=== building oasst1 SFT cache ===")
    oasst_dir = sft_dir / "oasst1"
    oasst_dir.mkdir(exist_ok=True)
    try:
        examples = (
            [{"role": msg.role, "content": msg.content} for msg in sample.messages]
            for sample in iter_oasst1_sft(split="train", seed=seed, shuffle_trees=False)
        )
        build_sft_cache(
            list(examples),
            str(oasst_dir),
            tokenizer=tokenizer,
            val_frac=DEFAULT_SFT_VAL_FRAC,
            seed=seed,
            source_name="OpenAssistant/oasst1",
        )
        _update_meta(
            oasst_dir / "meta.json",
            {
                "dataset_name": "OpenAssistant/oasst1",
                "dataset_config": None,
                "split_rule": f"torch randperm val_frac={DEFAULT_SFT_VAL_FRAC}, seed={seed}",
            },
        )
        meta = read_meta(str(oasst_dir / "meta.json"))
        print(f"✓ oasst1: {meta['train_examples']} train, {meta['val_examples']} val examples")
    except Exception as exc:
        print(f"✗ oasst1 failed: {exc}")

    _shutdown_arrow()
    print(f"\n✓ cache build complete: {cache_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build token caches for pretrain and SFT datasets")
    parser.add_argument("command", choices=["build-all", "build-sft-primer"], help="Command to run")
    parser.add_argument("--cache-dir", type=str, default="cache", help="Cache directory (default: cache)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--fineweb-train-tokens",
        type=int,
        default=DEFAULT_FINEWEB_TRAIN_TOKENS,
        help="FineWeb-Edu training token budget (default: 200M)",
    )
    parser.add_argument(
        "--fineweb-val-tokens",
        type=int,
        default=DEFAULT_FINEWEB_VAL_TOKENS,
        help="FineWeb-Edu validation token budget (default: 5M)",
    )
    parser.add_argument(
        "--shard-bytes",
        type=int,
        default=DEFAULT_SHARD_BYTES,
        help="Shard size in bytes (default: 128MB)",
    )
    parser.add_argument(
        "--roam-dir",
        type=str,
        default=None,
        help="Roam data directory (default: .roam-data)",
    )
    parser.add_argument(
        "--primer-jsonl",
        type=str,
        default=str(REPO_ROOT / "data" / "primer.jsonl"),
        help="Path to primer.jsonl file (default: data/primer.jsonl)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for SFT cache (default: cache/sft for build-sft-primer)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=DEFAULT_SFT_VAL_FRAC,
        help=f"Validation fraction (default: {DEFAULT_SFT_VAL_FRAC})",
    )
    parser.add_argument(
        "--t-max",
        type=int,
        default=1024,
        help="Maximum sequence length for truncation (default: 1024)",
    )

    args = parser.parse_args()
    if args.command == "build-all":
        build_all(
            cache_dir=args.cache_dir,
            seed=args.seed,
            fineweb_train_tokens=args.fineweb_train_tokens,
            fineweb_val_tokens=args.fineweb_val_tokens,
            shard_bytes=args.shard_bytes,
            roam_dir=args.roam_dir,
        )
    elif args.command == "build-sft-primer":
        cache_path = Path(args.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        tokenizer = _ensure_tokenizer(cache_path)

        out_dir = args.out_dir or str(cache_path / "sft" / "primer")

        primer_path = Path(args.primer_jsonl)
        if not primer_path.exists():
            print(f"error: primer.jsonl not found at {primer_path}")
            print("create a primer.jsonl file with chat messages in the specified format")
            sys.exit(1)

        print(f"\n=== building primer SFT cache ===")
        print(f"source: {primer_path}")
        print(f"output: {out_dir}")
        print(f"val_frac: {args.val_frac}")
        print(f"seed: {args.seed}")
        print(f"t_max: {args.t_max}")

        build_primer_sft_cache(
            str(primer_path),
            out_dir,
            tokenizer=tokenizer,
            val_frac=args.val_frac,
            seed=args.seed,
            t_max=args.t_max,
        )


if __name__ == "__main__":
    main()
