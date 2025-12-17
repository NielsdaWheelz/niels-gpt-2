"""Cache builders for pretrain and SFT datasets."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch

from niels_gpt.chat_template import format_chat
from niels_gpt.tokenizer import SPECIAL_TOKENS

from .formats import TOKEN_BYTES, TOKEN_DTYPE
from .meta import sha256_file, write_meta


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


def _tokenizer_meta(tokenizer) -> dict:
    special = tokenizer.special_token_ids()
    model_path = getattr(tokenizer, "model_path", None)
    tokenizer_sha256: str | None = None
    if model_path and Path(model_path).exists():
        tokenizer_sha256 = sha256_file(model_path)
    return {
        "tokenizer_sha256": tokenizer_sha256,
        "vocab_size": tokenizer.vocab_size,
        "special_token_ids": {
            "sys": special["sys"],
            "usr": special["usr"],
            "asst": special["asst"],
            "eot": special["eot"],
        },
    }


class _ShardWriter:
    """Stream-writing helper for fixed-size shards."""

    def __init__(self, shard_dir: Path, tokens_per_shard: int):
        self.shard_dir = shard_dir
        self.tokens_per_shard = max(1, tokens_per_shard)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self._shard_idx = 0
        self._tokens_in_shard = 0
        self._fh = None

    def _open_next(self) -> None:
        if self._fh:
            self._fh.close()
        shard_path = self.shard_dir / f"shard_{self._shard_idx:05d}.bin"
        self._fh = open(shard_path, "wb")
        self._shard_idx += 1
        self._tokens_in_shard = 0

    def write_tokens(self, tokens: list[int]) -> None:
        if not tokens:
            return
        offset = 0
        while offset < len(tokens):
            if self._fh is None or self._tokens_in_shard >= self.tokens_per_shard:
                self._open_next()
            space = self.tokens_per_shard - self._tokens_in_shard
            chunk = tokens[offset : offset + space]
            if chunk:
                arr = np.asarray(chunk, dtype="<u2")
                self._fh.write(arr.tobytes())
                self._tokens_in_shard += len(chunk)
                offset += len(chunk)
            else:
                break

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def _shuffle_with_buffer(
    texts: Iterable[str], buffer_size: int, generator: torch.Generator
) -> Iterator[str]:
    """Deterministic streaming shuffle using a fixed-size buffer."""
    buffer: list[str] = []
    for text in texts:
        buffer.append(text)
        if len(buffer) < buffer_size:
            continue
        idx = torch.randint(len(buffer), (1,), generator=generator).item()
        yield buffer.pop(idx)
    while buffer:
        idx = torch.randint(len(buffer), (1,), generator=generator).item()
        yield buffer.pop(idx)


def build_pretrain_cache(
    texts: Iterable[str],
    out_dir: str,
    *,
    tokenizer,
    max_train_tokens: int,
    max_val_tokens: int,
    shard_bytes: int,
    seed: int,
    shuffle_buffer: int | None,
    source_name: str | None = None,
    source_config: str | None = None,
    streaming: bool | None = None,
) -> None:
    """
    writes:
      out_dir/train/shard_*.bin
      out_dir/val/shard_*.bin
      out_dir/meta.json
    token dtype: uint16 little-endian
    deterministic given seed + identical input stream order.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tokens_per_shard = max(1, shard_bytes // TOKEN_BYTES)
    rng = torch.Generator().manual_seed(seed)

    text_iter: Iterator[str] = iter(texts)
    if shuffle_buffer and shuffle_buffer > 0:
        text_iter = _shuffle_with_buffer(text_iter, shuffle_buffer, rng)

    train_writer = _ShardWriter(out_path / "train", tokens_per_shard)
    val_writer = _ShardWriter(out_path / "val", tokens_per_shard)

    train_tokens = 0
    val_tokens = 0

    for text in text_iter:
        tokens = tokenizer.encode(text)
        if not tokens:
            continue

        if val_tokens < max_val_tokens:
            remaining_val = max_val_tokens - val_tokens
            if remaining_val > 0:
                to_val = tokens[:remaining_val]
                val_writer.write_tokens(to_val)
                val_tokens += len(to_val)
                tokens = tokens[remaining_val:]

        if train_tokens >= max_train_tokens:
            break

        if tokens and train_tokens < max_train_tokens:
            remaining_train = max_train_tokens - train_tokens
            to_train = tokens[:remaining_train]
            train_writer.write_tokens(to_train)
            train_tokens += len(to_train)

        if train_tokens >= max_train_tokens and val_tokens >= max_val_tokens:
            break

    train_writer.close()
    val_writer.close()

    meta = {
        "dataset_name": source_name,
        "dataset_config": source_config,
        "split_rule": "val tokens first, remaining to train",
        "token_dtype": TOKEN_DTYPE,
        "seed": seed,
        "shard_bytes": shard_bytes,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "streaming": streaming,
        "shuffle_buffer": shuffle_buffer,
        "train_tokens_target": max_train_tokens,
        "val_tokens_target": max_val_tokens,
    }
    meta.update(_tokenizer_meta(tokenizer))
    write_meta(str(out_path / "meta.json"), meta)


def build_sft_cache(
    examples: Iterable[list[Message]],
    out_dir: str,
    *,
    tokenizer,
    val_frac: float,
    seed: int,
) -> None:
    """
    writes:
      out_dir/train_tokens.bin, out_dir/train_idx.npy
      out_dir/val_tokens.bin,   out_dir/val_idx.npy
      out_dir/meta.json
    where idx is offsets into tokens.bin (int64), offsets-only (length inferred).
    deterministic given seed + identical input stream order.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_examples = list(examples)
    num_examples = len(all_examples)
    dropped_malformed = 0

    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_examples, generator=rng).tolist() if num_examples else []
    num_val = 0
    if num_examples > 1:
        num_val = max(1, int(num_examples * val_frac))
    val_indices = set(perm[:num_val])

    train_offsets: list[int] = []
    val_offsets: list[int] = []
    train_pos = 0
    val_pos = 0

    train_tokens_path = out_path / "train_tokens.bin"
    val_tokens_path = out_path / "val_tokens.bin"

    specials_set = set(tokenizer.special_token_ids().values())

    with open(train_tokens_path, "wb") as train_f, open(val_tokens_path, "wb") as val_f:
        for idx, messages in enumerate(all_examples):
            malformed = False
            seq_chunks: list[int] = []

            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if any(tok_str in content for tok_str in SPECIAL_TOKENS):
                    malformed = True
                    break
                content_ids = tokenizer.encode(content)
                if any(t in specials_set for t in content_ids):
                    malformed = True
                    break

                seq_chunks.extend(format_chat(tokenizer, [msg]))

            if malformed or not seq_chunks:
                dropped_malformed += 1
                continue

            seq = seq_chunks
            arr = np.asarray(seq, dtype="<u2")

            if idx in val_indices:
                val_offsets.append(val_pos)
                val_f.write(arr.tobytes())
                val_pos += len(seq)
            else:
                train_offsets.append(train_pos)
                train_f.write(arr.tobytes())
                train_pos += len(seq)

    np.save(out_path / "train_idx.npy", np.asarray(train_offsets, dtype=np.int64))
    np.save(out_path / "val_idx.npy", np.asarray(val_offsets, dtype=np.int64))

    meta = {
        "dataset_name": None,
        "dataset_config": None,
        "split_rule": f"torch.randperm val_frac={val_frac}",
        "token_dtype": TOKEN_DTYPE,
        "seed": seed,
        "val_frac": val_frac,
        "train_examples": len(train_offsets),
        "val_examples": len(val_offsets),
        "dropped_malformed_examples": dropped_malformed,
    }
    meta.update(_tokenizer_meta(tokenizer))
    write_meta(str(out_path / "meta.json"), meta)
