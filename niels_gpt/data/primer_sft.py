"""Primer SFT dataset loader and cache builder (PR-03)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np

from ..cache.formats import TOKEN_DTYPE
from ..cache.meta import sha256_file, write_meta
from .types import ChatMessage, ChatSample


def render_chat_sft(
    tokenizer: "SentencePieceTokenizer",  # type: ignore
    messages: list[dict],
    *,
    add_final_eot: bool = True,
) -> tuple[list[int], list[int]]:
    """
    Render chat messages to token IDs and labels with assistant-only loss masking.

    Args:
        tokenizer: SentencePieceTokenizer instance
        messages: List of message dicts with 'role' and 'content' keys
        add_final_eot: Whether to add final <|eot|> (default True)

    Returns:
        (input_ids, labels) where labels has -100 for non-assistant tokens

    Template:
        <|sys|> {system_content} <|eot|>
        <|usr|> {user_content} <|eot|>
        <|asst|> {assistant_content} <|eot|>

    Labeling rule:
        - Assistant content tokens AND assistant <|eot|> are labeled (loss-active)
        - Everything else is -100
    """
    if not messages:
        raise ValueError("messages must be non-empty")

    # Check for at least one assistant message
    has_assistant = any(msg.get("role") == "assistant" for msg in messages)
    if not has_assistant:
        raise ValueError("messages must contain at least one assistant message")

    special_ids = tokenizer.special_token_ids()
    role_to_id = {
        "system": special_ids["sys"],
        "user": special_ids["usr"],
        "assistant": special_ids["asst"],
    }
    eot_id = special_ids["eot"]

    input_ids: list[int] = []
    labels: list[int] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role not in role_to_id:
            raise ValueError(f"invalid role: {role}")

        # Add role token
        role_id = role_to_id[role]
        input_ids.append(role_id)
        labels.append(-100)

        # Encode content
        content_ids = tokenizer.encode(content)
        input_ids.extend(content_ids)

        # Label content tokens based on role
        if role == "assistant":
            # Assistant content is loss-active
            labels.extend(content_ids)
        else:
            # System/user content is masked
            labels.extend([-100] * len(content_ids))

        # Add <|eot|> token
        input_ids.append(eot_id)
        # Assistant <|eot|> is loss-active, others are masked
        if role == "assistant":
            labels.append(eot_id)
        else:
            labels.append(-100)

    if len(input_ids) != len(labels):
        raise RuntimeError("input_ids and labels length mismatch")

    return input_ids, labels


def load_primer_jsonl(path: str) -> list[dict]:
    """Load primer.jsonl and return list of examples."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                if "messages" not in example:
                    raise ValueError(f"line {line_num}: missing 'messages' key")
                if not isinstance(example["messages"], list):
                    raise ValueError(f"line {line_num}: 'messages' must be a list")
                if not example["messages"]:
                    raise ValueError(f"line {line_num}: 'messages' must be non-empty")

                # Validate each message
                for msg_idx, msg in enumerate(example["messages"]):
                    if not isinstance(msg, dict):
                        raise ValueError(f"line {line_num}, message {msg_idx}: must be a dict")
                    if set(msg.keys()) != {"role", "content"}:
                        raise ValueError(
                            f"line {line_num}, message {msg_idx}: must have exactly 'role' and 'content' keys"
                        )
                    if msg["role"] not in {"system", "user", "assistant"}:
                        raise ValueError(
                            f"line {line_num}, message {msg_idx}: invalid role '{msg['role']}'"
                        )
                    if not isinstance(msg["content"], str):
                        raise ValueError(
                            f"line {line_num}, message {msg_idx}: 'content' must be a string"
                        )

                examples.append(example)
            except json.JSONDecodeError as e:
                raise ValueError(f"line {line_num}: invalid JSON: {e}") from e

    if not examples:
        raise ValueError(f"no valid examples found in {path}")

    return examples


def split_primer_examples(
    examples: list[dict],
    *,
    val_frac: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """
    Split examples into train/val deterministically by example.

    Args:
        examples: List of primer examples
        val_frac: Fraction of examples for validation
        seed: Random seed for deterministic shuffle

    Returns:
        (train_examples, val_examples)
    """
    if not examples:
        return [], []

    N = len(examples)
    rng = random.Random(seed)
    indices = list(range(N))
    rng.shuffle(indices)

    n_val = int(N * val_frac)
    val_indices = set(indices[:n_val])

    train_examples = [examples[i] for i in range(N) if i not in val_indices]
    val_examples = [examples[i] for i in range(N) if i in val_indices]

    return train_examples, val_examples


def build_primer_sft_cache(
    primer_jsonl: str,
    out_dir: str,
    *,
    tokenizer,
    val_frac: float,
    seed: int,
    t_max: int,
    source_name: str = "primer",
) -> None:
    """
    Build SFT token cache for primer dataset.

    Args:
        primer_jsonl: Path to primer.jsonl file
        out_dir: Output directory (will create {out_dir}/primer/)
        tokenizer: SentencePieceTokenizer instance
        val_frac: Validation fraction
        seed: Random seed
        t_max: Maximum sequence length for truncation
        source_name: Source name for metadata (default "primer")

    Creates:
        out_dir/meta.json
        out_dir/train_tokens.bin, out_dir/train_idx.npy
        out_dir/val_tokens.bin, out_dir/val_idx.npy
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load and split examples
    all_examples = load_primer_jsonl(primer_jsonl)
    train_examples, val_examples = split_primer_examples(all_examples, val_frac=val_frac, seed=seed)

    special_ids = tokenizer.special_token_ids()

    def process_split(examples: list[dict], split: str) -> tuple[int, int, int]:
        """Process a split and return (num_examples, num_tokens, num_label_tokens)."""
        tokens_path = out_path / f"{split}_input_ids.bin"
        labels_path = out_path / f"{split}_labels.bin"
        idx_path = out_path / f"{split}_idx.npy"

        offsets: list[int] = []
        pos = 0
        num_tokens = 0
        num_label_tokens = 0
        dropped = 0

        with open(tokens_path, "wb") as f, open(labels_path, "wb") as lbl_f:
            for idx, example in enumerate(examples):
                messages = example["messages"]

                # Render to tokens and labels
                try:
                    input_ids, labels = render_chat_sft(tokenizer, messages, add_final_eot=True)
                except ValueError:
                    # Drop malformed examples
                    dropped += 1
                    continue

                # Truncate if needed (keep last T_max tokens)
                if len(input_ids) > t_max:
                    input_ids = input_ids[-t_max:]
                    labels = labels[-t_max:]

                # Drop if too short after truncation
                if len(input_ids) < 2:
                    dropped += 1
                    continue

                # Write tokens + labels
                offsets.append(pos)
                arr = np.asarray(input_ids, dtype="<u2")
                lbl_arr = np.asarray(labels, dtype="<i4")
                f.write(arr.tobytes())
                lbl_f.write(lbl_arr.tobytes())
                pos += len(input_ids)
                num_tokens += len(input_ids)

                # Count label tokens (not -100)
                num_label_tokens += sum(1 for label in labels if label != -100)

        # Save index
        np.save(idx_path, np.asarray(offsets, dtype=np.int64))

        return len(offsets), num_tokens, num_label_tokens

    # Process train and val
    train_count, train_tokens, train_label_tokens = process_split(train_examples, "train")
    val_count, val_tokens, val_label_tokens = process_split(val_examples, "val")

    # Write metadata
    model_path = getattr(tokenizer, "model_path", None)
    tokenizer_sha256: str | None = None
    if model_path and Path(model_path).exists():
        tokenizer_sha256 = sha256_file(model_path)

    meta = {
        "source": source_name,
        "tokenizer_sha256": tokenizer_sha256,
        "vocab_size": tokenizer.vocab_size,
        "special_token_ids": {
            "sys": special_ids["sys"],
            "usr": special_ids["usr"],
            "asst": special_ids["asst"],
            "eot": special_ids["eot"],
        },
        "splits": {
            "train": {
                "num_examples": train_count,
                "num_tokens": train_tokens,
                "num_label_tokens": train_label_tokens,
            },
            "val": {
                "num_examples": val_count,
                "num_tokens": val_tokens,
                "num_label_tokens": val_label_tokens,
            },
        },
        "val_frac": val_frac,
        "seed": seed,
        "T_max": t_max,
        "token_dtype": TOKEN_DTYPE,
        "label_dtype": "int32-le",
        "train_examples": train_count,
        "val_examples": val_count,
        "dataset_name": source_name,
        "dataset_config": None,
        "split_rule": f"by example, val_frac={val_frac}, seed={seed}",
    }

    write_meta(str(out_path / "meta.json"), meta)

    print(f"✓ primer cache built:")
    print(f"  train: {train_count} examples, {train_tokens} tokens ({train_label_tokens} labeled)")
    print(f"  val: {val_count} examples, {val_tokens} tokens ({val_label_tokens} labeled)")
    print(f"  output: {out_path}")


def iter_primer_sft(
    *,
    jsonl_path: str,
    split: str = "train",
    seed: int = 42,
    val_frac: float = 0.1,
    take: int | None = None,
) -> Iterator[ChatSample]:
    """
    Iterate over primer examples as ChatSample objects.

    Args:
        jsonl_path: Path to primer.jsonl
        split: "train" or "val"
        seed: Random seed for split
        val_frac: Validation fraction
        take: Optional limit on number of examples

    Yields:
        ChatSample objects
    """
    all_examples = load_primer_jsonl(jsonl_path)
    train_examples, val_examples = split_primer_examples(all_examples, val_frac=val_frac, seed=seed)

    examples = train_examples if split == "train" else val_examples

    count = 0
    for idx, example in enumerate(examples):
        if take is not None and count >= take:
            break

        messages = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in example["messages"]
        ]

        yield ChatSample(
            messages=messages,
            source="primer",
            meta={"index": idx, "split": split},
        )
        count += 1
