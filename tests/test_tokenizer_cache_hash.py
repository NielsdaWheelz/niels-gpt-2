import json
from pathlib import Path

import numpy as np
import pytest

from train.pretrain import _load_source_sharded as load_pretrain_source
from train.sft import _load_sft_source as load_sft_source


def _write_pretrain_cache(tmp: Path, name: str, tokenizer_sha: str, special: dict[str, int]) -> Path:
    source_dir = tmp / name
    train_dir = source_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    bin_path = train_dir / "shard_00000.bin"
    meta_path = source_dir / "meta.json"
    arr = np.asarray([1, 2, 3, 4], dtype="<u2")
    bin_path.write_bytes(arr.tobytes())
    meta = {
        "tokenizer_sha256": tokenizer_sha,
        "token_dtype": "uint16-le",
        "special_token_ids": special,
    }
    meta_path.write_text(json.dumps(meta))
    return bin_path


def _write_sft_cache(tmp: Path, name: str, tokenizer_sha: str, special: dict[str, int]) -> None:
    source_dir = tmp / name
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "train").mkdir(exist_ok=True)

    tokens_path = source_dir / "train_input_ids.bin"
    labels_path = source_dir / "train_labels.bin"
    idx_path = source_dir / "train_idx.npy"
    meta_path = source_dir / "meta.json"
    arr = np.asarray([1, 2, 3, 4], dtype="<u2")
    tokens_path.write_bytes(arr.tobytes())
    labels_path.write_bytes(np.asarray([1, 2, 3, 4], dtype="<i4").tobytes())
    np.save(idx_path, np.asarray([0], dtype=np.int64))
    meta = {
        "tokenizer_sha256": tokenizer_sha,
        "token_dtype": "uint16-le",
        "label_dtype": "int32-le",
        "special_token_ids": special,
        "source": name,
        "split": "train",
    }
    meta_path.write_text(json.dumps(meta))


def test_pretrain_cache_hash_mismatch(tmp_path: Path):
    special = {"sys": 1, "usr": 2, "asst": 3, "eot": 4}
    _write_pretrain_cache(tmp_path, "wikitext", "abc", special)
    with pytest.raises(ValueError, match="tokenizer hash mismatch"):
        load_pretrain_source(
            tmp_path,
            "wikitext",
            "train",
            T=2,
            expected_tokenizer_sha="def",
            expected_special_token_ids=special,
        )


def test_sft_cache_hash_mismatch(tmp_path: Path):
    special = {"sys": 1, "usr": 2, "asst": 3, "eot": 4}
    _write_sft_cache(tmp_path, "dolly15k", "abc", special)
    with pytest.raises(ValueError, match="tokenizer hash mismatch"):
        load_sft_source(
            tmp_path,
            "dolly15k",
            "train",
            T=4,
            device="cpu",
            assistant_only_loss=True,
            include_eot_in_loss=False,
            expected_tokenizer_sha="def",
            expected_special_token_ids=special,
        )

