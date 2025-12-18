from pathlib import Path

import pytest

from niels_gpt.settings import Settings
from niels_gpt.train.cache_validate import validate_token_caches


def test_invalid_key_rejected_pretrain_mix():
    with pytest.raises(ValueError) as exc:
        Settings.model_validate({"data": {"mix_pretrain": {"wiki": 1.0}}})
    msg = str(exc.value)
    assert "wiki" in msg
    assert "wikitext" in msg


def test_probabilities_must_sum_to_one():
    with pytest.raises(ValueError) as exc:
        Settings.model_validate({"data": {"mix_pretrain": {"wikitext": 0.9, "roam": 0.2}}})
    assert "must sum to 1.0" in str(exc.value)


def test_missing_cache_paths_listed(tmp_path: Path):
    cache_dir = tmp_path / "pretrain"
    with pytest.raises(FileNotFoundError) as exc:
        validate_token_caches(cache_dir, ["wikitext"])
    msg = str(exc.value)
    expected_meta = cache_dir / "wikitext" / "meta.json"
    expected_train = cache_dir / "wikitext" / "train"
    expected_val = cache_dir / "wikitext" / "val"
    for path in (expected_meta, expected_train, expected_val):
        assert str(path) in msg


def test_valid_cache_passes(tmp_path: Path):
    cache_dir = tmp_path / "pretrain"
    source_dir = cache_dir / "wikitext"
    train_dir = source_dir / "train"
    val_dir = source_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "meta.json").write_text("{}", encoding="utf-8")

    validate_token_caches(cache_dir, ["wikitext"])

