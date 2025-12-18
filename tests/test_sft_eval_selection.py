from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import train.sft as sft


class FakePretrainSource:
    def sample(self, device: str, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long)


class FakeSFTDataset:
    def __init__(self):
        self.calls = 0

    def get_batch(self, B: int, generator: torch.Generator):
        self.calls += 1
        x = torch.zeros((B, 4), dtype=torch.long)
        y_full = torch.zeros((B, 4), dtype=torch.long)
        y_masked = torch.zeros((B, 4), dtype=torch.long)
        return x, y_full, y_masked


def _train_cfg(seed: int = 0, B: int = 2, micro_B_eval: int | None = None):
    return SimpleNamespace(B=B, micro_B_eval=micro_B_eval, seed=seed)


def test_val_sft_source_sft_calls_both_evaluators(monkeypatch: pytest.MonkeyPatch):
    pretrain_calls: list[int] = []
    sft_calls: list[int] = []

    def fake_eval_pretrain(model, batch_fn, eval_batches):
        pretrain_calls.append(eval_batches)
        xs, ys = batch_fn()
        assert xs.shape[0] == 2
        assert ys.shape == xs.shape
        return 1.5

    def fake_eval_sft(model, batch_fn, eval_batches, ignore_index):
        sft_calls.append(eval_batches)
        batch = batch_fn()
        assert len(batch) == 2
        return 2.0

    monkeypatch.setattr(sft, "evaluate_pretrain", fake_eval_pretrain)
    monkeypatch.setattr(sft, "evaluate_sft", fake_eval_sft)

    fake_pretrain_source = FakePretrainSource()
    monkeypatch.setattr(sft, "_load_wikitext_val", lambda *args, **kwargs: fake_pretrain_source)

    fake_ds = FakeSFTDataset()
    monkeypatch.setattr(sft, "_load_sft_sources", lambda *args, **kwargs: {"mock": fake_ds})

    wikitext_source = sft._load_wikitext_val(Path("ignore"), T=4, expected_tokenizer_sha=None, expected_special_token_ids=None)
    sft_sources = sft._load_sft_sources(
        Path("ignore"),
        ["mock"],
        split="val",
        T=4,
        device="cpu",
        allow_missing_idx=True,
        assistant_only_loss=False,
        include_eot_in_loss=False,
        expected_tokenizer_sha=None,
        expected_special_token_ids=None,
    )
    val_mixture = sft.SFTMixture(sft_sources, {"mock": 1.0})

    train_cfg = _train_cfg(seed=0, B=2)

    pretrain_loss, sft_loss = sft.run_eval_tick(
        MagicMock(),
        device="cpu",
        train_cfg=train_cfg,
        eval_batches=2,
        val_source_choice="sft",
        wikitext_val_source=wikitext_source,
        val_mixture=val_mixture,
    )

    assert pretrain_loss == 1.5
    assert sft_loss == 2.0
    assert pretrain_calls == [2]
    assert sft_calls == [2]
    assert fake_ds.calls > 0


def test_val_sft_source_wikitext_only_calls_pretrain(monkeypatch: pytest.MonkeyPatch):
    pretrain_calls: list[int] = []

    def fake_eval_pretrain(model, batch_fn, eval_batches):
        pretrain_calls.append(eval_batches)
        xs, ys = batch_fn()
        assert xs.shape[0] == 2
        assert ys.shape == xs.shape
        return 1.25

    def fail_eval_sft(*args, **kwargs):
        raise AssertionError("evaluate_sft should not be called when val_source_choice='wikitext'")

    monkeypatch.setattr(sft, "evaluate_pretrain", fake_eval_pretrain)
    monkeypatch.setattr(sft, "evaluate_sft", fail_eval_sft)

    fake_pretrain_source = FakePretrainSource()
    monkeypatch.setattr(sft, "_load_wikitext_val", lambda *args, **kwargs: fake_pretrain_source)
    monkeypatch.setattr(sft, "_load_sft_sources", lambda *args, **kwargs: {"mock": FakeSFTDataset()})

    wikitext_source = sft._load_wikitext_val(Path("ignore"), T=4, expected_tokenizer_sha=None, expected_special_token_ids=None)

    train_cfg = _train_cfg(seed=0, B=2)

    pretrain_loss, sft_loss = sft.run_eval_tick(
        MagicMock(),
        device="cpu",
        train_cfg=train_cfg,
        eval_batches=3,
        val_source_choice="wikitext",
        wikitext_val_source=wikitext_source,
        val_mixture=None,
    )

    assert pretrain_loss == 1.25
    assert sft_loss is None
    assert pretrain_calls == [3]
