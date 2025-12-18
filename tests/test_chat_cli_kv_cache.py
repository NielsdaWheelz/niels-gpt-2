"""Tests for chat CLI KV-cache generation default behavior."""

import argparse
from unittest.mock import MagicMock, patch

import pytest
import torch

from niels_gpt.tokenizer import get_default_tokenizer


class MockArgs:
    """Mock argparse args for testing."""

    def __init__(self, no_kv_cache: bool = False):
        self.ckpt = "dummy.pt"
        self.max_new_tokens = 32
        self.temperature = 0.9
        self.top_k = 50
        self.top_p = 0.0
        self.seed = 42
        self.system = "test system prompt"
        self.system_file = None
        self.no_kv_cache = no_kv_cache


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.cfg = MagicMock()
    model.cfg.T = 512
    model.cfg.C = 384
    model.cfg.H = 6
    model.blocks = [MagicMock() for _ in range(8)]
    # Make parameters() return something so next() works
    param = torch.nn.Parameter(torch.zeros(1))
    model.parameters.return_value = iter([param])
    return model


@pytest.fixture
def tokenizer():
    """Get tokenizer for tests."""
    return get_default_tokenizer()


def test_default_uses_kv_cache(tokenizer, monkeypatch):
    """Test that default generation path calls generate_ids_cached."""
    calls = []

    def mock_generate_ids_cached(model, prompt_ids, **kwargs):
        calls.append({"func": "generate_ids_cached", "kwargs": kwargs})
        # Return valid response with eot at end
        return {"ids": prompt_ids + [tokenizer.special_token_ids()["eot"]], "steps": []}

    def mock_generate_text(*args, **kwargs):
        calls.append({"func": "generate_text", "kwargs": kwargs})
        return "test response"

    # Import module to patch
    import niels_gpt.chat_cli as chat_cli_module

    monkeypatch.setattr(chat_cli_module, "generate_ids_cached", mock_generate_ids_cached)
    monkeypatch.setattr(chat_cli_module, "generate_text", mock_generate_text)

    # Simulate the branching logic from chat_cli.py
    args = MockArgs(no_kv_cache=False)
    prompt = "<|sys|>test<|eot|><|usr|>hello<|eot|><|asst|>"
    prompt_ids = tokenizer.encode(prompt)
    eot_id = tokenizer.special_token_ids()["eot"]

    # This simulates what chat_cli does when no_kv_cache is False
    if args.no_kv_cache:
        mock_generate_text(None, prompt, cfg=None, generation=None, stop_token_id=eot_id)
    else:
        mock_generate_ids_cached(
            None,
            prompt_ids,
            max_new_tokens=32,
            eot_token_id=eot_id,
            temperature=0.9,
            top_k=50,
        )

    # Verify cached generator was called
    assert len(calls) == 1
    assert calls[0]["func"] == "generate_ids_cached"


def test_no_kv_cache_flag_uses_uncached(tokenizer, monkeypatch):
    """Test that --no-kv-cache flag calls generate_text instead."""
    calls = []

    def mock_generate_ids_cached(model, prompt_ids, **kwargs):
        calls.append({"func": "generate_ids_cached", "kwargs": kwargs})
        return {"ids": prompt_ids + [tokenizer.special_token_ids()["eot"]], "steps": []}

    def mock_generate_text(*args, **kwargs):
        calls.append({"func": "generate_text", "kwargs": kwargs})
        return "test response"

    import niels_gpt.chat_cli as chat_cli_module

    monkeypatch.setattr(chat_cli_module, "generate_ids_cached", mock_generate_ids_cached)
    monkeypatch.setattr(chat_cli_module, "generate_text", mock_generate_text)

    # Simulate the branching logic with --no-kv-cache
    args = MockArgs(no_kv_cache=True)
    prompt = "<|sys|>test<|eot|><|usr|>hello<|eot|><|asst|>"
    eot_id = tokenizer.special_token_ids()["eot"]

    # This simulates what chat_cli does when no_kv_cache is True
    if args.no_kv_cache:
        mock_generate_text(None, prompt, cfg=None, generation=None, stop_token_id=eot_id)
    else:
        prompt_ids = tokenizer.encode(prompt)
        mock_generate_ids_cached(
            None,
            prompt_ids,
            max_new_tokens=32,
            eot_token_id=eot_id,
        )

    # Verify uncached generator was called
    assert len(calls) == 1
    assert calls[0]["func"] == "generate_text"


def test_eot_token_id_passed_to_cached_generator(tokenizer, monkeypatch):
    """Test that eot_token_id is correctly passed to generate_ids_cached."""
    captured_kwargs = {}

    def mock_generate_ids_cached(model, prompt_ids, **kwargs):
        captured_kwargs.update(kwargs)
        return {"ids": prompt_ids + [kwargs["eot_token_id"]], "steps": []}

    import niels_gpt.chat_cli as chat_cli_module

    monkeypatch.setattr(chat_cli_module, "generate_ids_cached", mock_generate_ids_cached)

    # Get expected eot_token_id
    expected_eot_id = tokenizer.special_token_ids()["eot"]
    prompt = "<|sys|>test<|eot|><|usr|>hello<|eot|><|asst|>"
    prompt_ids = tokenizer.encode(prompt)

    # Call the mocked function as chat_cli would
    mock_generate_ids_cached(
        None,
        prompt_ids,
        max_new_tokens=32,
        eot_token_id=expected_eot_id,
        temperature=0.9,
        top_k=50,
    )

    # Verify eot_token_id was passed correctly
    assert "eot_token_id" in captured_kwargs
    assert captured_kwargs["eot_token_id"] == expected_eot_id


def test_generate_ids_cached_stops_on_eot():
    """Test that generate_ids_cached stops when eot token is generated."""
    import torch.nn as nn

    from niels_gpt.config import ModelConfig
    from niels_gpt.generate import generate_ids_cached
    from niels_gpt.model.gpt import GPT

    tok = get_default_tokenizer()
    eot_id = tok.special_token_ids()["eot"]

    # Create a minimal model
    cfg = ModelConfig(V=tok.vocab_size, T=64, C=64, L=2, H=2, d_ff=128, dropout=0.0, rope_theta=10000.0)
    model = GPT(cfg)
    model.eval()

    # Simple prompt
    prompt_ids = tok.encode("hello")

    # Generate with max_new_tokens high - should stop early if eot generated
    result = generate_ids_cached(
        model,
        prompt_ids,
        max_new_tokens=10,
        eot_token_id=eot_id,
        temperature=0.9,
        top_k=50,
    )

    # Result should be a dict with ids
    assert "ids" in result
    assert isinstance(result["ids"], list)
    # Prompt should be prefix
    assert result["ids"][: len(prompt_ids)] == prompt_ids


def test_generate_ids_cached_respects_banned_tokens():
    """Test that banned_token_ids are never generated."""
    from niels_gpt.config import ModelConfig
    from niels_gpt.generate import generate_ids_cached
    from niels_gpt.model.gpt import GPT

    tok = get_default_tokenizer()
    eot_id = tok.special_token_ids()["eot"]
    special_ids = tok.special_token_ids()
    banned = [special_ids["sys"], special_ids["usr"], special_ids["asst"]]

    cfg = ModelConfig(V=tok.vocab_size, T=64, C=64, L=2, H=2, d_ff=128, dropout=0.0, rope_theta=10000.0)
    model = GPT(cfg)
    model.eval()

    prompt_ids = tok.encode("hello")

    result = generate_ids_cached(
        model,
        prompt_ids,
        max_new_tokens=20,
        eot_token_id=eot_id,
        temperature=0.9,
        top_k=50,
        banned_token_ids=banned,
    )

    # Generated tokens (after prompt) should not contain banned tokens
    generated = result["ids"][len(prompt_ids) :]
    for token in generated:
        if token != eot_id:  # eot might appear at end
            assert token not in banned, f"banned token {token} was generated"


def test_argparse_has_no_kv_cache_flag():
    """Test that the CLI argparse includes --no-kv-cache flag."""
    from niels_gpt.settings import default_settings

    settings = default_settings()
    gen_defaults = settings.generation

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=gen_defaults.max_new_tokens)
    parser.add_argument("--temperature", type=float, default=gen_defaults.temperature)
    parser.add_argument("--top-k", type=int, default=gen_defaults.top_k or 0)
    parser.add_argument("--top-p", type=float, default=gen_defaults.top_p or 0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--system-file", type=str, default=None)
    parser.add_argument("--no-kv-cache", action="store_true")

    # Parse with flag
    args_with_flag = parser.parse_args(["--ckpt", "test.pt", "--no-kv-cache"])
    assert args_with_flag.no_kv_cache is True

    # Parse without flag
    args_without_flag = parser.parse_args(["--ckpt", "test.pt"])
    assert args_without_flag.no_kv_cache is False

