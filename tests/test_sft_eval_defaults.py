"""Tests for pr-02: sft eval defaults + dual-metric reporting."""

from __future__ import annotations

import pytest

from niels_gpt.settings import Settings


def test_default_val_sft_source_is_sft():
    """Test that the default val_sft_source is 'sft'."""
    settings = Settings()
    assert settings.data.val_sft_source == "sft"


def test_invalid_val_sft_source_rejected():
    """Test that invalid val_sft_source values are rejected."""
    with pytest.raises(ValueError, match="val_sft_source must be 'sft' or 'wikitext'"):
        Settings(data={"val_sft_source": "invalid"})


def test_valid_val_sft_source_sft():
    """Test that val_sft_source='sft' is accepted."""
    settings = Settings(data={"val_sft_source": "sft"})
    assert settings.data.val_sft_source == "sft"


def test_valid_val_sft_source_wikitext():
    """Test that val_sft_source='wikitext' is accepted."""
    settings = Settings(data={"val_sft_source": "wikitext"})
    assert settings.data.val_sft_source == "wikitext"
