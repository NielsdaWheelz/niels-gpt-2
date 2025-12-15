"""Scaffold tests to ensure basic package functionality."""

import torch

import niels_gpt
from niels_gpt.device import get_device
from niels_gpt.paths import (
    CHECKPOINT_DIR,
    CONFIG_DIR,
    PRIMER_PATH,
    REPO_ROOT,
    ROAM_DIR,
    ensure_dirs,
)
from niels_gpt.rng import make_generator, set_seed


def test_imports_work():
    """Test that all required imports work."""
    assert niels_gpt is not None
    assert get_device is not None
    assert REPO_ROOT is not None
    assert ROAM_DIR is not None
    assert PRIMER_PATH is not None
    assert CHECKPOINT_DIR is not None
    assert CONFIG_DIR is not None
    assert ensure_dirs is not None


def test_device_is_valid():
    """Test that get_device returns a valid device."""
    device = get_device()
    assert device in ("mps", "cpu"), f"device must be 'mps' or 'cpu', got {device!r}"


def test_seed_repeatability():
    """Test that seed repeatability works for torch."""
    set_seed(42)
    g1 = make_generator(123)
    result1 = torch.randint(0, 10, (5,), generator=g1)

    g2 = make_generator(123)
    result2 = torch.randint(0, 10, (5,), generator=g2)

    assert torch.equal(result1, result2), "generators with same seed must produce same results"


def test_ensure_dirs_creates_expected_dirs():
    """Test that ensure_dirs creates the expected directories."""
    ensure_dirs()
    assert CHECKPOINT_DIR.exists(), f"CHECKPOINT_DIR {CHECKPOINT_DIR} should exist"
    assert CONFIG_DIR.exists(), f"CONFIG_DIR {CONFIG_DIR} should exist"
