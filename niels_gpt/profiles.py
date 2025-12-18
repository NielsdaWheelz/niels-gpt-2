"""Configuration profile registry and metadata.

This module defines all available configuration profiles (presets) for training.
Profiles are organized by purpose: smoke tests, development, and production.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import niels_gpt.paths as paths

PhaseType = Literal["pretrain", "sft", "pipeline"]


@dataclass(frozen=True)
class ProfileMetadata:
    """Metadata describing a configuration profile."""

    name: str
    path: Path
    phase: PhaseType
    description: str
    notes: str = ""

    def exists(self) -> bool:
        """Check if the profile config file exists."""
        return self.path.exists()


# ----------------------------- profile registry ----------------------------- #

PROFILES: dict[str, ProfileMetadata] = {
    # Smoke tests - ultra-fast validation (2 steps, tiny models)
    "smoke-pretrain": ProfileMetadata(
        name="smoke-pretrain",
        path=paths.CONFIG_DIR / "smoke-pretrain.json",
        phase="pretrain",
        description="Ultra-fast pretrain validation (2 steps, tiny model: V=256, T=32, C=64, L=2)",
        notes="Runs in ~30s on CPU. Disables AMP for determinism. Use for CI/CD pipeline tests.",
    ),
    "smoke-sft": ProfileMetadata(
        name="smoke-sft",
        path=paths.CONFIG_DIR / "smoke-sft.json",
        phase="sft",
        description="Ultra-fast SFT validation (2 steps)",
        notes="Runs in ~30s on CPU. Disables AMP for determinism. Use for CI/CD pipeline tests.",
    ),
    # Development profiles - laptop-friendly iteration
    "dev": ProfileMetadata(
        name="dev",
        path=paths.CONFIG_DIR / "dev-pretrain.json",
        phase="pretrain",
        description="Development pretrain (1K steps, small model: V=256, T=128, C=256, L=4)",
        notes="Suitable for local iteration and quick experiments. Completes in hours on laptop.",
    ),
    "dev-sft": ProfileMetadata(
        name="dev-sft",
        path=paths.CONFIG_DIR / "dev-sft.json",
        phase="sft",
        description="Development SFT (small model, quick iteration)",
        notes="Suitable for local SFT experiments. Uses V=256 vocab for consistency with dev pretrain.",
    ),
    # Production profiles - full-scale training
    "prod": ProfileMetadata(
        name="prod",
        path=paths.CONFIG_DIR / "prod-pretrain.json",
        phase="pretrain",
        description="Production pretrain (full model: V=50257, T=512, C=384, L=8, H=6)",
        notes="GPT-2 vocab size, 8-layer model. Requires GPU. 10K+ steps.",
    ),
    "prod-sft": ProfileMetadata(
        name="prod-sft",
        path=paths.CONFIG_DIR / "prod-sft.json",
        phase="sft",
        description="Production SFT (10K steps, full model)",
        notes="Use after prod pretrain. Requires GPU.",
    ),
    # Minimal/baseline profiles
    "minimal": ProfileMetadata(
        name="minimal",
        path=paths.CONFIG_DIR / "minimal-pretrain.json",
        phase="pretrain",
        description="Minimal pretrain config (V=256, mostly defaults)",
        notes="Bare-bones config with minimal overrides. Good starting point for customization.",
    ),
    "minimal-sft": ProfileMetadata(
        name="minimal-sft",
        path=paths.CONFIG_DIR / "minimal-sft.json",
        phase="sft",
        description="Minimal SFT config (mostly defaults)",
        notes="Bare-bones SFT config. Good starting point for customization.",
    ),
    # Pipeline profiles
    "pipeline-dev": ProfileMetadata(
        name="pipeline-dev",
        path=paths.CONFIG_DIR / "pipeline-dev.json",
        phase="pipeline",
        description="Development pipeline (pretrain → SFT with small models)",
        notes="Runs full two-phase training with dev-sized models.",
    ),
    "pipeline-prod": ProfileMetadata(
        name="pipeline-prod",
        path=paths.CONFIG_DIR / "pipeline-prod.json",
        phase="pipeline",
        description="Production pipeline (pretrain → SFT with full models)",
        notes="Full two-phase training. Requires GPU and significant time.",
    ),
}


# ----------------------------- helper functions ----------------------------- #


def get_profile(name: str) -> ProfileMetadata:
    """Get profile metadata by name.

    Args:
        name: Profile name (e.g., 'smoke-pretrain', 'dev', 'prod')

    Returns:
        ProfileMetadata for the requested profile

    Raises:
        KeyError: If profile name is not found
    """
    if name not in PROFILES:
        raise KeyError(
            f"unknown profile '{name}'. Available profiles: {sorted(PROFILES.keys())}\n"
            f"Use --list-profiles to see all available profiles."
        )
    return PROFILES[name]


def list_profiles(phase: PhaseType | None = None) -> list[ProfileMetadata]:
    """List all available profiles, optionally filtered by phase.

    Args:
        phase: Optional phase filter ('pretrain', 'sft', or 'pipeline')

    Returns:
        List of ProfileMetadata objects, sorted by name
    """
    profiles = list(PROFILES.values())
    if phase:
        profiles = [p for p in profiles if p.phase == phase]
    return sorted(profiles, key=lambda p: p.name)


def format_profile_list(phase: PhaseType | None = None) -> str:
    """Format profile list as human-readable string.

    Args:
        phase: Optional phase filter

    Returns:
        Formatted string describing all profiles
    """
    profiles = list_profiles(phase)
    if not profiles:
        return "No profiles available."

    lines = ["Available profiles:"]
    lines.append("")

    # Group by category (smoke, dev, prod, minimal, pipeline)
    categories = {
        "Smoke Tests": [p for p in profiles if p.name.startswith("smoke")],
        "Development": [p for p in profiles if p.name.startswith("dev") or p.name == "dev"],
        "Production": [p for p in profiles if p.name.startswith("prod")],
        "Minimal/Baseline": [p for p in profiles if p.name.startswith("minimal")],
        "Pipelines": [p for p in profiles if p.name.startswith("pipeline")],
    }

    for category, cat_profiles in categories.items():
        if not cat_profiles:
            continue
        lines.append(f"{category}:")
        for profile in cat_profiles:
            status = "✓" if profile.exists() else "✗"
            lines.append(f"  [{status}] {profile.name:20s} ({profile.phase})")
            lines.append(f"      {profile.description}")
            if profile.notes:
                lines.append(f"      Notes: {profile.notes}")
        lines.append("")

    return "\n".join(lines)
