"""
bench_configs.py: Device-aware benchmark configuration grids.

Automatically selects appropriate grid based on device type and available memory.
"""

import torch

from niels_gpt.settings import default_settings


def _get_vram_gb() -> float:
    """Get CUDA VRAM in GB, or 0 if not available."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def _build_grid(
    candidate_T: list[int],
    candidate_models: list[dict[str, int]],
    checkpointing_modes: list[bool],
    amp: bool = True,
    amp_dtype: str = "fp16",
) -> list[dict]:
    """Build benchmark grid from components."""
    grid: list[dict] = []
    for T in candidate_T:
        for model_cfg in candidate_models:
            for ckpt in checkpointing_modes:
                d_ff = model_cfg.get("d_ff", 3 * model_cfg["C"])
                grid.append(
                    {
                        "T": T,
                        "C": model_cfg["C"],
                        "L": model_cfg["L"],
                        "H": model_cfg["H"],
                        "d_ff": d_ff,
                        "amp": amp,
                        "amp_dtype": amp_dtype,
                        "activation_checkpointing": ckpt,
                    }
                )
    return grid


def get_default_grid(device: str = "auto") -> list[dict]:
    """
    Return device-appropriate benchmark grid.

    Args:
        device: "cuda", "mps", "cpu", or "auto"

    Returns:
        List of benchmark configurations appropriate for the device's memory.

    Grid scaling by VRAM:
        - CUDA <10GB (8GB class: 2080, 3070, etc.):
          Conservative grid, T=[256,512], single model size
        - CUDA 10-16GB (12GB class: 3080, 4070, etc.):
          Medium grid, T=[256,512,1024], two model sizes
        - CUDA 16GB+ (24GB class: 3090, 4090, A100, etc.):
          Full grid, all T values and model sizes
        - MPS (Apple Silicon):
          Full grid (unified memory handles larger configs well)
        - CPU:
          Minimal grid for smoke testing only
    """
    settings = default_settings()
    bench = settings.benchmark

    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if device == "cuda":
        vram_gb = _get_vram_gb()
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
        print(f"CUDA device: {gpu_name} ({vram_gb:.1f} GB VRAM)")

        if vram_gb < 10:
            # 8GB class (RTX 2080, 3070, 4060, etc.)
            print("Using 8GB VRAM grid (conservative)")
            return _build_grid(
                candidate_T=[256, 512],
                candidate_models=[
                    {"C": 384, "L": 8, "H": 6},  # ~25M params
                ],
                checkpointing_modes=[False, True],
                amp=True,
                amp_dtype="fp16",
            )
        elif vram_gb < 16:
            # 12GB class (RTX 3080, 4070, etc.)
            print("Using 12GB VRAM grid (medium)")
            return _build_grid(
                candidate_T=[256, 512],
                candidate_models=[
                    {"C": 384, "L": 8, "H": 6},   # ~25M params
                    {"C": 512, "L": 8, "H": 8},   # ~50M params
                ],
                checkpointing_modes=[False, True],
                amp=True,
                amp_dtype="fp16",
            )
        else:
            # 24GB+ class (RTX 3090, 4090, A100, etc.)
            print("Using 24GB+ VRAM grid (full)")
            return _build_grid(
                candidate_T=[512, 1024],
                candidate_models=[
                    {"C": 384, "L": 8, "H": 6},
                    {"C": 512, "L": 8, "H": 8},
                    {"C": 512, "L": 12, "H": 8},
                ],
                checkpointing_modes=[False, True],
                amp=True,
                amp_dtype="fp16",
            )

    elif device == "mps":
        # Apple Silicon with unified memory - can handle larger configs
        print("Using MPS grid (unified memory)")
        return _build_grid(
            candidate_T=bench.candidate_T,
            candidate_models=bench.candidate_model_dims,
            checkpointing_modes=bench.checkpointing_modes,
            amp=False,  # AMP disabled on MPS (stability issues)
            amp_dtype="fp16",
        )

    else:
        # CPU - minimal grid for smoke testing
        print("Using CPU grid (minimal)")
        return _build_grid(
            candidate_T=[128],
            candidate_models=[
                {"C": 256, "L": 4, "H": 4},
            ],
            checkpointing_modes=[False],
            amp=False,
            amp_dtype="fp16",
        )
