"""
bench_configs.py: Default benchmark configuration grid.

Defines the default grid of model configs to benchmark.
"""


def get_default_grid() -> list[dict]:
    """
    Return the default benchmark grid.

    Grid includes:
    - T ∈ {512, 1024}
    - (C, L, H) ∈ {(384, 8, 6), (512, 8, 8), (512, 12, 8)}
    - amp = true, amp_dtype = fp16
    - activation_checkpointing ∈ {false, true}

    Returns a list of config dicts with keys: T, C, L, H, amp, activation_checkpointing
    """
    grid = []

    T_values = [512, 1024]
    model_configs = [
        {"C": 384, "L": 8, "H": 6},
        {"C": 512, "L": 8, "H": 8},
        {"C": 512, "L": 12, "H": 8},
    ]
    ckpt_values = [False, True]

    for T in T_values:
        for model_cfg in model_configs:
            for ckpt in ckpt_values:
                # Compute d_ff = 3 * C (default ratio)
                d_ff = 3 * model_cfg["C"]

                grid.append({
                    "T": T,
                    "C": model_cfg["C"],
                    "L": model_cfg["L"],
                    "H": model_cfg["H"],
                    "d_ff": d_ff,
                    "amp": True,
                    "amp_dtype": "fp16",
                    "activation_checkpointing": ckpt,
                })

    return grid
