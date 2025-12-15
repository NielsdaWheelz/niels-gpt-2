import random
from pathlib import Path


def list_roam_paths(root_dir: str) -> list[str]:
    """
    recursively list *.md under root_dir
    return absolute paths (as strings), sorted deterministically
    """
    root = Path(root_dir)
    md_files = sorted(root.rglob("*.md"))
    return [str(path.resolve()) for path in md_files]


def load_texts(paths: list[str]) -> list[str]:
    """
    read each file as utf-8 with errors="replace"
    returns list[str] same length/order
    """
    texts = []
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            texts.append(f.read())
    return texts


def split_roam_paths(
    paths: list[str],
    *,
    val_frac: float = 0.1,
    seed: int,
) -> tuple[list[str], list[str]]:
    """
    deterministic split by file path, not bytes
    uses seed for shuffle
    returns (train_paths, val_paths)

    rules:
    - if len(paths) < 2: val_paths == []
    - else val_size = max(1, int(len(paths) * val_frac))
    - ensure train and val are disjoint and cover all paths
    """
    if len(paths) < 2:
        return (paths, [])

    # Create a copy and shuffle deterministically
    shuffled = paths.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    # Calculate validation size
    val_size = max(1, int(len(paths) * val_frac))

    # Split
    val_paths = shuffled[:val_size]
    train_paths = shuffled[val_size:]

    return (train_paths, val_paths)
