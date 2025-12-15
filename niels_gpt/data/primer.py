import random


DIALOGUE_DELIM = "\n\n<dialogue>\n\n"


def load_primer_text(path: str) -> str:
    """returns raw text exactly as stored (utf-8, errors='replace')."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def split_primer_dialogues(
    text: str,
    *,
    val_frac: float = 0.1,
    seed: int,
) -> tuple[str, str]:
    """
    split by dialogue blocks (not bytes).
    delimiter: DIALOGUE_DELIM

    parse rules:
    - blocks = text.split(DIALOGUE_DELIM)
    - for each block: block_stripped = block.strip()
    - drop empty blocks (block_stripped == "")
    - split blocks deterministically using seed

    split rules:
    - if num_blocks < 2: val_text == ""
    - else val_size = max(1, int(num_blocks * val_frac))
    - return (train_text, val_text) as DIALOGUE_DELIM-joined blocks (no leading/trailing delimiter)
    """
    # Split and filter empty blocks
    blocks = text.split(DIALOGUE_DELIM)
    blocks = [block.strip() for block in blocks if block.strip() != ""]

    # Handle edge case
    if len(blocks) < 2:
        train_text = DIALOGUE_DELIM.join(blocks) if blocks else ""
        return (train_text, "")

    # Shuffle deterministically
    shuffled = blocks.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    # Calculate validation size
    val_size = max(1, int(len(blocks) * val_frac))

    # Split
    val_blocks = shuffled[:val_size]
    train_blocks = shuffled[val_size:]

    # Join back
    train_text = DIALOGUE_DELIM.join(train_blocks)
    val_text = DIALOGUE_DELIM.join(val_blocks)

    return (train_text, val_text)
