from typing import Iterator, Optional
from datasets import load_dataset
from .types import PretrainSample


def _select_text_field(sample: dict) -> str:
    """
    Select text field from sample dict using explicit whitelist.
    - tries "text", "content", "document" in order
    - else: raise ValueError with sample keys shown
    """
    for field in ["text", "content", "document"]:
        if field in sample and isinstance(sample[field], str):
            return sample[field]

    # No whitelisted string fields found
    raise ValueError(
        f"No text field found in sample. "
        f"Tried: text, content, document. "
        f"Available keys: {list(sample.keys())}"
    )


def iter_fineweb_edu(
    *,
    name: str = "CC-MAIN-2024-10",
    split: str = "train",
    streaming: bool = True,
    shuffle: bool = True,
    shuffle_buffer_size: int = 10_000,
    seed: int = 42,
    take: Optional[int] = None,
) -> Iterator[PretrainSample]:
    """
    yields PretrainSample(text=..., source="fineweb_edu", meta=...)
    requirements:
      - must use datasets.load_dataset(..., streaming=True) by default
      - if shuffle=True, must use iterable_dataset.shuffle(seed=..., buffer_size=...)
      - if take is not None, must yield at most take samples
    field selection rule:
      - prefer sample["text"]
      - else: pick the first string field in the sample dict
      - else: raise ValueError with sample keys shown
    """
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=name,
        split=split,
        streaming=streaming,
    )

    if shuffle:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

    count = 0
    for sample in ds:
        if take is not None and count >= take:
            break

        text = _select_text_field(sample)

        # Keep raw metadata from source (unfiltered)
        yield PretrainSample(
            text=text,
            source="fineweb_edu",
            meta=dict(sample),
        )
        count += 1
