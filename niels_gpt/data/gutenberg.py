from typing import Iterator, Optional
from datasets import load_dataset
from niels_gpt.settings import default_settings
from niels_gpt.special_tokens import assert_no_special_collision
from .types import PretrainSample


_DEFAULT_SHUFFLE = default_settings().reproducibility.dataset_shuffle_buffer_size


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


def iter_gutenberg(
    *,
    dataset_id: str = "nikolina-p/gutenberg_clean_en_splits",
    split: str = "train",
    streaming: bool = True,
    shuffle: bool = True,
    shuffle_buffer_size: int = _DEFAULT_SHUFFLE,
    seed: int = 42,
    take: Optional[int] = None,
) -> Iterator[PretrainSample]:
    """
    yields book texts as PretrainSample.

    field selection rule:
      - prefer "text"
      - else first string field
      - else error
    """
    ds = load_dataset(
        dataset_id,
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

        assert_no_special_collision(text, dataset="gutenberg", doc_index=count, field="text")

        # Keep raw metadata from source (unfiltered)
        yield PretrainSample(
            text=text,
            source="gutenberg",
            meta=dict(sample),
        )
        count += 1
