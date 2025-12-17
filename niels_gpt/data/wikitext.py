from typing import Iterator, Optional
from datasets import load_dataset
from .types import PretrainSample


def load_wikitext() -> dict[str, list[str]]:
    """
    loads via datasets.load_dataset("wikitext", "wikitext-103-raw-v1")
    returns keys: "train", "val", "test"
    maps hf "validation" -> "val"
    drops empty-only lines: keep line iff line.strip() != ""
    """
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")

    result = {}

    # Process train split
    result["train"] = [line for line in ds["train"]["text"] if line.strip() != ""]

    # Process validation split -> "val"
    result["val"] = [line for line in ds["validation"]["text"] if line.strip() != ""]

    # Process test split
    result["test"] = [line for line in ds["test"]["text"] if line.strip() != ""]

    return result


def iter_wikitext(
    *,
    config: str = "wikitext-103-raw-v1",
    split: str = "train",
    take: Optional[int] = None,
) -> Iterator[PretrainSample]:
    """non-streaming ok; yields text lines as samples; skips empty strings."""
    ds = load_dataset("wikitext", config, split=split)

    count = 0
    for idx, sample in enumerate(ds):
        if take is not None and count >= take:
            break

        text = sample["text"]
        if text.strip() == "":
            continue

        yield PretrainSample(
            text=text,
            source="wikitext",
            meta={"index": idx},
        )
        count += 1
