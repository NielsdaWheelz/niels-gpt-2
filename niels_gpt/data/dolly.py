from typing import Iterator, Optional
import random
from datasets import load_dataset
from .types import ChatSample, ChatMessage


def iter_dolly_sft(
    *,
    split: str = "train",
    seed: int = 42,
    shuffle: bool = True,
    take: Optional[int] = None,
    include_system: bool = False,
) -> Iterator[ChatSample]:
    """
    loads databricks/databricks-dolly-15k.
    mapping:
      - if include_system: add system msg "you are a helpful assistant."
      - user content:
          instruction
          + if context non-empty (after .strip()): "\\n\\ncontext:\\n{context}"
      - assistant content: response
    meta: raw metadata from source (unfiltered)
    """
    ds = load_dataset("databricks/databricks-dolly-15k", split=split)

    # Convert to list for shuffling if needed
    if shuffle:
        indices = list(range(len(ds)))
        rng = random.Random(seed)
        rng.shuffle(indices)
    else:
        indices = list(range(len(ds)))

    count = 0
    for idx in indices:
        if take is not None and count >= take:
            break

        sample = ds[idx]

        messages = []

        # Add system message if requested
        if include_system:
            system_text = "you are a helpful assistant."
            messages.append(ChatMessage(role="system", content=system_text))

        # Build user content
        user_content = sample["instruction"]
        if sample.get("context", "").strip():
            user_content += f"\n\ncontext:\n{sample['context']}"

        messages.append(ChatMessage(role="user", content=user_content))
        messages.append(ChatMessage(role="assistant", content=sample["response"]))

        # Keep raw metadata, add index
        meta = dict(sample)
        meta["index"] = idx

        yield ChatSample(
            messages=messages,
            source="dolly",
            meta=meta,
        )
        count += 1
