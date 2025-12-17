from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class PretrainSample:
    """
    A single pretrain text sample.

    Fields:
        text: Raw text content
        source: Dataset identifier (e.g., "fineweb_edu", "wikitext")
        meta: Raw metadata dict from source dataset (unfiltered, may contain any types)
    """
    text: str
    source: str
    meta: dict


@dataclass(frozen=True)
class ChatMessage:
    """A single chat message with role and content."""
    role: Role
    content: str


@dataclass(frozen=True)
class ChatSample:
    """
    A chat conversation sample.

    Fields:
        messages: List of ChatMessage objects forming a conversation
        source: Dataset identifier (e.g., "dolly", "oasst1")
        meta: Raw metadata dict from source dataset (unfiltered, may contain any types)
    """
    messages: list[ChatMessage]
    source: str
    meta: dict
