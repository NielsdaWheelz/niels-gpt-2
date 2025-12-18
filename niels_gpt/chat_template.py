from __future__ import annotations

from typing import Literal, TypedDict

from niels_gpt.tokenizer import SentencePieceTokenizer


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


def format_chat(
    tok: SentencePieceTokenizer,
    messages: list[Message],
) -> list[int]:
    """
    Format a complete conversation with all messages as completed turns.

    Each message becomes:
    - SYS_TOKEN {content} EOT for system
    - USR_TOKEN {content} EOT for user
    - ASST_TOKEN {content} EOT for assistant

    Args:
        tok: Tokenizer instance
        messages: List of messages with role and content

    Returns:
        Token IDs for the full conversation with EOT after each message
    """
    special = tok.special_token_ids()
    result = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # Add role token
        if role == "system":
            result.append(special["sys"])
        elif role == "user":
            result.append(special["usr"])
        elif role == "assistant":
            result.append(special["asst"])
        else:
            raise ValueError(f"Invalid role: {role}")

        # Add content tokens
        result.extend(tok.encode(content))

        # Add end-of-turn token
        result.append(special["eot"])

    return result


def format_prompt(
    tok: SentencePieceTokenizer,
    messages: list[Message],
) -> list[int]:
    """
    Format messages for generation.

    - All prior messages are included as completed turns (each ends with EOT)
    - Appends ASST_TOKEN only at the end (no EOT) so the model continues as assistant

    Args:
        tok: Tokenizer instance
        messages: List of messages with role and content

    Returns:
        Token IDs ending with ASST_TOKEN (no trailing EOT)
    """
    special = tok.special_token_ids()
    result = []

    # Add all messages as completed turns
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # Add role token
        if role == "system":
            result.append(special["sys"])
        elif role == "user":
            result.append(special["usr"])
        elif role == "assistant":
            result.append(special["asst"])
        else:
            raise ValueError(f"Invalid role: {role}")

        # Add content tokens
        result.extend(tok.encode(content))

        # Add end-of-turn token
        result.append(special["eot"])

    # Append assistant token for generation (no eot)
    result.append(special["asst"])

    return result


def extract_assistant_reply(
    tok: SentencePieceTokenizer,
    generated_ids: list[int],
) -> list[int]:
    """
    Extract the assistant reply tokens from generated output.

    Given full generated token ids (including prompt):
    - Find the last assistant role token
    - Return tokens after that role token
    - Up to (but not including) the first EOT after that
    - If no EOT exists after the last assistant token, return tokens to end

    Args:
        tok: Tokenizer instance
        generated_ids: Full generated token sequence including prompt

    Returns:
        Token IDs of the assistant reply only
    """
    special = tok.special_token_ids()
    asst_id = special["asst"]
    eot_id = special["eot"]

    # Find the last occurrence of the assistant token
    last_asst_idx = -1
    for i in range(len(generated_ids) - 1, -1, -1):
        if generated_ids[i] == asst_id:
            last_asst_idx = i
            break

    if last_asst_idx == -1:
        # No assistant token found, return empty
        return []

    # Start after the assistant token
    start_idx = last_asst_idx + 1

    # Find the first EOT after the assistant token
    end_idx = len(generated_ids)
    for i in range(start_idx, len(generated_ids)):
        if generated_ids[i] == eot_id:
            end_idx = i
            break

    # Return the slice (not including eot)
    return generated_ids[start_idx:end_idx]
