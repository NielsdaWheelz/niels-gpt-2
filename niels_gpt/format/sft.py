from __future__ import annotations

from typing import Iterable, List, Tuple

import torch

from niels_gpt.tokenizer import SPECIAL_TOKENS


class SFTFormatError(ValueError):
    def __init__(self, message: str, *, example_id: str | None = None):
        prefix = f"[example_id={example_id}] " if example_id is not None else ""
        super().__init__(prefix + message)


def _escape_special_token_literals(text: str, specials: Iterable[str]) -> str:
    """
    Replace any literal occurrences of special token strings to avoid them
    being encoded as structural tokens inside content.
    """
    escaped = text
    for tok in specials:
        escaped = escaped.replace(tok, tok.replace("|", "\\|"))
    return escaped


def serialize_chat_to_ids(
    ex: dict,
    *,
    tokenizer,
    default_system_text: str,
) -> list[int]:
    """
    Serialize a normalized chat example into token IDs using the token-native
    template. Always inject a system turn if one is missing.
    """
    messages: list[dict] = ex.get("messages", [])
    example_id = ex.get("id")
    special_ids = tokenizer.special_token_ids()
    sys_id = special_ids["sys"]
    usr_id = special_ids["usr"]
    asst_id = special_ids["asst"]
    eot_id = special_ids["eot"]

    role_to_id = {
        "system": sys_id,
        "user": usr_id,
        "assistant": asst_id,
        "prompter": usr_id,  # alias some datasets use
    }

    serialized: list[int] = []

    # ensure a leading system message
    if messages and messages[0].get("role") == "system":
        system_text = messages[0]["content"]
        start_idx = 1
    else:
        system_text = default_system_text
        start_idx = 0

    serialized.append(sys_id)
    serialized.extend(tokenizer.encode(_escape_special_token_literals(system_text, SPECIAL_TOKENS)))
    serialized.append(eot_id)

    for msg in messages[start_idx:]:
        try:
            role_raw = msg["role"]
            content = msg.get("content", "")
        except Exception as exc:  # pragma: no cover - defensive
            raise SFTFormatError("malformed message", example_id=example_id) from exc

        role = str(role_raw).lower()
        if role not in role_to_id:
            raise SFTFormatError(f"invalid role: {role}", example_id=example_id)

        serialized.append(role_to_id[role])
        serialized.extend(tokenizer.encode(_escape_special_token_literals(content, SPECIAL_TOKENS)))
        serialized.append(eot_id)

    return serialized


def sft_loss_mask_for_ids(
    ids: list[int],
    *,
    sys_id: int,
    usr_id: int,
    asst_id: int,
    eot_id: int,
) -> list[bool]:
    """
    Build an assistant-only loss mask matching the ids length.
    """
    mask: list[bool] = []
    in_assistant_span = False

    for token_id in ids:
        if token_id == asst_id:
            mask.append(False)
            in_assistant_span = True
            continue

        if in_assistant_span:
            mask.append(True)
            if token_id == eot_id:
                in_assistant_span = False
            continue

        mask.append(False)

    return mask


def _parse_segments(
    ids: list[int],
    sys_id: int,
    usr_id: int,
    asst_id: int,
    eot_id: int,
) -> list[tuple[str, int, int]]:
    """
    Return a list of (role, start_idx, end_idx) segments bounded by role tokens
    and the following eot. Inclusive end indices.
    """
    role_lookup = {
        sys_id: "sys",
        usr_id: "usr",
        asst_id: "asst",
    }
    segments: list[tuple[str, int, int]] = []
    i = 0
    n = len(ids)

    while i < n:
        token_id = ids[i]
        if token_id not in role_lookup:
            i += 1
            continue

        role = role_lookup[token_id]
        start = i
        i += 1
        while i < n and ids[i] != eot_id:
            i += 1
        if i < n:
            end = i
            i += 1  # consume eot
        else:
            end = n - 1
        segments.append((role, start, end))

    return segments


def _choose_window_start(
    segments: list[tuple[str, int, int]],
    *,
    target_start: int,
) -> int:
    """
    Pick a start index aligned to the nearest role token boundary at or after
    target_start, preferring assistant, then user, then system.
    """
    role_priority = ["asst", "usr", "sys"]
    starts_by_role = {r: [] for r in role_priority}

    for role, start, _ in segments:
        if start >= target_start and role in starts_by_role:
            starts_by_role[role].append(start)

    for role in role_priority:
        if starts_by_role[role]:
            return min(starts_by_role[role])

    # if no boundary after target, fall back to target (may be mid-span)
    return target_start


def pack_sft_ids_and_mask(
    ids: list[int],
    mask: list[bool],
    *,
    S: int,
    sys_id: int,
    usr_id: int,
    asst_id: int,
    eot_id: int,
    pad_id: int,
) -> tuple[list[int], list[bool]]:
    """
    Truncate/pad to length S while preserving the last assistant turn when
    possible. Drops oldest user+assistant pairs first, then hard-truncates,
    and finally pads with pad_id (mask False).
    """
    if len(ids) != len(mask):
        raise ValueError("ids and mask must have the same length")

    ids_work = list(ids)

    # drop whole user+assistant pairs after the system turn until length <= S
    while len(ids_work) > S:
        segments = _parse_segments(ids_work, sys_id, usr_id, asst_id, eot_id)
        system_idx = next((i for i, seg in enumerate(segments) if seg[0] == "sys"), None)
        last_asst_idx = max((i for i, seg in enumerate(segments) if seg[0] == "asst"), default=None)

        drop_range: tuple[int, int] | None = None
        start_search = system_idx + 1 if system_idx is not None else 0
        for seg_idx in range(start_search, len(segments) - 1):
            role_curr, start_curr, _ = segments[seg_idx]
            role_next, _, end_next = segments[seg_idx + 1]
            if role_curr == "usr" and role_next == "asst":
                if last_asst_idx is not None and seg_idx + 1 == last_asst_idx:
                    continue
                drop_range = (start_curr, end_next)
                break

        if drop_range is None:
            break

        start, end = drop_range
        del ids_work[start : end + 1]
        # mask will be recomputed after truncation; no need to delete in lockstep

    if len(ids_work) > S:
        segments = _parse_segments(ids_work, sys_id, usr_id, asst_id, eot_id)
        target_start = len(ids_work) - S

        # prefer starting at the last assistant turn if it fits entirely
        last_asst = None
        for role, start, end in reversed(segments):
            if role == "asst" and ids_work[end] == eot_id:
                last_asst = (start, end)
                break

        if last_asst is not None and (len(ids_work) - last_asst[0]) <= S:
            start_idx = max(last_asst[0], target_start)
        else:
            start_idx = _choose_window_start(segments, target_start=target_start)

        ids_work = ids_work[start_idx:]

    # recompute mask to avoid stale spans when hard-truncating mid-turn
    mask_work = sft_loss_mask_for_ids(
        ids_work,
        sys_id=sys_id,
        usr_id=usr_id,
        asst_id=asst_id,
        eot_id=eot_id,
    )

    # pad with pad_id (mask False) if too short
    if len(ids_work) < S:
        pad_len = S - len(ids_work)
        ids_work.extend([pad_id] * pad_len)
        mask_work.extend([False] * pad_len)

    return ids_work, mask_work


def collate_sft_batch(
    packed: list[tuple[list[int], list[bool]]],
    *,
    T: int,
    device: str,
    ignore_index: int = -100,
) -> tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]:
    """
    Collate packed samples (each length S=T+1) into tensors.
    """
    if not packed:
        raise ValueError("packed batch must be non-empty")

    ids_tensor = torch.tensor([item[0] for item in packed], dtype=torch.long, device=device)
    mask_tensor = torch.tensor([item[1] for item in packed], dtype=torch.bool, device=device)

    if ids_tensor.shape[1] != T + 1 or mask_tensor.shape[1] != T + 1:
        raise ValueError("each packed sample must have length T+1")

    x = ids_tensor[:, :T]
    y_raw = ids_tensor[:, 1:]
    loss_mask = mask_tensor[:, 1:]

    y = y_raw.clone()
    y[~loss_mask] = ignore_index

    return x, y, loss_mask


