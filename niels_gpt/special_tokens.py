"""Project-wide special token definitions and helpers."""

from typing import Sequence

SENTINEL_UUID = "84a5023f67d74cf29cc4001becde983c"

# Frozen sentinel strings (ASCII, no whitespace, project-specific).
SYS_TOKEN = f"<|ngpt_sys_{SENTINEL_UUID}|>"
USR_TOKEN = f"<|ngpt_usr_{SENTINEL_UUID}|>"
ASST_TOKEN = f"<|ngpt_asst_{SENTINEL_UUID}|>"
EOT_TOKEN = f"<|ngpt_eot_{SENTINEL_UUID}|>"

SPECIAL_TOKEN_NAMES = ("sys", "usr", "asst", "eot")
SPECIAL_TOKENS = (SYS_TOKEN, USR_TOKEN, ASST_TOKEN, EOT_TOKEN)


def _escape_snippet(snippet: str) -> str:
    """Escape snippet to ASCII-safe form for error messages."""
    return snippet.encode("unicode_escape").decode("ascii", errors="replace")


def find_special_collision(
    text: str,
    *,
    specials: Sequence[str] = SPECIAL_TOKENS,
) -> dict[str, int | str] | None:
    """
    Return collision details if any special token literal appears in text.
    Details: token, char_offset, byte_offset, snippet.
    """
    for tok in specials:
        pos = text.find(tok)
        if pos != -1:
            byte_offset = len(text[:pos].encode("utf-8", errors="replace"))
            snippet = _escape_snippet(text[pos : pos + 80])
            return {
                "token": tok,
                "char_offset": pos,
                "byte_offset": byte_offset,
                "snippet": snippet,
            }
    return None


def assert_no_special_collision(
    text: str,
    *,
    dataset: str,
    doc_index: int | str,
    field: str = "text",
    specials: Sequence[str] = SPECIAL_TOKENS,
) -> None:
    """
    Raise ValueError if any sentinel literal appears in text.
    """
    collision = find_special_collision(text, specials=specials)
    if collision:
        raise ValueError(
            "special token literal detected: "
            f"dataset={dataset}, doc_index={doc_index}, field={field}, "
            f"token={collision['token']}, byte_offset={collision['byte_offset']}, "
            f"snippet='{collision['snippet']}'"
        )

