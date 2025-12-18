import torch
import pytest

from niels_gpt.format.sft import (
    SFTFormatError,
    collate_sft_batch,
    pack_sft_ids_and_mask,
    serialize_chat_to_ids,
    sft_loss_mask_for_ids,
)
from niels_gpt.special_tokens import ASST_TOKEN, EOT_TOKEN, SYS_TOKEN, USR_TOKEN


class FakeTokenizer:
    def __init__(self):
        self.specials = {"sys": 0, "usr": 1, "asst": 2, "eot": 3}
        self._next_id = 10
        self._vocab: dict[str, int] = {}
        self._special_strings = {
            SYS_TOKEN: self.specials["sys"],
            USR_TOKEN: self.specials["usr"],
            ASST_TOKEN: self.specials["asst"],
            EOT_TOKEN: self.specials["eot"],
        }

    def special_token_ids(self):
        return self.specials.copy()

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        ids: list[int] = []
        for token in text.split(" "):
            if token in self._special_strings:
                ids.append(self._special_strings[token])
                continue
            if token not in self._vocab:
                self._vocab[token] = self._next_id
                self._next_id += 1
            ids.append(self._vocab[token])
        return ids


DEFAULT_SYSTEM = "you are a helpful assistant."


def test_serialize_injects_default_system():
    tok = FakeTokenizer()
    ex = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
    }
    ids = serialize_chat_to_ids(ex, tokenizer=tok, default_system_text=DEFAULT_SYSTEM)

    specials = tok.special_token_ids()
    expected = [
        specials["sys"],
        *tok.encode(DEFAULT_SYSTEM),
        specials["eot"],
        specials["usr"],
        *tok.encode("hello"),
        specials["eot"],
        specials["asst"],
        *tok.encode("hi there"),
        specials["eot"],
    ]

    assert ids == expected


def test_mask_marks_only_assistant_tokens():
    tok = FakeTokenizer()
    ex = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer ok"},
        ]
    }
    ids = serialize_chat_to_ids(ex, tokenizer=tok, default_system_text=DEFAULT_SYSTEM)
    specials = tok.special_token_ids()
    mask = sft_loss_mask_for_ids(
        ids,
        sys_id=specials["sys"],
        usr_id=specials["usr"],
        asst_id=specials["asst"],
        eot_id=specials["eot"],
    )

    sys_content = tok.encode("system prompt")
    user_content = tok.encode("question")
    asst_content = tok.encode("answer ok")

    expected_mask = (
        [False]  # sys token
        + [False] * len(sys_content)
        + [False]  # sys eot
        + [False]  # usr token
        + [False] * len(user_content)
        + [False]  # usr eot
        + [False]  # asst token
        + [True] * len(asst_content)
        + [False]  # assistant eot excluded from loss
    )

    assert mask == expected_mask


def test_truncation_drops_oldest_pairs_first():
    tok = FakeTokenizer()
    ex = {
        "messages": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
        ]
    }
    specials = tok.special_token_ids()
    ids = serialize_chat_to_ids(ex, tokenizer=tok, default_system_text="sys")
    mask = sft_loss_mask_for_ids(
        ids,
        sys_id=specials["sys"],
        usr_id=specials["usr"],
        asst_id=specials["asst"],
        eot_id=specials["eot"],
    )

    S = 15  # 3 turns (system + 2 pairs) of length 15 after dropping oldest pair
    packed_ids, packed_mask = pack_sft_ids_and_mask(
        ids,
        mask,
        S=S,
        sys_id=specials["sys"],
        usr_id=specials["usr"],
        asst_id=specials["asst"],
        eot_id=specials["eot"],
        pad_id=specials["eot"],
    )

    expected_ids = [
        specials["sys"],
        *tok.encode("sys"),
        specials["eot"],
        specials["usr"],
        *tok.encode("u2"),
        specials["eot"],
        specials["asst"],
        *tok.encode("a2"),
        specials["eot"],
        specials["usr"],
        *tok.encode("u3"),
        specials["eot"],
        specials["asst"],
        *tok.encode("a3"),
        specials["eot"],
    ]

    assert packed_ids == expected_ids
    assert len(packed_ids) == S
    assert len(packed_mask) == S


def test_hard_truncation_recomputes_mask_when_asst_token_dropped():
    tok = FakeTokenizer()
    specials = tok.special_token_ids()
    ex = {
        "messages": [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a b c d"},
        ]
    }
    ids = serialize_chat_to_ids(ex, tokenizer=tok, default_system_text="")
    mask = sft_loss_mask_for_ids(
        ids,
        sys_id=specials["sys"],
        usr_id=specials["usr"],
        asst_id=specials["asst"],
        eot_id=specials["eot"],
    )

    # force hard truncation to drop the assistant role token but keep content + eot
    S = 5
    packed_ids, packed_mask = pack_sft_ids_and_mask(
        ids,
        mask,
        S=S,
        sys_id=specials["sys"],
        usr_id=specials["usr"],
        asst_id=specials["asst"],
        eot_id=specials["eot"],
        pad_id=specials["eot"],
    )

    assert len(packed_ids) == S
    assert len(packed_mask) == S
    assert specials["asst"] not in packed_ids  # dropped
    assert not any(packed_mask), "mask must reset without asst_id in window"
    assert packed_ids[-1] == specials["eot"], "final eot preserved"


def test_hard_truncation_keeps_asst_when_span_fits():
    tok = FakeTokenizer()
    specials = tok.special_token_ids()
    ex = {
        "messages": [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a1 a2 a3 a4 a5 a6"},
        ]
    }
    ids = serialize_chat_to_ids(ex, tokenizer=tok, default_system_text="")
    mask = sft_loss_mask_for_ids(
        ids,
        sys_id=specials["sys"],
        usr_id=specials["usr"],
        asst_id=specials["asst"],
        eot_id=specials["eot"],
    )

    S = 10  # forces hard truncation but assistant span fits
    packed_ids, packed_mask = pack_sft_ids_and_mask(
        ids,
        mask,
        S=S,
        sys_id=specials["sys"],
        usr_id=specials["usr"],
        asst_id=specials["asst"],
        eot_id=specials["eot"],
        pad_id=specials["eot"],
    )

    assert packed_ids[0] == specials["asst"], "window must start at role boundary"
    assert packed_mask[0] is False
    assert specials["eot"] in packed_ids
    # mask should only turn True after the asst token within the window
    first_true = next((i for i, v in enumerate(packed_mask) if v), None)
    assert first_true is None or packed_ids[first_true - 1] == specials["asst"]


def test_padding_uses_eot_and_masks_out():
    tok = FakeTokenizer()
    specials = tok.special_token_ids()
    ex = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ]
    }
    ids = serialize_chat_to_ids(ex, tokenizer=tok, default_system_text="sys")
    mask = sft_loss_mask_for_ids(
        ids,
        sys_id=specials["sys"],
        usr_id=specials["usr"],
        asst_id=specials["asst"],
        eot_id=specials["eot"],
    )
    S = len(ids) + 3
    packed_ids, packed_mask = pack_sft_ids_and_mask(
        ids,
        mask,
        S=S,
        sys_id=specials["sys"],
        usr_id=specials["usr"],
        asst_id=specials["asst"],
        eot_id=specials["eot"],
        pad_id=specials["eot"],
    )

    assert packed_ids[-3:] == [specials["eot"]] * 3
    assert packed_mask[-3:] == [False, False, False]
    assert len(packed_ids) == S


def test_collate_aligns_and_applies_ignore_index():
    tok = FakeTokenizer()
    specials = tok.special_token_ids()
    ex = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
    }
    ids = serialize_chat_to_ids(ex, tokenizer=tok, default_system_text="")
    mask = sft_loss_mask_for_ids(
        ids,
        sys_id=specials["sys"],
        usr_id=specials["usr"],
        asst_id=specials["asst"],
        eot_id=specials["eot"],
    )

    S = len(ids)  # already fits
    packed = [
        pack_sft_ids_and_mask(
            ids,
            mask,
            S=S,
            sys_id=specials["sys"],
            usr_id=specials["usr"],
            asst_id=specials["asst"],
            eot_id=specials["eot"],
            pad_id=specials["eot"],
        )
    ]

    T = S - 1
    x, y, loss_mask = collate_sft_batch(packed, T=T, device="cpu", ignore_index=-100)

    expected_x = torch.tensor([ids[:T]], dtype=torch.long)
    y_raw = torch.tensor([ids[1:]], dtype=torch.long)
    expected_loss_mask = torch.tensor([mask[1:]], dtype=torch.bool)
    expected_y = y_raw.clone()
    expected_y[~expected_loss_mask] = -100

    assert torch.equal(x, expected_x)
    assert torch.equal(y, expected_y)
    assert torch.equal(loss_mask, expected_loss_mask)


def test_role_normalization_and_error_context():
    tok = FakeTokenizer()
    specials = tok.special_token_ids()
    # prompter alias + uppercase assistant accepted
    ex = {
        "id": "ex-123",
        "messages": [
            {"role": "PROMPTER", "content": "hi"},
            {"role": "ASSISTANT", "content": "ok"},
        ],
    }
    ids = serialize_chat_to_ids(ex, tokenizer=tok, default_system_text="")
    # layout: <sys> <eot> <usr> "hi" <eot> <asst> "ok" <eot>
    assert ids[2] == specials["usr"]  # prompter -> usr
    assert ids[-3] == specials["asst"]

    # unknown role raises with example id context
    bad = {"id": "bad-1", "messages": [{"role": "moderator", "content": "x"}]}
    try:
        serialize_chat_to_ids(bad, tokenizer=tok, default_system_text="")
    except SFTFormatError as exc:
        assert "example_id=bad-1" in str(exc)
    else:  # pragma: no cover - defensive
        assert False, "expected SFTFormatError"


def test_content_with_literal_special_tokens_is_escaped():
    tok = FakeTokenizer()
    content = f"please do not emit {USR_TOKEN} or {ASST_TOKEN} in replies"
    ex = {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": content},
            {"role": "assistant", "content": "ok"},
        ]
    }
    with pytest.raises(ValueError, match="special token literal detected"):
        serialize_chat_to_ids(ex, tokenizer=tok, default_system_text="")


