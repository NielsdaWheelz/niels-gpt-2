from src.chat_format import format_chat, extract_assistant_reply


def test_format_chat_basic():
    messages = [
        {"role": "system", "content": "x"},
        {"role": "user", "content": "y"},
    ]
    result = format_chat(messages)
    expected = "system: x\nuser: y\nassistant: "
    assert result == expected


def test_format_chat_last_empty_assistant_is_prompt():
    messages = [
        {"role": "system", "content": "x"},
        {"role": "user", "content": "y"},
        {"role": "assistant", "content": ""},
    ]
    result = format_chat(messages)
    expected = "system: x\nuser: y\nassistant: "
    assert result == expected


def test_format_chat_rejects_invalid_role():
    messages = [
        {"role": "invalid", "content": "test"},
    ]
    try:
        format_chat(messages)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_extract_assistant_reply_stops_at_next_tag():
    generated = "system: x\nuser: y\nassistant: hello there\nuser: z\nassistant: "
    reply = extract_assistant_reply(generated)
    assert reply == "hello there"


def test_extract_assistant_reply_uses_last_assistant_tag():
    generated = "assistant: first\nuser: u\nassistant: second\n"
    reply = extract_assistant_reply(generated)
    assert reply == "second"


def test_extract_assistant_reply_raises_if_missing():
    generated = "system: x\nuser: y\n"
    try:
        extract_assistant_reply(generated)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_extract_assistant_reply_all_empty_returns_empty():
    generated = "user: hello\nassistant: \n"
    reply = extract_assistant_reply(generated)
    assert reply == ""
