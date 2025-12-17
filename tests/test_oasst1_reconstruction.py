"""
Unit tests for oasst1 thread reconstruction logic.
No network access - uses synthetic data.
"""
from niels_gpt.data.oasst1 import _reconstruct_threads


def test_oasst1_single_thread():
    """Test reconstruction of a simple linear thread."""
    records = [
        {
            "message_id": "msg1",
            "parent_id": None,
            "message_tree_id": "tree1",
            "role": "prompter",
            "text": "Hello, what is AI?",
            "lang": "en",
        },
        {
            "message_id": "msg2",
            "parent_id": "msg1",
            "message_tree_id": "tree1",
            "role": "assistant",
            "text": "AI stands for Artificial Intelligence.",
            "lang": "en",
        },
    ]

    threads = _reconstruct_threads(records, english_only=True, max_messages=32)

    assert len(threads) == 1
    thread = threads[0]

    assert len(thread["messages"]) == 2
    assert thread["messages"][0].role == "user"
    assert thread["messages"][0].content == "Hello, what is AI?"
    assert thread["messages"][1].role == "assistant"
    assert thread["messages"][1].content == "AI stands for Artificial Intelligence."

    assert thread["tree_id"] == "tree1"
    assert thread["leaf_id"] == "msg2"
    assert thread["message_ids"] == ["msg1", "msg2"]


def test_oasst1_branching_thread():
    """Test reconstruction with two branches from the same root."""
    records = [
        {
            "message_id": "msg1",
            "parent_id": None,
            "message_tree_id": "tree1",
            "role": "prompter",
            "text": "What is Python?",
            "lang": "en",
        },
        {
            "message_id": "msg2a",
            "parent_id": "msg1",
            "message_tree_id": "tree1",
            "role": "assistant",
            "text": "Python is a snake.",
            "lang": "en",
        },
        {
            "message_id": "msg2b",
            "parent_id": "msg1",
            "message_tree_id": "tree1",
            "role": "assistant",
            "text": "Python is a programming language.",
            "lang": "en",
        },
    ]

    threads = _reconstruct_threads(records, english_only=True, max_messages=32)

    assert len(threads) == 2

    # Check that we have both branches
    leaf_ids = {t["leaf_id"] for t in threads}
    assert leaf_ids == {"msg2a", "msg2b"}

    # Both should have same root
    for thread in threads:
        assert thread["message_ids"][0] == "msg1"
        assert len(thread["messages"]) == 2


def test_oasst1_max_messages():
    """Test that max_messages is enforced."""
    records = [
        {
            "message_id": "msg1",
            "parent_id": None,
            "message_tree_id": "tree1",
            "role": "prompter",
            "text": "First message",
            "lang": "en",
        },
        {
            "message_id": "msg2",
            "parent_id": "msg1",
            "message_tree_id": "tree1",
            "role": "assistant",
            "text": "Second message",
            "lang": "en",
        },
        {
            "message_id": "msg3",
            "parent_id": "msg2",
            "message_tree_id": "tree1",
            "role": "prompter",
            "text": "Third message",
            "lang": "en",
        },
        {
            "message_id": "msg4",
            "parent_id": "msg3",
            "message_tree_id": "tree1",
            "role": "assistant",
            "text": "Fourth message",
            "lang": "en",
        },
    ]

    # Limit to 2 messages
    threads = _reconstruct_threads(records, english_only=True, max_messages=2)

    assert len(threads) == 1
    thread = threads[0]

    assert len(thread["messages"]) == 2
    assert thread["message_ids"] == ["msg1", "msg2"]


def test_oasst1_english_only_filter():
    """Test that english_only filters non-English messages."""
    records = [
        {
            "message_id": "msg1",
            "parent_id": None,
            "message_tree_id": "tree1",
            "role": "prompter",
            "text": "Hello",
            "lang": "en",
        },
        {
            "message_id": "msg2",
            "parent_id": "msg1",
            "message_tree_id": "tree1",
            "role": "assistant",
            "text": "Bonjour",
            "lang": "fr",
        },
    ]

    threads = _reconstruct_threads(records, english_only=True, max_messages=32)

    # Should get 0 threads because msg2 is filtered out, leaving msg1 with no children
    # But msg1 is still a leaf, so we should get 1 thread with just msg1
    assert len(threads) == 1
    assert len(threads[0]["messages"]) == 1
    assert threads[0]["message_ids"] == ["msg1"]


def test_oasst1_role_alternation():
    """Test that roles alternate correctly."""
    records = [
        {
            "message_id": "msg1",
            "parent_id": None,
            "message_tree_id": "tree1",
            "role": "prompter",
            "text": "Q1",
            "lang": "en",
        },
        {
            "message_id": "msg2",
            "parent_id": "msg1",
            "message_tree_id": "tree1",
            "role": "assistant",
            "text": "A1",
            "lang": "en",
        },
        {
            "message_id": "msg3",
            "parent_id": "msg2",
            "message_tree_id": "tree1",
            "role": "prompter",
            "text": "Q2",
            "lang": "en",
        },
        {
            "message_id": "msg4",
            "parent_id": "msg3",
            "message_tree_id": "tree1",
            "role": "assistant",
            "text": "A2",
            "lang": "en",
        },
    ]

    threads = _reconstruct_threads(records, english_only=True, max_messages=32)

    assert len(threads) == 1
    thread = threads[0]

    roles = [msg.role for msg in thread["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_oasst1_consecutive_same_role():
    """Test that threads with consecutive same-role messages are skipped."""
    records = [
        {
            "message_id": "msg1",
            "parent_id": None,
            "message_tree_id": "tree1",
            "role": "prompter",
            "text": "First question",
            "lang": "en",
        },
        {
            "message_id": "msg2",
            "parent_id": "msg1",
            "message_tree_id": "tree1",
            "role": "prompter",  # Same role as parent - invalid
            "text": "Follow-up without assistant response",
            "lang": "en",
        },
    ]

    threads = _reconstruct_threads(records, english_only=True, max_messages=32)

    # Thread should be skipped due to role alternation violation
    assert len(threads) == 0


def test_oasst1_empty_message_handling():
    """Test that empty/whitespace messages are skipped."""
    records = [
        {
            "message_id": "msg1",
            "parent_id": None,
            "message_tree_id": "tree1",
            "role": "prompter",
            "text": "Question",
            "lang": "en",
        },
        {
            "message_id": "msg2",
            "parent_id": "msg1",
            "message_tree_id": "tree1",
            "role": "assistant",
            "text": "   ",  # Whitespace only
            "lang": "en",
        },
        {
            "message_id": "msg3",
            "parent_id": "msg2",
            "message_tree_id": "tree1",
            "role": "prompter",
            "text": "Follow-up",
            "lang": "en",
        },
    ]

    threads = _reconstruct_threads(records, english_only=True, max_messages=32)

    # msg2 should be skipped, causing msg1 -> msg3 which violates alternation
    # So thread should be skipped entirely
    assert len(threads) == 0


def test_oasst1_whitespace_stripping():
    """Test that message content is stripped of leading/trailing whitespace."""
    records = [
        {
            "message_id": "msg1",
            "parent_id": None,
            "message_tree_id": "tree1",
            "role": "prompter",
            "text": "  Question with spaces  ",
            "lang": "en",
        },
        {
            "message_id": "msg2",
            "parent_id": "msg1",
            "message_tree_id": "tree1",
            "role": "assistant",
            "text": "\nAnswer with newlines\n",
            "lang": "en",
        },
    ]

    threads = _reconstruct_threads(records, english_only=True, max_messages=32)

    assert len(threads) == 1
    thread = threads[0]

    assert thread["messages"][0].content == "Question with spaces"
    assert thread["messages"][1].content == "Answer with newlines"
