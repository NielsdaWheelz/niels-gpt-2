"""
Unit tests for fineweb field selection logic.
No network access - uses synthetic data.
"""
import pytest
from niels_gpt.data.fineweb_edu import _select_text_field


def test_select_text_field_with_text():
    """Test that 'text' field is preferred."""
    sample = {"text": "This is the text", "content": "This is content"}
    result = _select_text_field(sample)
    assert result == "This is the text"


def test_select_text_field_with_content():
    """Test that 'content' field is used when 'text' is missing."""
    sample = {"content": "This is content", "url": "http://example.com"}
    result = _select_text_field(sample)
    assert result == "This is content"


def test_select_text_field_with_document():
    """Test that 'document' field is used when 'text' and 'content' are missing."""
    sample = {"document": "This is a document", "url": "http://example.com"}
    result = _select_text_field(sample)
    assert result == "This is a document"


def test_select_text_field_no_whitelisted_fields():
    """Test that ValueError is raised when no whitelisted fields exist."""
    sample = {"id": 123, "metadata": {"key": "value"}, "tags": ["a", "b"], "description": "Not in whitelist"}

    with pytest.raises(ValueError) as exc_info:
        _select_text_field(sample)

    # Check that error message mentions the attempted fields
    assert "No text field found" in str(exc_info.value)
    assert "text, content, document" in str(exc_info.value)
    assert "id" in str(exc_info.value)
