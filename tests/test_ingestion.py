"""
Unit tests — document ingestion (text extraction layer).
No Azure OpenAI or Typesense connection needed.
"""
import pytest
from app.ingestion.ingestor import _extract_text, ALLOWED_EXTENSIONS


def test_allowed_extensions():
    assert ".pdf" in ALLOWED_EXTENSIONS
    assert ".txt" in ALLOWED_EXTENSIONS
    assert ".exe" not in ALLOWED_EXTENSIONS


def test_extract_plain_text():
    content = b"Hello, world!\nSecond line."
    result  = _extract_text("doc.txt", content)
    assert "Hello, world!" in result
    assert "Second line" in result


def test_extract_markdown():
    content = b"# Heading\n\nSome paragraph text."
    result  = _extract_text("notes.md", content)
    assert "Heading" in result


def test_extract_csv():
    content = b"name,age\nAlice,30\nBob,25"
    result  = _extract_text("data.csv", content)
    assert "Alice" in result


def test_unsupported_extension_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        _extract_text("file.xyz", b"data")


def test_utf8_decoding_errors_handled():
    bad_bytes = b"Hello \xff\xfe world"
    result    = _extract_text("file.txt", bad_bytes)
    assert "Hello" in result
