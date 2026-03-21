import pytest

from backend.utils.chunker import chunk_text, split_sentences


class TestChunkText:
    def test_basic_chunking(self):
        text = "Первое предложение. Второе предложение. Третье предложение."
        chunks = chunk_text(text, size=50, overlap=10)
        assert len(chunks) >= 1
        for c in chunks:
            assert len(c) <= 50

    def test_empty_text(self):
        assert chunk_text("", size=100, overlap=10) == []

    def test_single_sentence(self):
        text = "Одно предложение."
        chunks = chunk_text(text, size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_overlap_must_be_less_than_size(self):
        with pytest.raises(ValueError):
            chunk_text("текст", size=10, overlap=10)

    def test_size_must_be_positive(self):
        with pytest.raises(ValueError):
            chunk_text("текст", size=0, overlap=0)


class TestSplitSentences:
    def test_basic(self):
        text = "Первое. Второе! Третье?"
        sentences = split_sentences(text)
        assert len(sentences) == 3

    def test_paragraph_split(self):
        text = "Абзац один.\n\nАбзац два."
        sentences = split_sentences(text)
        assert len(sentences) == 2

    def test_empty(self):
        assert split_sentences("") == []


class TestDocumentReader:
    def test_supported_extensions(self):
        from backend.utils.document_reader import SUPPORTED_EXTENSIONS
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS

    def test_read_nonexistent_file(self):
        from backend.utils.document_reader import read_document
        from pathlib import Path
        result = read_document(Path("/nonexistent/file.txt"))
        assert result == "" or result is None or isinstance(result, str)
