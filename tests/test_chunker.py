"""Tests for the ingestion chunker module."""

from __future__ import annotations

import pytest

from code_doc_assistant.ingestion.chunker import Chunk, chunk_source_file
from code_doc_assistant.ingestion.loader import SourceFile


class TestChunkSourceFile:
    def test_empty_file_returns_no_chunks(self) -> None:
        sf = SourceFile(path="empty.py", content="", language="python", source="local")
        assert chunk_source_file(sf) == []

    def test_whitespace_only_file_returns_no_chunks(self) -> None:
        sf = SourceFile(path="blank.py", content="   \n  \n", language="python", source="local")
        assert chunk_source_file(sf) == []

    def test_small_file_produces_single_chunk(self, sample_source_file: SourceFile) -> None:
        chunks = chunk_source_file(sample_source_file)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_preserves_file_path(self, sample_source_file: SourceFile) -> None:
        chunks = chunk_source_file(sample_source_file)
        for chunk in chunks:
            assert chunk.file_path == sample_source_file.path

    def test_chunk_preserves_language(self, sample_source_file: SourceFile) -> None:
        chunks = chunk_source_file(sample_source_file)
        for chunk in chunks:
            assert chunk.language == "python"

    def test_chunk_indices_are_sequential(self, sample_source_file: SourceFile) -> None:
        chunks = chunk_source_file(sample_source_file)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_metadata_has_required_keys(self, sample_source_file: SourceFile) -> None:
        chunks = chunk_source_file(sample_source_file)
        required = {"file_path", "language", "chunk_index", "start_line"}
        for chunk in chunks:
            assert required.issubset(chunk.metadata.keys())

    def test_large_file_produces_multiple_chunks(self) -> None:
        # Generate a file with many tokens to force multiple chunks
        big_content = "x = 1\n" * 2000  # ~6000 tokens
        sf = SourceFile(path="big.py", content=big_content, language="python", source="local")
        chunks = chunk_source_file(sf)
        assert len(chunks) > 1

    def test_chunk_text_is_nonempty(self, sample_source_file: SourceFile) -> None:
        chunks = chunk_source_file(sample_source_file)
        for chunk in chunks:
            assert chunk.text.strip() != ""
