"""Tests for the ChromaDB vector store.

All Azure OpenAI embedding calls are mocked; ChromaDB uses an ephemeral
in-memory client so no disk I/O takes place.
"""

from __future__ import annotations

import tempfile
from unittest.mock import patch

import pytest

from code_doc_assistant.ingestion.chunker import Chunk


# Patch embed_texts globally for vector store tests (no real API calls)
@pytest.fixture(autouse=True)
def mock_embeddings():
    """Replace embed_texts and embed_query with deterministic unit vectors."""
    import random

    def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        # Deterministic: seed on the joined text so same input → same vector
        rng = random.Random(hash(tuple(texts)))
        return [[rng.uniform(-1, 1) for _ in range(1536)] for _ in texts]

    def fake_embed_query(query: str) -> list[float]:
        rng = random.Random(hash(query))
        return [rng.uniform(-1, 1) for _ in range(1536)]

    with (
        patch("code_doc_assistant.embeddings.embedder.embed_texts", side_effect=fake_embed_texts),
        patch("code_doc_assistant.embeddings.embedder.embed_query", side_effect=fake_embed_query),
        patch("code_doc_assistant.retrieval.vector_store.embed_texts", side_effect=fake_embed_texts),
        patch("code_doc_assistant.retrieval.vector_store.embed_query", side_effect=fake_embed_query),
    ):
        yield


@pytest.fixture()
def vector_store(tmp_path):
    """Return a VectorStore backed by a temporary ChromaDB directory."""
    from unittest.mock import patch as _patch

    settings_patch = {
        "chroma_persist_dir": str(tmp_path / "chroma"),
        "chroma_collection_name": "test_collection",
        "top_k": 3,
    }

    # We patch get_settings to return an object with the temp path
    from code_doc_assistant.config import Settings

    fake_settings = Settings(
        azure_openai_api_key="test-key",
        azure_openai_endpoint="https://test.openai.azure.com/",
        azure_openai_api_version="2024-12-01-preview",
        azure_openai_chat_deployment="gpt-4o",
        azure_openai_embedding_deployment="text-embedding-ada-002",
        chroma_persist_dir=str(tmp_path / "chroma"),
        chroma_collection_name="test_collection",
        top_k=3,
    )

    with _patch("code_doc_assistant.retrieval.vector_store.get_settings", return_value=fake_settings):
        from code_doc_assistant.retrieval.vector_store import VectorStore

        store = VectorStore()
        yield store


def _make_chunk(index: int, text: str = "sample code", source: str = "local") -> Chunk:
    return Chunk(
        text=text,
        file_path=f"file_{index}.py",
        language="python",
        source=source,
        chunk_index=index,
        start_line=1,
        metadata={
            "file_path": f"file_{index}.py",
            "language": "python",
            "chunk_index": str(index),
            "start_line": "1",
            "source": source,
        },
    )


class TestVectorStore:
    def test_empty_store_has_zero_count(self, vector_store) -> None:
        assert vector_store.count() == 0

    def test_add_chunks_increases_count(self, vector_store) -> None:
        chunks = [_make_chunk(i) for i in range(5)]
        stored = vector_store.add_chunks(chunks)
        assert stored == 5
        assert vector_store.count() == 5

    def test_empty_chunks_returns_zero(self, vector_store) -> None:
        assert vector_store.add_chunks([]) == 0

    def test_search_returns_results(self, vector_store) -> None:
        chunks = [_make_chunk(i, text=f"Function {i}: does something interesting") for i in range(10)]
        vector_store.add_chunks(chunks)
        results = vector_store.search("function that does something")
        assert len(results) > 0

    def test_search_empty_store_returns_empty(self, vector_store) -> None:
        results = vector_store.search("any query")
        assert results == []

    def test_upsert_is_idempotent(self, vector_store) -> None:
        chunk = _make_chunk(0, "idempotent code")
        vector_store.add_chunks([chunk])
        vector_store.add_chunks([chunk])  # second upsert of same chunk
        assert vector_store.count() == 1

    def test_clear_resets_count(self, vector_store) -> None:
        chunks = [_make_chunk(i) for i in range(3)]
        vector_store.add_chunks(chunks)
        vector_store.clear()
        assert vector_store.count() == 0

    def test_retrieved_chunk_has_metadata(self, vector_store) -> None:
        chunk = _make_chunk(0, "class UserService: handles user operations")
        vector_store.add_chunks([chunk])
        results = vector_store.search("UserService")
        assert results[0].file_path == "file_0.py"
        assert results[0].language == "python"

    def test_search_respects_top_k(self, vector_store) -> None:
        chunks = [_make_chunk(i, text=f"Code snippet {i}") for i in range(20)]
        vector_store.add_chunks(chunks)
        results = vector_store.search("code snippet", top_k=3)
        assert len(results) <= 3

    def test_list_sources_empty(self, vector_store) -> None:
        assert vector_store.list_sources() == []

    def test_list_sources_returns_distinct_values(self, vector_store) -> None:
        chunks = (
            [_make_chunk(i, source="https://github.com/org/repo-a") for i in range(3)]
            + [_make_chunk(i, source="https://github.com/org/repo-b") for i in range(3, 6)]
        )
        vector_store.add_chunks(chunks)
        sources = vector_store.list_sources()
        assert sorted(sources) == ["https://github.com/org/repo-a", "https://github.com/org/repo-b"]

    def test_delete_by_source_removes_only_target(self, vector_store) -> None:
        repo_a = [_make_chunk(i, text=f"repo-a chunk {i}", source="https://github.com/org/repo-a") for i in range(4)]
        repo_b = [_make_chunk(i, text=f"repo-b chunk {i}", source="https://github.com/org/repo-b") for i in range(4, 8)]
        vector_store.add_chunks(repo_a + repo_b)
        assert vector_store.count() == 8

        removed = vector_store.delete_by_source("https://github.com/org/repo-a")

        assert removed == 4
        assert vector_store.count() == 4
        remaining_sources = vector_store.list_sources()
        assert remaining_sources == ["https://github.com/org/repo-b"]

    def test_delete_by_source_unknown_returns_zero(self, vector_store) -> None:
        vector_store.add_chunks([_make_chunk(0, source="local")])
        removed = vector_store.delete_by_source("https://github.com/org/nonexistent")
        assert removed == 0
        assert vector_store.count() == 1
