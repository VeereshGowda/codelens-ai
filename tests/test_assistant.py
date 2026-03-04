"""Tests for the conversational assistant.

All Azure OpenAI and ChromaDB interactions are mocked so no credentials or
network access are required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from code_doc_assistant.chat.assistant import Assistant, _trim_chunks_to_budget
from code_doc_assistant.retrieval.vector_store import RetrievedChunk


def _make_retrieved_chunk(text: str = "def foo(): pass", file_path: str = "a.py") -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        file_path=file_path,
        language="python",
        source="local",
        chunk_index=0,
        start_line=1,
        score=0.9,
        metadata={"file_path": file_path, "language": "python", "chunk_index": "0", "start_line": "1"},
    )


@pytest.fixture()
def mock_store():
    store = MagicMock()
    store.count.return_value = 5
    store.search.return_value = [_make_retrieved_chunk()]
    return store


@pytest.fixture()
def assistant(mock_store):
    return Assistant(mock_store)


class TestTrimChunksToBudget:
    def test_small_chunks_fit_in_budget(self) -> None:
        chunks = [_make_retrieved_chunk(f"x = {i}") for i in range(5)]
        result = _trim_chunks_to_budget(chunks, budget=10000)
        assert len(result) == 5

    def test_large_chunks_are_dropped(self) -> None:
        big = "a " * 10000  # many tokens
        chunks = [_make_retrieved_chunk(big) for _ in range(3)]
        result = _trim_chunks_to_budget(chunks, budget=50)
        # At most one chunk should fit
        assert len(result) <= 1

    def test_returns_dict_format(self) -> None:
        chunks = [_make_retrieved_chunk()]
        result = _trim_chunks_to_budget(chunks)
        assert "file_path" in result[0]
        assert "text" in result[0]


class TestAssistant:
    def test_ask_returns_string(self, assistant: Assistant, mock_store: MagicMock) -> None:
        with patch("code_doc_assistant.chat.assistant._get_openai_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client_fn.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices[0].message.content = "The `foo` function does nothing."
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 20
            mock_client.chat.completions.create.return_value = mock_response

            answer = assistant.ask("What does foo do?")

        assert isinstance(answer, str)
        assert "foo" in answer.lower()

    def test_ask_appends_to_history(self, assistant: Assistant, mock_store: MagicMock) -> None:
        with patch("code_doc_assistant.chat.assistant._get_openai_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client_fn.return_value = mock_client
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "It adds two numbers."
            mock_response.usage.prompt_tokens = 50
            mock_response.usage.completion_tokens = 10
            mock_client.chat.completions.create.return_value = mock_response

            assistant.ask("What does add() do?")

        # system + user + assistant = 3 messages
        assert len(assistant.history) == 3

    def test_reset_clears_history(self, assistant: Assistant) -> None:
        # Add a fake user message manually
        assistant._history.append({"role": "user", "content": "test"})
        assistant.reset()
        assert len(assistant.history) == 1  # only system prompt remains

    def test_ask_stream_yields_tokens(self, assistant: Assistant, mock_store: MagicMock) -> None:
        with patch("code_doc_assistant.chat.assistant._get_openai_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client_fn.return_value = mock_client

            def stream_chunks():
                for word in ["Hello", " world", "!"]:
                    chunk = MagicMock()
                    chunk.choices = [MagicMock()]
                    chunk.choices[0].delta.content = word
                    yield chunk

            mock_client.chat.completions.create.return_value = stream_chunks()

            tokens = list(assistant.ask_stream("Hi"))

        assert "".join(tokens) == "Hello world!"

    def test_history_property_returns_copy(self, assistant: Assistant) -> None:
        history = assistant.history
        history.append({"role": "user", "content": "injected"})
        # Original should be unaffected
        assert len(assistant.history) == 1
