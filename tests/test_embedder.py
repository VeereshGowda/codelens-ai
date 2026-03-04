"""Tests for the embeddings module.

All Azure OpenAI API calls are mocked so these tests run without credentials.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from code_doc_assistant.embeddings.embedder import _truncate, embed_query, embed_texts


class TestTruncate:
    def test_short_text_unchanged(self) -> None:
        text = "Hello, world!"
        result = _truncate(text, max_tokens=8191)
        assert result == text

    def test_long_text_is_truncated(self) -> None:
        # Create a text that exceeds the max token limit
        long_text = "word " * 10000  # well over 8191 tokens
        result = _truncate(long_text, max_tokens=100)
        assert len(result) < len(long_text)

    def test_empty_string(self) -> None:
        assert _truncate("") == ""


class TestEmbedTexts:
    def _make_mock_response(self, count: int) -> MagicMock:
        """Build a mock openai embeddings response for *count* inputs."""
        response = MagicMock()
        response.data = [
            MagicMock(embedding=[0.1] * 1536, index=i) for i in range(count)
        ]
        return response

    def test_empty_input_returns_empty(self) -> None:
        assert embed_texts([]) == []

    def test_single_text_returns_one_embedding(self) -> None:
        with patch("code_doc_assistant.embeddings.embedder._get_client") as mock_get_client:
            client = MagicMock()
            mock_get_client.return_value = client
            client.embeddings.create.return_value = self._make_mock_response(1)

            result = embed_texts(["hello world"])

        assert len(result) == 1
        assert len(result[0]) == 1536

    def test_multiple_texts_return_correct_count(self) -> None:
        texts = ["text one", "text two", "text three"]
        with patch("code_doc_assistant.embeddings.embedder._get_client") as mock_get_client:
            client = MagicMock()
            mock_get_client.return_value = client
            client.embeddings.create.return_value = self._make_mock_response(len(texts))

            result = embed_texts(texts)

        assert len(result) == len(texts)


class TestEmbedQuery:
    def test_returns_single_embedding(self) -> None:
        with patch("code_doc_assistant.embeddings.embedder.embed_texts") as mock_embed:
            mock_embed.return_value = [[0.1] * 1536]
            result = embed_query("What does authenticate() do?")

        assert len(result) == 1536
        mock_embed.assert_called_once_with(["What does authenticate() do?"])
