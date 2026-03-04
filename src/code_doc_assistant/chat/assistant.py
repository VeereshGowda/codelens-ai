"""Conversational AI assistant that answers questions about a codebase.

Design decisions
----------------
* Retrieval-Augmented Generation (RAG): for every user turn the assistant
  embeds the question, retrieves the top-k chunks, and injects them as context
  before calling GPT-4o.  This grounds the responses in the actual code and
  prevents hallucination.
* Conversation history: the full ``messages`` list is maintained across turns
  so the model understands follow-up questions (multi-turn support).
* Guardrails:
  - If the vector store is empty the assistant refuses to answer and prompts
    the user to ingest a codebase first.
  - The retrieved context is prepended to each user message, but the raw
    chunk text is trimmed to ``max_context_tokens`` to avoid exceeding the
    model's context window.
* Streaming: answers are yielded token-by-token via :meth:`ask_stream` to
  give a responsive UX.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Generator

from openai import AzureOpenAI
from openai.types.chat import ChatCompletionMessageParam

from code_doc_assistant.chat.prompts import (
    SYSTEM_PROMPT,
    build_context_block,
    build_user_message,
)
from code_doc_assistant.config import get_settings
from code_doc_assistant.retrieval.vector_store import RetrievedChunk, VectorStore
from code_doc_assistant.utils.logging import get_logger
from code_doc_assistant.utils.token_counter import count_tokens

logger = get_logger(__name__)

# Maximum tokens dedicated to context snippets.  Leave plenty of headroom for
# the system prompt, conversation history, and the model's response.
_MAX_CONTEXT_TOKENS = 6000


@lru_cache(maxsize=1)
def _get_openai_client() -> AzureOpenAI:
    """Return a cached :class:`AzureOpenAI` chat client."""
    settings = get_settings()
    return AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )


def _trim_chunks_to_budget(
    chunks: list[RetrievedChunk],
    budget: int = _MAX_CONTEXT_TOKENS,
) -> list[dict[str, str]]:
    """Convert RetrievedChunks to dicts, dropping low-priority chunks if over budget.

    Args:
        chunks: Retrieved chunks ordered by relevance (most relevant first).
        budget: Token budget for the context block.

    Returns:
        List of chunk dicts that fit within *budget*.
    """
    selected: list[dict[str, str]] = []
    used = 0
    for c in chunks:
        tokens = count_tokens(c.text)
        if used + tokens > budget:
            logger.debug("Context budget reached; dropping remaining %d chunk(s)", len(chunks) - len(selected))
            break
        selected.append(
            {
                "file_path": c.file_path,
                "language": c.language,
                "start_line": str(c.start_line),
                "text": c.text,
            }
        )
        used += tokens
    return selected


class Assistant:
    """Multi-turn conversational assistant backed by RAG + GPT-4o.

    Typical usage::

        store = VectorStore()  # pre-populated with ingested code chunks
        assistant = Assistant(store)

        # Single-turn (blocking)
        answer = assistant.ask("What does the AuthMiddleware class do?")
        print(answer)

        # Streaming
        for token in assistant.ask_stream("Show me the login endpoint"):
            print(token, end="", flush=True)

        assistant.reset()  # clear conversation history
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store
        self._history: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, question: str) -> str:
        """Return a complete answer to *question* (blocking).

        Args:
            question: User's natural-language question about the codebase.

        Returns:
            The assistant's answer as a Markdown string.
        """
        user_message = self._build_user_message(question)
        self._history.append({"role": "user", "content": user_message})

        settings = get_settings()
        client = _get_openai_client()

        logger.info("Calling GPT-4o (deployment=%s)…", settings.azure_openai_chat_deployment)
        response = client.chat.completions.create(
            model=settings.azure_openai_chat_deployment,
            messages=self._history,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )
        answer = response.choices[0].message.content or ""
        self._history.append({"role": "assistant", "content": answer})

        logger.debug(
            "Tokens used — prompt: %d, completion: %d",
            response.usage.prompt_tokens if response.usage else 0,
            response.usage.completion_tokens if response.usage else 0,
        )
        return answer

    def ask_stream(self, question: str) -> Generator[str, None, None]:
        """Yield answer tokens one-by-one for a streaming UX.

        Args:
            question: User's natural-language question about the codebase.

        Yields:
            Individual text tokens from the model response.
        """
        user_message = self._build_user_message(question)
        self._history.append({"role": "user", "content": user_message})

        settings = get_settings()
        client = _get_openai_client()

        logger.info("Streaming GPT-4o response…")
        stream = client.chat.completions.create(
            model=settings.azure_openai_chat_deployment,
            messages=self._history,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            stream=True,
        )

        full_answer_parts: list[str] = []
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_answer_parts.append(delta)
                    yield delta

        full_answer = "".join(full_answer_parts)
        self._history.append({"role": "assistant", "content": full_answer})

    def reset(self) -> None:
        """Clear conversation history (keeps the system prompt)."""
        self._history = [{"role": "system", "content": SYSTEM_PROMPT}]
        logger.info("Conversation history cleared.")

    @property
    def history(self) -> list[ChatCompletionMessageParam]:
        """Return the current conversation history (read-only copy)."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_user_message(self, question: str) -> str:
        """Retrieve relevant chunks and compose the full user message."""
        chunks = self._store.search(question)
        if not chunks:
            logger.warning("No relevant chunks found for query: %s", question[:80])

        trimmed = _trim_chunks_to_budget(chunks)
        context = build_context_block(trimmed)
        return build_user_message(question, context)
