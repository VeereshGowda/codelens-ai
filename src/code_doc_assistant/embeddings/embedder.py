"""Azure OpenAI text-embedding-ada-002 wrapper.

Design decisions
----------------
* Thin wrapper around ``openai.AzureOpenAI`` – no additional framework so the
  abstraction is transparent and easy to swap.
* Batching: the Azure Embeddings API accepts up to 2048 inputs per request;
  we use a conservative batch size of 512 to stay well within rate-limit
  budgets and to support retries on partial failures.
* Texts are truncated to 8191 tokens (the ada-002 limit) before being sent so
  that oversized chunks don't raise API errors.
* The client is built once and reused (singleton pattern via ``@lru_cache``).
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken
from openai import AzureOpenAI

from code_doc_assistant.config import get_settings
from code_doc_assistant.utils.logging import get_logger

logger = get_logger(__name__)

# text-embedding-ada-002 context window
_ADA_MAX_TOKENS = 8191
# Conservative batch size per API request
_BATCH_SIZE = 512


@lru_cache(maxsize=1)
def _get_client() -> AzureOpenAI:
    """Return a cached :class:`AzureOpenAI` client."""
    settings = get_settings()
    return AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )


def _truncate(text: str, max_tokens: int = _ADA_MAX_TOKENS) -> str:
    """Truncate *text* to at most *max_tokens* tokens."""
    try:
        enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    logger.debug("Truncating text from %d to %d tokens for embedding", len(tokens), max_tokens)
    return enc.decode(tokens[:max_tokens])


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using Azure OpenAI text-embedding-ada-002.

    Automatically batches requests and truncates inputs that exceed the model's
    context window.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors (each a list of floats) in the same order as
        the input texts.

    Raises:
        openai.OpenAIError: On API errors after retries.
    """
    if not texts:
        return []

    settings = get_settings()
    client = _get_client()
    deployment = settings.azure_openai_embedding_deployment

    # Pre-process: truncate oversized texts
    truncated = [_truncate(t) for t in texts]

    all_embeddings: list[list[float]] = []

    for batch_start in range(0, len(truncated), _BATCH_SIZE):
        batch = truncated[batch_start : batch_start + _BATCH_SIZE]
        logger.debug(
            "Embedding batch %d–%d / %d",
            batch_start,
            batch_start + len(batch) - 1,
            len(truncated),
        )
        response = client.embeddings.create(model=deployment, input=batch)
        # Responses are guaranteed to be returned in the same order as input
        batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """Embed a single query string.

    Args:
        query: User query to embed.

    Returns:
        Embedding vector as a list of floats.
    """
    return embed_texts([query])[0]
