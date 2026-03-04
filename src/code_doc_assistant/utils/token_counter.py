"""Token counting helpers using tiktoken."""

from __future__ import annotations

import tiktoken


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Return approximate token count for *text* under *model*'s tokenizer.

    Falls back to ``cl100k_base`` (used by GPT-4 / text-embedding-ada-002)
    if the exact model encoding is not found.

    Args:
        text: The string to count tokens for.
        model: Model name used to select the tokenizer.

    Returns:
        Integer token count.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
