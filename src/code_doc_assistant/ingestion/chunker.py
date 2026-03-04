"""Code-aware text chunking.

Design decisions
----------------
* Chunks are measured in **tokens** (not characters) using tiktoken so that
  context windows are never exceeded.
* A sliding-window approach with configurable ``chunk_size`` and
  ``chunk_overlap`` is used.  This is simpler than AST-based splitting and
  works across all languages, while the overlap preserves enough context so
  that a function signature or class header is usually included in the chunk
  that contains the body.
* Each :class:`Chunk` carries rich metadata so that the retrieval layer can
  surface precise source references back to the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import tiktoken

from code_doc_assistant.config import get_settings
from code_doc_assistant.ingestion.loader import SourceFile
from code_doc_assistant.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A tokenised slice of a source file ready to be embedded.

    Attributes:
        text: The raw text content of the chunk.
        file_path: Repo-relative path of the originating file.
        language: Language label derived from extension.
        source: ``"local"`` or GitHub URL.
        chunk_index: 0-based position of this chunk within the file.
        start_line: Approximate starting line number (1-based).
        metadata: Arbitrary key-value pairs forwarded to the vector store.
    """

    text: str
    file_path: str
    language: str
    source: str
    chunk_index: int
    start_line: int = 1
    metadata: dict[str, str] = field(default_factory=dict)


def _get_encoder(model: str = "gpt-4o") -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _split_by_tokens(
    text: str,
    enc: tiktoken.Encoding,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split *text* into overlapping token windows.

    Args:
        text: Full source text to split.
        enc: tiktoken encoder.
        chunk_size: Maximum number of tokens per chunk.
        chunk_overlap: Number of tokens shared between consecutive chunks.

    Returns:
        A list of decoded text chunks.
    """
    tokens = enc.encode(text)
    if not tokens:
        return []

    chunks: list[str] = []
    step = max(1, chunk_size - chunk_overlap)

    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        chunks.append(enc.decode(window))
        if start + chunk_size >= len(tokens):
            break

    return chunks


def _estimate_start_line(text_before: str) -> int:
    """Estimate 1-based line number by counting newlines in preceding text."""
    return text_before.count("\n") + 1


def chunk_source_file(source_file: SourceFile) -> list[Chunk]:
    """Break a :class:`SourceFile` into a list of :class:`Chunk` objects.

    Each chunk carries metadata linking it back to the originating file so
    that the LLM response can cite the exact source.

    Args:
        source_file: A loaded source file to be chunked.

    Returns:
        Ordered list of :class:`Chunk` objects (may be empty for blank files).
    """
    settings = get_settings()
    enc = _get_encoder()

    if not source_file.content.strip():
        logger.debug("Empty file skipped: %s", source_file.path)
        return []

    raw_chunks = _split_by_tokens(
        source_file.content,
        enc,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    chunks: list[Chunk] = []
    consumed = 0  # character offset into the original text

    for idx, chunk_text in enumerate(raw_chunks):
        start_line = _estimate_start_line(source_file.content[:consumed])
        chunks.append(
            Chunk(
                text=chunk_text,
                file_path=source_file.path,
                language=source_file.language,
                source=source_file.source,
                chunk_index=idx,
                start_line=start_line,
                metadata={
                    **source_file.metadata,
                    "chunk_index": str(idx),
                    "start_line": str(start_line),
                    "language": source_file.language,
                    "file_path": source_file.path,
                },
            )
        )
        consumed += len(chunk_text)

    logger.debug("File '%s' → %d chunk(s)", source_file.path, len(chunks))
    return chunks
