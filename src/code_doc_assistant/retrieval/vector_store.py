"""ChromaDB-backed vector store for code chunk retrieval.

Design decisions
----------------
* ChromaDB is used for its zero-infrastructure setup: it persists to a local
  directory, requires no Docker or network service, and is trivially swappable
  for a production store (Pinecone, Weaviate, Azure AI Search) by replacing
  this module.
* We supply our own embeddings (Azure OpenAI) rather than relying on
  ChromaDB's built-in embedding functions, giving full control and matching
  the embeddings used at query time.
* ``upsert`` behaviour: if a chunk with the same ID already exists it is
  overwritten, making re-ingestion idempotent.
* IDs are deterministic ``sha256(file_path + chunk_index)`` hashes so repeated
  ingestion of the same codebase produces stable, collision-free IDs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from code_doc_assistant.config import get_settings
from code_doc_assistant.embeddings.embedder import embed_query, embed_texts
from code_doc_assistant.ingestion.chunker import Chunk
from code_doc_assistant.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector store, enriched with a similarity score.

    Attributes:
        text: The raw chunk text.
        file_path: Repo-relative path of the originating file.
        language: Language label.
        source: ``"local"`` or GitHub URL.
        chunk_index: 0-based position within the file.
        start_line: Approximate starting line number (1-based).
        score: Cosine similarity score (higher is more similar).
        metadata: Full metadata dict from ChromaDB.
    """

    text: str
    file_path: str
    language: str
    source: str
    chunk_index: int
    start_line: int
    score: float
    metadata: dict[str, str]


def _chunk_id(source: str, file_path: str, chunk_index: int) -> str:
    """Generate a stable, unique ID for a chunk.

    The *source* (repo URL or ``"local"``) is included in the key so that two
    different repositories that contain identically-named files produce
    different IDs and never overwrite each other in the vector store.
    This makes multi-repo ingestion safe without requiring a store reset.
    """
    key = f"{source}::{file_path}::{chunk_index}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


class VectorStore:
    """Manages a ChromaDB collection for code chunk storage and retrieval.

    Example::

        store = VectorStore()
        store.add_chunks(chunks)
        results = store.search("How does authentication work?", top_k=5)
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready — collection '%s' has %d document(s)",
            settings.chroma_collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 256) -> int:
        """Embed and upsert *chunks* into the collection.

        Args:
            chunks: Chunks to store.
            batch_size: Number of chunks to embed + upsert per API call.

        Returns:
            Total number of chunks stored.
        """
        if not chunks:
            return 0

        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.text for c in batch]
            ids = [_chunk_id(c.source, c.file_path, c.chunk_index) for c in batch]
            metadatas = [c.metadata for c in batch]

            logger.debug("Embedding %d chunk(s) (batch %d–%d)…", len(batch), start, start + len(batch) - 1)
            embeddings = embed_texts(texts)

            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            total += len(batch)
            logger.debug("Upserted %d / %d chunk(s)", total, len(chunks))

        logger.info("Stored %d chunk(s) in the vector store", total)
        return total

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Find the most relevant chunks for *query* using cosine similarity.

        Args:
            query: Natural-language question or keyword query.
            top_k: Number of results to return.  Defaults to the value in
                   :class:`~code_doc_assistant.config.Settings`.

        Returns:
            Ordered list of :class:`RetrievedChunk` (most relevant first).
        """
        settings = get_settings()
        k = top_k or settings.top_k

        if self._collection.count() == 0:
            logger.warning("Vector store is empty — please ingest a codebase first.")
            return []

        query_embedding = embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []
        docs = results.get("documents") or [[]]
        metas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]

        for doc, meta, dist in zip(docs[0], metas[0], distances[0]):
            # ChromaDB cosine distance → similarity: score = 1 - distance
            score = 1.0 - float(dist)
            retrieved.append(
                RetrievedChunk(
                    text=doc,
                    file_path=meta.get("file_path", ""),
                    language=meta.get("language", ""),
                    source=meta.get("source", ""),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    start_line=int(meta.get("start_line", 1)),
                    score=score,
                    metadata=dict(meta),
                )
            )

        return retrieved

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of chunks stored."""
        return self._collection.count()

    def list_sources(self) -> list[str]:
        """Return all distinct source URLs / paths currently in the collection.

        Reads metadata from ChromaDB so the result is always authoritative and
        survives API restarts (no in-memory state required).
        """
        if self._collection.count() == 0:
            return []
        result = self._collection.get(include=["metadatas"])
        sources: set[str] = set()
        for meta in result.get("metadatas") or []:
            if meta and "source" in meta:
                sources.add(str(meta["source"]))
        return sorted(sources)

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks that belong to *source*.

        Args:
            source: Exact source string as stored during ingestion
                    (local path or GitHub URL).

        Returns:
            Number of chunks removed.
        """
        before = self._collection.count()
        self._collection.delete(where={"source": {"$eq": source}})
        after = self._collection.count()
        removed = before - after
        logger.info("Deleted %d chunk(s) for source '%s'", removed, source)
        return removed

    def clear(self) -> None:
        """Delete all documents from the collection (destructive!)."""
        settings = get_settings()
        self._client.delete_collection(settings.chroma_collection_name)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store cleared.")
