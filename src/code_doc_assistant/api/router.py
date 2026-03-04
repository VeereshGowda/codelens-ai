"""FastAPI REST endpoints for the Code Documentation Assistant.

Why REST endpoints + microservice design?
-----------------------------------------
The Streamlit UI and the business logic are separated into two processes:

  +-------------------+   HTTP requests   +---------------------------+
  |  Streamlit UI     | ----------------► |  FastAPI backend          |
  |  (port 8501)      | ◄---------------- |  (port 8000)              |
  +-------------------+   JSON / SSE      +---------------------------+

Advantages:
  - Clean separation: UI is a thin client; all AI/RAG logic lives in the API.
  - Enables other clients (CLI, CI/CD pipelines, VS Code extension).
  - Azure App Service health probes use GET /healthz.
  - Chat history is owned by the backend, not the browser.

Route layout
------------
All application routes are grouped under the /api prefix so nginx can do
a single prefix match to route traffic.  The /healthz probe lives at root
so it is reachable without the prefix for Azure App Service / Docker health checks.

  GET  /healthz              Liveness probe (no auth required).
  POST /api/ingest           Ingest a local directory or GitHub repo.
  POST /api/chat             Ask a question (blocking, full response).
  POST /api/chat/stream      Ask a question (streaming SSE response).
  POST /api/chat/reset       Clear the server-side conversation history.
  GET  /api/store/status     Return chunk count and last ingested source.
  DELETE /api/store          Clear the vector store (dev/reset).
"""

from __future__ import annotations

from fastapi import APIRouter, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json

from code_doc_assistant.chat.assistant import Assistant
from code_doc_assistant.guardrails.input_guard import GuardResult, InputGuard, get_input_guard
from code_doc_assistant.ingestion.chunker import chunk_source_file
from code_doc_assistant.ingestion.loader import load_github, load_local
from code_doc_assistant.retrieval.vector_store import VectorStore
from code_doc_assistant.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# App + router setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Code Documentation Assistant API",
    description=(
        "RAG-powered API that ingests a codebase and answers natural-language "
        "questions about it using Azure OpenAI GPT-4o + ChromaDB."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# All application routes live under /api so nginx can match a single prefix.
router = APIRouter(prefix="/api")

# Singletons: initialised once and reused across requests.
_store: VectorStore | None = None
_assistant: Assistant | None = None
_guard: InputGuard | None = None
_last_source: str = ""


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def _get_assistant() -> Assistant:
    global _assistant
    if _assistant is None:
        _assistant = Assistant(_get_store())
    return _assistant


def _get_guard() -> InputGuard:
    global _guard
    if _guard is None:
        _guard = get_input_guard()
    return _guard


def _enforce_guardrails(question: str) -> None:
    """Run input guardrails and raise HTTP 400 if the input is blocked."""
    result: GuardResult = _get_guard().check(question)
    if result.blocked:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input blocked ({result.category}): {result.reason}",
        )


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    source: str = Field(..., description="Local directory path or GitHub URL.")
    branch: str = Field(default="main", description="Git branch (GitHub only).")


class IngestResponse(BaseModel):
    files_loaded: int
    chunks_stored: int
    source: str
    message: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question about the codebase.")


class ChatResponse(BaseModel):
    answer: str
    question: str


class StoreStatus(BaseModel):
    chunk_count: int
    last_source: str
    sources: list[str]
    ready: bool


class DeleteSourceRequest(BaseModel):
    source: str = Field(..., description="Exact source string to remove (path or GitHub URL).")


# ---------------------------------------------------------------------------
# Root-level probe (no /api prefix -- reachable by Azure health checks)
# ---------------------------------------------------------------------------


@app.get("/healthz", tags=["ops"], summary="Liveness probe")
def healthz() -> dict[str, str]:
    """Return 200 OK.  Used by Azure App Service and Docker HEALTHCHECK."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# /api routes
# ---------------------------------------------------------------------------


@router.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["ingestion"],
    summary="Ingest a codebase into the vector store",
)
def ingest(request: IngestRequest) -> IngestResponse:
    """Load, chunk, embed, and upsert a codebase into ChromaDB.

    Ingestion is additive: multiple repos can be ingested without resetting
    the store.  Chunk IDs include the source URL so files with identical paths
    across repos never collide.
    """
    global _last_source, _assistant

    logger.info("Ingest request: source=%s branch=%s", request.source, request.branch)
    store = _get_store()

    try:
        is_github = request.source.startswith(("https://", "git@"))
        source_files = (
            list(load_github(request.source, branch=request.branch))
            if is_github
            else list(load_local(request.source))
        )
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Ingestion failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    all_chunks = [c for sf in source_files for c in chunk_source_file(sf)]
    stored = store.add_chunks(all_chunks)
    _last_source = request.source
    _assistant = None  # re-bind assistant to updated store on next request

    logger.info("Ingestion complete: %d files, %d chunks", len(source_files), stored)
    return IngestResponse(
        files_loaded=len(source_files),
        chunks_stored=stored,
        source=request.source,
        message=f"Ingested {len(source_files)} file(s) as {stored} chunk(s).",
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="Ask a question (blocking)",
)
def chat(request: ChatRequest) -> ChatResponse:
    """Return a full answer in a single JSON response (non-streaming)."""
    _enforce_guardrails(request.question)
    if _get_store().count() == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No codebase ingested.  Call POST /api/ingest first.",
        )
    logger.info("Chat (blocking): %s", request.question[:80])
    return ChatResponse(answer=_get_assistant().ask(request.question), question=request.question)


@router.post(
    "/chat/stream",
    tags=["chat"],
    summary="Ask a question and stream the answer (SSE)",
)
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream GPT-4o tokens as Server-Sent Events.

    Each event: data: <token>\\n\\n
    Final event: data: [DONE]\\n\\n
    """
    _enforce_guardrails(request.question)
    if _get_store().count() == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No codebase ingested.  Call POST /api/ingest first.",
        )

    def generate():
        for token in _get_assistant().ask_stream(request.question):
            yield f"data: {json.dumps(token)}\n\n"  # JSON-encode preserves \n in tokens
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post(
    "/chat/reset",
    tags=["chat"],
    summary="Clear server-side conversation history",
)
def chat_reset() -> dict[str, str]:
    """Reset the backend assistant history (keeps the system prompt)."""
    global _assistant
    if _assistant is not None:
        _assistant.reset()
    logger.info("Conversation history reset via API.")
    return {"message": "Conversation history cleared."}


@router.get(
    "/store/status",
    response_model=StoreStatus,
    tags=["ops"],
    summary="Vector store status",
)
def store_status() -> StoreStatus:
    """Return chunk count, all distinct ingested sources, and readiness flag."""
    store = _get_store()
    count = store.count()
    sources = store.list_sources()
    return StoreStatus(
        chunk_count=count,
        last_source=_last_source,
        sources=sources,
        ready=count > 0,
    )


@router.delete(
    "/store/source",
    tags=["ops"],
    summary="Remove all chunks for a specific ingested source",
)
def delete_source(request: DeleteSourceRequest) -> dict[str, str | int]:
    """Delete all indexed chunks that belong to *source*.

    Other ingested codebases are untouched.  The assistant singleton is reset
    so the next chat request picks up the updated store state.
    """
    global _assistant, _last_source
    removed = _get_store().delete_by_source(request.source)
    if removed == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No chunks found for source '{request.source}'.",
        )
    _assistant = None
    if _last_source == request.source:
        remaining = _get_store().list_sources()
        _last_source = remaining[-1] if remaining else ""
    logger.info("Removed %d chunk(s) for source '%s' via API.", removed, request.source)
    return {"message": f"Removed {removed} chunk(s) for '{request.source}'.", "chunks_removed": removed}


@router.delete(
    "/store",
    tags=["ops"],
    summary="Clear the entire vector store (destructive)",
)
def clear_store() -> dict[str, str]:
    """Delete ALL indexed chunks across all sources.  Protect with auth in production."""
    global _assistant, _last_source
    _get_store().clear()
    _assistant = None
    _last_source = ""
    logger.warning("Vector store cleared via API.")
    return {"message": "Vector store cleared."}


# Register the /api router with the main app
app.include_router(router)
