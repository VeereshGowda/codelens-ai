"""Streamlit web UI -- microservice client for the FastAPI backend.

This file contains ONLY presentation logic.  All business logic (ingestion,
embedding, retrieval, chat) lives in the FastAPI backend (api_server.py).
Communication happens over HTTP so the two processes can be deployed and
scaled independently.

Local development
-----------------
Start both services via:

    python main.py            # starts FastAPI on :8000, then Streamlit on :8501

Or individually:

    python main.py api        # FastAPI only
    python main.py ui         # Streamlit only (requires API already running)

Architecture note
-----------------
Streamlit <-> FastAPI:
  POST /api/ingest          -- ingest a codebase
  POST /api/chat/stream     -- streaming SSE chat
  POST /api/chat/reset      -- reset conversation history
  DELETE /api/store         -- clear the vector store
  GET  /api/store/status    -- chunk count + readiness flag
"""

from __future__ import annotations

import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

from code_doc_assistant.config import get_settings
from code_doc_assistant.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

settings = get_settings()
API_BASE = settings.api_base_url.rstrip("/")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Code Documentation Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages: list[dict[str, str]] = []

if "ingested_source" not in st.session_state:
    st.session_state.ingested_source: str = ""

if "ingest_success_msg" not in st.session_state:
    st.session_state.ingest_success_msg: str = ""

# ---------------------------------------------------------------------------
# API helper functions
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def api_store_status() -> dict:
    """GET /api/store/status -- cached for 60 s to avoid hammering the API on every re-render.

    Call ``api_store_status.clear()`` after any mutating action (ingest /
    clear / delete source) so the sidebar reflects changes immediately.
    """
    try:
        r = requests.get(f"{API_BASE}/api/store/status", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning("Could not reach API: %s", exc)
        return {"chunk_count": 0, "last_source": "", "sources": [], "ready": False}


def api_ingest(source: str, branch: str = "main") -> dict:
    """POST /api/ingest -- returns IngestResponse dict or raises."""
    r = requests.post(
        f"{API_BASE}/api/ingest",
        json={"source": source, "branch": branch},
        timeout=600,          # ingestion can take several minutes for large repos
    )
    r.raise_for_status()
    return r.json()


def api_clear_store() -> None:
    """DELETE /api/store."""
    requests.delete(f"{API_BASE}/api/store", timeout=10).raise_for_status()


def api_delete_source(source: str) -> dict:
    """DELETE /api/store/source -- remove all chunks for one specific source."""
    r = requests.delete(
        f"{API_BASE}/api/store/source",
        json={"source": source},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def api_chat_reset() -> None:
    """POST /api/chat/reset."""
    requests.post(f"{API_BASE}/api/chat/reset", timeout=10).raise_for_status()


def api_chat_stream(question: str):
    """POST /api/chat/stream -- yields text tokens from SSE response.

    Tokens are JSON-encoded by the server so that newline characters
    (markdown headers, blank lines between sections) are preserved exactly.
    """
    import json as _json
    with requests.post(
        f"{API_BASE}/api/chat/stream",
        json={"question": question},
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if line.startswith("data: "):
                payload = line[6:]
                if not payload or payload == "[DONE]":
                    return
                try:
                    yield _json.loads(payload)  # decode JSON to restore \n, etc.
                except _json.JSONDecodeError:
                    yield payload  # fall back to raw text if not valid JSON


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")

    # Single cached call for the entire render — reused by both the
    # connectivity badge and the store-status section below.
    status_data = api_store_status()
    api_ok = status_data.get("chunk_count", 0) >= 0  # any successful response

    st.caption(
        f"🔗 **Endpoint:** Azure OpenAI (configured)  \n"
        f"🤖 **Chat model:** `{settings.azure_openai_chat_deployment}`  \n"
        f"📐 **Embedding model:** `{settings.azure_openai_embedding_deployment}`  \n"
        f"🖧 **API backend:** `{API_BASE}`"
    )

    st.divider()
    st.subheader("📥 Ingest a Codebase")

    source_type = st.radio(
        "Source type",
        options=["Local directory", "GitHub URL"],
        horizontal=True,
    )

    source_input: str = ""
    branch: str = "main"
    if source_type == "Local directory":
        source_input = st.text_input(
            "Directory path",
            placeholder="C:/projects/my-repo or /home/user/repo",
        )
    else:
        source_input = st.text_input(
            "GitHub repository URL",
            placeholder="https://github.com/owner/repo",
        )
        branch = st.text_input("Branch", value="main")

    col_ingest, col_clear = st.columns(2)

    # Show deferred success message from the previous ingest run (before st.rerun)
    if st.session_state.ingest_success_msg:
        st.success(st.session_state.ingest_success_msg)
        st.session_state.ingest_success_msg = ""

    with col_ingest:
        ingest_btn = st.button("🚀 Ingest", use_container_width=True, type="primary")
    with col_clear:
        clear_btn = st.button("🗑️ Clear Store", use_container_width=True)

    # ---- Ingest ----
    if ingest_btn and source_input:
        with st.spinner("Ingesting codebase… this may take a few minutes for large repos."):
            try:
                result = api_ingest(source_input, branch=branch)
                api_store_status.clear()  # bust cache so sidebar refreshes immediately
                st.session_state.ingested_source = source_input
                st.session_state.messages = []
                st.session_state.ingest_success_msg = (
                    f"✅ Ingested **{result['files_loaded']}** file(s) → "
                    f"**{result['chunks_stored']}** chunk(s) stored."
                )
                st.rerun()  # re-render so status_data is fetched fresh
            except requests.HTTPError as exc:
                try:
                    detail = exc.response.json().get("detail", str(exc))
                except Exception:
                    detail = str(exc)
                st.error(f"❌ Ingestion failed: {detail}")
            except Exception as exc:
                st.error(f"❌ Could not reach API backend: {exc}")

    if ingest_btn and not source_input:
        st.warning("Please enter a source path or URL.")

    # ---- Clear store ----
    if clear_btn:
        try:
            api_clear_store()
            api_store_status.clear()  # bust cache
            st.session_state.messages = []
            st.session_state.ingested_source = ""
            st.session_state.ingest_success_msg = "✅ Vector store cleared."
            st.rerun()  # re-render so status_data is fetched fresh
        except Exception as exc:
            st.error(f"❌ Clear failed: {exc}")

    # ---- Store status ---- reuses status_data fetched at sidebar entry (cached 60 s)
    st.divider()
    st.subheader("📊 Store Status")
    chunk_count = status_data.get("chunk_count", 0)
    ingested_sources: list[str] = status_data.get("sources", [])

    if chunk_count:
        st.metric("Chunks indexed", chunk_count)
        st.caption(f"{len(ingested_sources)} codebase(s) loaded")

        if ingested_sources:
            st.write("**Ingested sources:**")
            for src in ingested_sources:
                # Display: "owner/repo" for GitHub URLs, folder name for local paths
                is_github = src.startswith(("https://", "git@"))
                if is_github:
                    # extract the last two path components: owner/repo
                    # handles https://github.com/owner/repo and similar
                    parts = src.rstrip("/").rstrip(".git").rsplit("/", 2)
                    display = "/".join(parts[-2:]) if len(parts) >= 2 else src
                else:
                    from pathlib import Path as _Path
                    display = _Path(src).name or src  # e.g. "my-repo" from "/home/user/my-repo"
                col_label, col_btn = st.columns([4, 1])
                with col_label:
                    st.caption(f"`{display}`", help=src)
                with col_btn:
                    if st.button("✖", key=f"remove_{src}", help=f"Remove: {src}"):
                        try:
                            result = api_delete_source(src)
                            api_store_status.clear()  # bust cache
                            st.session_state.ingest_success_msg = result.get("message", "✅ Removed.")
                            if st.session_state.get("ingested_source") == src:
                                st.session_state.ingested_source = ""
                            st.rerun()
                        except requests.HTTPError as exc:
                            try:
                                detail = exc.response.json().get("detail", str(exc))
                            except Exception:
                                detail = str(exc)
                            st.error(f"❌ {detail}")
                        except Exception as exc:
                            st.error(f"❌ {exc}")
    else:
        st.info("No codebase ingested yet.")

    # ---- Reset conversation ----
    st.divider()
    if st.button("🔄 Reset Conversation", use_container_width=True):
        try:
            api_chat_reset()
            st.session_state.messages = []
            st.success("Conversation reset.")
        except Exception as exc:
            st.error(f"❌ Reset failed: {exc}")

# ---------------------------------------------------------------------------
# Main panel -- Chat
# ---------------------------------------------------------------------------

st.title("🔍 Code Documentation Assistant")
st.caption(
    "Ask questions about your codebase. "
    "Ingest a local directory or GitHub repo using the sidebar first."
)

# Replay conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input -- disabled when store is empty.
# Re-use the value already fetched inside the sidebar block (still cached).
store_ready = api_store_status().get("ready", False)

if prompt := st.chat_input(
    "Ask a question about the codebase…",
    disabled=not store_ready,
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            streamed_text = st.write_stream(api_chat_stream(prompt))
        except Exception as exc:
            streamed_text = f"❌ Error communicating with API backend: {exc}"
            st.error(streamed_text)

    st.session_state.messages.append({"role": "assistant", "content": streamed_text or ""})

if not store_ready:
    st.info("⬅️ Ingest a codebase from the sidebar to start chatting.")
