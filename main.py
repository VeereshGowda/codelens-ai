"""Process manager and CLI for the Code Documentation Assistant.

Usage
-----
Start both services (recommended for local development):

    python main.py              # FastAPI on :8000, then Streamlit on :8501

Start individual services:

    python main.py api          # FastAPI / uvicorn only
    python main.py ui           # Streamlit only (API must already be running)

Ingest a codebase via the running API:

    python main.py ingest <path-or-github-url> [branch]

Deployment note
---------------
In production (Azure App Service), FastAPI and Streamlit run in separate
containers (Dockerfile.api / Dockerfile.ui).  main.py is for local
development only.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# Locate the python + streamlit executables inside the active virtual env.
# Using sys.executable guarantees we stay inside the uv-managed .venv.
_PYTHON = sys.executable
_VENV_BIN = Path(_PYTHON).parent


def _streamlit_cmd() -> list[str]:
    """Return the streamlit command for the current environment."""
    st_path = _VENV_BIN / "streamlit"
    if st_path.exists():
        return [str(st_path)]
    # Fallback: run as a module (works if bin dir is not on PATH)
    return [_PYTHON, "-m", "streamlit"]


def _wait_for_health(url: str, timeout: int = 30) -> bool:
    """Poll *url* every 0.5 s until it returns HTTP 200 or *timeout* expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def cmd_api() -> subprocess.Popen:  # type: ignore[type-arg]
    """Start the FastAPI / uvicorn server and return its Popen handle."""
    print("▶  Starting FastAPI backend on http://localhost:8000 …")
    proc = subprocess.Popen(
        [
            _PYTHON, "-m", "uvicorn",
            "api_server:app",
            "--host", "0.0.0.0",
            "--port", "8000",
        ]
    )
    print("   Waiting for API to be ready …", end="", flush=True)
    if not _wait_for_health("http://localhost:8000/healthz", timeout=30):
        print(" FAILED")
        proc.terminate()
        raise RuntimeError("FastAPI did not start within 30 s. Check logs above.")
    print(" ready ✓")
    return proc


def cmd_ui() -> subprocess.Popen:  # type: ignore[type-arg]
    """Start the Streamlit UI and return its Popen handle."""
    print("▶  Starting Streamlit UI on http://localhost:8501 …")
    st_cmd = _streamlit_cmd()
    proc = subprocess.Popen(
        st_cmd + [
            "run", "app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
        ]
    )
    return proc


def cmd_serve() -> None:
    """Start FastAPI first, wait for health, then start Streamlit."""
    api_proc = cmd_api()
    ui_proc = cmd_ui()

    print()
    print("=" * 55)
    print("  FastAPI  →  http://localhost:8000")
    print("  Docs     →  http://localhost:8000/docs")
    print("  Streamlit→  http://localhost:8501")
    print("  Press Ctrl+C to stop both services.")
    print("=" * 55)
    print()

    try:
        api_proc.wait()
        ui_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down …")
        ui_proc.terminate()
        api_proc.terminate()
        ui_proc.wait()
        api_proc.wait()
        print("Stopped.")


def cmd_ingest(source: str, branch: str = "main") -> None:
    """Call POST /api/ingest on the running backend to ingest a codebase.

    Args:
        source: Local path or GitHub https/ssh URL.
        branch: Branch to clone (GitHub only).
    """
    from code_doc_assistant.config import get_settings
    api_base = get_settings().api_base_url.rstrip("/")
    url = f"{api_base}/api/ingest"

    print(f"Ingesting '{source}' via {url} …")
    try:
        r = requests.post(url, json={"source": source, "branch": branch}, timeout=600)
        r.raise_for_status()
        data = r.json()
        print(f"✓ {data['message']}")
    except requests.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        print(f"✗ Ingestion failed: {detail}")
        sys.exit(1)
    except Exception as exc:
        print(f"✗ Could not reach API at {api_base}: {exc}")
        print("  Make sure the API server is running:  python main.py api")
        sys.exit(1)


def main() -> None:
    """CLI dispatcher."""
    args = sys.argv[1:]

    if not args:
        # Default: start both services
        cmd_serve()
        return

    command = args[0]

    if command == "serve":
        cmd_serve()

    elif command == "api":
        proc = cmd_api()
        print("FastAPI running. Press Ctrl+C to stop.")
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()

    elif command == "ui":
        proc = cmd_ui()
        print("Streamlit running. Press Ctrl+C to stop.")
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()

    elif command == "ingest":
        if len(args) < 2:
            print("Usage: python main.py ingest <path-or-github-url> [branch]")
            sys.exit(1)
        source = args[1]
        branch = args[2] if len(args) > 2 else "main"
        cmd_ingest(source, branch=branch)

    else:
        print(
            f"Unknown command: '{command}'\n\n"
            "Available commands:\n"
            "  python main.py              Start FastAPI + Streamlit\n"
            "  python main.py api          Start FastAPI only\n"
            "  python main.py ui           Start Streamlit only\n"
            "  python main.py ingest <src> Ingest via running API\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
