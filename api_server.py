"""Programmatic entry point for the FastAPI server.

Run the API server::

    uv run python api_server.py
    # or
    uv run uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Swagger docs available at http://localhost:8000/docs
"""

from dotenv import load_dotenv

load_dotenv()

from code_doc_assistant.api.router import app  # noqa: F401 – re-export for uvicorn

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
