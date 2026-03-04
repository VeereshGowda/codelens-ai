"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings  # type: ignore[import]

# Load .env file at import time so all settings have values available
load_dotenv()


class Settings(BaseSettings):
    """All configuration for the Code Documentation Assistant.

    Values are read from environment variables (case-insensitive) or from a
    .env file in the working directory.  Provide at minimum
    AZURE_OPENAI_API_KEY to start the application.
    """

    # ------------------------------------------------------------------
    # Azure OpenAI
    # ------------------------------------------------------------------
    azure_openai_api_key: str = Field(..., description="Azure OpenAI subscription key")
    azure_openai_endpoint: str = Field(
        default="https://gem-openai-mm-we-001.openai.azure.com/",
        description="Azure OpenAI resource endpoint",
    )
    azure_openai_api_version: str = Field(
        default="2024-12-01-preview",
        description="Azure OpenAI REST API version",
    )
    azure_openai_chat_deployment: str = Field(
        default="gpt-4o",
        description="Deployment name for the chat completion model",
    )
    azure_openai_embedding_deployment: str = Field(
        default="text-embedding-ada-002",
        description="Deployment name for the embedding model",
    )

    # ------------------------------------------------------------------
    # ChromaDB vector store
    # ------------------------------------------------------------------
    chroma_persist_dir: str = Field(
        default="./chroma_db",
        description="Directory where ChromaDB persists data",
    )
    chroma_collection_name: str = Field(
        default="code_docs",
        description="Name of the ChromaDB collection",
    )

    # ------------------------------------------------------------------
    # Chunking / ingestion
    # ------------------------------------------------------------------
    max_file_size_mb: int = Field(
        default=5,
        description="Files larger than this (MB) are skipped during ingestion",
    )
    chunk_size: int = Field(
        default=1000,
        description="Target number of tokens per chunk",
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap in tokens between consecutive chunks",
    )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    top_k: int = Field(
        default=5,
        description="Number of chunks to retrieve per query",
    )

    # ------------------------------------------------------------------
    # LLM generation
    # ------------------------------------------------------------------
    max_tokens: int = Field(default=2048, description="Max tokens in the LLM response")
    temperature: float = Field(default=0.2, description="Sampling temperature")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    log_level: str = Field(default="INFO", description="Python logging level")

    # ------------------------------------------------------------------
    # Microservice: Streamlit → FastAPI base URL
    # ------------------------------------------------------------------
    api_base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL of the FastAPI backend (used by the Streamlit client)",
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton)."""
    return Settings()  # type: ignore[call-arg]
