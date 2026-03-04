"""Load source files from a local directory or a remote GitHub repository.

Design decisions
----------------
* Keep it dependency-light: use ``gitpython`` for cloning, ``pathlib`` for
  local traversal.  No LangChain document loaders are used so that the
  ingestion pipeline is fully transparent and testable.
* Filter to ``SUPPORTED_EXTENSIONS`` to skip binary/irrelevant files.
* Respect ``max_file_size_mb`` to avoid embedding enormous auto-generated
  files (e.g., lock files, minified JS bundles).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import git

from code_doc_assistant.config import get_settings
from code_doc_assistant.utils.logging import get_logger

logger = get_logger(__name__)

# File extensions that are treated as source code / documentation.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Python
        ".py", ".pyi",
        # JS / TS
        ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
        # Web
        ".html", ".htm", ".css", ".scss", ".sass",
        # JVM
        ".java", ".kt", ".scala", ".groovy",
        # Systems
        ".c", ".cpp", ".cc", ".h", ".hpp", ".rs", ".go",
        # .NET
        ".cs", ".fs", ".vb",
        # Ruby / PHP / Swift / Dart
        ".rb", ".php", ".swift", ".dart",
        # Shell
        ".sh", ".bash", ".zsh", ".fish", ".ps1",
        # Config / data
        ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".env",
        # Docs
        ".md", ".mdx", ".rst", ".txt",
        # SQL
        ".sql",
        # Docker
        ".dockerfile",
    }
)

# Directories that are always skipped during traversal.
IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git", ".github", ".venv", "venv", "__pycache__", "node_modules",
        ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build",
        "*.egg-info", ".tox", "coverage_html",
    }
)


@dataclass
class SourceFile:
    """A single source file ready to be chunked.

    Attributes:
        path: Absolute or repo-relative path of the file.
        content: Full decoded text content of the file.
        language: Derived from the file extension (e.g. ``"python"``).
        source: ``"local"`` or the GitHub URL string.
    """

    path: str
    content: str
    language: str
    source: str
    metadata: dict[str, str] = field(default_factory=dict)


def _extension_to_language(ext: str) -> str:
    """Map a file extension to a human-readable language label."""
    mapping: dict[str, str] = {
        ".py": "python", ".pyi": "python",
        ".js": "javascript", ".jsx": "javascript",
        ".mjs": "javascript", ".cjs": "javascript",
        ".ts": "typescript", ".tsx": "typescript",
        ".java": "java", ".kt": "kotlin", ".scala": "scala",
        ".c": "c", ".cpp": "cpp", ".cc": "cpp",
        ".h": "c", ".hpp": "cpp",
        ".cs": "csharp", ".fs": "fsharp",
        ".go": "go", ".rs": "rust",
        ".rb": "ruby", ".php": "php",
        ".sh": "shell", ".bash": "shell", ".zsh": "shell",
        ".ps1": "powershell",
        ".sql": "sql",
        ".md": "markdown", ".mdx": "markdown",
        ".rst": "rst",
        ".yaml": "yaml", ".yml": "yaml",
        ".json": "json", ".toml": "toml",
        ".html": "html", ".htm": "html",
        ".css": "css", ".scss": "scss",
        ".dockerfile": "dockerfile",
    }
    return mapping.get(ext.lower(), "text")


def _is_ignored_dir(path: Path) -> bool:
    """Return True if any part of *path* matches an ignored directory name."""
    return any(part in IGNORED_DIRS for part in path.parts)


def load_local(directory: str | Path) -> Iterator[SourceFile]:
    """Recursively yield :class:`SourceFile` objects from a local directory.

    Args:
        directory: Path to the root directory to ingest.

    Yields:
        :class:`SourceFile` for each eligible source file found.
    """
    settings = get_settings()
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    root = Path(directory).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    logger.info("Loading files from local directory: %s", root)
    count = 0

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if _is_ignored_dir(file_path.relative_to(root)):
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if file_path.stat().st_size > max_bytes:
            logger.debug("Skipping large file (%d bytes): %s", file_path.stat().st_size, file_path)
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Cannot read %s: %s", file_path, exc)
            continue

        relative = str(file_path.relative_to(root))
        language = _extension_to_language(file_path.suffix)
        source_label = str(root)  # resolved absolute path — unique per ingested directory
        yield SourceFile(
            path=relative,
            content=content,
            language=language,
            source=source_label,
            metadata={"source": source_label, "file_path": relative, "language": language},
        )
        count += 1

    logger.info("Loaded %d files from %s", count, root)


def load_github(repo_url: str, branch: str = "main") -> Iterator[SourceFile]:
    """Clone a GitHub repository and yield :class:`SourceFile` objects.

    The repo is cloned into a temporary directory which is cleaned up after
    iteration (when the generator is garbage-collected).

    Args:
        repo_url: HTTPS or SSH URL of the GitHub repository.
        branch: Branch name to checkout (default: ``"main"``).

    Yields:
        :class:`SourceFile` for each eligible source file found.
    """
    logger.info("Cloning GitHub repository: %s (branch=%s)", repo_url, branch)
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            git.Repo.clone_from(repo_url, tmp_dir, branch=branch, depth=1)
        except git.GitCommandError as exc:
            # Some repos use 'master' as default branch.  Try falling back.
            if branch == "main":
                logger.warning("Branch 'main' not found, retrying with 'master'…")
                git.Repo.clone_from(repo_url, tmp_dir, branch="master", depth=1)
            else:
                raise RuntimeError(f"Failed to clone {repo_url}: {exc}") from exc

        yield from (
            SourceFile(
                path=f.path,
                content=f.content,
                language=f.language,
                source=repo_url,
                metadata={
                    **f.metadata,
                    "source": repo_url,   # override temp-dir path set by load_local
                    "repo_url": repo_url,
                    "branch": branch,
                },
            )
            for f in load_local(tmp_dir)
        )
