"""Structured logging setup for the application.

Logs are written to two sinks:
  - **stderr**: Rich-formatted coloured output for local development.
  - **logs/app.log**: Plain rotating file log for persistent observability and
    production inspection (e.g. streamed to Azure Monitor).
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)

# Directory where file logs are stored; created automatically if absent.
_LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
_LOG_FILE = _LOG_DIR / "app.log"
_MAX_BYTES = 10 * 1024 * 1024   # 10 MB per file
_BACKUP_COUNT = 5               # keep up to 5 rotated files

_configured = False


def _ensure_log_dir() -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a logger that writes to both the terminal (Rich) and a log file.

    A :class:`~logging.handlers.RotatingFileHandler` is attached to the root
    logger the first time this function is called; subsequent calls reuse the
    same handlers via ``force=False``.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
        level: The minimum log level string (e.g. ``"DEBUG"``, ``"INFO"``).

    Returns:
        A :class:`logging.Logger` instance.
    """
    global _configured
    if not _configured:
        _ensure_log_dir()

        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = RotatingFileHandler(
            _LOG_FILE,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level.upper())

        rich_handler = RichHandler(
            console=_console,
            rich_tracebacks=True,
            show_path=True,
        )

        logging.basicConfig(
            level=level.upper(),
            format="%(message)s",
            datefmt="[%X]",
            handlers=[rich_handler, file_handler],
            force=True,
        )
        _configured = True

    return logging.getLogger(name)
