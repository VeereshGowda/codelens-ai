"""Pytest fixtures shared across the test suite."""

from __future__ import annotations

import pytest

from code_doc_assistant.ingestion.loader import SourceFile


@pytest.fixture()
def sample_source_file() -> SourceFile:
    """A simple Python source file fixture."""
    return SourceFile(
        path="mymodule/example.py",
        content='"""Module docstring."""\n\ndef add(a: int, b: int) -> int:\n    """Return a + b."""\n    return a + b\n\n\nclass Calculator:\n    """A simple calculator."""\n\n    def multiply(self, x: int, y: int) -> int:\n        """Return x * y."""\n        return x * y\n',
        language="python",
        source="local",
        metadata={"source": "local", "file_path": "mymodule/example.py", "language": "python"},
    )
