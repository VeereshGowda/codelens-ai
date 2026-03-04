"""Tests for the ingestion loader module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from code_doc_assistant.ingestion.loader import (
    SUPPORTED_EXTENSIONS,
    SourceFile,
    _extension_to_language,
    _is_ignored_dir,
    load_local,
)


class TestExtensionToLanguage:
    def test_python(self) -> None:
        assert _extension_to_language(".py") == "python"

    def test_typescript(self) -> None:
        assert _extension_to_language(".ts") == "typescript"

    def test_unknown_extension(self) -> None:
        assert _extension_to_language(".xyz") == "text"

    def test_case_insensitive(self) -> None:
        assert _extension_to_language(".PY") == "python"


class TestIsIgnoredDir:
    def test_node_modules_ignored(self) -> None:
        p = Path("project/node_modules/some_file.js")
        assert _is_ignored_dir(p)

    def test_venv_ignored(self) -> None:
        p = Path("project/.venv/pydantic/__init__.py")
        assert _is_ignored_dir(p)

    def test_source_file_not_ignored(self) -> None:
        p = Path("src/mymodule/utils.py")
        assert not _is_ignored_dir(p)


class TestLoadLocal:
    def test_loads_python_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "main.py").write_text("print('hello')", encoding="utf-8")
            files = list(load_local(tmp))
        assert len(files) == 1
        assert files[0].language == "python"
        assert files[0].content == "print('hello')"

    def test_skips_unsupported_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "image.png").write_bytes(b"\x89PNG")
            files = list(load_local(tmp))
        assert files == []

    def test_skips_ignored_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            node_mod = Path(tmp) / "node_modules"
            node_mod.mkdir()
            (node_mod / "index.js").write_text("module.exports={}", encoding="utf-8")
            files = list(load_local(tmp))
        assert files == []

    def test_raises_on_missing_directory(self) -> None:
        with pytest.raises(FileNotFoundError):
            list(load_local("/nonexistent/path/that/does/not/exist"))

    def test_source_file_has_correct_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "utils.py").write_text("pass", encoding="utf-8")
            files = list(load_local(tmp))
        sf = files[0]
        assert isinstance(sf, SourceFile)
        # source is the resolved absolute path of the ingested directory
        assert sf.source == str(Path(tmp).resolve())
        assert "file_path" in sf.metadata
        assert "language" in sf.metadata
        assert sf.metadata["source"] == sf.source

    def test_loads_multiple_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "code.py").write_text("x = 1", encoding="utf-8")
            (Path(tmp) / "README.md").write_text("# Title", encoding="utf-8")
            (Path(tmp) / "config.yaml").write_text("key: value", encoding="utf-8")
            files = list(load_local(tmp))
        assert len(files) == 3
