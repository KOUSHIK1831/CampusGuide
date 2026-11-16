"""Safe, text-first loaders for local RAG documents.

PDFs must contain extractable text; OCR is deliberately out of MVP scope.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".json", ".pdf", ".docx"})
MAX_FILE_BYTES = 15 * 1024 * 1024


def _json_records(data: Any) -> list[str]:
    if isinstance(data, dict) and isinstance(data.get("documents"), list):
        data = data["documents"]
    if isinstance(data, list):
        return [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in data]
    return [json.dumps(data, ensure_ascii=False, sort_keys=True)]


def load_document(path: str | Path) -> list[dict]:
    """Return text units with source and 1-based page for PDFs.

    Raises for unsupported formats, unreadable documents and image-only PDFs.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document type: {suffix}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"Document exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB limit: {path.name}")

    source = path.name
    records: list[tuple[int | None, str]] = []
    if suffix in {".txt", ".md", ".markdown"}:
        records = [(None, path.read_text(encoding="utf-8-sig"))]
    elif suffix == ".json":
        records = [(None, text) for text in _json_records(json.loads(path.read_text(encoding="utf-8-sig")))]
    elif suffix == ".pdf":
        import fitz  # PyMuPDF; optional until PDFs are ingested
        with fitz.open(str(path)) as pdf:
            if pdf.needs_pass:
                raise ValueError(f"Encrypted PDF is not supported: {source}")
            records = [(i + 1, page.get_text("text")) for i, page in enumerate(pdf)]
    elif suffix == ".docx":
        from docx import Document
        document = Document(str(path))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                paragraphs.append(" | ".join(cell.text for cell in row.cells))
        records = [(None, "\n".join(paragraphs))]

    units = [
        {"text": text.strip(), "source": source, "page": page}
        for page, text in records if text and text.strip()
    ]
    if not units:
        raise ValueError(f"No extractable text in {source}; scanned PDFs require OCR")
    return units


def load_directory(directory: str | Path) -> tuple[list[dict], list[str]]:
    """Load a corpus, collecting per-file errors instead of losing healthy documents."""
    directory = Path(directory)
    if not directory.exists():
        return [], []
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    units: list[dict] = []
    errors: list[str] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            units.extend(load_document(path))
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError, ImportError) as exc:
            errors.append(f"{path.name}: {exc}")
    return units, errors


def split_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    """Split on nearby whitespace with a guaranteed forward progress invariant."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("chunk_size must be positive and 0 <= overlap < chunk_size")
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            split_at = text.rfind(" ", start + chunk_size // 2, end)
            if split_at > start:
                end = split_at + 1
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def chunk_documents(units: list[dict], chunk_size: int = 900, overlap: int = 120) -> list[dict]:
    """Create stable chunk IDs and preserve the original document/page metadata."""
    result: list[dict] = []
    for unit_id, unit in enumerate(units):
        for chunk_id, part in enumerate(split_text(unit["text"], chunk_size, overlap)):
            result.append({
                "id": f"{unit_id}:{chunk_id}", "text": part,
                "source": unit["source"], "page": unit.get("page"),
            })
    return result
