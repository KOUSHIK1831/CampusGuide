"""Text-first, memory-safe document loaders for the CampusGuide RAG app.

The interactive app consumes uploaded bytes directly, without writing private
content to a shared directory. The CLI can still load local files by path.
"""
from __future__ import annotations

import json
import re
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".json", ".pdf", ".docx"})
MAX_FILE_BYTES = 15 * 1024 * 1024
POLICY_FIELDS = ("policy_id", "version", "effective_date")
_POLICY_HEADER = re.compile(r"(Policy ID|Version|Effective date):[ \t]*(\S.{0,79})", re.IGNORECASE)


def _json_records(data: Any) -> list[str]:
    if isinstance(data, dict) and isinstance(data.get("documents"), list):
        data = data["documents"]
    if isinstance(data, list):
        return [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in data]
    return [json.dumps(data, ensure_ascii=False, sort_keys=True)]


def _policy_metadata(first_record: str) -> dict[str, str]:
    """Accept explicitly labeled fields near the top of a document."""
    headers: dict[str, set[str]] = {}
    for line in first_record.strip().split("\n", 12)[:12]:
        match = _POLICY_HEADER.fullmatch(line.strip())
        if match:
            field = match[1].lower().replace(" ", "_")
            headers.setdefault(field, set()).add(match[2])
    metadata = {field: values.pop() for field, values in headers.items() if len(values) == 1}
    if "effective_date" in metadata:
        value = metadata["effective_date"]
        try:
            # Require the documented YYYY-MM-DD form, including zero padding.
            if date.fromisoformat(value).isoformat() != value:
                metadata.pop("effective_date")
        except ValueError:
            metadata.pop("effective_date")
    return metadata


def load_document_bytes(filename: str, data: bytes) -> list[dict]:
    """Extract text and source/page metadata from one in-memory upload.

    File contents are processed without persisting the original document.
    Scanned PDF images are deliberately unsupported (no OCR).
    """
    source = filename.replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(source).suffix.lower()
    if not source or suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document type: {source}")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"Document exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB limit: {source}")
    if not data:
        raise ValueError(f"Empty document: {source}")

    records: list[tuple[int | None, str]] = []
    if suffix in {".txt", ".md", ".markdown", ".json"}:
        try:
            decoded = data.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeError as exc:
            raise ValueError(f"Document must be UTF-8 encoded: {source}") from exc
        if suffix == ".json":
            try:
                records = [(None, part) for part in _json_records(json.loads(decoded))]
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON document: {source}") from exc
        else:
            records = [(None, decoded)]
    elif suffix == ".pdf":
        import fitz  # PyMuPDF
        try:
            with fitz.open(stream=data, filetype="pdf") as pdf:
                if pdf.needs_pass:
                    raise ValueError(f"Encrypted PDF is not supported: {source}")
                records = [(i + 1, page.get_text("text")) for i, page in enumerate(pdf)]
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Unreadable PDF document: {source}") from exc
    elif suffix == ".docx":
        from docx import Document
        try:
            document = Document(BytesIO(data))
            paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
            for table in document.tables:
                for row in table.rows:
                    paragraphs.append(" | ".join(cell.text for cell in row.cells))
            records = [(None, "\n".join(paragraphs))]
        except Exception as exc:
            raise ValueError(f"Unreadable DOCX document: {source}") from exc

    metadata = _policy_metadata(records[0][1]) if records and suffix != ".json" else {}
    units = [
        {"text": content.strip(), "source": source, "page": page, **metadata}
        for page, content in records if content and content.strip()
    ]
    if not units:
        raise ValueError(f"No extractable text in {source}; scanned PDFs require OCR")
    return units


def load_document(path: str | Path) -> list[dict]:
    """Load a local document file; used by CLI ingestion and evaluation."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"Document exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB limit: {path.name}")
    return load_document_bytes(path.name, path.read_bytes())


def load_directory(directory: str | Path) -> tuple[list[dict], list[str]]:
    """Load a local corpus, collecting per-file errors instead of losing good documents."""
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
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            errors.append(f"{path.name}: {exc}")
    return units, errors


def split_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    """Split on nearby whitespace with guaranteed forward progress."""
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
    """Create stable chunk IDs and preserve each uploaded document's metadata."""
    result: list[dict] = []
    for unit_id, unit in enumerate(units):
        for chunk_id, part in enumerate(split_text(unit["text"], chunk_size, overlap)):
            result.append({
                "id": f"{unit_id}:{chunk_id}", "text": part,
                "source": unit["source"], "page": unit.get("page"),
                **{field: unit[field] for field in POLICY_FIELDS if unit.get(field)},
            })
    return result
