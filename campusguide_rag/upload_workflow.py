"""Validation and extraction for isolated, in-memory Streamlit upload sessions."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

from .ingestion import MAX_FILE_BYTES, SUPPORTED_EXTENSIONS, chunk_documents, load_document_bytes

MAX_UPLOAD_FILES = 10
MAX_TOTAL_BYTES = 40 * 1024 * 1024
UploadedFileData = tuple[str, bytes]


def upload_fingerprint(files: Sequence[UploadedFileData]) -> str | None:
    """Stable signature used to invalidate an old index when uploads change."""
    if not files:
        return None
    digest = hashlib.sha256()
    for name, data in files:
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()


def prepare_uploaded_documents(files: Sequence[UploadedFileData]) -> tuple[list[dict], int]:
    """Validate all files first, then extract/chunk without writing to disk.

    An invalid file rejects the whole batch so a user never accidentally
    chats against an incomplete or stale collection of documents.
    """
    if not files:
        raise ValueError("Upload at least one document first.")
    if len(files) > MAX_UPLOAD_FILES:
        raise ValueError(f"Upload at most {MAX_UPLOAD_FILES} documents at a time.")
    total_bytes = 0
    seen: set[str] = set()
    for name, data in files:
        basename = name.replace("\\", "/").rsplit("/", 1)[-1]
        if not basename or Path(basename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file: {basename or 'unnamed file'}")
        if basename.casefold() in seen:
            raise ValueError(f"Duplicate filename: {basename}. Rename one of the documents.")
        seen.add(basename.casefold())
        if len(data) > MAX_FILE_BYTES:
            raise ValueError(f"File {basename} exceeds the 15 MB limit.")
        total_bytes += len(data)
    if total_bytes > MAX_TOTAL_BYTES:
        raise ValueError("Combined uploads exceed the 40 MB limit.")

    units: list[dict] = []
    for name, data in files:
        units.extend(load_document_bytes(name, data))
    chunks = chunk_documents(units)
    if not chunks:
        raise ValueError("No extractable text found in these documents.")
    return chunks, len(files)
