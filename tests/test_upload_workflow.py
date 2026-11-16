"""Regressions for BYOK upload parsing and per-session document isolation."""
from io import BytesIO

import numpy as np
import pytest

from campusguide_rag.ingestion import load_document_bytes
from campusguide_rag.upload_workflow import (
    prepare_uploaded_documents, upload_fingerprint, MAX_TOTAL_BYTES,
)
from campusguide_rag.vector_store import VectorStore


def test_txt_upload_without_persisting_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    chunks, count = prepare_uploaded_documents([("notes.txt", b"Meeting held on Tuesday")])
    assert count == 1
    assert chunks[0]["source"] == "notes.txt"
    assert "Tuesday" in chunks[0]["text"]
    assert list(tmp_path.iterdir()) == []


def test_markdown_and_json_uploads():
    files = [("guide.md", b"# Guide\nHall at Block A"),
             ("facts.json", b'{"documents":[{"text":"Library at B"}]}')]
    chunks, count = prepare_uploaded_documents(files)
    assert count == 2
    assert {item["source"] for item in chunks} == {"guide.md", "facts.json"}


def test_pdf_upload_page_tracking():
    fitz = pytest.importorskip("fitz")
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((50, 50), "Admission closes in July")
    data = pdf.tobytes()
    pdf.close()
    chunks, count = prepare_uploaded_documents([("brochure.pdf", data)])
    assert count == 1
    assert chunks[0]["page"] == 1
    assert "Admission" in chunks[0]["text"]


def test_docx_upload_in_memory():
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("Scholarship form deadline")
    buffer = BytesIO()
    document.save(buffer)
    chunks, _ = prepare_uploaded_documents([("form.docx", buffer.getvalue())])
    assert "Scholarship" in chunks[0]["text"]


def test_invalid_and_duplicate_uploads_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_uploaded_documents([("Book.TXT", b"a"), ("book.txt", b"b")])
    with pytest.raises(ValueError, match="Unsupported"):
        prepare_uploaded_documents([("script.py", b"print(1)")])
    with pytest.raises(ValueError, match="No extractable text"):
        load_document_bytes("blank.md", b"  ")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_document_bytes("bad.json", b"{not valid")


def test_upload_limits():
    with pytest.raises(ValueError, match="at most"):
        prepare_uploaded_documents([(f"file{i}.txt", b"hi") for i in range(11)])
    with pytest.raises(ValueError, match="15 MB"):
        prepare_uploaded_documents([("huge.txt", b"x" * (15 * 1024 * 1024 + 1))])
    assert MAX_TOTAL_BYTES == 40 * 1024 * 1024


def test_fingerprint_changes_when_upload_changes():
    assert upload_fingerprint([]) is None
    assert upload_fingerprint([("first.txt", b"a")]) != upload_fingerprint([("first.txt", b"b")])
    assert upload_fingerprint([("first.txt", b"a")]) != upload_fingerprint([("second.txt", b"a")])


class FakeEncoder:
    def encode(self, texts):
        return np.asarray([[1.0, 0.0] if "secret" in t else [0.0, 1.0] for t in texts], dtype=np.float32)


def test_two_session_stores_do_not_share_uploaded_chunks():
    encoder = FakeEncoder()
    first_chunks, _ = prepare_uploaded_documents([("first.txt", b"private secret notes")])
    second_chunks, _ = prepare_uploaded_documents([("second.txt", b"public notes")])
    first = VectorStore(encoder=encoder, model_name="fake", backend="numpy")
    second = VectorStore(encoder=encoder, model_name="fake", backend="numpy")
    first.build(first_chunks)
    second.build(second_chunks)
    assert first.search("secret", min_score=0.5)[0]["source"] == "first.txt"
    assert second.search("secret", min_score=0.5) == []
    assert second.stats()["documents"] == 1


def test_bad_docx_upload_rejects_entire_batch_without_partial_index():
    """BYOK upload rejects incomplete batches; directory CLI skips bad files."""
    with pytest.raises(ValueError, match="Unreadable DOCX document"):
        load_document_bytes("bad.docx", b"not-a-valid-docx-file")
    with pytest.raises(ValueError, match="Unreadable DOCX document"):
        prepare_uploaded_documents([
            ("good.txt", b"Valid campus information"),
            ("bad.docx", b"not-a-valid-docx-file"),
        ])
