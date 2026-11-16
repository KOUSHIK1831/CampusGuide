import json

import pytest

from campusguide_rag.ingestion import (
    MAX_FILE_BYTES, chunk_documents, load_directory, load_document, split_text,
)


def test_txt_and_markdown(tmp_path):
    (tmp_path / "rules.txt").write_text("  Semester rules for all students.  ", encoding="utf-8")
    (tmp_path / "guide.md").write_text("# Guide\nBring a valid card.", encoding="utf-8")
    units, errors = load_directory(tmp_path)
    assert errors == []
    assert {u["source"] for u in units} == {"rules.txt", "guide.md"}
    assert all(u["text"] == u["text"].strip() for u in units)


def test_json_array_and_documents_object(tmp_path):
    path = tmp_path / "faq.json"
    path.write_text(json.dumps({"documents": [{"question": "Where?", "answer": "Library"}, {"answer": "Office"}]}))
    units = load_document(path)
    assert len(units) == 2
    assert "Library" in units[0]["text"]


def test_pdf_page_numbers(tmp_path):
    fitz = pytest.importorskip("fitz")
    file = tmp_path / "campus.pdf"
    document = fitz.open()
    p = document.new_page()
    p.insert_text((72, 72), "Semester policies page one")
    p = document.new_page()
    p.insert_text((72, 72), "Placement policies page two")
    document.save(file)
    document.close()
    units = load_document(file)
    assert [x["page"] for x in units] == [1, 2]
    assert "Placement" in units[1]["text"]


def test_docx_paragraphs_and_tables(tmp_path):
    docx = pytest.importorskip("docx")
    file = tmp_path / "rules.docx"
    document = docx.Document()
    document.add_paragraph("Campus services handbook")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Library"
    table.cell(0, 1).text = "Four books"
    document.save(file)
    text = load_document(file)[0]["text"]
    assert "Campus services" in text
    assert "Four books" in text


def test_invalid_and_scanned_files(tmp_path):
    file = tmp_path / "bin.exe"
    file.write_bytes(b"x")
    with pytest.raises(ValueError, match="Unsupported"):
        load_document(file)
    fitz = pytest.importorskip("fitz")
    pdf = fitz.open()
    pdf.new_page()
    pdf.save(tmp_path / "scan.pdf")
    pdf.close()
    with pytest.raises(ValueError, match="extractable"):
        load_document(tmp_path / "scan.pdf")


def test_chunking_preserves_source_and_limit():
    units = [{"source": "semester.txt", "page": 3, "text": "semester " * 40}]
    chunks = chunk_documents(units, chunk_size=80, overlap=12)
    assert len(chunks) > 1
    assert len({x["id"] for x in chunks}) == len(chunks)
    assert all(x["source"] == "semester.txt" and x["page"] == 3 for x in chunks)
    assert all(len(x["text"]) <= 80 for x in chunks)
    assert split_text("", 20, 5) == []
    with pytest.raises(ValueError):
        split_text("abc", 20, 20)


def test_reject_oversized_file(tmp_path):
    file = tmp_path / "big.txt"
    with file.open("wb") as f:
        f.truncate(MAX_FILE_BYTES + 1)
    with pytest.raises(ValueError, match="limit"):
        load_document(file)
