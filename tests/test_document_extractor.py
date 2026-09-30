"""Tests de l'extraction TXT / DOCX / PDF (documents synthétiques uniquement)."""

import pytest

from core.document_extractor import ExtractionError, extract_document
from tests.synthetic_docs import SAMPLE_LINES, make_docx, make_pdf

SAMPLE_TEXT = "\n".join(SAMPLE_LINES) + "\n"


def write(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def full_text(result):
    return "\n".join(p["text"] for p in result["pages"])


# --- TXT ------------------------------------------------------------------------

def test_txt_utf8_preserved_exactly(tmp_path):
    result = extract_document(write(tmp_path, "e.txt", SAMPLE_TEXT.encode("utf-8")))
    assert result["format"] == "txt" and result["mime_type"] == "text/plain"
    assert full_text(result) == SAMPLE_TEXT
    assert result["pages"][0]["page"] is None
    assert result["warnings"] == []


def test_txt_bom_and_windows_newlines_are_technical_only(tmp_path):
    data = b"\xef\xbb\xbf" + "Enquêté : euh…\r\nbon\r\n".encode("utf-8")
    result = extract_document(write(tmp_path, "e.txt", data))
    assert full_text(result) == "Enquêté : euh…\nbon\n"


def test_txt_encoding_fallback_warns(tmp_path):
    result = extract_document(write(tmp_path, "e.txt", "Enquêté : c’était ça".encode("cp1252")))
    assert full_text(result) == "Enquêté : c’était ça"
    assert result["encoding"] == "cp1252"
    assert [w["code"] for w in result["warnings"]] == ["ENCODING_FALLBACK"]


def test_txt_utf16_with_bom(tmp_path):
    result = extract_document(write(tmp_path, "e.txt", "Enquêté : oui".encode("utf-16")))
    assert full_text(result) == "Enquêté : oui"


def test_txt_empty(tmp_path):
    result = extract_document(write(tmp_path, "e.txt", b""))
    assert full_text(result) == ""


def test_source_file_is_not_modified(tmp_path):
    path = write(tmp_path, "e.txt", b"\xef\xbb\xbfA\r\nB")
    before = path.read_bytes()
    extract_document(path)
    assert path.read_bytes() == before


# --- DOCX -----------------------------------------------------------------------

def test_docx_paragraphs_in_order(tmp_path):
    result = extract_document(write(tmp_path, "e.docx", make_docx(SAMPLE_LINES)))
    assert result["format"] == "docx"
    assert full_text(result) == "\n".join(SAMPLE_LINES)


def test_docx_tables_are_kept(tmp_path):
    data = make_docx(["Avant", "Après"], table_rows=[["Âge", "20 ans"], ["Filière", "Socio"]])
    result = extract_document(write(tmp_path, "e.docx", data))
    text = full_text(result)
    for fragment in ("Avant", "Après", "Âge\t20 ans", "Filière\tSocio"):
        assert fragment in text
    assert "DOCX_TABLE_FLATTENED" in [w["code"] for w in result["warnings"]]


def test_docx_empty(tmp_path):
    result = extract_document(write(tmp_path, "e.docx", make_docx([])))
    assert full_text(result).strip() == ""


def test_docx_corrupted(tmp_path):
    with pytest.raises(ExtractionError) as info:
        extract_document(write(tmp_path, "e.docx", b"ceci n'est pas un docx"))
    assert info.value.code == "EXTRACTION_FAILED"


# --- PDF ------------------------------------------------------------------------

def test_pdf_text_by_page(tmp_path):
    data = make_pdf([SAMPLE_LINES[:4], SAMPLE_LINES[4:]])
    result = extract_document(write(tmp_path, "e.pdf", data))
    assert result["format"] == "pdf" and result["page_count"] == 2
    assert [p["page"] for p in result["pages"]] == [1, 2]
    # Une ligne vide n'a pas d'existence dans un PDF : on compare les lignes non vides
    assert result["pages"][0]["text"].splitlines() == [l for l in SAMPLE_LINES[:4] if l]
    assert result["pages"][1]["text"].splitlines() == SAMPLE_LINES[4:]
    assert result["empty_pages"] == [] and result["warnings"] == []


def test_pdf_blank_page_is_reported(tmp_path):
    result = extract_document(write(tmp_path, "e.pdf", make_pdf([["Enquêteur : a"], [], ["Enquêté : b"]])))
    assert result["empty_pages"] == [2]
    assert [w["code"] for w in result["warnings"]] == ["EMPTY_PDF_PAGES"]


def test_pdf_without_text(tmp_path):
    result = extract_document(write(tmp_path, "e.pdf", make_pdf([[]])))
    assert result["empty_pages"] == [1]
    assert full_text(result) == ""


def test_pdf_corrupted(tmp_path):
    with pytest.raises(ExtractionError) as info:
        extract_document(write(tmp_path, "e.pdf", b"%PDF-1.4\nn'importe quoi"))
    assert info.value.code == "EXTRACTION_FAILED"


# --- Formats ----------------------------------------------------------------------

@pytest.mark.parametrize("name", ["e.doc", "e.odt", "e.exe", "sans_extension"])
def test_unsupported_extension(tmp_path, name):
    with pytest.raises(ExtractionError) as info:
        extract_document(write(tmp_path, name, b"x"))
    assert info.value.code == "UNSUPPORTED_FORMAT"
