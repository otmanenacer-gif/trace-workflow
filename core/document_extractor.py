"""Extraction du texte brut des entretiens (TXT, DOCX, PDF), sans OCR ni IA.

Le texte est rendu tel qu'il est lu dans le fichier. Seules deux
transformations techniques sont appliquées, et elles sont documentées :
- les fins de ligne Windows / Mac (\\r\\n, \\r) deviennent \\n ;
- l'éventuel BOM (marque d'ordre des octets) en tête de TXT est retiré.
Aucune faute, hésitation ou répétition n'est corrigée.

Le résultat est une liste de « blocs » ordonnés : une page pour un PDF,
le document entier pour un TXT ou un DOCX (page = None).
"""

import io
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from core.schemas import make_warning

SUPPORTED_FORMATS = ("txt", "docx", "pdf")

MIME_TYPES = {
    "txt": "text/plain",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

# Encodages de repli pour les TXT qui ne sont pas en UTF-8, dans cet ordre.
# latin-1 ne peut jamais échouer : c'est le dernier recours.
TXT_FALLBACK_ENCODINGS = ("cp1252", "latin-1")


class ExtractionError(Exception):
    """Fichier illisible. `code` renvoie à core.schemas.WARNING_CODES."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _library_version(name: str) -> str:
    try:
        return f"{name} {version(name)}"
    except PackageNotFoundError:
        return name


def normalize_newlines(text: str) -> str:
    """Transformation technique documentée : \\r\\n et \\r deviennent \\n."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


# --- TXT --------------------------------------------------------------------

def _decode_txt(data: bytes) -> tuple[str, str, list[dict]]:
    """Décode un TXT : UTF-8 (avec ou sans BOM), UTF-16 avec BOM, sinon repli."""
    warnings = []
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16"), "utf-16", warnings
    try:
        encoding = "utf-8-sig" if data.startswith(b"\xef\xbb\xbf") else "utf-8"
        return data.decode(encoding), encoding, warnings
    except UnicodeDecodeError:
        pass
    for encoding in TXT_FALLBACK_ENCODINGS:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        warnings.append(make_warning(
            "ENCODING_FALLBACK",
            f"Fichier non UTF-8 : décodé en {encoding}. Vérifier les caractères accentués.",
            encoding=encoding,
        ))
        return text, encoding, warnings
    raise AssertionError("latin-1 ne peut pas échouer")  # pragma: no cover


def extract_txt(data: bytes) -> dict:
    text, encoding, warnings = _decode_txt(data)
    return {
        "pages": [{"page": None, "text": normalize_newlines(text)}],
        "encoding": encoding,
        "extractor": "python (décodage texte)",
        "warnings": warnings,
    }


# --- DOCX -------------------------------------------------------------------

def _docx_block_texts(parent, document, warnings: list) -> list[str]:
    """Renvoie les textes des éléments de `parent` dans l'ordre du document.

    Paragraphes : texte tel quel. Tableaux : une ligne de texte par ligne de
    tableau, cellules séparées par une tabulation. Contrôles de contenu (sdt) :
    parcourus récursivement. Tout autre élément porteur de texte est récupéré
    en texte brut avec un avertissement, pour que rien ne disparaisse.
    """
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    texts = []
    for child in parent.iterchildren():
        tag = child.tag
        if tag == qn("w:p"):
            texts.append(Paragraph(child, document).text)
        elif tag == qn("w:tbl"):
            if not any(w["code"] == "DOCX_TABLE_FLATTENED" for w in warnings):
                warnings.append(make_warning("DOCX_TABLE_FLATTENED"))
            for row in Table(child, document).rows:
                seen, cells = set(), []
                for cell in row.cells:
                    # Les cellules fusionnées sont renvoyées plusieurs fois : dédoublonnage
                    if id(cell._tc) in seen:
                        continue
                    seen.add(id(cell._tc))
                    cells.append("\n".join(p.text for p in cell.paragraphs))
                texts.append("\t".join(cells))
        elif tag == qn("w:sdt"):
            content = child.find(qn("w:sdtContent"))
            if content is not None:
                texts.extend(_docx_block_texts(content, document, warnings))
        elif tag == qn("w:sectPr"):
            continue  # mise en page, sans texte
        else:
            raw = "".join(t.text or "" for t in child.iter(qn("w:t")))
            if raw.strip():
                warnings.append(make_warning("DOCX_UNKNOWN_ELEMENT", element=tag.split("}")[-1]))
                texts.append(raw)
    return texts


def extract_docx(data: bytes) -> dict:
    try:
        import docx

        document = docx.Document(io.BytesIO(data))
    except Exception as exc:  # fichier non zip, XML invalide, etc.
        raise ExtractionError("EXTRACTION_FAILED", f"DOCX illisible : {type(exc).__name__}: {exc}") from exc
    warnings: list[dict] = []
    texts = _docx_block_texts(document.element.body, document, warnings)
    return {
        "pages": [{"page": None, "text": normalize_newlines("\n".join(texts))}],
        "encoding": None,
        "extractor": _library_version("python-docx"),
        "warnings": warnings,
    }


# --- PDF --------------------------------------------------------------------

def extract_pdf(data: bytes) -> dict:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ExtractionError("EXTRACTION_FAILED", "PDF protégé par un mot de passe.")
        pages = [
            {"page": number, "text": normalize_newlines(page.extract_text() or "")}
            for number, page in enumerate(reader.pages, start=1)
        ]
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError("EXTRACTION_FAILED", f"PDF illisible : {type(exc).__name__}: {exc}") from exc
    if not pages:
        raise ExtractionError("EXTRACTION_FAILED", "PDF illisible : aucune page trouvée.")

    warnings = []
    empty_pages = [p["page"] for p in pages if not p["text"].strip()]
    if empty_pages and len(empty_pages) < len(pages):
        warnings.append(make_warning(
            "EMPTY_PDF_PAGES",
            f"Pages sans texte extractible : {', '.join(map(str, empty_pages))}.",
            pages=empty_pages,
        ))
    return {
        "pages": pages,
        "encoding": None,
        "extractor": _library_version("pypdf"),
        "warnings": warnings,
    }


# --- Point d'entrée -----------------------------------------------------------

EXTRACTORS = {"txt": extract_txt, "docx": extract_docx, "pdf": extract_pdf}


def extract_document(path: Path) -> dict:
    """Extrait le texte d'un fichier d'entretien. Le fichier n'est jamais modifié.

    Renvoie un dictionnaire :
      format, mime_type, extractor, encoding,
      pages       : [{"page": int | None, "text": str}, ...] dans l'ordre ;
      page_count  : nombre de pages (PDF) ou None ;
      empty_pages : numéros des pages PDF sans texte ;
      warnings    : avertissements (voir core.schemas).

    Lève ExtractionError si le format est interdit ou le fichier illisible.
    Un fichier lisible mais vide n'est PAS une exception : le validateur
    le signale (NO_TEXT_EXTRACTED).
    """
    path = Path(path)
    fmt = path.suffix.lower().lstrip(".")
    if fmt not in EXTRACTORS:
        raise ExtractionError("UNSUPPORTED_FORMAT", f"Format « .{fmt} » non pris en charge ({', '.join(SUPPORTED_FORMATS)}).")
    result = EXTRACTORS[fmt](path.read_bytes())
    is_pdf = fmt == "pdf"
    result.update({
        "format": fmt,
        "mime_type": MIME_TYPES[fmt],
        "page_count": len(result["pages"]) if is_pdf else None,
        "empty_pages": [p["page"] for p in result["pages"] if not p["text"].strip()] if is_pdf else [],
    })
    return result
