from __future__ import annotations

from pathlib import Path


def load_text_from_document(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8")

    if suffix == ".pdf":
        return _read_pdf(path)

    if suffix in {".docx", ".dox"}:
        return _read_docx(path)

    raise ValueError(
        f"Неподдерживаемый формат файла: {suffix}. Поддерживаются: .txt, .pdf, .docx, .dox"
    )


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Для чтения PDF установите зависимость pypdf: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text.strip())

    return "\n\n".join(parts).strip()


def _read_docx(path: Path) -> str:
    try:
        from docx import Document  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Для чтения DOCX/DOX установите python-docx: pip install python-docx"
        ) from exc

    doc = Document(str(path))
    parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(parts).strip()
