"""
Сервис чтения документов.

Поддерживаемые форматы: PDF, DOCX, TXT.
Включает нормализацию текста и декодинг битой кириллицы из PDF.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# ── Нормализация текста ─────────────────────────────────────


def normalize_whitespace(text: str) -> str:
    """Схлопывает лишние пробелы и переносы строк."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fix_pdf_artifacts(text: str) -> str:
    """Склеивает разорванные переносом слова и убирает лишние пробелы перед пунктуацией."""
    # Склейка переноса: «сло-\nво» -> «слово»
    text = re.sub(
        r"([A-Za-zА-Яа-яЁё])\-\n([A-Za-zА-Яа-яЁё])",
        r"\1\2",
        text,
    )
    # Разрядка: «П р и м е р» -> «Пример»
    text = re.sub(
        r"(?:(?<=\s)|^)([A-Za-zА-Яа-яЁё](?:\s+[A-Za-zА-Яа-яЁё]){2,})(?=\s|$)",
        lambda m: m.group(1).replace(" ", ""),
        text,
    )
    # Пробел перед пунктуацией: «слово .» -> «слово.»
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


# ── Декодинг подстановочной кириллицы (битые cmap в PDF) ────

_CYRILLIC_SUBSTITUTION_MAP = {
    "\x10": "А", "\x11": "Б", "\x12": "В", "\x13": "Г",
    "\x14": "Д", "\x15": "Е", "\x16": "Ж", "\x17": "З",
    "\x18": "И", "\x19": "Й", "\x1a": "К", "\x1b": "Л",
    "\x1c": "М", "\x1d": "Н", "\x1e": "О", "\x1f": "П",
    "!": "Р", "\"": "С", "#": "Т", "$": "У",
    "%": "Ф", "&": "Х", "'": "Ц", "(": "Ч",
    ")": "Ш", "*": "Щ", "+": "Ъ", ",": "Ы",
    "-": "Ь", ".": "Э", "/": "Ю",
    "0": "а", "1": "б", "2": "в", "3": "г",
    "4": "д", "5": "е", "6": "ж", "7": "з",
    "8": "и", "9": "й", ":": "к", ";": "л",
    "<": "м", "=": "н", ">": "о", "?": "п",
    "@": "р", "A": "с", "B": "т", "C": "у",
    "D": "ф", "E": "х", "F": "ц", "G": "ч",
    "H": "ш", "I": "щ", "J": "ъ", "K": "ы",
    "L": "ь", "M": "э", "N": "ю", "O": "я",
}

_MAPPED_CHARS = set(_CYRILLIC_SUBSTITUTION_MAP.keys())


def _decode_token(match: re.Match) -> str:  # type: ignore[type-arg]
    """Декодирует один токен, если он похож на подстановочную кириллицу."""
    token = match.group(0)
    if "http" in token.lower() or "/" in token:
        return token
    hits = sum(1 for ch in token if ch in _MAPPED_CHARS)
    if hits < 3:
        return token
    if hits / max(1, len(token)) < 0.55:
        return token
    return "".join(_CYRILLIC_SUBSTITUTION_MAP.get(ch, ch) for ch in token)


def decode_substitution_cyrillic(text: str) -> str:
    """Декодирует последовательности подстановочного алфавита обратно в кириллицу."""
    return re.sub(r"[^\s]+", _decode_token, text)


def normalize_text(text: str) -> str:
    """Полный пайплайн нормализации: декодинг -> артефакты -> пробелы."""
    text = decode_substitution_cyrillic(text)
    text = fix_pdf_artifacts(text)
    text = normalize_whitespace(text)
    return text


# ── Читалки по форматам ─────────────────────────────────────


def read_txt(path: Path) -> str:
    """Читает текстовый файл."""
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_pymupdf(path: Path) -> str:
    """Читает PDF через PyMuPDF (fitz) — более точное извлечение."""
    import fitz  # pylint: disable=import-outside-toplevel

    parts: List[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            blocks = page.get_text("blocks")
            blocks = sorted(
                blocks, key=lambda b: (round(b[1], 1), round(b[0], 1))
            )
            page_lines = [
                (block[4] or "").strip()
                for block in blocks
                if (block[4] or "").strip()
            ]
            parts.append("\n".join(page_lines))
    return "\n\n".join(parts)


def read_pdf_pypdf(path: Path) -> str:
    """Читает PDF через pypdf — fallback если PyMuPDF недоступен."""
    from pypdf import PdfReader  # pylint: disable=import-outside-toplevel

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def read_pdf(path: Path) -> str:
    """Читает PDF: сначала PyMuPDF, при ошибке — pypdf."""
    try:
        return read_pdf_pymupdf(path)
    except Exception:  # pylint: disable=broad-except
        return read_pdf_pypdf(path)


def read_docx(path: Path) -> str:
    """Читает DOCX через python-docx."""
    from docx import Document  # pylint: disable=import-outside-toplevel

    doc = Document(str(path))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


# ── Единая точка входа ──────────────────────────────────────

_READERS = {
    ".txt": read_txt,
    ".pdf": read_pdf,
    ".docx": read_docx,
}


def read_document(path: Path) -> str:
    """
    Читает документ любого поддерживаемого формата и возвращает
    нормализованный текст.

    Raises:
        ValueError: если расширение файла не поддерживается.
    """
    ext = path.suffix.lower()
    reader = _READERS.get(ext)
    if reader is None:
        raise ValueError(
            f"Unsupported extension: {ext}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    raw = reader(path)
    return normalize_text(raw)


def collect_files(paths: list[str]) -> List[Path]:
    """
    Собирает список файлов из переданных путей.

    Принимает как файлы, так и директории (рекурсивно).
    Возвращает отсортированный список уникальных путей.
    """
    files: List[Path] = []
    for item in paths:
        path = Path(item).expanduser().resolve()
        if not path.exists():
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
        elif path.is_dir():
            for file_path in path.rglob("*"):
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
                ):
                    files.append(file_path.resolve())
    return sorted(set(files))
