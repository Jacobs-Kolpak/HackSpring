from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Set

SUPPORTED_EXTENSIONS: Set[str] = {".pdf", ".docx", ".txt"}


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fix_pdf_artifacts(text: str) -> str:
    text = re.sub(
        r"([A-Za-zА-Яа-яЁё])\-\n([A-Za-zА-Яа-яЁё])", r"\1\2", text
    )
    text = re.sub(
        r"(?:(?<=\s)|^)"
        r"([A-Za-zА-Яа-яЁё](?:\s+[A-Za-zА-Яа-яЁё]){2,})"
        r"(?=\s|$)",
        lambda m: m.group(1).replace(" ", ""),
        text,
    )
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


_CYRILLIC_MAP: Dict[str, str] = {
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
_MAPPED_CHARS = set(_CYRILLIC_MAP.keys())


def decode_substitution_cyrillic(text: str) -> str:
    def _decode(match: "re.Match[str]") -> str:
        token = match.group(0)
        if "http" in token.lower() or "/" in token:
            return token
        hits = sum(1 for ch in token if ch in _MAPPED_CHARS)
        if hits < 3 or hits / max(1, len(token)) < 0.55:
            return token
        return "".join(_CYRILLIC_MAP.get(ch, ch) for ch in token)

    result: str = re.sub(r"[^\s]+", _decode, text)
    return result


def normalize_text(text: str) -> str:
    text = decode_substitution_cyrillic(text)
    text = fix_pdf_artifacts(text)
    return normalize_whitespace(text)


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    try:
        return _read_pdf_pymupdf(path)
    except Exception:  # pylint: disable=broad-except
        return _read_pdf_pypdf(path)


def _read_pdf_pymupdf(path: Path) -> str:
    import fitz  # pylint: disable=import-outside-toplevel

    parts: List[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            blocks = sorted(
                page.get_text("blocks"),
                key=lambda b: (round(b[1], 1), round(b[0], 1)),
            )
            lines = [
                str(b[4]).strip() for b in blocks if str(b[4]).strip()
            ]
            parts.append("\n".join(lines))
    result: str = "\n\n".join(parts)
    return result


def _read_pdf_pypdf(path: Path) -> str:
    from pypdf import PdfReader  # pylint: disable=import-outside-toplevel

    reader = PdfReader(str(path))
    result: str = "\n\n".join(p.extract_text() or "" for p in reader.pages)
    return result


def read_docx(path: Path) -> str:
    from docx import Document  # pylint: disable=import-outside-toplevel

    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


_READERS: Dict[str, Callable[[Path], str]] = {
    ".txt": read_txt,
    ".pdf": read_pdf,
    ".docx": read_docx,
}


def read_document(path: Path) -> str:
    ext = path.suffix.lower()
    reader = _READERS.get(ext)
    if reader is None:
        raise ValueError(f"Unsupported extension: {ext}")
    return normalize_text(reader(path))


def collect_files(paths: List[str]) -> List[Path]:
    files: List[Path] = []
    for item in paths:
        path = Path(item).expanduser().resolve()
        if not path.exists():
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
        elif path.is_dir():
            for fp in path.rglob("*"):
                if fp.is_file() and fp.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(fp.resolve())
    return sorted(set(files))
