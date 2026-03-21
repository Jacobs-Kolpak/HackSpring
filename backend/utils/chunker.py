from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.utils.document_reader import read_document


@dataclass
class Chunk:
    chunk_id: str
    source_path: str
    source_name: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def split_sentences(text: str) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    sentences: List[str] = []
    for para in paragraphs:
        for part in re.split(r"(?<=[.!?])\s+", para):
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def _flush(chunks: List[str], current: str) -> str:
    if current:
        chunks.append(current.strip())
    return ""


def _chunk_long_sentence(
    chunks: List[str], sent: str, size: int, overlap: int
) -> str:
    pos = 0
    while pos < len(sent):
        piece = sent[pos:pos + size].strip()
        if piece:
            chunks.append(piece)
        pos += max(1, size - overlap)
    return ""


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= size:
        raise ValueError("overlap must be < size")

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current = ""

    for sent in sentences:
        if len(sent) > size:
            current = _flush(chunks, current)
            current = _chunk_long_sentence(chunks, sent, size, overlap)
            continue

        candidate = f"{current} {sent}".strip() if current else sent
        if len(candidate) <= size:
            current = candidate
        elif current:
            chunks.append(current.strip())
            tail = current[-overlap:].strip()
            current = f"{tail} {sent}".strip() if tail else sent
        else:
            current = sent

    if current:
        chunks.append(current.strip())
    return chunks


def make_chunk_id(path: Path, index: int, text: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{path}:{index}:{text}"))


def build_chunks(
    paths: List[Path],
    size: int,
    overlap: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Chunk]:
    result: List[Chunk] = []
    shared = dict(metadata or {})

    for path in paths:
        text = read_document(path)
        if not text:
            continue
        for idx, piece in enumerate(chunk_text(text, size, overlap)):
            result.append(Chunk(
                chunk_id=make_chunk_id(path, idx, piece),
                source_path=str(path),
                source_name=path.name,
                chunk_index=idx,
                text=piece,
                metadata=shared,
            ))
    return result
