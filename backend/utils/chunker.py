from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.utils.document_reader import read_document_segments


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
    source_name_overrides: Optional[Dict[str, str]] = None,
) -> List[Chunk]:
    result: List[Chunk] = []
    shared = dict(metadata or {})
    overrides = dict(source_name_overrides or {})

    for path in paths:
        segments = read_document_segments(path)
        if not segments:
            continue
        resolved = str(path.resolve())
        source_name = overrides.get(resolved) or overrides.get(str(path)) or path.name
        chunk_index = 0
        for segment in segments:
            if not segment.text:
                continue
            chunk_metadata = dict(shared)
            chunk_metadata.update(segment.metadata)
            for piece in chunk_text(segment.text, size, overlap):
                result.append(Chunk(
                    chunk_id=make_chunk_id(path, chunk_index, piece),
                    source_path=str(path),
                    source_name=source_name,
                    chunk_index=chunk_index,
                    text=piece,
                    metadata=dict(chunk_metadata),
                ))
                chunk_index += 1
    return result
