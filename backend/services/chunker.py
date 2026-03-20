"""
Сервис чанкинга текста.

Sentence-aware разбиение с настраиваемым overlap.
Параметры берутся из settings.rag (chunk_size, chunk_overlap).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.services.document_reader import read_document


@dataclass
class Chunk:
    """Единица текста после разбиения документа."""

    chunk_id: str
    source_path: str
    source_name: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Разбиение на предложения ────────────────────────────────


def split_sentences(text: str) -> List[str]:
    """Разбивает текст на предложения по абзацам и пунктуации."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    sentences: List[str] = []
    for para in paragraphs:
        parts = re.split(r"(?<=[.!?])\s+", para)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


# ── Sentence-aware чанкинг ──────────────────────────────────


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """
    Разбивает текст на чанки с учётом границ предложений.

    Args:
        text: исходный текст.
        chunk_size: максимальный размер чанка в символах.
        chunk_overlap: размер перекрытия между чанками.

    Returns:
        Список текстовых чанков.

    Raises:
        ValueError: при некорректных параметрах.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        # Предложение длиннее чанка — режем принудительно
        if len(sentence) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(sentence):
                piece = sentence[start : start + chunk_size].strip()
                if piece:
                    chunks.append(piece)
                start += max(1, chunk_size - chunk_overlap)
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        # Текущий чанк полон — сохраняем и начинаем новый с overlap
        if current:
            chunks.append(current.strip())
            overlap_tail = current[-chunk_overlap:].strip()
            current = (
                f"{overlap_tail} {sentence}".strip()
                if overlap_tail
                else sentence
            )
        else:
            current = sentence

    if current:
        chunks.append(current.strip())

    return chunks


# ── Генерация стабильных ID ─────────────────────────────────


def make_chunk_id(path: Path, chunk_index: int, text: str) -> str:
    """Генерирует детерминированный UUID5 для чанка."""
    stable_key = f"{path}:{chunk_index}:{text}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))


# ── Построение чанков из файлов ─────────────────────────────


def build_chunks(
    paths: List[Path],
    chunk_size: int,
    chunk_overlap: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Chunk]:
    """
    Читает документы и разбивает их на чанки.

    Args:
        paths: список путей к документам.
        chunk_size: размер чанка в символах.
        chunk_overlap: перекрытие между чанками.
        metadata: дополнительные метаданные для всех чанков.

    Returns:
        Список объектов Chunk.
    """
    result: List[Chunk] = []
    shared_metadata = dict(metadata or {})

    for path in paths:
        text = read_document(path)
        if not text:
            continue

        pieces = chunk_text(
            text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        for idx, piece in enumerate(pieces):
            result.append(
                Chunk(
                    chunk_id=make_chunk_id(path, idx, piece),
                    source_path=str(path),
                    source_name=path.name,
                    chunk_index=idx,
                    text=piece,
                    metadata=shared_metadata,
                )
            )

    return result
