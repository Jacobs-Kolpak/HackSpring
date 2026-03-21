from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from backend.utils.document_reader import read_document
from backend.utils.llm import generate_text


def summarize(
    text: str,
    *,
    topic: str = "Без названия",
    max_sentences: int = 10,
    model: Optional[str] = None,
) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return "Недостаточно данных для суммаризации."

    chunk = cleaned[:9000]
    prompt = (
        "Сделай суммаризацию этого текста на русском языке.\n"
        f"Тема: {topic}\n"
        f"Желаемая длина: {max(3, min(max_sentences, 100))} предложений.\n"
        f"Текст:\n{chunk}\n"
        "Суммаризация:"
    )
    result = generate_text(
        prompt,
        system="Ты — ассистент для суммаризации текстов. "
               "Пиши кратко и по делу на русском языке.",
        model=model,
        temperature=0.2,
        max_tokens=2000,
    )
    result = re.sub(
        r"(?is)^.*?суммаризац(?:ия|ию)\s*:\s*", "", result
    ).strip()
    if not result:
        raise RuntimeError("LLM вернул пустой результат.")
    return result


def summarize_file(
    path: Path,
    *,
    topic: str = "Без названия",
    max_sentences: int = 10,
    model: Optional[str] = None,
) -> str:
    text = read_document(path)
    if not text.strip():
        raise ValueError("В файле не найден текст.")
    return summarize(
        text, topic=topic, max_sentences=max_sentences, model=model
    )
