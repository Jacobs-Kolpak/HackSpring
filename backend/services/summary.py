from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from backend.utils.document_reader import read_document
from backend.utils.llm import generate_text


DEFAULT_TEMPLATE = (
    "Сделай суммаризацию этого текста на русском языке.\n"
    "Тема: {topic}\n"
    "Желаемая длина: {max_sentences} предложений.\n"
    "Текст:\n{text}\n"
    "Суммаризация:"
)

DEFAULT_SYSTEM = (
    "Ты — ассистент для суммаризации текстов. "
    "Пиши кратко и по делу на русском языке."
)


def summarize(
    text: str,
    *,
    topic: str = "Без названия",
    max_sentences: int = 10,
    model: Optional[str] = None,
    template: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return "Недостаточно данных для суммаризации."

    chunk = cleaned[:9000]
    bounded = max(3, min(max_sentences, 100))

    tmpl = template or DEFAULT_TEMPLATE
    prompt = tmpl.format(
        topic=topic,
        max_sentences=bounded,
        text=chunk,
    )

    result = generate_text(
        prompt,
        system=system_prompt or DEFAULT_SYSTEM,
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
    template: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    text = read_document(path)
    if not text.strip():
        raise ValueError("В файле не найден текст.")
    return summarize(
        text,
        topic=topic,
        max_sentences=max_sentences,
        model=model,
        template=template,
        system_prompt=system_prompt,
    )
