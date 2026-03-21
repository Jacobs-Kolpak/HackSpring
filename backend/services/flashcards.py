from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.utils.document_reader import read_document
from backend.utils.llm import generate_text


def build_study_context(
    chunks: List[Dict[str, Any]], max_chars: int = 18000
) -> str:
    blocks: List[str] = []
    used = 0
    for item in chunks:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        source_name = str(item.get("source_name", "unknown"))
        chunk_index = item.get("chunk_index", -1)
        block = f"[Source: {source_name} | chunk={chunk_index}]\n{text}"
        extra = len(block) + (2 if blocks else 0)
        if not blocks and extra > max_chars:
            reserve = len(f"[Source: {source_name} | chunk={chunk_index}]\n")
            allowed_text = max(0, max_chars - reserve - 3)
            truncated = text[:allowed_text] + "..." if allowed_text > 0 else text[:max_chars]
            return f"[Source: {source_name} | chunk={chunk_index}]\n{truncated}"
        if used + extra > max_chars:
            break
        blocks.append(block)
        used += extra
    return "\n\n".join(blocks)


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    if not text:
        raise ValueError("Empty model response")

    fence_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL
    )
    if fence_match:
        text = fence_match.group(1).strip()

    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start: end + 1]

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Model response must be a JSON object")
    return data


def _sanitize_cards(raw: List[Any]) -> List[Dict[str, str]]:
    cards: List[Dict[str, str]] = []
    for card in raw:
        if not isinstance(card, dict):
            continue
        q = str(card.get("question", "")).strip()
        a = str(card.get("answer", "")).strip()
        if q and a:
            cards.append({"question": q, "answer": a})
    return cards


def _sanitize_tests(raw: List[Any]) -> List[Dict[str, Any]]:  # pylint: disable=too-many-branches
    tests: List[Dict[str, Any]] = []
    for test in raw:
        if not isinstance(test, dict):
            continue
        question = str(test.get("question", "")).strip()
        options = test.get("options", [])
        if not question or not isinstance(options, list):
            continue
        normalized = [str(o).strip() for o in options if str(o).strip()]
        if len(normalized) < 2:
            continue
        correct_index = test.get("correct_index")
        if (
            not isinstance(correct_index, int)
            or correct_index < 0
            or correct_index >= len(normalized)
        ):
            correct_index = 0
        explanation = str(test.get("explanation", "")).strip()
        tests.append({
            "question": question,
            "options": normalized,
            "correct_index": correct_index,
            "explanation": explanation,
        })
    return tests


def _sanitize_pack(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "flashcards": _sanitize_cards(data.get("flashcards") or []),
        "tests": _sanitize_tests(data.get("tests") or []),
    }


def generate_flash_materials(
    context: str,
    *,
    topic: str = "Без названия",
    cards_count: int = 10,
    tests_count: int = 5,
    language: str = "ru",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    system_prompt = (
        "Ты ассистент-методист. Твоя задача: сгенерировать карточки и тесты "
        "строго на основе контекста. Никаких фактов вне контекста. Верни только JSON."
    )
    user_prompt = (
        f"Тема: {topic}\n"
        f"Язык ответа: {language}\n"
        f"Сгенерируй:\n"
        f"- flashcards: {cards_count} карточек question/answer\n"
        f"- tests: {tests_count} тестовых вопросов\n"
        "Для каждого теста: question, options (ровно 4), "
        "correct_index (0..3), explanation.\n"
        "Формат строго:\n"
        '{"flashcards":[{"question":"...","answer":"..."}],'
        '"tests":[{"question":"...","options":["A","B","C","D"],'
        '"correct_index":1,"explanation":"..."}]}\n\n'
        f"Контекст:\n{context}"
    )
    raw = generate_text(
        user_prompt,
        system=system_prompt,
        model=model,
        temperature=0.2,
        max_tokens=4096,
    )
    parsed = _extract_json_object(raw)
    return _sanitize_pack(parsed)


def generate_from_text(
    text: str,
    *,
    topic: str = "Без названия",
    cards_count: int = 10,
    tests_count: int = 5,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    context = text[:18000]
    return generate_flash_materials(
        context,
        topic=topic,
        cards_count=cards_count,
        tests_count=tests_count,
        model=model,
    )


def generate_from_file(
    path: Path,
    *,
    topic: str = "Без названия",
    cards_count: int = 10,
    tests_count: int = 5,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    text = read_document(path)
    if not text.strip():
        raise ValueError("В файле не найден текст.")
    return generate_from_text(
        text,
        topic=topic,
        cards_count=cards_count,
        tests_count=tests_count,
        model=model,
    )
