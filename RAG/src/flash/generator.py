import json
import re
from typing import Any, Dict, List, Optional


def build_study_context(chunks: List[dict], max_chars: int = 18000) -> str:
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
            truncated = (text[:allowed_text] + "...") if allowed_text > 0 else text[:max_chars]
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

    # Support markdown fenced JSON output.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Fallback: find first JSON object in text.
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Model response must be a JSON object")
    return data


def _sanitize_pack(data: Dict[str, Any]) -> Dict[str, Any]:
    cards: List[Dict[str, str]] = []
    tests: List[Dict[str, Any]] = []

    for card in data.get("flashcards", []) or []:
        if not isinstance(card, dict):
            continue
        q = str(card.get("question", "")).strip()
        a = str(card.get("answer", "")).strip()
        if q and a:
            cards.append({"question": q, "answer": a})

    for test in data.get("tests", []) or []:
        if not isinstance(test, dict):
            continue
        question = str(test.get("question", "")).strip()
        options = test.get("options", [])
        if not question or not isinstance(options, list):
            continue

        normalized_options = [str(opt).strip() for opt in options if str(opt).strip()]
        if len(normalized_options) < 2:
            continue

        correct_index = test.get("correct_index")
        if not isinstance(correct_index, int) or correct_index < 0 or correct_index >= len(normalized_options):
            correct_index = 0

        explanation = str(test.get("explanation", "")).strip()
        tests.append(
            {
                "question": question,
                "options": normalized_options,
                "correct_index": correct_index,
                "explanation": explanation,
            }
        )

    return {"flashcards": cards, "tests": tests}


def generate_flash_materials(
    *,
    topic: str,
    context: str,
    model: str,
    llm_url: Optional[str],
    llm_api_key: str,
    cards_count: int,
    tests_count: int,
    language: str = "ru",
) -> Dict[str, Any]:
    from openai import OpenAI

    base_url = None
    if llm_url:
        base = llm_url.rstrip("/")
        base_url = base if base.endswith("/v1") else f"{base}/v1"

    if llm_url:
        client = OpenAI(api_key=llm_api_key, base_url=base_url, timeout=180.0, max_retries=2)
    else:
        client = OpenAI(api_key=llm_api_key, timeout=180.0, max_retries=2)

    system_prompt = (
        "Ты ассистент-методист. Твоя задача: сгенерировать карточки и тесты строго на основе контекста. "
        "Никаких фактов вне контекста. Верни только JSON."
    )
    user_prompt = (
        f"Тема: {topic}\n"
        f"Язык ответа: {language}\n"
        f"Сгенерируй:\n"
        f"- flashcards: {cards_count} карточек question/answer\n"
        f"- tests: {tests_count} тестовых вопросов\n"
        "Для каждого теста: question, options (ровно 4), correct_index (0..3), explanation.\n"
        "Формат строго:\n"
        '{"flashcards":[{"question":"...","answer":"..."}],'
        '"tests":[{"question":"...","options":["A","B","C","D"],"correct_index":1,"explanation":"..."}]}\n\n'
        f"Контекст:\n{context}"
    )

    resp = client.responses.create(
        model=model,
        temperature=0.2,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    parsed = _extract_json_object(resp.output_text)
    return _sanitize_pack(parsed)
