from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

from backend.core.config import settings
from backend.utils.llm import generate_text

_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_WORD_RE = re.compile(r"[a-zа-яё0-9]+", flags=re.IGNORECASE)
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?%?")
_LAT_RE = re.compile(r"[a-z]", flags=re.IGNORECASE)
_CYR_RE = re.compile(r"[а-яё]", flags=re.IGNORECASE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOPWORDS = {
    "это", "как", "что", "или", "для", "при", "без", "под", "над", "после", "перед",
    "про", "из", "до", "мы", "вы", "они", "она", "оно", "его", "ее", "их", "наш", "ваш",
    "который", "которая", "которые", "также", "чтобы", "если", "быть", "есть", "был", "была",
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "you", "your",
}


def _extract_json(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response does not contain JSON object")
    fragment = text[start: end + 1]
    try:
        return json.loads(fragment)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", fragment)
        cleaned = re.sub(r"'", '"', cleaned)
        cleaned = re.sub(r"\n", " ", cleaned)
        return json.loads(cleaned)


def _has_cyrillic(text: str) -> bool:
    return bool(_CYRILLIC_RE.search(text or ""))


def _short_text(text: str, limit: int) -> str:
    clean = " ".join(str(text or "").split())
    if not clean:
        return ""
    if len(clean) <= limit:
        return clean
    trimmed = clean[:limit].rsplit(" ", 1)[0].strip()
    return trimmed or clean[:limit].strip()


def _to_float(raw: Any, fallback: float = 0.0) -> float:
    if raw is None:
        return fallback
    if isinstance(raw, str):
        text = raw.strip().replace(",", ".")
        pct = text.endswith("%")
        if pct:
            text = text[:-1].strip()
        try:
            val = float(text)
        except ValueError:
            return fallback
        if pct:
            val /= 100.0
        return val
    try:
        return float(raw)
    except (TypeError, ValueError):
        return fallback


def _norm_point_id(prefix: str, idx: int) -> str:
    return f"{prefix}-{idx}"


def _parse_models(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    rows = [x.strip() for x in raw.replace(";", ",").split(",")]
    return [x for x in rows if x]


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _resolve_models(requested_model: Optional[str]) -> List[str]:
    cfg = settings.infographic
    preferred = _parse_models(requested_model)
    configured = _parse_models(cfg.models)
    base = [settings.llm.model]
    resolved = _dedupe_keep_order(preferred + configured + base)
    if not resolved:
        resolved = [settings.llm.model]
    limit = max(1, cfg.max_model_candidates)
    return resolved[:limit]


def _tokenize(text: str) -> List[str]:
    tokens = [t.lower() for t in _WORD_RE.findall(_clean_ocr_noise(text))]
    return [
        t for t in tokens
        if len(t) >= 3
        and t not in _STOPWORDS
        and _is_clean_token(t)
    ]


def _clean_ocr_noise(text: str) -> str:
    clean = str(text or "")
    clean = re.sub(r"[^\S\r\n]+", " ", clean)
    clean = re.sub(r"^.*\.{4,}\s*\d+\s*$", "", clean, flags=re.MULTILINE)
    clean = re.sub(r"\b[а-яёa-z]{1,3}[эьъ][а-яёa-z]{0,3}\b", " ", clean, flags=re.IGNORECASE)
    return clean


def _is_clean_token(token: str) -> bool:
    if len(token) < 3:
        return False
    has_lat = bool(_LAT_RE.search(token))
    has_cyr = bool(_CYR_RE.search(token))
    if has_lat and has_cyr:
        return False
    if re.search(r"[ьъэ]{2,}", token):
        return False
    if re.search(r"[a-z]{1,2}[ьъэ]|[ьъэ][a-z]{1,2}", token, flags=re.IGNORECASE):
        return False
    vowels_ru = set("аеёиоуыэюя")
    vowels_en = set("aeiouy")
    if len(token) <= 4 and not any(ch in vowels_ru or ch in vowels_en for ch in token):
        return False
    return True


def _extract_theme_phrases(text: str) -> List[str]:
    tokens = _tokenize(text)
    phrases: List[str] = []
    for i in range(len(tokens) - 1):
        left, right = tokens[i], tokens[i + 1]
        if len(left) < 4 or len(right) < 4:
            continue
        if not _is_clean_token(left) or not _is_clean_token(right):
            continue
        phrases.append(f"{left} {right}")
    if len(phrases) < 4:
        phrases.extend(tokens[:8])
    return phrases


def _extract_content_insights(results: List[Dict[str, Any]], limit: int = 5) -> List[str]:
    corpus_text = " ".join(_clean_ocr_noise(str(row.get("text") or "")) for row in results)
    token_freq: Counter[str] = Counter(_tokenize(corpus_text))
    top_terms = {tok for tok, _ in token_freq.most_common(25)}

    candidates: List[tuple[float, str]] = []
    seen: set[str] = set()

    for row in results:
        text = _clean_ocr_noise(str(row.get("text") or ""))
        for raw_sent in _SENT_SPLIT_RE.split(text):
            sent = _short_text(" ".join(raw_sent.split()), 170)
            if len(sent) < 45:
                continue
            if not _has_cyrillic(sent):
                continue
            low = sent.lower()
            if re.search(r"\.{4,}\s*\d+$", low):
                continue

            sent_tokens = [t for t in _tokenize(sent) if _is_clean_token(t)]
            if len(sent_tokens) < 4:
                continue

            thematic_hits = sum(1 for t in sent_tokens if t in top_terms)
            numeric_hits = len(_NUM_RE.findall(sent))
            diversity = len(set(sent_tokens))
            score = thematic_hits * 1.2 + numeric_hits * 1.1 + diversity * 0.1

            norm = low.strip(" .")
            if norm in seen:
                continue
            seen.add(norm)
            candidates.append((score, sent))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [txt for _, txt in candidates[:limit]]


def _pad_points(
    data: List[Dict[str, Any]],
    *,
    prefix: str,
    min_len: int,
    name_prefix: str,
) -> List[Dict[str, Any]]:
    out = list(data)
    idx = len(out) + 1
    while len(out) < min_len:
        out.append(
            {
                "id": _norm_point_id(prefix, idx),
                "name": f"{name_prefix} {idx}",
                "value": 0.0,
            }
        )
        idx += 1
    return out


def _build_facts(results: List[Dict[str, Any]], max_points: int) -> Dict[str, Any]:
    scores = [max(0.0, _to_float(r.get("score"), 0.0)) for r in results]
    score_total = sum(scores) or 1.0

    src_acc: Dict[str, Dict[str, float]] = defaultdict(lambda: {"sum": 0.0, "cnt": 0.0})
    src_chunk_count: Counter[str] = Counter()
    theme_counter: Counter[str] = Counter()
    line_points: List[Dict[str, Any]] = []
    numbers_count = 0
    lengths: List[int] = []

    for idx, row in enumerate(results, 1):
        src = _short_text(str(row.get("source_name") or f"Источник {idx}"), 64)
        s = max(0.0, _to_float(row.get("score"), 0.0))
        src_acc[src]["sum"] += s
        src_acc[src]["cnt"] += 1
        src_chunk_count[src] += 1

        text = str(row.get("text") or "")
        lengths.append(len(text))
        numbers_count += len(_NUM_RE.findall(text))
        for phrase in _extract_theme_phrases(text):
            theme_counter[phrase] += max(0.1, s)

        line_points.append(
            {
                "id": _norm_point_id("line", idx),
                "name": f"Чанк {idx}",
                "value": round(s * 100.0, 2),
            }
        )

    bar_raw = theme_counter.most_common(max(8, max_points))
    bar_total = sum(weight for _, weight in bar_raw[:max_points]) or 1.0
    bar_data: List[Dict[str, Any]] = []
    for idx, (name, weight) in enumerate(bar_raw[:max_points], 1):
        share = (weight / bar_total) * 100.0
        bar_data.append(
            {
                "id": _norm_point_id("bar", idx),
                "name": _short_text(name, 48),
                "value": round(share, 2),
            }
        )
    bar_data = _pad_points(bar_data, prefix="bar", min_len=4, name_prefix="Тема")

    pie_data: List[Dict[str, Any]] = []
    src_rank = src_chunk_count.most_common(max(6, max_points))
    src_total = sum(cnt for _, cnt in src_rank[:4]) or 1
    for idx, (source, cnt) in enumerate(src_rank[:4], 1):
        pie_data.append(
            {
                "id": _norm_point_id("pie", idx),
                "name": _short_text(source, 32),
                "value": round((cnt / src_total) * 100.0, 2),
            }
        )
    pie_data = _pad_points(pie_data, prefix="pie", min_len=3, name_prefix="Источник")
    line_points = _pad_points(line_points[:max_points], prefix="line", min_len=4, name_prefix="Чанк")

    unique_sources = len(src_acc)
    avg_score_pct = round((mean(scores) if scores else 0.0) * 100.0, 2)

    metrics = [
        {
            "id": "metric-1",
            "name": "Показатель 1",
            "value": len(results),
            "unit": "шт",
            "description": "Количество отобранных фрагментов контекста.",
        },
        {
            "id": "metric-2",
            "name": "Показатель 2",
            "value": unique_sources,
            "unit": "шт",
            "description": "Число источников в выборке.",
        },
        {
            "id": "metric-3",
            "name": "Показатель 3",
            "value": avg_score_pct,
            "unit": "%",
            "description": "Средний интегральный score по retrieval.",
        },
        {
            "id": "metric-4",
            "name": "Показатель 4",
            "value": numbers_count,
            "unit": "шт",
            "description": "Количество обнаруженных числовых упоминаний.",
        },
    ]

    insights = _extract_content_insights(results, limit=5)
    if len(insights) < 3:
        if bar_data:
            insights.append(
                f"Ключевая тема контекста: {bar_data[0]['name']} ({bar_data[0]['value']:.1f}% тематического веса)."
            )
        if len(bar_data) > 2:
            top_themes = ", ".join(x["name"] for x in bar_data[:3])
            insights.append(f"В документе доминируют темы: {top_themes}.")
        insights.append(
            f"Средний релевантный score составляет {avg_score_pct:.1f}%, анализ охватывает {len(results)} чанков."
        )

    return {
        "title": "Инфографика",
        "subtitle": "Агрегированный контекст источников",
        "metrics": metrics,
        "bar_chart": {
            "title": "График 1",
            "x_axis_label": "Категория",
            "y_axis_label": "Значение",
            "legend": "Распределение по категориям",
            "data": bar_data,
        },
        "pie_chart": {
            "title": "График 2",
            "legend": "Доли категорий",
            "data": pie_data,
        },
        "line_chart": {
            "title": "График 3",
            "x_axis_label": "Шаг",
            "y_axis_label": "Значение",
            "legend": "Динамика показателя",
            "data": line_points,
        },
        "key_insights": insights[:5],
    }


def _build_llm_prompt(query: str, results: List[Dict[str, Any]], facts: Dict[str, Any]) -> str:
    compact_results = [
        {
            "rank": i + 1,
            "source_name": row.get("source_name", "unknown"),
            "chunk_index": row.get("chunk_index", -1),
            "score": round(_to_float(row.get("score"), 0.0), 4),
            "text": _short_text(str(row.get("text") or ""), 550),
        }
        for i, row in enumerate(results)
    ]
    return (
        "Построй содержательную инфографику по контексту источников. Нужен не шаблон, а осмысленная аналитика.\n\n"
        "Что важно:\n"
        "1) Анализируй текст чанков по сути, выделяй реальные темы, факты и числовые зависимости.\n"
        "2) Сформируй связный сторителлинг: заголовок, подзаголовок, графики и инсайты должны логически совпадать.\n"
        "3) Все подписи и выводы на русском языке.\n"
        "4) Используй только данные из контекста; если чисел мало, допускается аккуратная агрегация на основе контекста.\n"
        "5) В key_insights пиши содержательные тезисы по сути документа (как мини-выжимка), а не тех. комментарии про чанки.\n"
        "6) Верни только JSON без markdown.\n\n"
        "Верни JSON с общей структурой (без жестких количественных ограничений):\n"
        "{\n"
        "  \"title\": \"...\",\n"
        "  \"subtitle\": \"...\",\n"
        "  \"metrics\": [\n"
        "    {\"id\":\"...\",\"name\":\"...\",\"value\":123,\"unit\":\"...\",\"description\":\"...\"}\n"
        "  ],\n"
        "  \"bar_chart\": {\n"
        "    \"title\":\"...\",\n"
        "    \"x_axis_label\":\"...\",\n"
        "    \"y_axis_label\":\"...\",\n"
        "    \"legend\":\"...\",\n"
        "    \"data\":[{\"id\":\"...\",\"name\":\"...\",\"value\":12.3}]\n"
        "  },\n"
        "  \"pie_chart\": {\n"
        "    \"title\":\"...\",\n"
        "    \"legend\":\"...\",\n"
        "    \"data\":[{\"id\":\"...\",\"name\":\"...\",\"value\":25.0}]\n"
        "  },\n"
        "  \"line_chart\": {\n"
        "    \"title\":\"...\",\n"
        "    \"x_axis_label\":\"...\",\n"
        "    \"y_axis_label\":\"...\",\n"
        "    \"legend\":\"...\",\n"
        "    \"data\":[{\"id\":\"...\",\"name\":\"...\",\"value\":7.5}]\n"
        "  },\n"
        "  \"key_insights\": [\"Содержательный вывод 1 по сути документа\", \"Содержательный вывод 2\"]\n"
        "}\n\n"
        f"Фокус запроса пользователя:\n{query}\n\n"
        f"facts_seed (базовые агрегаты, можно использовать, но не копировать механически):\n"
        f"{json.dumps(facts, ensure_ascii=False)}\n\n"
        f"retrieval_context:\n{json.dumps(compact_results, ensure_ascii=False)}"
    )


_SYSTEM_PROMPT = (
    "Ты старший аналитик BI-дашбордов. Возвращай строго валидный JSON по заданной схеме, "
    "без markdown, без комментариев, без лишнего текста. Все поля на русском."
)


def _call_model_for_plan(model: str, prompt: str) -> Dict[str, Any]:
    raw_text = generate_text(
        prompt,
        system=_SYSTEM_PROMPT,
        model=model,
        temperature=0.1,
        max_tokens=4096,
    )
    try:
        return _extract_json(raw_text)
    except ValueError:
        repair_text = generate_text(
            raw_text,
            system="Исправь ответ до строго валидного JSON. Верни только JSON.",
            model=model,
            temperature=0,
        )
        return _extract_json(repair_text)


def _normalize_points(
    raw: Any,
    *,
    prefix: str,
    min_len: int,
    max_len: int,
    fallback: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for idx, item in enumerate(raw, 1):
            if not isinstance(item, dict):
                continue
            name = _short_text(str(item.get("name") or ""), 48)
            value = _to_float(item.get("value"), -1.0)
            if not name or value < 0:
                continue
            rows.append(
                {
                    "id": str(item.get("id") or _norm_point_id(prefix, idx)),
                    "name": name,
                    "value": round(value, 2),
                }
            )
            if len(rows) >= max_len:
                break

    if len(rows) < min_len:
        rows = fallback[:max_len]

    return rows


def _normalize_metrics(raw: Any, fallback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for idx, item in enumerate(raw, 1):
            if not isinstance(item, dict):
                continue
            name = _short_text(str(item.get("name") or ""), 42)
            unit = _short_text(str(item.get("unit") or ""), 12)
            desc = _short_text(str(item.get("description") or ""), 140)
            value = item.get("value")
            if isinstance(value, str) and value.strip():
                value = _short_text(value, 24)
            elif isinstance(value, (int, float)):
                value = round(float(value), 2)
            else:
                continue
            if not name:
                continue
            out.append(
                {
                    "id": str(item.get("id") or f"metric-{idx}"),
                    "name": name,
                    "value": value,
                    "unit": unit,
                    "description": desc,
                }
            )
            if len(out) >= 8:
                break

    if len(out) < 2:
        return fallback
    return out


def _normalize_plan(raw_plan: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    title = _short_text(str(raw_plan.get("title") or ""), 90)
    subtitle = _short_text(str(raw_plan.get("subtitle") or ""), 120)
    if not title or not _has_cyrillic(title):
        title = fallback["title"]
    if not subtitle or not _has_cyrillic(subtitle):
        subtitle = fallback["subtitle"]

    fb_bar = fallback["bar_chart"]
    fb_pie = fallback["pie_chart"]
    fb_line = fallback["line_chart"]

    bar = raw_plan.get("bar_chart") if isinstance(raw_plan.get("bar_chart"), dict) else {}
    pie = raw_plan.get("pie_chart") if isinstance(raw_plan.get("pie_chart"), dict) else {}
    line = raw_plan.get("line_chart") if isinstance(raw_plan.get("line_chart"), dict) else {}

    bar_data = _normalize_points(
        bar.get("data"), prefix="bar", min_len=2, max_len=12, fallback=fb_bar["data"],
    )
    pie_data = _normalize_points(
        pie.get("data"), prefix="pie", min_len=2, max_len=10, fallback=fb_pie["data"],
    )
    line_data = _normalize_points(
        line.get("data"), prefix="line", min_len=2, max_len=20, fallback=fb_line["data"],
    )

    insights: List[str] = []
    raw_insights = raw_plan.get("key_insights")
    if isinstance(raw_insights, list):
        for item in raw_insights:
            txt = _short_text(str(item or ""), 140)
            if txt and _has_cyrillic(txt):
                insights.append(txt)
            if len(insights) >= 10:
                break
    if len(insights) < 2:
        insights = fallback["key_insights"][:5]

    return {
        "title": title,
        "subtitle": subtitle,
        "metrics": _normalize_metrics(raw_plan.get("metrics"), fallback["metrics"]),
        "bar_chart": {
            "title": _short_text(str(bar.get("title") or fb_bar["title"]), 72),
            "x_axis_label": _short_text(str(bar.get("x_axis_label") or fb_bar["x_axis_label"]), 42),
            "y_axis_label": _short_text(str(bar.get("y_axis_label") or fb_bar["y_axis_label"]), 42),
            "legend": _short_text(str(bar.get("legend") or fb_bar["legend"]), 140),
            "data": bar_data,
        },
        "pie_chart": {
            "title": _short_text(str(pie.get("title") or fb_pie["title"]), 72),
            "legend": _short_text(str(pie.get("legend") or fb_pie["legend"]), 140),
            "data": pie_data,
        },
        "line_chart": {
            "title": _short_text(str(line.get("title") or fb_line["title"]), 72),
            "x_axis_label": _short_text(str(line.get("x_axis_label") or fb_line["x_axis_label"]), 42),
            "y_axis_label": _short_text(str(line.get("y_axis_label") or fb_line["y_axis_label"]), 42),
            "legend": _short_text(str(line.get("legend") or fb_line["legend"]), 140),
            "data": line_data,
        },
        "key_insights": insights,
    }


def _merge_plans(base: Dict[str, Any], variants: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(base)
    for plan in variants:
        if len(merged.get("key_insights", [])) < 6:
            for item in plan.get("key_insights", []):
                if item not in merged["key_insights"]:
                    merged["key_insights"].append(item)
                    if len(merged["key_insights"]) >= 6:
                        break
    return merged


def generate_infographic_payload(
    query: str,
    results: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    max_topics: Optional[int] = None,
) -> Dict[str, Any]:
    if not results:
        raise ValueError("No retrieval results")

    cfg = settings.infographic
    requested = max_topics if max_topics is not None else cfg.default_max_topics
    max_points = max(4, min(requested, cfg.max_topics_limit))

    fallback = _build_facts(results, max_points=max_points)
    prompt = _build_llm_prompt(query, results, fallback)

    models = _resolve_models(model)
    normalized_candidates: List[Dict[str, Any]] = []
    errors: List[str] = []

    for mdl in models:
        try:
            raw = _call_model_for_plan(mdl, prompt)
            normalized = _normalize_plan(raw, fallback)
            normalized_candidates.append(normalized)
        except Exception as exc:
            errors.append(f"{mdl}: {exc}")

    final_plan = fallback
    if normalized_candidates:
        final_plan = _merge_plans(normalized_candidates[0], normalized_candidates[1:])

    return {
        "query": query,
        "title": final_plan["title"],
        "subtitle": final_plan["subtitle"],
        "metrics": final_plan["metrics"],
        "bar_chart": final_plan["bar_chart"],
        "pie_chart": final_plan["pie_chart"],
        "line_chart": final_plan["line_chart"],
        "key_insights": final_plan["key_insights"],
        "used_chunks": len(results),
        "models_used": models,
        "failed_models": errors,
        "download": {
            "mime_type": "image/png",
            "strategy": "frontend",
            "hint": "PNG формируется на фронтенде из этого JSON и скачивается в браузере.",
        },
    }
