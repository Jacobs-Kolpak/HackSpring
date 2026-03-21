from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from io import BytesIO, StringIO
from typing import Any, Dict, List, Optional

from backend.utils.llm import generate_text

SYSTEM_PROMPT = (
    "Ты ассистент по структурированию данных. "
    "Извлекай табличную информацию из неструктурированного текста. "
    "Возвращай только JSON и ничего кроме JSON."
)


@dataclass
class ExtractedTable:
    title: str
    columns: List[str]
    rows: List[List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "columns": self.columns,
            "rows": self.rows,
        }


def _build_prompt(text: str, hint: Optional[str] = None) -> str:
    hint_part = f"Контекст/подсказка: {hint}\n" if hint else ""
    return (
        "Преобразуй текст в таблицу. "
        "Выдели сущности и поля, сохрани смысл, не выдумывай факты.\n"
        "Требования:\n"
        "1) Ответ только в JSON-формате.\n"
        "2) Структура JSON строго:\n"
        "{\n"
        '  "title": "Название таблицы",\n'
        '  "columns": ["col1", "col2"],\n'
        '  "rows": [["v11", "v12"], ["v21", "v22"]]\n'
        "}\n"
        "3) rows[i] должен иметь ту же длину, что и columns.\n"
        "4) Если данных мало, верни 1-2 колонки и минимально нужные строки.\n"
        "5) Пустые значения заполняй пустой строкой.\n"
        f"{hint_part}"
        f"Текст:\n{text}"
    )


def _parse_json(raw: str) -> Dict[str, Any]:
    if not raw:
        raise RuntimeError("LLM returned empty response")
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise RuntimeError("LLM response is not valid JSON") from None
        return json.loads(raw[start : end + 1])


def _normalize_table(payload: Dict[str, Any]) -> ExtractedTable:
    columns_raw = payload.get("columns")
    rows_raw = payload.get("rows")

    if not isinstance(columns_raw, list) or not columns_raw:
        raise RuntimeError("JSON missing non-empty 'columns' array")
    if not isinstance(rows_raw, list):
        raise RuntimeError("JSON missing 'rows' array")

    columns = [
        str(v).strip() if str(v).strip() else f"column_{i + 1}"
        for i, v in enumerate(columns_raw)
    ]
    width = len(columns)

    rows: List[List[str]] = []
    for row in rows_raw:
        if not isinstance(row, list):
            continue
        values = [str(v).strip() for v in row]
        if len(values) < width:
            values.extend([""] * (width - len(values)))
        elif len(values) > width:
            values = values[:width]
        rows.append(values)

    title = str(payload.get("title") or "Extracted Table").strip()
    return ExtractedTable(title=title, columns=columns, rows=rows)


def extract_table(
    text: str,
    *,
    hint: Optional[str] = None,
) -> ExtractedTable:
    if not text or not text.strip():
        raise ValueError("Input text is empty")

    prompt = _build_prompt(text, hint)
    raw = generate_text(
        prompt,
        system=SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=4096,
    )
    payload = _parse_json(raw)
    return _normalize_table(payload)


def table_to_csv_text(table: ExtractedTable) -> str:
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(table.columns)
    writer.writerows(table.rows)
    return buffer.getvalue()


def table_to_xlsx_bytes(table: ExtractedTable) -> bytes:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = table.title[:31] if table.title else "Sheet1"
    ws.append(table.columns)
    for row in table.rows:
        ws.append(row)

    stream = BytesIO()
    wb.save(stream)
    return stream.getvalue()
