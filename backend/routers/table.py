from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from backend.services import table as table_service
from backend.utils.document_reader import read_document

router = APIRouter(prefix="/api/jacobs/table", tags=["Table"])


class TextRequest(BaseModel):
    text: str
    hint: Optional[str] = None
    format: Optional[str] = "json"


def _read_upload(file: UploadFile, content: bytes) -> str:
    suffix = file.filename.rsplit(".", 1)[-1] if file.filename else "txt"
    with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    text = read_document(Path(tmp_path))
    if not text.strip():
        raise ValueError("Could not extract text from file")
    return text


def _format_response(table: table_service.ExtractedTable, fmt: str) -> Response:
    if fmt == "csv":
        return Response(
            content=table_service.table_to_csv_text(table).encode("utf-8"),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=table.csv"},
        )
    if fmt == "xlsx":
        return Response(
            content=table_service.table_to_xlsx_bytes(table),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=table.xlsx"},
        )
    return Response(
        content=__import__("json").dumps(
            table.to_dict(), ensure_ascii=False, indent=2,
        ).encode("utf-8"),
        media_type="application/json",
    )


@router.post("/text")
async def table_from_text(body: TextRequest):
    try:
        result = table_service.extract_table(body.text, hint=body.hint)
        return _format_response(result, body.format or "json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/file")
async def table_from_file(
    file: UploadFile = File(...),
    hint: Optional[str] = Form(None),
    format: Optional[str] = Form("json"),
):
    try:
        content = await file.read()
        text = _read_upload(file, content)
        result = table_service.extract_table(text, hint=hint)
        return _format_response(result, format or "json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
