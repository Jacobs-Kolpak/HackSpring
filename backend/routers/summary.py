from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from backend.services import summary as summary_service
from backend.utils.document_reader import SUPPORTED_EXTENSIONS

router = APIRouter(prefix="/api/jacobs/summary", tags=["Summary"])


class SummaryResponse(BaseModel):
    summary: str
    source: str
    model: str
    meta: Dict[str, Any] = {}


@router.post("/file", response_model=SummaryResponse)
async def from_file(
    file: UploadFile = File(...),
    topic: str = Form("Без названия"),
    max_sentences: int = Form(10),
    model: Optional[str] = Form(None),
    template: Optional[str] = Form(None),
    system_prompt: Optional[str] = Form(None),
) -> SummaryResponse:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        temp_path = Path(tmp.name)
    try:
        from backend.core.config import settings

        used_model = model or settings.llm.model
        result = summary_service.summarize_file(
            temp_path,
            topic=topic,
            max_sentences=max_sentences,
            model=used_model,
            template=template,
            system_prompt=system_prompt,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    return SummaryResponse(
        summary=result,
        source="file",
        model=used_model,
        meta={"filename": file.filename, "custom_template": template is not None},
    )
