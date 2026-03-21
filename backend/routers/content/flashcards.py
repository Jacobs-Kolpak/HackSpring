from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from backend.services.content import flashcards as flash_service
from backend.utils.document_reader import SUPPORTED_EXTENSIONS

router = APIRouter(prefix="/api/jacobs/flashcards", tags=["Flashcards"])


class FlashResponse(BaseModel):
    flashcards: List[Dict[str, str]]
    tests: List[Dict[str, Any]]
    source: str
    model: str
    meta: Dict[str, Any] = {}


@router.post("/file", response_model=FlashResponse)
async def from_file(
    file: UploadFile = File(...),
    topic: str = Form("Без названия"),
    cards_count: int = Form(10),
    tests_count: int = Form(5),
    model: Optional[str] = Form(None),
) -> FlashResponse:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        temp_path = Path(tmp.name)
    try:
        from backend.core.config import settings

        used_model = model or settings.llm.model
        result = flash_service.generate_from_file(
            temp_path,
            topic=topic,
            cards_count=cards_count,
            tests_count=tests_count,
            model=used_model,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    return FlashResponse(
        flashcards=result["flashcards"],
        tests=result["tests"],
        source="file",
        model=used_model,
        meta={
            "filename": file.filename,
            "flashcards_count": len(result["flashcards"]),
            "tests_count": len(result["tests"]),
        },
    )
