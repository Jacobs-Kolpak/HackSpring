from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.services import podcast as podcast_service
from backend.services import rag as rag_service
from backend.services.podcast import PodcastConfig, normalize_pace, normalize_tone
from backend.utils.document_reader import SUPPORTED_EXTENSIONS, read_document

router = APIRouter(prefix="/api/podcast", tags=["Podcast"])

_AUDIO_DIR = Path("data/audio")


class TextPodcastRequest(BaseModel):
    text: str = Field(..., min_length=1)
    topic: str = Field(default="Без названия")
    tone: str = Field(default="scientific")
    pace: str = Field(default="normal")
    model: Optional[str] = None


class QueryPodcastRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection: Optional[str] = None
    top_k: int = Field(default=8, ge=1, le=50)
    min_score: float = Field(default=0.15, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    source_name: Optional[str] = None
    topic: str = Field(default="Без названия")
    tone: str = Field(default="scientific")
    pace: str = Field(default="normal")
    model: Optional[str] = None


class DialogueResponse(BaseModel):
    dialogue: str
    source: str
    model: str
    has_audio: bool
    audio_url: Optional[str] = None
    meta: Dict[str, Any] = {}


def _make_config(tone: str, pace: str) -> PodcastConfig:
    return PodcastConfig(
        tone=normalize_tone(tone),
        pace=normalize_pace(pace),
    )


def _try_audio(dialogue: str, pace: str) -> tuple[bool, Optional[str]]:
    _AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"podcast_{uuid.uuid4().hex[:12]}.wav"
    audio_path = _AUDIO_DIR / filename
    ok = podcast_service.save_audio(
        dialogue, audio_path, pace=normalize_pace(pace)
    )
    if ok:
        return True, f"/api/podcast/audio/{filename}"
    return False, None


@router.post("/text", response_model=DialogueResponse)
async def from_text(payload: TextPodcastRequest) -> DialogueResponse:
    try:
        cfg = _make_config(payload.tone, payload.pace)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    from backend.core.config import settings  # pylint: disable=import-outside-toplevel

    model = payload.model or settings.llm.model
    dialogue = podcast_service.generate_dialogue(
        payload.text, topic=payload.topic, config=cfg, model=model
    )
    has_audio, audio_url = _try_audio(dialogue, payload.pace)

    return DialogueResponse(
        dialogue=dialogue,
        source="text",
        model=model,
        has_audio=has_audio,
        audio_url=audio_url,
    )


@router.post("/file", response_model=DialogueResponse)
async def from_file(
    file: UploadFile = File(...),
    topic: str = Form("Без названия"),
    tone: str = Form("scientific"),
    pace: str = Form("normal"),
    model: Optional[str] = Form(None),
) -> DialogueResponse:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат")

    try:
        cfg = _make_config(tone, pace)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        temp_path = Path(tmp.name)
    try:
        text = read_document(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)

    if not text.strip():
        raise HTTPException(status_code=400, detail="В файле нет текста")

    from backend.core.config import settings  # pylint: disable=import-outside-toplevel

    used_model = model or settings.llm.model
    dialogue = podcast_service.generate_dialogue(
        text, topic=topic, config=cfg, model=used_model
    )
    has_audio, audio_url = _try_audio(dialogue, pace)

    return DialogueResponse(
        dialogue=dialogue,
        source="file",
        model=used_model,
        has_audio=has_audio,
        audio_url=audio_url,
        meta={"filename": file.filename},
    )


@router.post("/query", response_model=DialogueResponse)
async def from_query(payload: QueryPodcastRequest) -> DialogueResponse:
    try:
        cfg = _make_config(payload.tone, payload.pace)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    results: List[Dict[str, Any]] = rag_service.retrieve(
        query=payload.query,
        collection=payload.collection,
        top_k=payload.top_k,
        min_score=payload.min_score,
        dense_weight=payload.dense_weight,
        source_name=payload.source_name,
    )
    if not results:
        raise HTTPException(status_code=404, detail="Релевантных чанков не найдено")

    text = "\n\n".join(r["text"] for r in results if r.get("text"))

    from backend.core.config import settings  # pylint: disable=import-outside-toplevel

    used_model = payload.model or settings.llm.model
    dialogue = podcast_service.generate_dialogue(
        text, topic=payload.topic, config=cfg, model=used_model
    )
    has_audio, audio_url = _try_audio(dialogue, payload.pace)

    return DialogueResponse(
        dialogue=dialogue,
        source="query",
        model=used_model,
        has_audio=has_audio,
        audio_url=audio_url,
        meta={"query": payload.query, "used_chunks": len(results)},
    )


@router.get("/audio/{filename}")
async def get_audio(filename: str) -> FileResponse:
    path = _AUDIO_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Аудио не найдено")
    return FileResponse(path, media_type="audio/wav", filename=filename)
