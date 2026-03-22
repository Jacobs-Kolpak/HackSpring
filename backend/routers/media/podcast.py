from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.core.config import settings

router = APIRouter(prefix="/api/jacobs/podcast", tags=["Podcast"])

_AUDIO_DIR = Path("data/audio")

AVAILABLE_SPEAKERS = ["aidar", "baya", "eugene", "kseniya", "xenia"]


class DialogueResponse(BaseModel):
    dialogue: str
    source: str
    model: str
    has_audio: bool
    audio_url: Optional[str] = None
    audio_error: Optional[str] = None
    meta: Dict[str, Any] = {}


class SpeakersResponse(BaseModel):
    speakers: List[str]


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


@router.get("/speakers", response_model=SpeakersResponse)
async def list_speakers() -> SpeakersResponse:
    return SpeakersResponse(speakers=AVAILABLE_SPEAKERS)


@router.post("/file", response_model=DialogueResponse)
async def from_file(
    file: UploadFile = File(...),
    topic: str = Form("Без названия"),
    tone: str = Form("scientific"),
    pace: str = Form("normal"),
    speaker_1: str = Form("baya"),
    speaker_2: str = Form("xenia"),
    model: Optional[str] = Form(None),
) -> DialogueResponse:
    if not settings.video.api_key:
        raise HTTPException(
            status_code=500,
            detail="VIDEO_API_KEY is not configured (shared with podcast service)",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Файл пустой")

    target_url = f"{_normalize_base_url(settings.video.base_url)}/v1/podcast-from-file"
    headers = {"X-API-Key": settings.video.api_key}
    files_payload = {
        "file": (
            file.filename or "document.txt",
            file_bytes,
            file.content_type or "application/octet-stream",
        )
    }
    data = {
        "topic": topic,
        "tone": tone,
        "pace": pace,
        "silero_speaker_1": speaker_1,
        "silero_speaker_2": speaker_2,
    }

    try:
        async with httpx.AsyncClient(timeout=settings.video.timeout_seconds) as client:
            upstream = await client.post(
                target_url,
                headers=headers,
                data=data,
                files=files_payload,
            )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Podcast service unavailable: {exc}",
        ) from exc

    content_type = upstream.headers.get("content-type", "")

    if upstream.status_code >= 400:
        if "application/json" in content_type:
            try:
                payload = upstream.json()
                detail = str(payload.get("detail") or payload)
            except ValueError:
                detail = upstream.text or "Podcast service error"
        else:
            detail = upstream.text or "Podcast service error"
        return DialogueResponse(
            dialogue="",
            source="file",
            model="external",
            has_audio=False,
            audio_error=f"Ошибка внешнего сервиса: {detail}",
            meta={"filename": file.filename, "speakers": [speaker_1, speaker_2]},
        )

    # Save WAV response to local file
    _AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"podcast_{uuid.uuid4().hex[:12]}.wav"
    audio_path = _AUDIO_DIR / filename
    audio_path.write_bytes(upstream.content)

    return DialogueResponse(
        dialogue="",
        source="file",
        model="external",
        has_audio=True,
        audio_url=f"/api/jacobs/podcast/audio/{filename}",
        meta={"filename": file.filename, "speakers": [speaker_1, speaker_2]},
    )


@router.get("/audio/{filename}")
async def get_audio(filename: str) -> FileResponse:
    path = _AUDIO_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Аудио не найдено")
    return FileResponse(path, media_type="audio/wav", filename=filename)
