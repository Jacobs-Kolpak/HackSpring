from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.services.media import podcast as podcast_service
from backend.services.media.podcast import (
    AVAILABLE_SPEAKERS,
    PodcastConfig,
    normalize_pace,
    normalize_tone,
)
from backend.utils.document_reader import SUPPORTED_EXTENSIONS, read_document

router = APIRouter(prefix="/api/jacobs/podcast", tags=["Podcast"])

_AUDIO_DIR = Path("data/audio")


class DialogueResponse(BaseModel):
    dialogue: str
    source: str
    model: str
    has_audio: bool
    audio_url: Optional[str] = None
    meta: Dict[str, Any] = {}


class SpeakersResponse(BaseModel):
    speakers: List[str]


def _make_config(
    tone: str,
    pace: str,
    speaker_1: str = "baya",
    speaker_2: str = "xenia",
) -> PodcastConfig:
    return PodcastConfig(
        tone=normalize_tone(tone),
        pace=normalize_pace(pace),
        silero_speaker_1=speaker_1,
        silero_speaker_2=speaker_2,
    )


def _try_audio(
    dialogue: str,
    pace: str,
    speaker_1: str = "baya",
    speaker_2: str = "xenia",
) -> tuple[bool, Optional[str]]:
    _AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"podcast_{uuid.uuid4().hex[:12]}.wav"
    audio_path = _AUDIO_DIR / filename
    ok = podcast_service.save_audio(
        dialogue,
        audio_path,
        pace=normalize_pace(pace),
        silero_speaker_1=speaker_1,
        silero_speaker_2=speaker_2,
    )
    if ok:
        return True, f"/api/jacobs/podcast/audio/{filename}"
    return False, None


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
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат")

    try:
        cfg = _make_config(tone, pace, speaker_1, speaker_2)
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

    from backend.core.config import settings

    used_model = model or settings.llm.model
    dialogue = podcast_service.generate_dialogue(
        text, topic=topic, config=cfg, model=used_model
    )
    has_audio, audio_url = _try_audio(dialogue, pace, speaker_1, speaker_2)

    return DialogueResponse(
        dialogue=dialogue,
        source="file",
        model=used_model,
        has_audio=has_audio,
        audio_url=audio_url,
        meta={"filename": file.filename, "speakers": [speaker_1, speaker_2]},
    )


@router.get("/audio/{filename}")
async def get_audio(filename: str) -> FileResponse:
    path = _AUDIO_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Аудио не найдено")
    return FileResponse(path, media_type="audio/wav", filename=filename)
