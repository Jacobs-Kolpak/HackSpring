from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.core.config import settings
from backend.services import video as video_service
from backend.services.video import VideoConfig
from backend.utils.document_reader import SUPPORTED_EXTENSIONS

router = APIRouter(prefix="/api/jacobs/video", tags=["Video"])


class VideoMetaResponse(BaseModel):
    topic: str
    slides_count: int
    duration_sec: float
    resolution: str
    file_path: str


@router.post("/file", response_model=VideoMetaResponse)
async def video_from_file(
    file: UploadFile = File(...),
    topic: str = Form("Без названия"),
    max_sentences: int = Form(8),
    speaker: str = Form("baya"),
    width: int = Form(1280),
    height: int = Form(720),
    fps: int = Form(24),
    model: Optional[str] = Form(None),
) -> VideoMetaResponse:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат файла")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        temp_path = Path(tmp.name)

    try:
        config = VideoConfig(
            max_sentences=max(3, min(max_sentences, 20)),
            width=max(640, width),
            height=max(360, height),
            fps=max(12, fps),
            speaker=speaker,
        )
        file_path, metadata = video_service.generate_video_from_file(
            temp_path, topic=topic, config=config, model=model,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)

    return VideoMetaResponse(**metadata)


@router.get("/download")
async def download_video(file_path: str) -> FileResponse:
    path = Path(file_path).resolve()
    allowed_root = Path(settings.video.output_dir).resolve()
    if allowed_root not in path.parents:
        raise HTTPException(status_code=403, detail="Access to file is forbidden")
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".mp4":
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(
        path=str(path),
        media_type="video/mp4",
        filename=path.name,
    )
