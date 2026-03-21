from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from backend.core.config import settings
from backend.utils.document_reader import SUPPORTED_EXTENSIONS

router = APIRouter(prefix="/api/jacobs/video", tags=["Video"])


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


@router.post("/from-file")
async def generate_video_from_file(
    file: UploadFile = File(...),
    topic: str = Form("Без названия"),
    summary_template: str = Form("summary"),
    max_sentences: int = Form(8),
    speaker: str = Form("xenia"),
    width: int = Form(1280),
    height: int = Form(720),
    fps: int = Form(24),
) -> Response:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат")

    if not settings.video.api_key:
        raise HTTPException(status_code=500, detail="VIDEO_API_KEY is not configured")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Файл пустой")

    target_url = f"{_normalize_base_url(settings.video.base_url)}/v1/video-from-file"
    headers = {"X-API-Key": settings.video.api_key}
    files = {
        "file": (
            file.filename or "document.txt",
            file_bytes,
            file.content_type or "application/octet-stream",
        )
    }
    data = {
        "topic": topic,
        "summary_template": summary_template,
        "max_sentences": str(max_sentences),
        "speaker": speaker,
        "width": str(width),
        "height": str(height),
        "fps": str(fps),
    }

    try:
        async with httpx.AsyncClient(timeout=settings.video.timeout_seconds) as client:
            upstream = await client.post(
                target_url,
                headers=headers,
                data=data,
                files=files,
            )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Video service unavailable: {exc}",
        ) from exc

    content_type = upstream.headers.get("content-type", "")
    if upstream.status_code >= 400:
        detail: str
        if "application/json" in content_type:
            try:
                payload = upstream.json()
                detail = str(payload.get("detail") or payload)
            except ValueError:
                detail = upstream.text or "Video service error"
        else:
            detail = upstream.text or "Video service error"
        raise HTTPException(status_code=upstream.status_code, detail=detail)

    response_headers: dict[str, str] = {}
    content_disposition = upstream.headers.get("content-disposition")
    if content_disposition:
        response_headers["Content-Disposition"] = content_disposition

    return Response(
        content=upstream.content,
        media_type=content_type or "video/mp4",
        headers=response_headers,
    )
