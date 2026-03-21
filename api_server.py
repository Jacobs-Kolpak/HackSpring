from __future__ import annotations

import os
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from common.document_loader import load_text_from_document
from common.remote_llm import RemoteLLMClient, RemoteLLMConfig
from summury.summarizer import FormalSummarizer, SummaryConfig
from video import TextToVideoConfig, TextToVideoGenerator

app = FastAPI(title="HackSpring API", version="1.0.0")
load_dotenv()


# Директория для результатов генерации видео.
JOBS_DIR = Path(os.getenv("HACKSPRING_JOBS_DIR", "/tmp/hackspring_jobs"))


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise HTTPException(status_code=500, detail=f"Не задана переменная окружения {name}")
    return value


def _get_allowed_api_keys() -> set[str]:
    raw = (
        os.getenv("SERVICE_API_KEYS", "")
        or os.getenv("CLIENT_API_KEYS", "")
        or os.getenv("API_KEYS", "")
    )
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    if not keys:
        fallback = os.getenv("SERVICE_API_KEY", "").strip()
        if fallback:
            keys.add(fallback)
    return keys


def _require_client_api_key(x_api_key: Optional[str]) -> None:
    keys = _get_allowed_api_keys()
    if not keys:
        raise HTTPException(
            status_code=500,
            detail=(
                "Не настроены клиентские API-ключи. "
                "Укажите SERVICE_API_KEYS='key1,key2' или SERVICE_API_KEY='key1'."
            ),
        )
    if not x_api_key or x_api_key.strip() not in keys:
        raise HTTPException(status_code=401, detail="Неверный API-ключ")


def _normalize_summary_template(value: str) -> str:
    template = (value or "").strip().lower()
    mapping = {
        "executive": "executive",
        "исполнительное резюме": "executive",
        "detailed": "detailed",
        "детализированный": "detailed",
        "summary": "summary",
        "саммари": "summary",
    }
    if template not in mapping:
        allowed = "executive|detailed|summary или исполнительное резюме|детализированный|саммари"
        raise HTTPException(status_code=400, detail=f"Некорректный summary_template. Используйте {allowed}.")
    return mapping[template]


def _build_remote_client() -> RemoteLLMClient:
    host = _required_env("REMOTE_LLM_HOST")
    port_raw = _required_env("REMOTE_LLM_PORT")
    api_key = _required_env("HACKAI_API_KEY")
    model = os.getenv("REMOTE_LLM_MODEL", "").strip()
    endpoint_path = os.getenv("REMOTE_LLM_PATH", "").strip()

    try:
        port = int(port_raw)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="REMOTE_LLM_PORT должен быть целым числом") from exc

    base_url = f"{host.rstrip('/')}:{port}"
    return RemoteLLMClient(
        RemoteLLMConfig(
            base_url=base_url,
            api_key=api_key,
            model=model,
            endpoint_path=endpoint_path,
        )
    )


def _extract_input_text(text: Optional[str], file: Optional[UploadFile]) -> str:
    if text and text.strip():
        return text.strip()

    if file is None:
        raise HTTPException(status_code=400, detail="Передайте text или file")

    suffix = Path(file.filename or "input.txt").suffix or ".txt"
    try:
        raw = file.file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать файл: {exc}") from exc

    if not raw:
        raise HTTPException(status_code=400, detail="Загруженный файл пуст")

    with NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp.flush()
        try:
            parsed = load_text_from_document(Path(tmp.name))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Ошибка чтения документа: {exc}") from exc

    if not parsed.strip():
        raise HTTPException(status_code=400, detail="В документе не найден текст")
    return parsed.strip()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/summary")
def generate_summary(
    text: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
    topic: str = Form(default="Без названия"),
    summary_template: str = Form(default="summary"),
    max_sentences: int = Form(default=8),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict[str, str]:
    _require_client_api_key(x_api_key)

    source_text = _extract_input_text(text=text, file=file)
    template = _normalize_summary_template(summary_template)

    remote_client = _build_remote_client()
    summarizer = FormalSummarizer(remote_llm_client=remote_client)
    result = summarizer.summarize(
        text=source_text,
        topic=topic,
        config=SummaryConfig(max_sentences=max_sentences, template=template),
    )

    return {
        "topic": topic,
        "summary_template": template,
        "summary": result,
    }


@app.post("/v1/video-from-text")
def generate_video(
    text: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
    topic: str = Form(default="Без названия"),
    summary_template: str = Form(default="summary"),
    max_sentences: int = Form(default=8),
    speaker: str = Form(default="xenia"),
    width: int = Form(default=1280),
    height: int = Form(default=720),
    fps: int = Form(default=24),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    _require_client_api_key(x_api_key)

    source_text = _extract_input_text(text=text, file=file)
    template = _normalize_summary_template(summary_template)

    remote_client = _build_remote_client()
    generator = TextToVideoGenerator(remote_llm_client=remote_client)

    job_id = uuid.uuid4().hex
    out_dir = JOBS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "video.mp4"

    try:
        produced = generator.generate(
            source_text=source_text,
            topic=topic,
            output_path=output_path,
            config=TextToVideoConfig(
                summary_template=template,
                max_sentences=max_sentences,
                speaker=speaker,
                width=max(640, width),
                height=max(360, height),
                fps=max(12, fps),
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации видео: {exc}") from exc

    return FileResponse(
        path=produced,
        media_type="video/mp4",
        filename=f"video_{job_id}.mp4",
    )
