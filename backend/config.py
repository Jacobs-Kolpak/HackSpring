"""
Единый конфиг-ридер проекта HackSpring.

Читает .env файл из корня проекта и предоставляет типизированные
настройки через pydantic-settings.  Каждый модуль (auth, rag, tts …)
получает свою секцию, но все живут в одном Settings-объекте.

Использование:
    from backend.config import settings
    print(settings.DATABASE_URL)
    print(settings.llm.model)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from functools import cached_property

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Корень проекта — на уровень выше backend/
BASE_DIR = Path(__file__).resolve().parent.parent


# ── Вложенные секции ────────────────────────────────────────


class LLMSettings(BaseSettings):
    """Настройки LLM-провайдера."""

    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")

    provider: str = "openai"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.7
    max_tokens: int = 4096


class RAGSettings(BaseSettings):
    """Настройки RAG-пайплайна."""

    model_config = SettingsConfigDict(env_prefix="RAG_", extra="ignore")

    # Chunking
    chunk_size: int = 900
    chunk_overlap: int = 180

    # Embeddings
    embeddings_model: str = "Qwen3-Embedding-0.6B"
    embedder_url: str = ""
    embedder_api_key: str = ""

    # Vector store
    vector_store_path: str = str(BASE_DIR / "data" / "vector_store")
    collection: str = "docs"

    # Retrieval
    top_k: int = 5
    fetch_k: int = 40
    min_score: float = 0.20
    dense_weight: float = 0.70

    # Rerank
    rerank_blend: float = 0.35
    rerank_url: str = ""
    rerank_api_key: str = ""
    rerank_model: str = ""


class TTSSettings(BaseSettings):
    """Настройки Text-to-Speech (audio_summary)."""

    model_config = SettingsConfigDict(env_prefix="TTS_", extra="ignore")

    engine: str = "silero"
    model: str = "v4_ru"
    speaker: str = "baya"
    sample_rate: int = 48000
    output_dir: str = str(BASE_DIR / "data" / "audio")


class UploadSettings(BaseSettings):
    """Настройки загрузки файлов."""

    model_config = SettingsConfigDict(env_prefix="UPLOAD_", extra="ignore")

    max_size_mb: int = 50
    allowed_extensions: str = ".pdf,.docx,.txt,.csv,.xlsx"
    dir: str = str(BASE_DIR / "data" / "uploads")

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [ext.strip() for ext in self.allowed_extensions.split(",")]

    @property
    def max_size_bytes(self) -> int:
        return self.max_size_mb * 1024 * 1024


# ── Главный Settings ────────────────────────────────────────


class Settings(BaseSettings):
    """Корневые настройки приложения.  Читает .env автоматически."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- App --
    APP_NAME: str = "HackSpring"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _parse_cors(cls, v: object) -> object:
        if isinstance(v, str):
            return json.loads(v)
        return v

    # -- Database --
    DATABASE_URL: str = "sqlite:///./auth.db"

    # -- JWT / Auth --
    SECRET_KEY: str = "change-me-to-random-64-char-string"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # -- Logging --
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    # -- Вложенные секции (инициализируются из того же .env, кэшируются) --
    @cached_property
    def llm(self) -> LLMSettings:
        return LLMSettings(_env_file=str(BASE_DIR / ".env"))

    @cached_property
    def rag(self) -> RAGSettings:
        return RAGSettings(_env_file=str(BASE_DIR / ".env"))

    @cached_property
    def tts(self) -> TTSSettings:
        return TTSSettings(_env_file=str(BASE_DIR / ".env"))

    @cached_property
    def upload(self) -> UploadSettings:
        return UploadSettings(_env_file=str(BASE_DIR / ".env"))


# Синглтон — импортируй именно его
settings = Settings()
