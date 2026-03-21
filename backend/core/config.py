from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = str(BASE_DIR / ".env")


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")

    provider: str = "openai"
    api_key: str = ""
    model: str = "gpt-oss-20b"
    base_url: str = "https://hackai.centrinvest.ru:6630"
    temperature: float = 0.7
    max_tokens: int = 4096


class RAGSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", extra="ignore")

    chunk_size: int = 900
    chunk_overlap: int = 180
    embeddings_model: str = "Qwen3-Embedding-0.6B"
    embedder_url: str = ""
    embedder_api_key: str = ""
    vector_store_path: str = str(BASE_DIR / "data" / "vector_store")
    collection: str = "docs"
    top_k: int = 5
    fetch_k: int = 40
    min_score: float = 0.20
    dense_weight: float = 0.70
    rerank_blend: float = 0.35
    rerank_url: str = ""
    rerank_api_key: str = ""
    rerank_model: str = ""


class TTSSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TTS_", extra="ignore")

    engine: str = "silero"
    model: str = "v4_ru"
    speaker: str = "baya"
    sample_rate: int = 48000
    output_dir: str = str(BASE_DIR / "data" / "audio")


class UploadSettings(BaseSettings):
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


class InfographicSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFOGRAPHIC_", extra="ignore")

    output_dir: str = str(BASE_DIR / "data" / "infographics")
    default_max_topics: int = 6
    max_topics_limit: int = 12
    models: str = ""
    auto_discover_models: bool = False
    max_model_candidates: int = 8


class PresentationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PRESENTATION_", extra="ignore")

    output_dir: str = str(BASE_DIR / "data" / "presentations")
    default_max_slides: int = 8
    max_slides_limit: int = 15
    max_bullets_per_slide: int = 5


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "HackSpring"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _parse_cors(cls, v: object) -> object:  # noqa: N805
        if isinstance(v, str):
            return json.loads(v)
        return v

    DATABASE_URL: str = "sqlite:///./auth.db"
    SECRET_KEY: str = "change-me-to-random-64-char-string"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    @cached_property
    def llm(self) -> LLMSettings:
        return LLMSettings(_env_file=_ENV_FILE)  # type: ignore[call-arg]

    @cached_property
    def rag(self) -> RAGSettings:
        return RAGSettings(_env_file=_ENV_FILE)  # type: ignore[call-arg]

    @cached_property
    def tts(self) -> TTSSettings:
        return TTSSettings(_env_file=_ENV_FILE)  # type: ignore[call-arg]

    @cached_property
    def upload(self) -> UploadSettings:
        return UploadSettings(_env_file=_ENV_FILE)  # type: ignore[call-arg]

    @cached_property
    def infographic(self) -> InfographicSettings:
        return InfographicSettings(_env_file=_ENV_FILE)  # type: ignore[call-arg]

    @cached_property
    def presentation(self) -> PresentationSettings:
        return PresentationSettings(_env_file=_ENV_FILE)  # type: ignore[call-arg]


settings = Settings()
