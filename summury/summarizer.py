from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from common.remote_llm import RemoteLLMClient


@dataclass
class SummaryConfig:
    max_sentences: int = 100


class FormalSummarizer:
    """Простая суммаризация через удаленный LLM API."""

    def __init__(self, remote_llm_client: RemoteLLMClient):
        self.remote_llm_client = remote_llm_client

    def summarize(self, text: str, topic: str, config: Optional[SummaryConfig] = None) -> str:
        cfg = config or SummaryConfig()
        cleaned = self._clean_text(text)

        if not cleaned:
            return "Недостаточно данных для формирования суммаризации."
        return self._summarize_remote(cleaned, topic=topic, max_sentences=cfg.max_sentences)

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _summarize_remote(self, text: str, topic: str, max_sentences: int) -> str:
        chunk = text[:9000]
        prompt = (
            "Сделай мне суммаризацию этого файла на русском языке.\n"
            f"Тема: {topic}\n"
            f"Желаемая длина: {max(3, min(max_sentences, 100))} предложений.\n"
            f"Текст файла:\n{chunk}\n"
            "Суммаризация:"
        )
        raw = self.remote_llm_client.generate(prompt=prompt, max_new_tokens=2000, temperature=0.2)
        cleaned = re.sub(r"(?is)^.*?суммаризац(?:ия|ию)\s*:\s*", "", raw).strip()
        if not cleaned:
            raise RuntimeError("Удаленный LLM вернул пустое резюме.")
        return cleaned
