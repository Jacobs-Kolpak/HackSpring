from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SummaryConfig:
    max_sentences: int = 5
    language: str = "ru"
    style: str = "official"


class FormalSummarizer:
    """Генерация формализованного саммари по шаблону.

    1) Пытается использовать простую LLM-модель через transformers pipeline.
    2) При недоступности модели использует извлекающий (extractive) fallback.
    """

    def __init__(self, model_name: str = "IlyaGusev/rut5_base_sum_gazeta"):
        self.model_name = model_name
        self._hf_summarizer = self._load_hf_pipeline()

    def _load_hf_pipeline(self):
        try:
            from transformers import pipeline  # type: ignore

            return pipeline("summarization", model=self.model_name)
        except Exception:
            return None

    def summarize(self, text: str, topic: str, config: Optional[SummaryConfig] = None) -> str:
        cfg = config or SummaryConfig()
        cleaned = self._clean_text(text)

        if not cleaned:
            return self._format_template(topic=topic, summary="Недостаточно данных для формирования резюме.")

        if self._hf_summarizer is not None:
            try:
                summary = self._summarize_hf(cleaned)
            except Exception:
                summary = self._summarize_extractive(cleaned, cfg.max_sentences)
        else:
            summary = self._summarize_extractive(cleaned, cfg.max_sentences)

        return self._format_template(topic=topic, summary=summary)

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _summarize_hf(self, text: str) -> str:
        # Многие модели имеют ограничение по длине входа — ограничиваем безопасно.
        chunk = text[:3500]
        result = self._hf_summarizer(
            chunk,
            max_length=220,
            min_length=80,
            do_sample=False,
        )
        return result[0]["summary_text"].strip()

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _summarize_extractive(self, text: str, max_sentences: int) -> str:
        sentences = self._split_sentences(text)
        if not sentences:
            return text[:400]

        # Простой скоринг: частота значимых слов + длина предложения.
        words = re.findall(r"[А-Яа-яA-Za-z0-9-]+", text.lower())
        stopwords = {
            "и", "в", "во", "на", "с", "со", "по", "о", "об", "для", "а", "но", "или",
            "к", "ко", "из", "за", "под", "при", "что", "это", "как", "так", "от", "до",
            "the", "and", "of", "to", "in", "for", "a", "is", "on", "with", "that",
        }
        freq = {}
        for w in words:
            if len(w) < 3 or w in stopwords:
                continue
            freq[w] = freq.get(w, 0) + 1

        def score(sentence: str) -> float:
            tokens = re.findall(r"[А-Яа-яA-Za-z0-9-]+", sentence.lower())
            if not tokens:
                return 0.0
            value = sum(freq.get(t, 0) for t in tokens)
            length_penalty = 1.0 + (max(0, len(tokens) - 28) / 40.0)
            return value / length_penalty

        ranked = sorted(enumerate(sentences), key=lambda it: score(it[1]), reverse=True)
        top_idx = sorted(i for i, _ in ranked[:max_sentences])
        selected = [sentences[i] for i in top_idx]
        return " ".join(selected)

    @staticmethod
    def _format_template(topic: str, summary: str) -> str:
        points = FormalSummarizer._to_points(summary, limit=4)
        conclusions = FormalSummarizer._build_conclusions(summary)
        recommendations = FormalSummarizer._build_recommendations(summary)

        return (
            "ФОРМАЛИЗОВАННОЕ РЕЗЮМЕ\n"
            f"Тема: {topic}\n"
            "Цель документа: Кратко представить ключевые факты и выводы исходного материала.\n\n"
            "1. Ключевые положения:\n"
            f"{points}\n\n"
            "2. Выводы:\n"
            f"{conclusions}\n\n"
            "3. Рекомендации:\n"
            f"{recommendations}\n"
        )

    @staticmethod
    def _to_points(summary: str, limit: int = 4) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", summary)
        clean = [s.strip() for s in sentences if s.strip()]
        if not clean:
            return "- Существенные положения не выявлены."
        return "\n".join(f"- {s}" for s in clean[:limit])

    @staticmethod
    def _build_conclusions(summary: str) -> str:
        text = summary.strip()
        if not text:
            return "- По предоставленным данным формирование вывода затруднено."
        sentences = re.split(r"(?<=[.!?])\s+", text)
        core = " ".join(s.strip() for s in sentences[:2] if s.strip())
        core = core.rstrip(" ,.;:")
        if not core:
            return "- По предоставленным данным формирование вывода затруднено."
        return f"- Анализ показывает, что {core}"

    @staticmethod
    def _build_recommendations(summary: str) -> str:
        has_risk = any(w in summary.lower() for w in ["риск", "проблем", "ограничен", "дефицит", "ошибк"])
        if has_risk:
            return (
                "- Организовать контрольные мероприятия по выявленным рискам.\n"
                "- Зафиксировать ответственных и сроки исполнения."
            )
        return (
            "- Использовать полученные выводы при планировании дальнейших мероприятий.\n"
            "- Обновлять резюме по мере поступления новых данных."
        )
