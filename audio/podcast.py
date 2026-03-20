from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class PodcastConfig:
    tone: str = "scientific"  # scientific | popular
    pace: str = "normal"      # slow | normal | fast
    max_facts: int = 6
    model_name: str = "google/flan-t5-small"
    max_new_tokens: int = 220


class PodcastGenerator:
    """Легковесная генерация аудиопересказа в формате диалога двух ведущих."""

    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.model_name = model_name
        self._generator = self._load_generator()

    def _load_generator(self):
        try:
            from transformers import pipeline  # type: ignore

            return pipeline("text2text-generation", model=self.model_name)
        except Exception:
            return None

    def generate_dialogue(self, text: str, topic: str, config: Optional[PodcastConfig] = None) -> str:
        cfg = config or PodcastConfig()
        clean = self._clean(text)

        # 1) Пытаемся сгенерировать диалог через модель.
        if self._generator is not None:
            generated = self._generate_with_model(clean, topic, cfg)
            if generated:
                return generated

        # 2) При недоступности модели — fallback.
        facts = self._extract_key_facts(clean, cfg.max_facts)

        if not facts:
            facts = ["Данных недостаточно для содержательного пересказа."]

        intro_1, intro_2, bridge, outro = self._style_phrases(cfg.tone, topic)

        lines: List[str] = []
        lines.append(f"Ведущий 1: {intro_1}")
        lines.append(f"Ведущий 2: {intro_2}")

        for i, fact in enumerate(facts, start=1):
            if i % 2 == 1:
                lines.append(f"Ведущий 1: {bridge} {fact}")
            else:
                lines.append(f"Ведущий 2: {bridge} {fact}")

        lines.append(f"Ведущий 2: {outro}")
        return "\n".join(lines)

    def _generate_with_model(self, text: str, topic: str, config: PodcastConfig) -> Optional[str]:
        prompt = self._build_prompt(text=text, topic=topic, tone=config.tone)
        try:
            chunk = prompt[:3500]
            result = self._generator(
                chunk,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
            )
            if not result:
                return None
            raw = result[0].get("generated_text", "").strip()
            normalized = self._normalize_dialogue(raw)
            return normalized if normalized else None
        except Exception:
            return None

    def save_audio(self, dialogue: str, output_path: Path, pace: str = "normal") -> bool:
        """Пытается сохранить WAV через pyttsx3. Возвращает True при успехе."""
        try:
            import pyttsx3  # type: ignore

            output_path.parent.mkdir(parents=True, exist_ok=True)
            engine = pyttsx3.init()

            base_rate = engine.getProperty("rate") or 180
            factor = {"slow": 0.85, "normal": 1.0, "fast": 1.2}.get(pace, 1.0)
            engine.setProperty("rate", int(base_rate * factor))

            # В легковесной версии озвучиваем общий текст с пометками ведущих.
            engine.save_to_file(dialogue, str(output_path))
            engine.runAndWait()
            return output_path.exists() and output_path.stat().st_size > 0
        except Exception:
            return False

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _extract_key_facts(self, text: str, limit: int) -> List[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        words = re.findall(r"[А-Яа-яA-Za-z0-9-]+", text.lower())
        stop = {
            "и", "в", "во", "на", "с", "со", "по", "о", "об", "для", "а", "но", "или",
            "к", "ко", "из", "за", "под", "при", "что", "это", "как", "так", "от", "до",
            "the", "and", "of", "to", "in", "for", "a", "is", "on", "with", "that",
        }

        freq = {}
        for w in words:
            if len(w) < 3 or w in stop:
                continue
            freq[w] = freq.get(w, 0) + 1

        def score(sent: str) -> float:
            tokens = re.findall(r"[А-Яа-яA-Za-z0-9-]+", sent.lower())
            if not tokens:
                return 0.0
            return sum(freq.get(t, 0) for t in tokens) / max(len(tokens), 1)

        ranked = sorted(enumerate(sentences), key=lambda p: score(p[1]), reverse=True)
        best_idx = sorted(i for i, _ in ranked[:limit])
        return [sentences[i] for i in best_idx]

    @staticmethod
    def _style_phrases(tone: str, topic: str):
        if tone == "popular":
            return (
                f"Привет! Сегодня коротко и понятно разбираем тему: {topic}.",
                "Сделаем это в формате живого диалога и выделим главное.",
                "Проще говоря,",
                "На этом все, главное мы разобрали."
            )
        return (
            f"Добрый день. Рассматриваем материал по теме: {topic}.",
            "Сфокусируемся на ключевых фактах и практических выводах.",
            "Зафиксируем следующий тезис:",
            "Итог: основные положения представлены в структурированном виде."
        )

    @staticmethod
    def _build_prompt(text: str, topic: str, tone: str) -> str:
        tone_hint = "научный, нейтральный и точный" if tone == "scientific" else "популярный, живой и простой"
        return (
            "Сгенерируй короткий диалог двух ведущих на русском языке.\n"
            "Формат строго:\n"
            "Ведущий 1: ...\n"
            "Ведущий 2: ...\n"
            "8-12 реплик, без нумерации.\n"
            f"Тон: {tone_hint}.\n"
            f"Тема: {topic}.\n"
            f"Материал:\n{text}\n"
        )

    @staticmethod
    def _normalize_dialogue(raw: str) -> str:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        normalized: List[str] = []
        for line in lines:
            if line.startswith("Ведущий 1:") or line.startswith("Ведущий 2:"):
                normalized.append(line)
            elif line.startswith("Speaker 1:"):
                normalized.append(line.replace("Speaker 1:", "Ведущий 1:", 1))
            elif line.startswith("Speaker 2:"):
                normalized.append(line.replace("Speaker 2:", "Ведущий 2:", 1))

        if len(normalized) < 4:
            return ""
        return "\n".join(normalized)
