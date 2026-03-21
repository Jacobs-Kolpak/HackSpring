from __future__ import annotations

import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from common.remote_llm import RemoteLLMClient


@dataclass
class PodcastConfig:
    tone: str = "scientific"  # scientific | everyday
    pace: str = "normal"      # slow | normal | fast
    silero_speaker_1: str = "baya"
    silero_speaker_2: str = "xenia"


class PodcastGenerator:
    """Генерация диалога и озвучка через удаленный LLM + Silero TTS."""

    _silero_model = None
    _silero_sample_rate = 48000

    def __init__(self, remote_llm_client: RemoteLLMClient):
        self.remote_llm_client = remote_llm_client

    def generate_dialogue(self, text: str, topic: str, config: Optional[PodcastConfig] = None) -> str:
        cfg = config or PodcastConfig()
        clean = self._prepare_source_for_prompt(text)
        prompt = self._build_prompt(text=clean, topic=topic, tone=cfg.tone, pace=cfg.pace)

        raw = self.remote_llm_client.generate(prompt=prompt, temperature=0.3)
        normalized = self._normalize_dialogue(raw)
        if not normalized:
            normalized = self._coerce_dialogue_from_free_text(raw)

        if not normalized or self._is_low_quality_dialogue(normalized):
            raise RuntimeError("Удаленный LLM вернул некорректный диалог. Проверьте remote-model/промпт.")
        return normalized

    def save_audio(
        self,
        dialogue: str,
        output_path: Path,
        pace: str = "normal",
        tts_model: str = "silero",
        silero_speaker_1: str = "baya",
        silero_speaker_2: str = "xenia",
    ) -> bool:
        _ = pace
        _ = tts_model
        try:
            import numpy as np  # type: ignore
        except Exception:
            return False

        model, sample_rate = self._load_silero_model()
        if model is None or sample_rate is None:
            return False

        chunks = self._split_dialogue_for_tts(dialogue)
        if not chunks:
            # Без меток спикеров читаем как один монолог
            clean = re.sub(r"Ведущий\s*[12]\s*:\s*", "", dialogue).strip()
            chunks = [(1, clean)] if clean else []

        wav_parts: List["np.ndarray"] = []
        pause = np.zeros(int(0.14 * sample_rate), dtype=np.float32)

        for speaker_id, text_part in chunks:
            text = text_part.strip()
            if not text:
                continue

            speaker = silero_speaker_1 if speaker_id == 1 else silero_speaker_2
            try:
                audio = model.apply_tts(
                    text=text,
                    speaker=speaker,
                    sample_rate=sample_rate,
                    put_accent=True,
                    put_yo=True,
                )
            except Exception:
                return False

            wav = audio.detach().cpu().numpy().astype(np.float32)
            wav_parts.append(wav)
            wav_parts.append(pause)

        if not wav_parts:
            return False

        full_audio = np.concatenate(wav_parts)
        full_audio = np.clip(full_audio, -1.0, 1.0)
        wav_int16 = (full_audio * 32767.0).astype(np.int16)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(wav_int16.tobytes())

        return output_path.exists() and output_path.stat().st_size > 44

    @classmethod
    def _load_silero_model(cls):
        if cls._silero_model is not None:
            return cls._silero_model, cls._silero_sample_rate
        try:
            import torch  # type: ignore

            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language="ru",
                speaker="v4_ru",
                trust_repo=True,
            )
            cls._silero_model = model
            cls._silero_sample_rate = 48000
            return cls._silero_model, cls._silero_sample_rate
        except Exception:
            return None, None

    @staticmethod
    def _clean(text: str) -> str:
        normalized = re.sub(r"[-‐‑]\s*\n\s*", "", text)
        normalized = normalized.replace("\n", " ")
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = re.sub(r"([.!?])([А-ЯA-Z])", r"\1 \2", normalized)
        normalized = re.sub(r"([a-zа-я])([A-ZА-Я])", r"\1 \2", normalized)
        normalized = re.sub(r"([A-Za-z])([А-Яа-я])", r"\1 \2", normalized)
        normalized = re.sub(r"([А-Яа-я])([A-Za-z])", r"\1 \2", normalized)
        normalized = re.sub(r"\b\d+\s*$", "", normalized)
        return normalized

    @staticmethod
    def _prepare_source_for_prompt(text: str) -> str:
        clean = PodcastGenerator._clean(text)
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
        selected = []
        for sentence in parts:
            if len(sentence) < 25:
                continue
            if re.fullmatch(r"[\W\d_]+", sentence):
                continue
            selected.append(sentence)
            if len(selected) >= 18:
                break
        if not selected:
            return clean[:3500]
        return " ".join(selected)[:3500]

    @staticmethod
    def _build_prompt(text: str, topic: str, tone: str, pace: str) -> str:
        tone_hint = "научный" if tone == "scientific" else "повседневный"
        pace_hint = "медленный" if pace == "slow" else ("быстрый" if pace == "fast" else "средний")
        return (
            "Сделай читаемый диалог двух ведущих по этому тексту.\n"
            f"Стиль: {tone_hint}.\n"
            f"Темп речи: {pace_hint}.\n"
            "Требования:\n"
            "- Только русский язык, без латиницы и обрывков слов.\n"
            "- 30-35 реплик, короткие и понятные.\n"
            "- Каждая строка строго начинается с 'Ведущий 1:' или 'Ведущий 2:'.\n"
            "- Не вставляй служебный текст, JSON, комментарии и номера страниц.\n"
            f"Тема: {topic}.\n"
            f"Текст:\n{text}\n"
            "Диалог:"
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
        if len(normalized) < 6:
            return ""
        return "\n".join(normalized)

    @staticmethod
    def _coerce_dialogue_from_free_text(raw: str) -> str:
        text = re.sub(r"\s+", " ", raw).strip()
        if not text:
            return ""
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(parts) < 6:
            return ""
        lines: List[str] = []
        speaker = 1
        for sentence in parts:
            sentence = sentence.strip(' "\'')
            if len(sentence) < 8:
                continue
            lines.append(f"Ведущий {speaker}: {sentence}")
            speaker = 2 if speaker == 1 else 1
        return "\n".join(lines) if len(lines) >= 6 else ""

    @staticmethod
    def _is_low_quality_dialogue(dialogue: str) -> bool:
        lines = [line.strip() for line in dialogue.splitlines() if line.strip()]
        if len(lines) < 6:
            return True
        has_1 = any(line.startswith("Ведущий 1:") for line in lines)
        has_2 = any(line.startswith("Ведущий 2:") for line in lines)
        return not (has_1 and has_2)

    @staticmethod
    def _split_dialogue_for_tts(dialogue: str) -> List[tuple[int, str]]:
        chunks: List[tuple[int, str]] = []
        for line in dialogue.splitlines():
            clean = line.strip()
            if not clean:
                continue
            match = re.match(r"^Ведущий\s*([12])\s*:\s*(.+)$", clean)
            if not match:
                continue
            # В озвучку передаем только реплику, без префикса "Ведущий"
            chunks.append((int(match.group(1)), match.group(2).strip()))
        return chunks
