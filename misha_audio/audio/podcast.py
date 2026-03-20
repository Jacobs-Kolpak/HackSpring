from __future__ import annotations

import re
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from common.remote_llm import RemoteLLMClient


@dataclass
class PodcastConfig:
    tone: str = "scientific"  # scientific | everyday
    pace: str = "normal"      # slow | normal | fast
    max_facts: int = 6


class PodcastGenerator:
    """Генерация аудиопересказа в формате диалога двух ведущих через удаленный LLM API."""

    def __init__(self, remote_llm_client: RemoteLLMClient):
        self.remote_llm_client = remote_llm_client

    def generate_dialogue(self, text: str, topic: str, config: Optional[PodcastConfig] = None) -> str:
        cfg = config or PodcastConfig()
        clean = self._clean(text)
        context = clean
        prompt = self._build_prompt(text=context, topic=topic, tone=cfg.tone, pace=cfg.pace)

        raw = self.remote_llm_client.generate(
            prompt=prompt,
            temperature=0.3,
        )
        normalized = self._normalize_dialogue(raw)
        if not normalized:
            normalized = self._coerce_dialogue_from_free_text(raw)

        if not normalized or self._is_low_quality_dialogue(normalized):
            raise RuntimeError(
                "Удаленный LLM вернул некорректный диалог. "
                "Проверьте remote-model/промпт."
            )
        return normalized

    def save_audio(self, dialogue: str, output_path: Path, pace: str = "normal") -> bool:
        try:
            import pyttsx3  # type: ignore

            output_path.parent.mkdir(parents=True, exist_ok=True)
            engine = pyttsx3.init()

            base_rate = engine.getProperty("rate") or 180
            factor = {"slow": 0.85, "normal": 1.0, "fast": 1.2}.get(pace, 1.0)
            tts_rate = int(base_rate * factor)
            chunks = self._split_dialogue_for_tts(dialogue)

            if not chunks:
                engine.setProperty("rate", tts_rate)
                clean_dialogue = re.sub(r"Ведущий\s*[12]\s*:", "", dialogue)
                engine.save_to_file(clean_dialogue, str(output_path))
                engine.runAndWait()
                return output_path.exists() and output_path.stat().st_size > 0

            voices = engine.getProperty("voices") or []
            voice_1, voice_2 = self._pick_two_voices(voices)
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_files: List[Path] = []
                for idx, (speaker, text_part) in enumerate(chunks, start=1):
                    tmp_file = Path(tmpdir) / f"part_{idx:03d}.wav"
                    engine.setProperty("rate", tts_rate)
                    if speaker == 1 and voice_1:
                        engine.setProperty("voice", voice_1)
                    elif speaker == 2 and voice_2:
                        engine.setProperty("voice", voice_2)
                    engine.save_to_file(text_part, str(tmp_file))
                    temp_files.append(tmp_file)

                engine.runAndWait()
                if self._concat_wav_files(temp_files, output_path):
                    return True

            engine = pyttsx3.init()
            engine.setProperty("rate", tts_rate)
            clean_dialogue = re.sub(r"Ведущий\s*[12]\s*:", "", dialogue)
            engine.save_to_file(clean_dialogue, str(output_path))
            engine.runAndWait()
            return output_path.exists() and output_path.stat().st_size > 0
        except Exception:
            return False

    @staticmethod
    def _clean(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        normalized = re.sub(r"([.!?])([А-ЯA-Z])", r"\1 \2", normalized)
        return normalized

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        prepared = text.replace("●", "\n").replace("•", "\n").replace("▪", "\n").replace("◦", "\n")
        parts = re.split(r"(?<=[.!?])\s+|\n+", prepared)
        return [s.strip() for s in parts if s.strip()]

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
        facts = [self._normalize_fact(sentences[i]) for i in best_idx]
        return [fact for fact in facts if fact][:limit]

    def _prepare_model_context(self, text: str, limit: int = 6) -> str:
        facts = self._extract_key_facts(text, limit)
        if not facts:
            return text[:1800]
        return "\n".join(f"- {fact}" for fact in facts[:limit])

    @staticmethod
    def _build_prompt(text: str, topic: str, tone: str, pace: str) -> str:
        tone_hint = "научный" if tone == "scientific" else "повседневный"
        pace_hint = "медленный" if pace == "slow" else ("быстрый" if pace == "fast" else "средний")
        return (
            "Сделай диалог двух ведущих по этому тексту.\n"
            f"Стиль: {tone_hint}.\n"
            f"Темп речи: {pace_hint}.\n"
            "Формат строк: только 'Ведущий 1: ...' и 'Ведущий 2: ...'.\n"
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
    def _normalize_fact(text: str) -> str:
        fact = text.strip().lstrip("-*•●▪◦ ")
        fact = re.sub(r"\s+", " ", fact)
        if len(re.findall(r"[А-Яа-яA-Za-z0-9-]+", fact)) < 5:
            return ""
        return fact[:220].rstrip(" ,;:")

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
            chunks.append((int(match.group(1)), match.group(2).strip()))
        return chunks

    @staticmethod
    def _pick_two_voices(voices: list) -> tuple[Optional[str], Optional[str]]:
        if not voices:
            return None, None
        if len(voices) == 1:
            voice_id = getattr(voices[0], "id", None)
            return voice_id, voice_id
        return getattr(voices[0], "id", None), getattr(voices[1], "id", None)

    @staticmethod
    def _concat_wav_files(parts: List[Path], output_path: Path) -> bool:
        valid_parts = [p for p in parts if p.exists() and p.stat().st_size > 44]
        if not valid_parts:
            return False
        try:
            with wave.open(str(valid_parts[0]), "rb") as first:
                params = first.getparams()
                frames = [first.readframes(first.getnframes())]
            for path in valid_parts[1:]:
                with wave.open(str(path), "rb") as wf:
                    if wf.getnchannels() != params.nchannels or wf.getframerate() != params.framerate:
                        continue
                    frames.append(wf.readframes(wf.getnframes()))

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output_path), "wb") as out:
                out.setparams(params)
                for data in frames:
                    out.writeframes(data)
            return output_path.exists() and output_path.stat().st_size > 44
        except Exception:
            return False
