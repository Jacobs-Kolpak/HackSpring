from __future__ import annotations

import re
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from backend.utils.llm import generate_text


@dataclass
class PodcastConfig:
    tone: str = "scientific"
    pace: str = "normal"


_TONE_MAP = {
    "scientific": "scientific", "science": "scientific",
    "научный": "scientific", "научно": "scientific",
    "everyday": "everyday", "повседневный": "everyday",
    "повседневно": "everyday", "popular": "everyday",
    "популярный": "everyday", "простыми": "everyday",
}

_PACE_MAP = {
    "slow": "slow", "медленно": "slow",
    "normal": "normal", "нормально": "normal",
    "fast": "fast", "быстро": "fast",
}


def normalize_tone(value: str) -> str:
    key = value.strip().lower()
    if key not in _TONE_MAP:
        raise ValueError(
            f"Некорректный tone: {value}. "
            "Допустимо: scientific|everyday|научный|повседневный"
        )
    return _TONE_MAP[key]


def normalize_pace(value: str) -> str:
    key = value.strip().lower()
    if key not in _PACE_MAP:
        raise ValueError(
            f"Некорректный pace: {value}. "
            "Допустимо: slow|normal|fast|медленно|нормально|быстро"
        )
    return _PACE_MAP[key]


def generate_dialogue(
    text: str,
    *,
    topic: str = "Без названия",
    config: Optional[PodcastConfig] = None,
    model: Optional[str] = None,
) -> str:
    cfg = config or PodcastConfig()
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        raise ValueError("Пустой текст для генерации диалога.")

    tone_hint = "научный" if cfg.tone == "scientific" else "повседневный"
    pace_hint = {
        "slow": "медленный", "normal": "средний", "fast": "быстрый"
    }.get(cfg.pace, "средний")

    prompt = (
        "Сделай диалог двух ведущих по этому тексту.\n"
        f"Стиль: {tone_hint}.\n"
        f"Темп речи: {pace_hint}.\n"
        "Формат строк: только 'Ведущий 1: ...' и 'Ведущий 2: ...'.\n"
        "Минимум 8 реплик.\n"
        f"Тема: {topic}.\n"
        f"Текст:\n{cleaned[:6000]}\n"
        "Диалог:"
    )
    raw = generate_text(
        prompt,
        system="Ты генератор подкастов. Пиши диалоги на русском языке "
               "строго в формате 'Ведущий 1: ...' и 'Ведущий 2: ...'.",
        model=model,
        temperature=0.3,
        max_tokens=3000,
    )

    dialogue = _normalize_dialogue(raw)
    if not dialogue:
        dialogue = _coerce_from_free_text(raw)

    if not dialogue or _is_low_quality(dialogue):
        raise RuntimeError(
            "LLM вернул некорректный диалог. "
            "Попробуйте другую модель или текст."
        )
    return dialogue


def _normalize_dialogue(raw: str) -> str:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    result: List[str] = []
    for line in lines:
        if line.startswith("Ведущий 1:") or line.startswith("Ведущий 2:"):
            result.append(line)
        elif line.startswith("Speaker 1:"):
            result.append(line.replace("Speaker 1:", "Ведущий 1:", 1))
        elif line.startswith("Speaker 2:"):
            result.append(line.replace("Speaker 2:", "Ведущий 2:", 1))
    return "\n".join(result) if len(result) >= 6 else ""


def _coerce_from_free_text(raw: str) -> str:
    text = re.sub(r"\s+", " ", raw).strip()
    if not text:
        return ""
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(parts) < 6:
        return ""
    lines: List[str] = []
    speaker = 1
    for sent in parts:
        sent = sent.strip(' "\'')
        if len(sent) < 8:
            continue
        lines.append(f"Ведущий {speaker}: {sent}")
        speaker = 2 if speaker == 1 else 1
    return "\n".join(lines) if len(lines) >= 6 else ""


def _is_low_quality(dialogue: str) -> bool:
    lines = [line.strip() for line in dialogue.splitlines() if line.strip()]
    if len(lines) < 6:
        return True
    has_1 = any(line.startswith("Ведущий 1:") for line in lines)
    has_2 = any(line.startswith("Ведущий 2:") for line in lines)
    return not (has_1 and has_2)


def save_audio(
    dialogue: str,
    output_path: Path,
    pace: str = "normal",
) -> bool:
    try:
        import pyttsx3  # pylint: disable=import-outside-toplevel
    except ImportError:
        return False

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        engine = pyttsx3.init()

        base_rate = engine.getProperty("rate") or 180
        factor = {"slow": 0.85, "normal": 1.0, "fast": 1.2}.get(pace, 1.0)
        tts_rate = int(base_rate * factor)

        chunks = _split_for_tts(dialogue)
        if not chunks:
            engine.setProperty("rate", tts_rate)
            clean = re.sub(r"Ведущий\s*[12]\s*:", "", dialogue)
            engine.save_to_file(clean, str(output_path))
            engine.runAndWait()
            return output_path.exists() and output_path.stat().st_size > 0

        voices = engine.getProperty("voices") or []
        voice_1, voice_2 = _pick_voices(voices)

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

            if _concat_wav(temp_files, output_path):
                return True

        engine = pyttsx3.init()
        engine.setProperty("rate", tts_rate)
        clean = re.sub(r"Ведущий\s*[12]\s*:", "", dialogue)
        engine.save_to_file(clean, str(output_path))
        engine.runAndWait()
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception:  # pylint: disable=broad-except
        return False


def _split_for_tts(
    dialogue: str,
) -> List[Tuple[int, str]]:
    chunks: List[Tuple[int, str]] = []
    for line in dialogue.splitlines():
        clean = line.strip()
        if not clean:
            continue
        match = re.match(r"^Ведущий\s*([12])\s*:\s*(.+)$", clean)
        if match:
            chunks.append((int(match.group(1)), match.group(2).strip()))
    return chunks


def _pick_voices(
    voices: list,
) -> Tuple[Optional[str], Optional[str]]:
    if not voices:
        return None, None
    if len(voices) == 1:
        vid = getattr(voices[0], "id", None)
        return vid, vid
    return getattr(voices[0], "id", None), getattr(voices[1], "id", None)


def _concat_wav(parts: List[Path], output_path: Path) -> bool:
    valid = [p for p in parts if p.exists() and p.stat().st_size > 44]
    if not valid:
        return False
    try:
        with wave.open(str(valid[0]), "rb") as first:
            params = first.getparams()
            frames = [first.readframes(first.getnframes())]
        for path in valid[1:]:
            with wave.open(str(path), "rb") as wf:
                if (wf.getnchannels() != params.nchannels
                        or wf.getframerate() != params.framerate):
                    continue
                frames.append(wf.readframes(wf.getnframes()))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as out:
            out.setparams(params)
            for data in frames:
                out.writeframes(data)
        return output_path.exists() and output_path.stat().st_size > 44
    except Exception:  # pylint: disable=broad-except
        return False
