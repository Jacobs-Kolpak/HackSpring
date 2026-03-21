from __future__ import annotations

import logging
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from backend.utils.llm import generate_text

logger = logging.getLogger(__name__)


@dataclass
class PodcastConfig:
    tone: str = "scientific"
    pace: str = "normal"
    silero_speaker_1: str = "baya"
    silero_speaker_2: str = "xenia"


AVAILABLE_SPEAKERS = ["aidar", "baya", "eugene", "kseniya", "xenia"]

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

_silero_model = None
_silero_sample_rate = 48000
_silero_load_error: str = ""


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


def _clean_source(text: str) -> str:
    normalized = re.sub(r"[-‐‑]\s*\n\s*", "", text)
    normalized = normalized.replace("\n", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"([.!?])([А-ЯA-Z])", r"\1 \2", normalized)
    normalized = re.sub(r"([a-zа-я])([A-ZА-Я])", r"\1 \2", normalized)
    normalized = re.sub(r"([A-Za-z])([А-Яа-я])", r"\1 \2", normalized)
    normalized = re.sub(r"([А-Яа-я])([A-Za-z])", r"\1 \2", normalized)
    normalized = re.sub(r"\b\d+\s*$", "", normalized)
    return normalized


def _prepare_source(text: str) -> str:
    clean = _clean_source(text)
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


def generate_dialogue(
    text: str,
    *,
    topic: str = "Без названия",
    config: Optional[PodcastConfig] = None,
    model: Optional[str] = None,
) -> str:
    cfg = config or PodcastConfig()
    cleaned = _prepare_source(text)
    if not cleaned:
        raise ValueError("Пустой текст для генерации диалога.")

    tone_hint = "научный" if cfg.tone == "scientific" else "повседневный"
    pace_hint = {
        "slow": "медленный", "normal": "средний", "fast": "быстрый"
    }.get(cfg.pace, "средний")

    prompt = (
        "Сделай читаемый диалог двух ведущих по этому тексту.\n"
        f"Стиль: {tone_hint}.\n"
        f"Темп речи: {pace_hint}.\n"
        "Требования:\n"
        "- Только русский язык, без латиницы и обрывков слов.\n"
        "- 30-35 реплик, короткие и понятные.\n"
        "- Каждая строка строго начинается с 'Ведущий 1:' или 'Ведущий 2:'.\n"
        "- Не вставляй служебный текст, JSON, комментарии и номера страниц.\n"
        f"Тема: {topic}.\n"
        f"Текст:\n{cleaned}\n"
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


def _load_silero_model():  # type: ignore[no-untyped-def]
    global _silero_model, _silero_sample_rate, _silero_load_error  # noqa: PLW0603
    if _silero_load_error:
        return None, None
    if _silero_model is not None:
        return _silero_model, _silero_sample_rate
    try:
        import ssl  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
        from silero import silero_tts  # pylint: disable=import-outside-toplevel

        # Отключаем проверку SSL — внутри контейнера бывают проблемы с сертификатами
        ssl._create_default_https_context = ssl._create_unverified_context

        logger.info("Loading Silero TTS via silero pip package...")
        model, symbols, sample_rate, example_text, apply_tts = silero_tts(
            language="ru",
            speaker="v4_ru",
        )
        logger.info("Silero loaded via pip package OK")

        _silero_model = model
        _silero_sample_rate = sample_rate
        model.apply_tts(text="Тест.", speaker="baya", sample_rate=sample_rate)
        logger.info("Silero TTS test passed")
        return _silero_model, _silero_sample_rate
    except Exception as exc:  # pylint: disable=broad-except
        _silero_load_error = f"{type(exc).__name__}: {exc}"
        logger.error("Silero load failed: %s", _silero_load_error)
        return None, None


def _sanitize_tts_text(text: str) -> str:
    cleaned = re.sub(r"[^\w\s.,!?;:()\"'«»—-]", " ", text, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:280]


def save_audio(
    dialogue: str,
    output_path: Path,
    pace: str = "normal",
    silero_speaker_1: str = "baya",
    silero_speaker_2: str = "xenia",
) -> bool:
    _ = pace
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
    except ImportError:
        return False

    model, sample_rate = _load_silero_model()
    if model is None or sample_rate is None:
        logger.error("Silero model not loaded, load_error=%s", _silero_load_error)
        return False

    logger.info("Silero model loaded OK, sample_rate=%s", sample_rate)

    chunks = _split_for_tts(dialogue)
    if not chunks:
        clean = re.sub(r"Ведущий\s*[12]\s*:\s*", "", dialogue).strip()
        chunks = [(1, clean)] if clean else []

    logger.info("TTS chunks count: %d", len(chunks))

    wav_parts: list = []
    pause = np.zeros(int(0.14 * sample_rate), dtype=np.float32)
    success_chunks = 0

    for speaker_id, text_part in chunks:
        text_part = text_part.strip()
        if not text_part:
            continue

        speaker = silero_speaker_1 if speaker_id == 1 else silero_speaker_2
        try:
            audio = model.apply_tts(
                text=text_part,
                speaker=speaker,
                sample_rate=sample_rate,
                put_accent=True,
                put_yo=True,
            )
        except Exception as exc1:
            logger.warning(
                "TTS failed for chunk (speaker=%s, len=%d): %s. Text: %.80s",
                speaker, len(text_part), exc1, text_part,
            )
            safe_text = _sanitize_tts_text(text_part)
            if not safe_text:
                continue
            try:
                audio = model.apply_tts(
                    text=safe_text,
                    speaker=speaker,
                    sample_rate=sample_rate,
                    put_accent=True,
                    put_yo=True,
                )
            except Exception as exc2:
                logger.warning(
                    "TTS sanitized retry also failed: %s. Safe text: %.80s",
                    exc2, safe_text,
                )
                continue

        wav = audio.detach().cpu().numpy().astype(np.float32)
        wav_parts.append(wav)
        wav_parts.append(pause)
        success_chunks += 1

    if not wav_parts or success_chunks == 0:
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
