from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unicodedata
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.core.config import settings
from backend.services import podcast as podcast_service
from backend.services import summary as summary_service


@dataclass
class VideoConfig:
    max_sentences: int = 8
    width: int = 1280
    height: int = 720
    fps: int = 24
    speaker: str = "baya"


def generate_video(
    text: str,
    *,
    topic: str = "Без названия",
    config: Optional[VideoConfig] = None,
    model: Optional[str] = None,
) -> Tuple[Path, Dict[str, Any]]:
    cfg = config or VideoConfig()

    if not text.strip():
        raise RuntimeError("Пустой входной текст: невозможно собрать видео.")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Не найден ffmpeg. Установите ffmpeg и повторите запуск.")

    summary = summary_service.summarize(
        text,
        topic=topic,
        max_sentences=cfg.max_sentences,
        model=model,
    )

    slides = _split_into_slides(summary)
    if not slides:
        slides = [summary.strip() or "Нет данных для отображения"]
    narration = _build_narration_from_slides(slides)

    with tempfile.TemporaryDirectory(prefix="text_to_video_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        audio_path = tmp_dir / "narration.wav"
        slides_dir = tmp_dir / "slides"
        slides_dir.mkdir(parents=True, exist_ok=True)

        ok, tts_debug = _try_save_audio_with_fallbacks(
            dialogue=narration,
            output_path=audio_path,
            preferred_speaker=cfg.speaker,
        )
        if not ok:
            raise RuntimeError(
                "Не удалось сгенерировать аудио. "
                f"Детали: {tts_debug}"
            )

        audio_duration = _get_wav_duration_sec(audio_path)

        image_paths = _render_slides(
            slides=slides,
            out_dir=slides_dir,
            width=cfg.width,
            height=cfg.height,
        )
        concat_file = _build_concat_file(
            image_paths=image_paths,
            slide_texts=slides,
            audio_duration=audio_duration,
            out_path=tmp_dir / "slides.txt",
        )

        video_cfg = settings.video
        output_dir = Path(video_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_topic = re.sub(r"[^a-zA-Z0-9._-]+", "_", topic[:40]).strip("._") or "video"
        output_path = output_dir / f"{safe_topic}_{stamp}.mp4"

        _render_video_with_ffmpeg(
            concat_file=concat_file,
            audio_file=audio_path,
            output_path=output_path,
            fps=cfg.fps,
        )

    metadata = {
        "topic": topic,
        "slides_count": len(slides),
        "duration_sec": round(audio_duration, 2),
        "resolution": f"{cfg.width}x{cfg.height}",
        "path": str(output_path),
    }
    return output_path, metadata


def generate_video_from_file(
    file_path: Path,
    *,
    topic: str = "Без названия",
    config: Optional[VideoConfig] = None,
    model: Optional[str] = None,
) -> Tuple[Path, Dict[str, Any]]:
    from backend.utils.document_reader import read_document

    text = read_document(file_path)
    if not text.strip():
        raise ValueError("В файле не найден текст.")
    return generate_video(text, topic=topic, config=config, model=model)


def _build_narration_from_slides(slides: List[str]) -> str:
    clean_slides = [s.strip() for s in slides if s.strip()]
    if not clean_slides:
        return "Ведущий 1: Краткое изложение документа недоступно."
    spoken_slides = [_clean_for_tts(s) for s in clean_slides]
    return "\n".join(f"Ведущий 1: {s}" for s in spoken_slides if s.strip())


def _try_save_audio_with_fallbacks(
    dialogue: str, output_path: Path, preferred_speaker: str
) -> tuple[bool, str]:
    from backend.services.podcast import _silero_load_error  # pylint: disable=import-outside-toplevel

    if _silero_load_error:
        return False, f"Silero не загрузился: {_silero_load_error}"

    candidate_speakers = [preferred_speaker, "xenia", "aidar"]
    deduped: List[str] = []
    for sp in candidate_speakers:
        normalized = sp.strip()
        if normalized and normalized not in deduped:
            deduped.append(normalized)

    narration_preview = dialogue[:200].replace("\n", " ")

    ok = podcast_service.save_audio(
        dialogue=dialogue,
        output_path=output_path,
        pace="normal",
        silero_speaker_1=deduped[0],
        silero_speaker_2=deduped[0],
    )
    if ok:
        return True, ""

    short_dialogue = _shorten_dialogue(dialogue, max_chars=140)
    for speaker in deduped:
        ok = podcast_service.save_audio(
            dialogue=short_dialogue,
            output_path=output_path,
            pace="normal",
            silero_speaker_1=speaker,
            silero_speaker_2=speaker,
        )
        if ok:
            return True, ""

    if _silero_load_error:
        return False, f"Silero не загрузился: {_silero_load_error}"

    return False, (
        f"Все {len(deduped)} голоса не смогли озвучить текст. "
        f"Narration preview: {narration_preview}"
    )


def _shorten_dialogue(dialogue: str, max_chars: int) -> str:
    lines = [line.strip() for line in dialogue.splitlines() if line.strip()]
    short_lines: List[str] = []
    for line in lines:
        prefix = "Ведущий 1: "
        if line.startswith("Ведущий 1:"):
            text = line.replace("Ведущий 1:", "", 1).strip()
        elif line.startswith("Ведущий 2:"):
            text = line.replace("Ведущий 2:", "", 1).strip()
        else:
            text = line
        pieces = _chunk_sentences([text], max_chars=max_chars)
        for piece in pieces:
            if piece.strip():
                short_lines.append(prefix + piece.strip())
    return "\n".join(short_lines)


def _chunk_sentences(sentences: List[str], max_chars: int) -> List[str]:
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        clean = sentence.strip()
        if not clean:
            continue
        if len(clean) > max_chars:
            words = clean.split()
            piece = ""
            for word in words:
                extra = 1 if piece else 0
                if len(piece) + len(word) + extra > max_chars:
                    if piece:
                        chunks.append(piece.strip())
                    piece = word
                else:
                    piece = f"{piece} {word}".strip()
            if piece:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.append(piece.strip())
            continue

        candidate = f"{current} {clean}".strip() if current else clean
        if len(candidate) > max_chars:
            if current:
                chunks.append(current.strip())
            current = clean
        else:
            current = candidate

    if current:
        chunks.append(current.strip())
    return chunks


def _split_into_slides(summary: str) -> List[str]:
    lines = [line.strip() for line in summary.splitlines() if line.strip()]
    points: List[str] = []

    if len(lines) >= 2:
        for line in lines:
            cleaned = _normalize_slide_point(line)
            if cleaned:
                points.append(cleaned)
    else:
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
        for part in parts:
            cleaned = _normalize_slide_point(part)
            if cleaned:
                points.append(cleaned)

    if not points:
        cleaned = _normalize_slide_point(summary)
        points = [cleaned] if cleaned else []

    points = _expand_points_for_readability(points, max_chars=190)
    return points[:12]


def _normalize_slide_point(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"[*_`#]+", " ", cleaned)
    cleaned = re.sub(r"^\s*[-–—•]+\s*", "", cleaned)
    cleaned = re.sub(r"^\s*\d+\s*[.)-:]\s*", "", cleaned)
    cleaned = re.sub(r"^\s*[a-zа-я]\s*[.)-:]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _expand_points_for_readability(points: List[str], max_chars: int) -> List[str]:
    expanded: List[str] = []
    for point in points:
        text = point.strip()
        if not text:
            continue
        if len(text) <= max_chars:
            expanded.append(text)
            continue
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(parts) <= 1:
            expanded.extend(_chunk_sentences([text], max_chars=max_chars))
            continue
        chunks = _chunk_sentences(parts, max_chars=max_chars)
        for chunk in chunks:
            if chunk.strip():
                expanded.append(chunk.strip())
    return expanded


def _render_slides(
    slides: List[str], out_dir: Path, width: int, height: int
) -> List[Path]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError(
            "Pillow не установлен. Добавьте Pillow>=10.0.0 в requirements.txt"
        ) from exc

    paths: List[Path] = []
    for i, text in enumerate(slides, start=1):
        image = Image.new("RGB", (width, height), color=(185, 230, 178))
        draw = ImageDraw.Draw(image)

        draw.rectangle(
            [(30, 30), (width - 30, height - 30)], outline=(0, 0, 0), width=6
        )
        draw.rectangle(
            [(80, 90), (width - 80, height - 90)],
            fill=(235, 249, 230),
            outline=(0, 0, 0),
            width=4,
        )

        clean_text = _clean_text_for_slide(text)
        max_w = width - 240
        max_h = height - 280
        body_font, wrapped, spacing = _fit_text_to_box(
            draw=draw,
            image_font_module=ImageFont,
            text=clean_text,
            max_width=max_w,
            max_height=max_h,
        )
        bbox = draw.multiline_textbbox(
            (0, 0), wrapped, font=body_font, spacing=spacing, align="center"
        )
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = int((width - text_w) / 2)
        text_y = int((height - text_h) / 2)
        draw.multiline_text(
            (text_x, text_y),
            wrapped,
            fill=(5, 5, 5),
            font=body_font,
            spacing=spacing,
            align="center",
        )

        out_path = out_dir / f"slide_{i:02d}.png"
        image.save(out_path)
        paths.append(out_path)

    return paths


def _clean_text_for_slide(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"[\x00-\x1f\x7f]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _wrap_text(text: str, width: int = 70) -> str:
    words = text.split()
    lines: List[str] = []
    current: List[str] = []
    current_len = 0

    for word in words:
        extra = 1 if current else 0
        if current_len + len(word) + extra > width:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + extra

    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def _fit_text_to_box(draw, image_font_module, text: str, max_width: int, max_height: int):
    for size in (36, 34, 32, 30, 28, 26, 24, 22):
        font = _load_font(image_font_module, size=size)
        rough_width = max(36, int(max_width / max(size * 0.56, 1)))
        wrapped = _wrap_text(text, width=rough_width)
        spacing = max(8, int(size * 0.35))
        if _text_fits(draw, wrapped, font, spacing, max_width, max_height):
            return font, wrapped, spacing

        trimmed = _trim_to_fit(draw, wrapped, font, spacing, max_width, max_height)
        if trimmed:
            return font, trimmed, spacing

    fallback_font = _load_font(image_font_module, size=22)
    fallback = _trim_to_fit(
        draw, _wrap_text(text, width=42), fallback_font, 8, max_width, max_height
    )
    return fallback_font, (fallback or "Слишком много текста для слайда."), 8


def _text_fits(
    draw, text: str, font, spacing: int, max_width: int, max_height: int
) -> bool:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
    return (bbox[2] - bbox[0]) <= max_width and (bbox[3] - bbox[1]) <= max_height


def _trim_to_fit(
    draw, text: str, font, spacing: int, max_width: int, max_height: int
) -> str:
    if not text.strip():
        return ""
    lines = text.splitlines()
    while lines:
        candidate = "\n".join(lines)
        if _text_fits(draw, candidate, font, spacing, max_width, max_height):
            return candidate
        lines.pop()
    return ""


def _load_font(image_font_module, size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            try:
                return image_font_module.truetype(font_path, size=size)
            except Exception:
                continue
    return image_font_module.load_default()


def _build_concat_file(
    image_paths: List[Path],
    slide_texts: List[str],
    audio_duration: float,
    out_path: Path,
) -> Path:
    if not image_paths:
        raise RuntimeError("Нет изображений для сборки видео.")

    durations = _allocate_slide_durations(
        slide_texts=slide_texts,
        slides_count=len(image_paths),
        audio_duration=audio_duration,
    )
    lines: List[str] = []
    for idx, image in enumerate(image_paths):
        lines.append(f"file '{image.as_posix()}'")
        lines.append(f"duration {durations[idx]:.3f}")
    lines.append(f"file '{image_paths[-1].as_posix()}'")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _allocate_slide_durations(
    slide_texts: List[str], slides_count: int, audio_duration: float
) -> List[float]:
    if slides_count <= 0:
        return []
    if audio_duration <= 0:
        return [2.0] * slides_count

    weights: List[float] = []
    for i in range(slides_count):
        text = slide_texts[i] if i < len(slide_texts) else ""
        weights.append(max(1.0, float(len(text.strip()))))

    total_weight = sum(weights) or float(slides_count)
    return [audio_duration * (w / total_weight) for w in weights]


def _render_video_with_ffmpeg(
    concat_file: Path, audio_file: Path, output_path: Path, fps: int
) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-i", str(audio_file),
        "-map", "0:v:0", "-map", "1:a:0",
        "-r", str(fps),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-filter:a", "volume=2.0",
        "-movflags", "+faststart", "-shortest",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg завершился с ошибкой: {stderr or exc}") from exc


def _get_wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate <= 0:
            return 1.0
        return frames / float(rate)


def _clean_for_tts(text: str) -> str:
    cleaned = re.sub(r"[A-Za-z][A-Za-z0-9'/_-]*", " ", text)
    cleaned = re.sub(r"[^\w\s.,!?;:()«»—-]", " ", cleaned, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
