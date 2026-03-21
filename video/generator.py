from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List

from audio import PodcastGenerator
from common.remote_llm import RemoteLLMClient
from summury.summarizer import FormalSummarizer, SummaryConfig


@dataclass
class TextToVideoConfig:
    summary_template: str = "executive"
    max_sentences: int = 8
    width: int = 1280
    height: int = 720
    fps: int = 24
    speaker: str = "baya"


class TextToVideoGenerator:
    """Генерация видео из текста: summary -> озвучка -> слайды -> mp4."""

    def __init__(self, remote_llm_client: RemoteLLMClient):
        self.remote_llm_client = remote_llm_client
        self.summarizer = FormalSummarizer(remote_llm_client=remote_llm_client)
        self.podcast_generator = PodcastGenerator(remote_llm_client=remote_llm_client)

    def generate(self, source_text: str, topic: str, output_path: Path, config: TextToVideoConfig | None = None) -> Path:
        cfg = config or TextToVideoConfig()

        if not source_text.strip():
            raise RuntimeError("Пустой входной текст: невозможно собрать видео.")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("Не найден ffmpeg. Установите ffmpeg и повторите запуск.")

        summary = self.summarizer.summarize(
            text=source_text,
            topic=topic,
            config=SummaryConfig(max_sentences=cfg.max_sentences, template=cfg.summary_template),
        )

        slides = self._split_into_slides(summary)
        if not slides:
            slides = [summary.strip() or "Нет данных для отображения"]
        narration = self._build_narration_from_slides(slides)
        with tempfile.TemporaryDirectory(prefix="text_to_video_") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            audio_path = tmp_dir / "narration.wav"
            slides_dir = tmp_dir / "slides"
            slides_dir.mkdir(parents=True, exist_ok=True)

            ok = self._try_save_audio_with_fallbacks(
                dialogue=narration,
                output_path=audio_path,
                preferred_speaker=cfg.speaker,
            )
            if not ok:
                load_error = self.podcast_generator.get_silero_load_error()
                details = f" Детали загрузки Silero: {load_error}" if load_error else ""
                raise RuntimeError(
                    "Не удалось сгенерировать аудио даже после fallback-режимов. "
                    "Попробуйте указать --speaker xenia или --speaker aidar."
                    + details
                )

            audio_duration = self._get_wav_duration_sec(audio_path)

            image_paths = self._render_slides(
                slides=slides,
                out_dir=slides_dir,
                width=cfg.width,
                height=cfg.height,
            )
            concat_file = self._build_concat_file(
                image_paths=image_paths,
                slide_texts=slides,
                audio_duration=audio_duration,
                out_path=tmp_dir / "slides.txt",
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._render_video_with_ffmpeg(
                concat_file=concat_file,
                audio_file=audio_path,
                output_path=output_path,
                fps=cfg.fps,
            )

        return output_path

    @staticmethod
    def _build_narration_from_slides(slides: List[str]) -> str:
        clean_slides = [s.strip() for s in slides if s.strip()]
        if not clean_slides:
            return "Ведущий 1: Краткое изложение документа недоступно."
        spoken_slides = [TextToVideoGenerator._transliterate_latin_words(s) for s in clean_slides]
        return "\n".join(f"Ведущий 1: {s}" for s in spoken_slides)

    def _try_save_audio_with_fallbacks(self, dialogue: str, output_path: Path, preferred_speaker: str) -> bool:
        # В некоторых окружениях отдельные голоса или слишком длинные реплики ломают TTS.
        candidate_speakers = [preferred_speaker, "xenia", "kseniya", "aidar", "baya", "eugene"]
        deduped_speakers: List[str] = []
        for speaker in candidate_speakers:
            normalized = speaker.strip()
            if normalized and normalized not in deduped_speakers:
                deduped_speakers.append(normalized)

        dialogues = [dialogue, self._shorten_dialogue(dialogue, max_chars=140)]
        for current_dialogue in dialogues:
            for speaker in deduped_speakers:
                ok = self.podcast_generator.save_audio(
                    dialogue=current_dialogue,
                    output_path=output_path,
                    pace="normal",
                    tts_model="silero",
                    silero_speaker_1=speaker,
                    silero_speaker_2=speaker,
                )
                if ok:
                    return True
        return False

    @staticmethod
    def _shorten_dialogue(dialogue: str, max_chars: int) -> str:
        lines = [line.strip() for line in dialogue.splitlines() if line.strip()]
        short_lines: List[str] = []
        for line in lines:
            prefix = "Ведущий 1: "
            text = line
            if line.startswith("Ведущий 1:"):
                text = line.replace("Ведущий 1:", "", 1).strip()
            elif line.startswith("Ведущий 2:"):
                text = line.replace("Ведущий 2:", "", 1).strip()
            pieces = TextToVideoGenerator._chunk_sentences([text], max_chars=max_chars)
            for piece in pieces:
                if piece.strip():
                    short_lines.append(prefix + piece.strip())
        return "\n".join(short_lines)

    @staticmethod
    def _chunk_sentences(sentences: List[str], max_chars: int) -> List[str]:
        chunks: List[str] = []
        current = ""
        for sentence in sentences:
            clean = sentence.strip()
            if not clean:
                continue
            if len(clean) > max_chars:
                # Режем длинный хвост на безопасные фрагменты.
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

    @staticmethod
    def _split_into_slides(summary: str) -> List[str]:
        lines = [line.strip() for line in summary.splitlines() if line.strip()]
        points: List[str] = []

        if len(lines) >= 2:
            for line in lines:
                cleaned = TextToVideoGenerator._normalize_slide_point(line)
                if cleaned:
                    points.append(cleaned)
        else:
            parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
            for part in parts:
                cleaned = TextToVideoGenerator._normalize_slide_point(part)
                if cleaned:
                    points.append(cleaned)

        if not points:
            cleaned = TextToVideoGenerator._normalize_slide_point(summary)
            points = [cleaned] if cleaned else []

        points = TextToVideoGenerator._expand_points_for_readability(points, max_chars=190)

        # Один пункт на слайд.
        return points[:12]

    @staticmethod
    def _render_slides(slides: List[str], out_dir: Path, width: int, height: int) -> List[Path]:
        try:
            from PIL import Image, ImageDraw, ImageFont  # type: ignore
        except Exception as exc:
            raise RuntimeError("Не хватает Pillow для генерации изображений. Установите: pip install pillow") from exc

        paths: List[Path] = []
        for i, text in enumerate(slides, start=1):
            _ = i
            image = Image.new("RGB", (width, height), color=(185, 230, 178))
            draw = ImageDraw.Draw(image)

            # Черная внешняя рамка и светлый текстовый блок для лучшей читаемости.
            draw.rectangle([(30, 30), (width - 30, height - 30)], outline=(0, 0, 0), width=6)
            draw.rectangle([(80, 90), (width - 80, height - 90)], fill=(235, 249, 230), outline=(0, 0, 0), width=4)

            clean_text = TextToVideoGenerator._clean_text_for_slide(text)
            max_w = width - 240
            max_h = height - 280
            body_font, wrapped, spacing = TextToVideoGenerator._fit_text_to_box(
                draw=draw,
                image_font_module=ImageFont,
                text=clean_text,
                max_width=max_w,
                max_height=max_h,
            )
            bbox = draw.multiline_textbbox((0, 0), wrapped, font=body_font, spacing=spacing, align="center")
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_x = int((width - text_w) / 2)
            text_y = int((height - text_h) / 2)
            draw.multiline_text((text_x, text_y), wrapped, fill=(5, 5, 5), font=body_font, spacing=spacing, align="center")

            out_path = out_dir / f"slide_{i:02d}.png"
            image.save(out_path)
            paths.append(out_path)

        return paths

    @staticmethod
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

    @staticmethod
    def _fit_text_to_box(draw, image_font_module, text: str, max_width: int, max_height: int):
        for size in (36, 34, 32, 30, 28, 26, 24, 22):
            font = TextToVideoGenerator._load_font(image_font_module, size=size)
            rough_width = max(36, int(max_width / max(size * 0.56, 1)))
            wrapped = TextToVideoGenerator._wrap_text(text, width=rough_width)
            spacing = max(8, int(size * 0.35))
            if TextToVideoGenerator._text_fits(draw, wrapped, font, spacing, max_width, max_height):
                return font, wrapped, spacing

            trimmed = TextToVideoGenerator._trim_to_fit(
                draw=draw,
                text=wrapped,
                font=font,
                spacing=spacing,
                max_width=max_width,
                max_height=max_height,
            )
            if trimmed:
                return font, trimmed, spacing

        fallback_font = TextToVideoGenerator._load_font(image_font_module, size=22)
        fallback = TextToVideoGenerator._trim_to_fit(
            draw=draw,
            text=TextToVideoGenerator._wrap_text(text, width=42),
            font=fallback_font,
            spacing=8,
            max_width=max_width,
            max_height=max_height,
        )
        return fallback_font, (fallback or "Слишком много текста для слайда."), 8

    @staticmethod
    def _text_fits(draw, text: str, font, spacing: int, max_width: int, max_height: int) -> bool:
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width <= max_width and height <= max_height

    @staticmethod
    def _trim_to_fit(draw, text: str, font, spacing: int, max_width: int, max_height: int) -> str:
        if not text.strip():
            return ""
        lines = text.splitlines()
        if not lines:
            return ""

        while lines:
            candidate = "\n".join(lines)
            if TextToVideoGenerator._text_fits(draw, candidate, font, spacing, max_width, max_height):
                return candidate
            lines.pop()

        return ""

    @staticmethod
    def _build_concat_file(image_paths: List[Path], slide_texts: List[str], audio_duration: float, out_path: Path) -> Path:
        if not image_paths:
            raise RuntimeError("Нет изображений для сборки видео.")

        durations = TextToVideoGenerator._allocate_slide_durations(
            slide_texts=slide_texts,
            slides_count=len(image_paths),
            audio_duration=audio_duration,
        )
        lines: List[str] = []

        for idx, image in enumerate(image_paths):
            lines.append(f"file '{image.as_posix()}'")
            lines.append(f"duration {durations[idx]:.3f}")

        # Для ffmpeg concat последний кадр нужно продублировать.
        lines.append(f"file '{image_paths[-1].as_posix()}'")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    @staticmethod
    def _render_video_with_ffmpeg(concat_file: Path, audio_file: Path, output_path: Path, fps: int) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-i",
            str(audio_file),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-filter:a",
            "volume=2.0",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(f"ffmpeg завершился с ошибкой: {stderr or exc}") from exc

    @staticmethod
    def _get_wav_duration_sec(path: Path) -> float:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return 1.0
            return frames / float(rate)

    @staticmethod
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

    @staticmethod
    def _clean_text_for_slide(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        normalized = re.sub(r"[\x00-\x1f\x7f]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _normalize_slide_point(text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""

        # Убираем markdown-выделения и префиксы списков/нумерации.
        cleaned = re.sub(r"[*_`#]+", " ", cleaned)
        cleaned = re.sub(r"^\s*[-–—•]+\s*", "", cleaned)
        cleaned = re.sub(r"^\s*\d+\s*[.)-:]\s*", "", cleaned)
        cleaned = re.sub(r"^\s*[a-zа-я]\s*[.)-:]\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _transliterate_latin_words(text: str) -> str:
        def repl(match) -> str:
            token = match.group(0)
            return TextToVideoGenerator._transliterate_token(token)

        return re.sub(r"[A-Za-z][A-Za-z0-9'/-]*", repl, text)

    @staticmethod
    def _transliterate_token(token: str) -> str:
        core = re.sub(r"[^A-Za-z]", "", token)
        if not core:
            return token

        if core.isupper() and len(core) <= 6:
            return TextToVideoGenerator._transliterate_acronym(core)

        return TextToVideoGenerator._transliterate_word(core.lower())

    @staticmethod
    def _transliterate_acronym(word: str) -> str:
        letter_map = {
            "A": "эй",
            "B": "би",
            "C": "си",
            "D": "ди",
            "E": "и",
            "F": "эф",
            "G": "джи",
            "H": "эйч",
            "I": "ай",
            "J": "джей",
            "K": "кей",
            "L": "эл",
            "M": "эм",
            "N": "эн",
            "O": "оу",
            "P": "пи",
            "Q": "кью",
            "R": "ар",
            "S": "эс",
            "T": "ти",
            "U": "ю",
            "V": "ви",
            "W": "дабл-ю",
            "X": "икс",
            "Y": "уай",
            "Z": "зед",
        }
        return " ".join(letter_map.get(ch, ch.lower()) for ch in word)

    @staticmethod
    def _transliterate_word(word: str) -> str:
        rules = [
            ("shch", "щ"),
            ("sch", "щ"),
            ("yo", "ё"),
            ("zh", "ж"),
            ("kh", "х"),
            ("ts", "ц"),
            ("ch", "ч"),
            ("sh", "ш"),
            ("yu", "ю"),
            ("ya", "я"),
            ("ye", "е"),
            ("ju", "ю"),
            ("ja", "я"),
            ("ph", "ф"),
            ("th", "т"),
            ("qu", "кв"),
            ("ck", "к"),
        ]
        single = {
            "a": "а",
            "b": "б",
            "c": "к",
            "d": "д",
            "e": "е",
            "f": "ф",
            "g": "г",
            "h": "х",
            "i": "и",
            "j": "дж",
            "k": "к",
            "l": "л",
            "m": "м",
            "n": "н",
            "o": "о",
            "p": "п",
            "q": "к",
            "r": "р",
            "s": "с",
            "t": "т",
            "u": "у",
            "v": "в",
            "w": "в",
            "x": "кс",
            "y": "й",
            "z": "з",
        }

        result: List[str] = []
        i = 0
        while i < len(word):
            applied = False
            for src, dst in rules:
                if word.startswith(src, i):
                    result.append(dst)
                    i += len(src)
                    applied = True
                    break
            if applied:
                continue

            ch = word[i]
            result.append(single.get(ch, ch))
            i += 1

        out = "".join(result)
        out = re.sub(r"йе", "е", out)
        out = re.sub(r"ий$", "и", out)
        return out

    @staticmethod
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
                expanded.extend(TextToVideoGenerator._chunk_sentences([text], max_chars=max_chars))
                continue

            chunks = TextToVideoGenerator._chunk_sentences(parts, max_chars=max_chars)
            for chunk in chunks:
                if chunk.strip():
                    expanded.append(chunk.strip())
        return expanded

    @staticmethod
    def _allocate_slide_durations(slide_texts: List[str], slides_count: int, audio_duration: float) -> List[float]:
        if slides_count <= 0:
            return []
        if audio_duration <= 0:
            return [2.0] * slides_count

        weights: List[float] = []
        for i in range(slides_count):
            text = slide_texts[i] if i < len(slide_texts) else ""
            weights.append(max(1.0, float(len(text.strip()))))

        total_weight = sum(weights) or float(slides_count)
        durations = [audio_duration * (w / total_weight) for w in weights]
        return durations
