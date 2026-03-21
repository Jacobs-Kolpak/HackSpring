from __future__ import annotations

import argparse
import os
from pathlib import Path

from common.document_loader import load_text_from_document
from common.remote_llm import RemoteLLMClient, RemoteLLMConfig
from video import TextToVideoConfig, TextToVideoGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Генерация видео из текстового файла (отдельный запуск, без main.py)")
    parser.add_argument("--input", type=Path, required=True, help="Путь к исходному файлу (.txt/.pdf/.docx/.dox)")
    parser.add_argument("--topic", type=str, default="Без названия", help="Тема материала")
    parser.add_argument("--output", type=Path, default=Path("data/video.mp4"), help="Куда сохранить итоговое видео")

    parser.add_argument("--remote-host", type=str, required=True, help="Хост удаленного AI-сервиса")
    parser.add_argument("--llm-port", type=int, required=True, help="Порт LLM-сервиса")
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("HACKAI_API_KEY", ""),
        help="API ключ (или env HACKAI_API_KEY)",
    )
    parser.add_argument("--remote-model", type=str, default="", help="Имя модели (опционально)")
    parser.add_argument("--remote-path", type=str, default="", help="Явный endpoint API, например /api/v1/chat/completions")

    parser.add_argument(
        "--summary-template",
        type=str,
        default="executive",
        help="Шаблон summary: executive|detailed|summary или исполнительное резюме|детализированный|саммари",
    )
    parser.add_argument("--max-sentences", type=int, default=8, help="Количество предложений в сценарии")
    parser.add_argument("--speaker", type=str, default="baya", help="Голос Silero")
    parser.add_argument("--width", type=int, default=1280, help="Ширина видео")
    parser.add_argument("--height", type=int, default=720, help="Высота видео")
    parser.add_argument("--fps", type=int, default=24, help="Кадров в секунду")

    return parser.parse_args()


def normalize_summary_template(value: str) -> str:
    template = value.strip().lower()
    mapping = {
        "executive": "executive",
        "исполнительное резюме": "executive",
        "detailed": "detailed",
        "детализированный": "detailed",
        "summary": "summary",
        "саммари": "summary",
    }
    if template not in mapping:
        allowed = "executive|detailed|summary или исполнительное резюме|детализированный|саммари"
        raise SystemExit(f"Некорректный --summary-template: {value}. Используйте {allowed}.")
    return mapping[template]


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Файл не найден: {args.input}")

    api_key = (args.api_key or "").strip()
    if not api_key:
        raise SystemExit("Укажите --api-key или переменную HACKAI_API_KEY")

    try:
        source_text = load_text_from_document(args.input)
    except Exception as exc:
        raise SystemExit(f"Ошибка чтения файла {args.input}: {exc}") from exc

    if not source_text.strip():
        raise SystemExit(f"В файле {args.input} не найден текст")

    base_url = f"{args.remote_host.rstrip('/')}:{args.llm_port}"
    remote_client = RemoteLLMClient(
        RemoteLLMConfig(
            base_url=base_url,
            api_key=api_key,
            model=args.remote_model.strip(),
            endpoint_path=args.remote_path.strip(),
        )
    )

    generator = TextToVideoGenerator(remote_llm_client=remote_client)
    output = generator.generate(
        source_text=source_text,
        topic=args.topic,
        output_path=args.output,
        config=TextToVideoConfig(
            summary_template=normalize_summary_template(args.summary_template),
            max_sentences=args.max_sentences,
            width=max(640, args.width),
            height=max(360, args.height),
            fps=max(12, args.fps),
            speaker=args.speaker,
        ),
    )

    print(f"Видео сохранено: {output}")


if __name__ == "__main__":
    main()
