from __future__ import annotations

import argparse
import os
from pathlib import Path

from audio import PodcastConfig, PodcastGenerator
from common.document_loader import load_text_from_document
from common.remote_llm import RemoteLLMClient, RemoteLLMConfig
from summury.summarizer import FormalSummarizer, SummaryConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Генерация формализованного summary и аудиопересказа через удаленный LLM API"
    )
    parser.add_argument("--task", choices=["summary", "audio"], default="summary", help="Режим работы")
    parser.add_argument("--input", type=Path, help="Путь к исходному файлу (.txt/.pdf/.docx/.dox)")
    parser.add_argument("--text", type=str, help="Исходный текст напрямую")
    parser.add_argument("--topic", type=str, default="Без названия", help="Тема материала")
    parser.add_argument("--llm-provider", choices=["remote"], default="remote", help="Источник LLM (только remote)")
    parser.add_argument("--remote-host", type=str, default=None, help="Хост удаленного AI-сервиса")
    parser.add_argument("--llm-port", type=int, default=None, help="Порт LLM-сервиса")
    parser.add_argument("--api-key", type=str, default=os.getenv("HACKAI_API_KEY", ""), help="API ключ (или env HACKAI_API_KEY)")
    parser.add_argument("--remote-model", type=str, default="", help="Имя модели для удаленного LLM (опционально)")
    parser.add_argument(
        "--remote-path",
        type=str,
        default="",
        help="Явный путь endpoint LLM API (например /api/v1/chat/completions)",
    )

    # Summary
    parser.add_argument("--max-sentences", type=int, default=5, help="Максимум предложений в кратком саммари")
    parser.add_argument("--output", type=Path, help="Куда сохранить итоговый summary (txt)")

    # Audio
    parser.add_argument(
        "--tone",
        type=str,
        default="scientific",
        help="Тональность диалога: scientific|popular или научный|популярный",
    )
    parser.add_argument(
        "--pace",
        type=str,
        default="normal",
        help="Темп речи: slow|normal|fast или медленно|нормально|быстро",
    )
    parser.add_argument(
        "--audio-max-new-tokens",
        type=int,
        default=320,
        help="Максимум новых токенов для аудио-диалога",
    )
    parser.add_argument("--dialog-output", type=Path, default=Path("audio/dialogue.txt"), help="Куда сохранить текст диалога")
    parser.add_argument("--audio-output", type=Path, default=Path("audio/podcast.wav"), help="Куда сохранить audio wav")

    return parser.parse_args()


def read_source(args: argparse.Namespace) -> str:
    if args.text:
        return args.text

    if args.input and args.input.exists():
        try:
            text = load_text_from_document(args.input)
        except Exception as exc:
            raise SystemExit(f"Ошибка чтения входного файла {args.input}: {exc}") from exc

        if not text.strip():
            raise SystemExit(f"В файле {args.input} не найден текст для обработки")
        return text

    raise SystemExit("Укажите --text или корректный --input <file.txt|file.pdf|file.docx|file.dox>")


def run_summary(args: argparse.Namespace, source_text: str) -> None:
    remote_client = _build_remote_client(args)
    summarizer = FormalSummarizer(remote_llm_client=remote_client)
    result = summarizer.summarize(
        text=source_text,
        topic=args.topic,
        config=SummaryConfig(max_sentences=args.max_sentences),
    )

    if args.output:
        args.output.write_text(result, encoding="utf-8")
        print(f"Summary сохранен: {args.output}")
    else:
        print(result)


def run_audio(args: argparse.Namespace, source_text: str) -> None:
    tone = _normalize_tone(args.tone)
    pace = _normalize_pace(args.pace)
    remote_client = _build_remote_client(args)
    generator = PodcastGenerator(remote_llm_client=remote_client)
    try:
        dialogue = generator.generate_dialogue(
            text=source_text,
            topic=args.topic,
            config=PodcastConfig(
                tone=tone,
                pace=pace,
                max_new_tokens=args.audio_max_new_tokens,
            ),
        )
    except RuntimeError as exc:
        raise SystemExit(f"Ошибка генерации подкаста: {exc}") from exc

    args.dialog_output.parent.mkdir(parents=True, exist_ok=True)
    args.dialog_output.write_text(dialogue, encoding="utf-8")

    ok = generator.save_audio(dialogue, output_path=args.audio_output, pace=pace)

    print(f"Диалог сохранен: {args.dialog_output}")
    if ok:
        print(f"Аудио сохранено: {args.audio_output}")
    else:
        print("Аудио не создано (вероятно, не установлен pyttsx3/espeak). Текст диалога готов.")


def _normalize_tone(value: str) -> str:
    tone = value.strip().lower()
    mapping = {
        "scientific": "scientific",
        "science": "scientific",
        "научный": "scientific",
        "научно": "scientific",
        "popular": "popular",
        "популярный": "popular",
        "простыми": "popular",
    }
    if tone not in mapping:
        allowed = "scientific|popular или научный|популярный"
        raise SystemExit(f"Некорректный --tone: {value}. Используйте {allowed}.")
    return mapping[tone]


def _normalize_pace(value: str) -> str:
    pace = value.strip().lower()
    mapping = {
        "slow": "slow",
        "normal": "normal",
        "fast": "fast",
        "медленно": "slow",
        "нормально": "normal",
        "быстро": "fast",
    }
    if pace not in mapping:
        allowed = "slow|normal|fast или медленно|нормально|быстро"
        raise SystemExit(f"Некорректный --pace: {value}. Используйте {allowed}.")
    return mapping[pace]


def _build_remote_client(args: argparse.Namespace) -> RemoteLLMClient:
    api_key = (args.api_key or "").strip()
    if not api_key:
        raise SystemExit("Укажите --api-key или переменную HACKAI_API_KEY для remote-режима.")

    base_url = f"{args.remote_host.rstrip('/')}:{args.llm_port}"
    return RemoteLLMClient(
        RemoteLLMConfig(
            base_url=base_url,
            api_key=api_key,
            model=args.remote_model.strip(),
            endpoint_path=args.remote_path.strip(),
        )
    )


def main() -> None:
    args = parse_args()
    source_text = read_source(args)

    if args.task == "audio":
        run_audio(args, source_text)
    else:
        run_summary(args, source_text)


if __name__ == "__main__":
    main()
