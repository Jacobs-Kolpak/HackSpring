from __future__ import annotations

import argparse
from pathlib import Path

from audio import PodcastConfig, PodcastGenerator
from common.document_loader import load_text_from_document
from summury.summarizer import FormalSummarizer, SummaryConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Генерация формализованного summary и аудиопересказа"
    )
    parser.add_argument("--task", choices=["summary", "audio"], default="summary", help="Режим работы")
    parser.add_argument("--input", type=Path, help="Путь к исходному файлу (.txt/.pdf/.docx/.dox)")
    parser.add_argument("--text", type=str, help="Исходный текст напрямую")
    parser.add_argument("--topic", type=str, default="Без названия", help="Тема материала")

    # Summary
    parser.add_argument("--max-sentences", type=int, default=5, help="Максимум предложений в кратком саммари")
    parser.add_argument("--model", type=str, default="IlyaGusev/rut5_base_sum_gazeta", help="Имя модели transformers")
    parser.add_argument("--output", type=Path, help="Куда сохранить итоговый summary (txt)")

    # Audio
    parser.add_argument("--tone", choices=["scientific", "popular"], default="scientific", help="Тональность диалога")
    parser.add_argument("--pace", choices=["slow", "normal", "fast"], default="normal", help="Темп речи")
    parser.add_argument("--audio-model", type=str, default="google/flan-t5-small", help="Модель для генерации диалога")
    parser.add_argument("--audio-max-new-tokens", type=int, default=220, help="Максимум новых токенов для аудио-диалога")
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
    summarizer = FormalSummarizer(model_name=args.model)
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
    generator = PodcastGenerator(model_name=args.audio_model)
    dialogue = generator.generate_dialogue(
        text=source_text,
        topic=args.topic,
        config=PodcastConfig(
            tone=args.tone,
            pace=args.pace,
            model_name=args.audio_model,
            max_new_tokens=args.audio_max_new_tokens,
        ),
    )

    args.dialog_output.parent.mkdir(parents=True, exist_ok=True)
    args.dialog_output.write_text(dialogue, encoding="utf-8")

    ok = generator.save_audio(dialogue, output_path=args.audio_output, pace=args.pace)

    print(f"Диалог сохранен: {args.dialog_output}")
    if ok:
        print(f"Аудио сохранено: {args.audio_output}")
    else:
        print("Аудио не создано (вероятно, не установлен pyttsx3/espeak). Текст диалога готов.")


def main() -> None:
    args = parse_args()
    source_text = read_source(args)

    if args.task == "audio":
        run_audio(args, source_text)
    else:
        run_summary(args, source_text)


if __name__ == "__main__":
    main()
