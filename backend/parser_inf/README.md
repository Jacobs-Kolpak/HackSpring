# parser_inf

Модуль продвинутого парсинга веб-источников для последующей загрузки в RAG как обычных `.txt` файлов.

## Что умеет сейчас

- Краулит сайт (`BFS`) с лимитами по `max_pages` и `max_depth`.
- Фильтрует ссылки (пропускает binary/doc/media URL).
- Учитывает `robots.txt` (опционально).
- Делает retry/backoff на временных HTTP-ошибках.
- Выбирает лучший текст из нескольких open-source экстракторов через quality scoring.
- Опционально включает browser fallback (`playwright`) для JS-heavy страниц.
- Опционально включает нейро-ранжирование контент-блоков через `transformers` zero-shot.

## Extraction pipeline

`_pick_best_extraction(...)` запускает несколько стратегий и выбирает лучшую по качеству:

1. `trafilatura`
2. `readability-lxml`
3. `goose3`
4. `newspaper3k`
5. `transformers` block ranker (опционально)
6. `BeautifulSoup` raw fallback

Для каждой стратегии считается `quality_score` (длина, структура предложений, лексическое разнообразие, шум навигации).  
Берется вариант с максимальным score.

## API модуля

```python
from backend.parser_inf import (
    ParserConfig,
    parse_website,
    summarize_extractors,
    write_pages_to_txt,
)

cfg = ParserConfig(
    max_pages=12,
    max_depth=2,
    use_transformer_ranker=True,
    enable_browser_fallback=True,
)

pages = parse_website("https://example.com", config=cfg)
stats = summarize_extractors(pages)
paths = write_pages_to_txt(pages, folder=tmp_path)
```

`ParsedPage` содержит:
- `url`
- `title`
- `text`
- `extractor` (какой метод победил)
- `quality_score`
- `depth`
- `meta` (например `status_code`, `content_type`)

## Рекомендуемые зависимости

Базовые:
- `httpx`
- `beautifulsoup4`
- `trafilatura`
- `readability-lxml`

Расширенные:
- `goose3`
- `newspaper3k`
- `playwright` (для JS-render fallback)
- `transformers` + `torch` (нейро-ранжирование блоков)

## Интеграция в RAG (дальше)

1. Оркестратор получает URL и параметры crawl.
2. Вызывает `parse_website(...)`.
3. Пишет результаты через `write_pages_to_txt(...)` во временную директорию.
4. Передает эти пути в существующий `rag_service.ingest(paths=[...])`.
5. Сохраняет в metadata источник `source_kind=web`, `source_url`, `extractor`, `quality_score`.

## Ограничения

- Для сайтов с жестким anti-bot/captcha без отдельной инфраструктуры гарантий нет.
- Browser fallback повышает покрытие, но увеличивает время и стоимость.
- Нейро-режим (`transformers`) требует скачивания модели и заметно тяжелее CPU/RAM.
