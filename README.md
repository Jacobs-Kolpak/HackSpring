# HackSpring — AI-платформа для исследований

> Альтернатива NotebookLM с RAG, mind maps, audio summaries, flashcards, презентациями.
> Хакатон Центр Инвест 2026.
>
> **Команда:** Данил (RAG, Mind Map, Flashcards, Web Parser), Миша (Summary, Podcast + TTS, Table Extraction), Яковкин (Презентации)

---

## Быстрый старт

```bash
cp .env.example .env
docker compose up --build -d
```

Приложение: `http://localhost:8000`
Документация (Swagger): `http://localhost:8000/docs`

---

## Архитектура

```
backend/
├── core/           # config, database, security
├── utils/          # document_reader, chunker, embeddings, llm, web_parser
├── services/       # rag, mindmap, summary, podcast, flashcards, presentation, table, parser
├── routers/        # auth, rag, mindmap, summary, podcast, flashcards, presentation, table, parser
└── schemas.py      # Pydantic-модели
```

---

## API Endpoints

### Health

| Метод | URL | Описание |
|-------|-----|----------|
| GET | `/` | Информация о приложении |
| GET | `/health` | Health check |

```bash
curl http://localhost:8000/
curl http://localhost:8000/health
```

---

### Auth — `/api/jacobs/auth`

| Метод | URL | Auth | Описание |
|-------|-----|------|----------|
| POST | `/api/jacobs/auth/register` | — | Регистрация |
| POST | `/api/jacobs/auth/login` | — | Логин (получение токенов) |
| POST | `/api/jacobs/auth/refresh` | Bearer (refresh) | Обновление access token |
| GET | `/api/jacobs/auth/me` | Bearer (access) | Текущий пользователь |
| POST | `/api/jacobs/auth/logout` | — | Выход |
| GET | `/api/jacobs/auth/status` | — | Статус системы авторизации |

#### Регистрация

```bash
curl -X POST http://localhost:8000/api/jacobs/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "username": "myuser",
    "password": "mypassword123",
    "role": "researcher"
  }'
```

Роли: `researcher`, `government`, `student` (по умолчанию `student`).

**Ответ:**
```json
{
  "email": "user@example.com",
  "username": "myuser",
  "id": 1,
  "role": "researcher",
  "is_active": true,
  "created_at": "2026-03-21T12:00:00",
  "updated_at": null
}
```

#### Логин

```bash
curl -X POST http://localhost:8000/api/jacobs/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "mypassword123"
  }'
```

**Ответ:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

#### Текущий пользователь

```bash
# Сохраняем токен в переменную (удобно):
TOKEN=$(curl -s -X POST http://localhost:8000/api/jacobs/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"mypassword123"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Используем:
curl http://localhost:8000/api/jacobs/auth/me \
  -H "Authorization: Bearer $TOKEN"
```

**Ответ:**
```json
{
  "user": {
    "email": "user@example.com",
    "username": "myuser",
    "id": 1,
    "role": "researcher",
    "is_active": true,
    "created_at": "2026-03-21T12:00:00",
    "updated_at": null
  },
  "is_authenticated": true
}
```

#### Обновить токен

```bash
curl -X POST http://localhost:8000/api/jacobs/auth/refresh \
  -H "Authorization: Bearer $REFRESH_TOKEN"
```

#### Выход

```bash
curl -X POST http://localhost:8000/api/jacobs/auth/logout
```

#### Статус системы

```bash
curl http://localhost:8000/api/jacobs/auth/status
```

---

### RAG — `/api/jacobs/rag`

| Метод | URL | Описание |
|-------|-----|----------|
| POST | `/api/jacobs/rag/ingest` | Загрузка и индексация документов |
| POST | `/api/jacobs/rag/retrieve` | Поиск релевантных чанков |
| POST | `/api/jacobs/rag/ask` | Поиск + ответ LLM |

#### Индексация документов

```bash
curl -X POST http://localhost:8000/api/jacobs/rag/ingest \
  -F "files=@document.pdf" \
  -F "files=@notes.txt" \
  -F "collection=docs_ci" \
  -F "chunk_size=900" \
  -F "chunk_overlap=180"
```

Поддерживаемые форматы: `.pdf`, `.docx`, `.txt`

Параметры (все кроме `files` — опциональные, берутся из `.env`):
- `files` — файлы для загрузки (multipart)
- `collection` — название коллекции в Qdrant
- `chunk_size` — размер чанка (символы)
- `chunk_overlap` — перекрытие между чанками

**Ответ:**
```json
{
  "indexed_files": 2,
  "inserted_chunks": 47,
  "collection": "docs_ci"
}
```

#### Поиск чанков

```bash
curl -X POST http://localhost:8000/api/jacobs/rag/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Какие инвестиционные программы доступны?",
    "collection": "docs_ci",
    "top_k": 5,
    "min_score": 0.2,
    "dense_weight": 0.7
  }'
```

Параметры (все кроме `query` — опциональные):
- `query` — текст запроса
- `collection` — коллекция для поиска
- `top_k` — количество результатов (1–50)
- `fetch_k` — кандидатов для rerank (1–200)
- `min_score` — минимальный score (0.0–1.0)
- `dense_weight` — вес dense vs BM25 (0.0–1.0)
- `source_name` — фильтр по имени файла

**Ответ:**
```json
{
  "query": "Какие инвестиционные программы доступны?",
  "returned": 3,
  "results": [
    {
      "rank": 1,
      "score": 0.85,
      "dense_score": 0.92,
      "bm25_score": 0.71,
      "api_rerank_score": 0.0,
      "chunk_id": "abc-123",
      "source_name": "document.pdf",
      "source_path": "/tmp/document.pdf",
      "chunk_index": 5,
      "text": "Текст релевантного чанка..."
    }
  ]
}
```

#### Вопрос-ответ (RAG + LLM)

```bash
curl -X POST http://localhost:8000/api/jacobs/rag/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Какие инвестиционные программы доступны?",
    "collection": "docs_ci",
    "top_k": 5,
    "min_score": 0.2,
    "model": "gpt-oss-20b"
  }'
```

Параметры: все из `/retrieve` + `model` (опционально, по умолчанию из `.env`).

**Ответ:**
```json
{
  "query": "Какие инвестиционные программы доступны?",
  "answer": "Согласно документам, доступны следующие программы...",
  "model": "gpt-oss-20b",
  "used_chunks": 5,
  "results": [...]
}
```

---

### Mind Map — `/api/jacobs/mindmap`

| Метод | URL | Описание |
|-------|-----|----------|
| POST | `/api/jacobs/mindmap/text` | Mind map из текста |
| POST | `/api/jacobs/mindmap/file` | Mind map из файла |
| POST | `/api/jacobs/mindmap/query` | Mind map из RAG-поиска |

#### Из текста

```bash
curl -X POST http://localhost:8000/api/jacobs/mindmap/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Центр Инвест — крупнейший банк юга России. Банк предоставляет кредиты, вклады и инвестиционные услуги. Кредитование малого бизнеса является приоритетным направлением.",
    "top_concepts": 14,
    "min_freq": 2,
    "min_edge": 1
  }'
```

Параметры:
- `text` — текст для анализа
- `top_concepts` — макс. кол-во концептов (1–100, по умолчанию 14)
- `min_freq` — минимальная частота слова (по умолчанию 2)
- `min_edge` — минимальный вес ребра (по умолчанию 1)

**Ответ:**
```json
{
  "nodes": [
    {
      "id": "банк",
      "label": "банк",
      "title": "банк: 3",
      "value": 3,
      "relevance": 0.95
    }
  ],
  "edges": [
    {
      "from": "банк",
      "to": "кредит",
      "value": 2,
      "title": "Совместные упоминания: 2"
    }
  ],
  "meta": {
    "source": "text",
    "nodes_count": 8,
    "edges_count": 12
  }
}
```

#### Из файла

```bash
curl -X POST http://localhost:8000/api/jacobs/mindmap/file \
  -F "file=@document.pdf" \
  -F "top_concepts=20" \
  -F "min_freq=2" \
  -F "min_edge=1"
```

#### Из RAG-запроса

```bash
curl -X POST http://localhost:8000/api/jacobs/mindmap/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "инвестиционные программы",
    "collection": "docs_ci",
    "top_k": 8,
    "min_score": 0.15,
    "dense_weight": 0.6,
    "top_concepts": 14,
    "min_freq": 2,
    "min_edge": 1
  }'
```

Параметры: `query` + параметры поиска + параметры mind map.

---

### Summary — `/api/jacobs/summary`

| Метод | URL | Описание |
|-------|-----|----------|
| POST | `/api/jacobs/summary/text` | Суммаризация из текста |
| POST | `/api/jacobs/summary/file` | Суммаризация из файла |
| POST | `/api/jacobs/summary/query` | Суммаризация из RAG-поиска |

#### Из текста

```bash
curl -X POST http://localhost:8000/api/jacobs/summary/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Длинный текст для суммаризации...",
    "topic": "Инвестиции",
    "max_sentences": 5
  }'
```

Параметры:
- `text` — текст для суммаризации
- `topic` — тема (по умолчанию "Без названия")
- `max_sentences` — макс. предложений (1–100, по умолчанию 10)
- `model` — модель LLM (опционально)
- `template` — кастомный промпт (опционально, см. ниже)
- `system_prompt` — кастомный системный промпт (опционально)

**Ответ:**
```json
{
  "summary": "Краткое содержание текста...",
  "source": "text",
  "model": "gpt-oss-20b",
  "meta": {}
}
```

#### Из файла

```bash
curl -X POST http://localhost:8000/api/jacobs/summary/file \
  -F "file=@document.pdf" \
  -F "topic=Финансы" \
  -F "max_sentences=10"
```

#### Из RAG-запроса

```bash
curl -X POST http://localhost:8000/api/jacobs/summary/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "инвестиционные программы",
    "collection": "docs_ci",
    "top_k": 8,
    "topic": "Инвестиции",
    "max_sentences": 10
  }'
```

#### Кастомный промпт

Фронтенд может отправить свой промпт целиком в поле `template`. Текст документа будет автоматически добавлен в конец.

```bash
curl -X POST http://localhost:8000/api/jacobs/summary/file \
  -F "file=@document.pdf" \
  -F "template=Выдели ключевые тезисы из этого текста в формате нумерованного списка." \
  -F "system_prompt=Ты — эксперт по анализу документов."
```

Два режима работы `template`:
1. **Простой промпт** — если в `template` нет `{text}`, текст документа добавляется автоматически в конец
2. **Шаблон с плейсхолдерами** — если `template` содержит `{text}`, используется как шаблон (`{topic}`, `{max_sentences}`, `{text}`)

---

### Podcast — `/api/jacobs/podcast`

| Метод | URL | Описание |
|-------|-----|----------|
| POST | `/api/jacobs/podcast/text` | Подкаст из текста |
| POST | `/api/jacobs/podcast/file` | Подкаст из файла |
| POST | `/api/jacobs/podcast/query` | Подкаст из RAG-поиска |
| GET | `/api/jacobs/podcast/audio/{filename}` | Скачать аудиофайл |

#### Из текста

```bash
curl -X POST http://localhost:8000/api/jacobs/podcast/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Текст для генерации подкаста...",
    "topic": "Финансы",
    "tone": "scientific",
    "pace": "normal"
  }'
```

Параметры:
- `text` — текст для подкаста
- `topic` — тема (по умолчанию "Без названия")
- `tone` — тональность: `scientific`|`everyday` (или `научный`|`повседневный`)
- `pace` — темп: `slow`|`normal`|`fast` (или `медленно`|`нормально`|`быстро`)
- `model` — модель LLM (опционально)

**Ответ:**
```json
{
  "dialogue": "Ведущий 1: Сегодня мы обсудим...\nВедущий 2: Да, это интересная тема...",
  "source": "text",
  "model": "gpt-oss-20b",
  "has_audio": true,
  "audio_url": "/api/jacobs/podcast/audio/podcast_a1b2c3d4e5f6.wav",
  "meta": {}
}
```

#### Из файла

```bash
curl -X POST http://localhost:8000/api/jacobs/podcast/file \
  -F "file=@document.pdf" \
  -F "topic=Финансы" \
  -F "tone=everyday" \
  -F "pace=normal"
```

#### Из RAG-запроса

```bash
curl -X POST http://localhost:8000/api/jacobs/podcast/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "инвестиционные программы",
    "collection": "docs_ci",
    "top_k": 8,
    "topic": "Инвестиции",
    "tone": "scientific",
    "pace": "normal"
  }'
```

#### Скачать аудио

```bash
curl -O http://localhost:8000/api/jacobs/podcast/audio/podcast_a1b2c3d4e5f6.wav
```

---

### Flashcards — `/api/jacobs/flashcards`

| Метод | URL | Описание |
|-------|-----|----------|
| POST | `/api/jacobs/flashcards/file` | Генерация карточек и тестов из файла |

#### Из файла

```bash
curl -X POST http://localhost:8000/api/jacobs/flashcards/file \
  -F "file=@document.pdf" \
  -F "topic=Инвестиции" \
  -F "cards_count=10" \
  -F "tests_count=5"
```

Параметры:
- `file` — файл (PDF/DOCX/TXT)
- `topic` — тема (по умолчанию "Без названия")
- `cards_count` — количество карточек (1–50, по умолчанию 10)
- `tests_count` — количество тестов (1–30, по умолчанию 5)

**Ответ:**
```json
{
  "topic": "Инвестиции",
  "flashcards": [
    {"question": "Что такое RAG?", "answer": "Retrieval-Augmented Generation..."}
  ],
  "tests": [
    {
      "question": "Какой метод используется для поиска?",
      "options": ["BM25", "TF-IDF", "Dense", "Все перечисленные"],
      "correct_index": 3,
      "explanation": "Используется гибридный подход..."
    }
  ],
  "model": "gpt-oss-20b",
  "meta": {"filename": "document.pdf"}
}
```

---

### Presentation — `/api/jacobs/presentation`

| Метод | URL | Описание |
|-------|-----|----------|
| POST | `/api/jacobs/presentation/generate` | Генерация PPTX из RAG-запроса |
| GET | `/api/jacobs/presentation/download/{filename}` | Скачать презентацию |

#### Генерация презентации

```bash
curl -X POST http://localhost:8000/api/jacobs/presentation/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Подготовка к собеседованию",
    "top_k": 8,
    "max_slides": 6
  }'
```

Параметры:
- `query` — запрос для поиска в RAG
- `collection` — коллекция Qdrant (опционально)
- `top_k` — кол-во чанков для контекста (по умолчанию 8)
- `max_slides` — макс. слайдов (1–20, по умолчанию 6)
- `model` — модель LLM (опционально)

**Ответ:**
```json
{
  "title": "Подготовка к онлайн-собеседованию",
  "slides_count": 4,
  "download_url": "/api/jacobs/presentation/download/presentation_20260321_083308.pptx",
  "meta": {
    "title": "...",
    "slides_count": 4,
    "path": "/app/data/presentations/presentation_20260321_083308.pptx"
  }
}
```

#### Скачать презентацию

```bash
curl -O http://localhost:8000/api/jacobs/presentation/download/presentation_20260321_083308.pptx
```

---

### Table — `/api/jacobs/table`

| Метод | URL | Описание |
|-------|-----|----------|
| POST | `/api/jacobs/table/text` | Таблица из текста |
| POST | `/api/jacobs/table/file` | Таблица из файла (JSON/CSV/XLSX) |

#### Из текста

```bash
curl -X POST http://localhost:8000/api/jacobs/table/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Москва - 12 млн жителей, Ростов-на-Дону - 1.1 млн, Краснодар - 1 млн",
    "hint": "города и население",
    "format": "json"
  }'
```

Параметры:
- `text` — текст для структурирования
- `hint` — подсказка для LLM (опционально)
- `format` — формат ответа: `json` | `csv` | `xlsx` (по умолчанию `json`)

**Ответ (format=json):**
```json
{
  "title": "Население городов",
  "columns": ["Город", "Население"],
  "rows": [
    ["Москва", "12 млн"],
    ["Ростов-на-Дону", "1.1 млн"],
    ["Краснодар", "1 млн"]
  ]
}
```

#### Из файла

```bash
curl -X POST http://localhost:8000/api/jacobs/table/file \
  -F "file=@document.pdf" \
  -F "hint=подсказка" \
  -F "format=json"
```

#### Скачать как CSV

```bash
curl -X POST http://localhost:8000/api/jacobs/table/file \
  -F "file=@document.pdf" \
  -F "format=csv" \
  -o table.csv
```

#### Скачать как XLSX

```bash
curl -X POST http://localhost:8000/api/jacobs/table/file \
  -F "file=@document.pdf" \
  -F "format=xlsx" \
  -o table.xlsx
```

---

### Parser — `/api/jacobs/parser`

| Метод | URL | Описание |
|-------|-----|----------|
| POST | `/api/jacobs/parser/parse` | Парсинг веб-страниц |
| POST | `/api/jacobs/parser/ingest` | Парсинг + загрузка в RAG |

#### Парсинг сайта

```bash
curl -X POST http://localhost:8000/api/jacobs/parser/parse \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "max_pages": 5,
    "max_depth": 1,
    "same_domain": true
  }'
```

Параметры:
- `url` — начальный URL для парсинга
- `max_pages` — макс. страниц (по умолчанию 10)
- `max_depth` — глубина обхода (по умолчанию 1)
- `same_domain` — только тот же домен (по умолчанию `true`)
- `min_chars` — мин. символов для страницы (по умолчанию 280)

**Ответ:**
```json
{
  "pages": [
    {
      "url": "https://example.com",
      "title": "Example Page",
      "text": "Извлечённый текст...",
      "extractor": "trafilatura",
      "quality_score": 0.87,
      "depth": 0
    }
  ],
  "stats": {
    "total_pages": 1,
    "avg_quality": 0.87,
    "extractors": {"trafilatura": 1}
  }
}
```

#### Парсинг + загрузка в RAG

```bash
curl -X POST http://localhost:8000/api/jacobs/parser/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "max_pages": 5,
    "max_depth": 1
  }'
```

**Ответ:**
```json
{
  "parsed_pages": 3,
  "indexed_files": 3,
  "inserted_chunks": 24,
  "collection": "docs_ci",
  "stats": {"total_pages": 3, "avg_quality": 0.82, "extractors": {"trafilatura": 2, "bs4-main": 1}}
}
```

---

## Полный сценарий тестирования

```bash
# 0. Запуск
docker compose up --build -d

# 1. Health check
curl http://localhost:8000/health

# 2. Регистрация
curl -X POST http://localhost:8000/api/jacobs/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","username":"tester","password":"pass123"}'

# 3. Логин + сохраняем токен
TOKEN=$(curl -s -X POST http://localhost:8000/api/jacobs/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"pass123"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

echo "Token: $TOKEN"

# 4. Проверяем авторизацию
curl http://localhost:8000/api/jacobs/auth/me \
  -H "Authorization: Bearer $TOKEN"

# 5. Индексация документа
curl -X POST http://localhost:8000/api/jacobs/rag/ingest \
  -F "files=@your_document.pdf"

# 6. Вопрос-ответ через RAG
curl -X POST http://localhost:8000/api/jacobs/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "ваш вопрос"}'

# 7. Суммаризация файла
curl -X POST http://localhost:8000/api/jacobs/summary/file \
  -F "file=@your_document.pdf" \
  -F "topic=Тема" \
  -F "max_sentences=5"

# 8. Подкаст из файла
curl -X POST http://localhost:8000/api/jacobs/podcast/file \
  -F "file=@your_document.pdf" \
  -F "topic=Тема" \
  -F "tone=scientific" \
  -F "pace=normal"

# 9. Flashcards из файла
curl -X POST http://localhost:8000/api/jacobs/flashcards/file \
  -F "file=@your_document.pdf" \
  -F "topic=Тема" \
  -F "cards_count=10" \
  -F "tests_count=5"

# 10. Презентация из RAG
curl -X POST http://localhost:8000/api/jacobs/presentation/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "ваш запрос", "max_slides": 6}'

# 11. Mind map из файла
curl -X POST http://localhost:8000/api/jacobs/mindmap/file \
  -F "file=@your_document.pdf"

# 12. Таблица из файла (JSON)
curl -X POST http://localhost:8000/api/jacobs/table/file \
  -F "file=@your_document.pdf" \
  -F "hint=ключевые данные" \
  -F "format=json"

# 13. Таблица из файла (скачать XLSX)
curl -X POST http://localhost:8000/api/jacobs/table/file \
  -F "file=@your_document.pdf" \
  -F "format=xlsx" \
  -o table.xlsx

# 14. Парсинг сайта
curl -X POST http://localhost:8000/api/jacobs/parser/parse \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "max_pages": 3}'

# 15. Парсинг + загрузка в RAG
curl -X POST http://localhost:8000/api/jacobs/parser/ingest \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "max_pages": 3}'
```

---

## Конфигурация (.env)

| Переменная | Описание | По умолчанию |
|-----------|----------|--------------|
| `APP_NAME` | Название приложения | `HackSpring` |
| `DEBUG` | Режим отладки | `true` |
| `DATABASE_URL` | PostgreSQL connection string | — |
| `SECRET_KEY` | Ключ для JWT | — |
| `LLM_MODEL` | Модель LLM | `gpt-oss-20b` |
| `LLM_BASE_URL` | URL LLM API | `https://hackai.centrinvest.ru:6630` |
| `LLM_API_KEY` | API ключ LLM | `hackaton2026` |
| `RAG_EMBEDDINGS_MODEL` | Модель эмбеддингов | `Qwen3-Embedding-0.6B` |
| `RAG_EMBEDDER_URL` | URL embeddings API | `https://hackai.centrinvest.ru:6620` |
| `RAG_EMBEDDER_API_KEY` | API ключ эмбеддингов | `hackaton2026` |
| `RAG_COLLECTION` | Коллекция Qdrant | `docs_ci` |
| `RAG_CHUNK_SIZE` | Размер чанка | `900` |
| `RAG_CHUNK_OVERLAP` | Перекрытие чанков | `180` |
| `RAG_TOP_K` | Кол-во результатов | `5` |
| `RAG_MIN_SCORE` | Мин. score | `0.20` |
| `RAG_DENSE_WEIGHT` | Вес dense поиска | `0.70` |

Полный список — в `.env.example`.

---

## Docker Compose

Сервисы:
- **db** — PostgreSQL 16
- **qdrant** — Qdrant v1.12.1 (порт 6333)
- **app** — FastAPI (порт 8000)

```bash
docker compose up --build -d    # запуск
docker compose logs -f app      # логи приложения
docker compose down              # остановка
```

---

## Стек

- **Backend:** FastAPI, SQLAlchemy, Pydantic v2, pydantic-settings
- **Auth:** JWT (access + refresh tokens), bcrypt
- **RAG:** Qdrant, fastembed, BM25 rerank, OpenAI-compatible LLM
- **Mind Map:** TF + co-occurrence graph, vis-network
- **Summary:** Суммаризация через LLM
- **Podcast:** Генерация диалога + TTS (Silero, 5 голосов: aidar, baya, eugene, kseniya, xenia)
- **Flashcards:** Генерация карточек и тестов через LLM
- **Presentation:** Генерация PPTX презентаций через LLM + python-pptx
- **Table:** Извлечение структурированных таблиц из текста/файлов (JSON/CSV/XLSX)
- **Parser:** BFS-краулер сайтов с auto-extraction (trafilatura, readability, bs4) + ingest в RAG
- **Infra:** Docker Compose, PostgreSQL, Qdrant
