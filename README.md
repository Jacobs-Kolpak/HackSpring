# HackSpring

Платформа для работы с документами на базе LLM и RAG.

HackSpring помогает загружать документы, искать по ним релевантные фрагменты и генерировать учебные/аналитические материалы: summary, mindmap, flashcards, таблицы, подкаст, инфографику, презентации и видео.

## Содержание

- [Что делает проект](#что-делает-проект)
- [Стек](#стек)
- [Архитектура (кратко)](#архитектура-кратко)
- [Структура репозитория](#структура-репозитория)
- [Быстрый старт (Docker)](#быстрый-старт-docker)
- [Локальный запуск (без Docker)](#локальный-запуск-без-docker)
- [Переменные окружения](#переменные-окружения)
- [API карта](#api-карта)
- [Рабочий сценарий (curl)](#рабочий-сценарий-curl)
- [Частые ошибки](#частые-ошибки)
- [Тесты и качество](#тесты-и-качество)

## Что делает проект

- Индексирует `.pdf`, `.docx`, `.txt` в векторное хранилище.
- Отвечает на вопросы по документам через RAG (`/api/jacobs/rag/ask`).
- Генерирует контент по файлу или RAG-результатам:
  - summary,
  - mindmap,
  - flashcards + тесты,
  - таблицы (JSON/CSV/XLSX),
  - подкаст (текст + `.wav`),
  - инфографику (структурированный JSON для фронтенда),
  - презентации `.pptx` (в том числе по пользовательскому шаблону),
  - видео (через внешний video-service).
- Поддерживает регистрацию, логин и JWT-токены.

## Стек

- Backend: `FastAPI`, `SQLAlchemy`, `Pydantic v2`, `Uvicorn`
- Auth: `JWT` (`python-jose`), `bcrypt`
- RAG: `Qdrant client`, `fastembed`, `BM25` + гибридный rerank
- LLM API: OpenAI-compatible (`openai` SDK)
- Media: `python-pptx`, `silero`, `torch`, `Pillow`, `ffmpeg`
- Frontend: `React 18`, `TypeScript`, `Vite`, `Tailwind`
- Infra: `Docker Compose`, `PostgreSQL`, `Nginx`

## Архитектура (кратко)

1. Документы читаются и нормализуются (`backend/utils/document_reader.py`).
2. Текст режется на чанки (`backend/utils/chunker.py`).
3. Чанки векторизуются (`backend/utils/embeddings.py`) и сохраняются в Qdrant-хранилище.
4. При запросе выполняется retrieve + hybrid rerank (`backend/services/rag/service.py`).
5. Отобранные чанки идут в генеративные сервисы (`backend/services/content/*`, `backend/services/media/*`).

Важно:
- В текущей реализации RAG использует **локальное файловое хранилище Qdrant** (`RAG_VECTOR_STORE_PATH`, по умолчанию `./data/vector_store`).
- Контейнер `qdrant` в `docker-compose.yml` сейчас не используется backend-кодом напрямую.

## Структура репозитория

```text
backend/
  core/         # конфиг, БД, безопасность
  routers/      # FastAPI роуты
  services/     # бизнес-логика модулей
  utils/        # чтение документов, embeddings, LLM-утилиты
frontend/
  src/app/      # страницы, роутинг, контексты, UI
video/          # отдельный video-модуль (не основной backend router)
data/           # outputs: uploads, audio, presentations, vector_store
config/         # конфиг линтера/типизации/pytest
```

## Быстрый старт (Docker)

### 1. Подготовка

```bash
cp .env.example .env
```

Заполните в `.env` минимум:

- `SECRET_KEY`
- `LLM_BASE_URL`
- `LLM_API_KEY` (или `HACKAI_API_KEY` / `OPENAI_API_KEY`)
- `RAG_EMBEDDER_URL` и `RAG_EMBEDDER_API_KEY` (если embeddings берете удаленно)

Опционально:
- `VIDEO_API_KEY` — для `/api/jacobs/video/from-file`

### 2. SSL-сертификаты (опционально)

SSL **не обязателен** для запуска. Без сертификатов проект работает по HTTP на порту 3000.

Если нужен HTTPS, положите файлы в `docker/ssl/`:

```
docker/ssl/
  fullchain.crt   # сертификат + цепочка CA
  private.key     # приватный ключ
```

> **Важно:** Папка `docker/ssl/` добавлена в `.gitignore`. Сертификаты нужно копировать на сервер вручную через `scp`, **не через git**.

При запуске entrypoint автоматически определит наличие сертификатов:
- **Есть сертификаты** — запуск с HTTP (порт 3000) + HTTPS (порт 443) + редирект (порт 80 -> 443)
- **Нет сертификатов** — запуск только HTTP (порт 3000)

### 3. Запуск

```bash
docker compose up --build -d
```

### 4. Проверка

```bash
curl http://localhost:8000/health
```

Ожидаемо: `{"status":"healthy", ...}`

### 5. URL сервисов

| Режим | URL |
|---|---|
| Frontend (HTTP) | `http://localhost:3000` |
| Frontend (HTTPS, если есть SSL) | `https://your-domain.ru` |
| Backend API | `http://localhost:8000` |
| Swagger | `http://localhost:8000/docs` |
| ReDoc | `http://localhost:8000/redoc` |

### 6. Полезные команды

```bash
docker compose logs -f app
docker compose logs -f frontend
docker compose down
```

## Локальный запуск (без Docker)

### Требования

- Python `3.11+`
- Node.js `20+`
- PostgreSQL (или SQLite для быстрого старта)
- `ffmpeg` в PATH (для части медиа-сценариев)

### Backend

```bash
cp .env.example .env
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Если нужен быстрый запуск без PostgreSQL:

```bash
sed -i '' 's|^DATABASE_URL=.*|DATABASE_URL=sqlite:///./auth.db|' .env
```

### Frontend

```bash
cd frontend
npm ci
npm run dev
```

Frontend в dev-режиме поднимется на `http://localhost:5173`.

По умолчанию Vite проксирует `/api/*` на `http://localhost:8000`, поэтому `VITE_API_BASE_URL` обычно не требуется.

## Переменные окружения

Полный шаблон: [`.env.example`](./.env.example).

Ниже ключевые группы:

### Базовые

- `APP_NAME`, `APP_VERSION`, `DEBUG`, `HOST`, `PORT`, `CORS_ORIGINS`
- `DATABASE_URL`
- `SECRET_KEY`, `ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES`, `REFRESH_TOKEN_EXPIRE_DAYS`

### LLM

- `LLM_PROVIDER`
- `LLM_BASE_URL`
- `LLM_API_KEY` (или env fallback: `HACKAI_API_KEY` / `OPENAI_API_KEY`)
- `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`

### RAG

- `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`
- `RAG_EMBEDDINGS_MODEL`
- `RAG_EMBEDDER_URL`, `RAG_EMBEDDER_API_KEY`
- `RAG_VECTOR_STORE_PATH`, `RAG_COLLECTION`
- `RAG_TOP_K`, `RAG_FETCH_K`, `RAG_MIN_SCORE`, `RAG_DENSE_WEIGHT`
- `RAG_RERANK_URL`, `RAG_RERANK_API_KEY`, `RAG_RERANK_MODEL`, `RAG_RERANK_BLEND`

### Media

- Презентации: `PRESENTATION_OUTPUT_DIR`, `PRESENTATION_DEFAULT_MAX_SLIDES`, `PRESENTATION_MAX_SLIDES_LIMIT`, `PRESENTATION_MAX_BULLETS_PER_SLIDE`
- Аудио/TTS: `TTS_ENGINE`, `TTS_MODEL`, `TTS_SPEAKER`, `TTS_SAMPLE_RATE`, `TTS_OUTPUT_DIR`
- Видео-прокси: `VIDEO_BASE_URL`, `VIDEO_API_KEY`, `VIDEO_TIMEOUT_SECONDS`
- Инфографика (опционально): `INFOGRAPHIC_OUTPUT_DIR`, `INFOGRAPHIC_DEFAULT_MAX_TOPICS`, `INFOGRAPHIC_MAX_TOPICS_LIMIT`, `INFOGRAPHIC_MODELS`, `INFOGRAPHIC_AUTO_DISCOVER_MODELS`, `INFOGRAPHIC_MAX_MODEL_CANDIDATES`

### Upload/Logs

- `UPLOAD_MAX_SIZE_MB`, `UPLOAD_ALLOWED_EXTENSIONS`, `UPLOAD_DIR`
- `LOG_LEVEL`, `LOG_FORMAT`

## API карта

Базовый префикс большинства роутов: `/api/jacobs/*`.

### System

- `GET /`
- `GET /health`

### Auth (`/api/jacobs/auth`)

- `POST /register`
- `POST /login`
- `POST /refresh`
- `GET /me`
- `POST /logout`
- `GET /status`

### RAG (`/api/jacobs/rag`)

- `POST /ingest`
- `POST /retrieve`
- `POST /ask`

### Content

- Mindmap: `POST /api/jacobs/mindmap/file`
- Summary: `POST /api/jacobs/summary/file`
- Flashcards: `POST /api/jacobs/flashcards/file`
- Table: `POST /api/jacobs/table/text`, `POST /api/jacobs/table/file`

### Media

- Podcast:
  - `GET /api/jacobs/podcast/speakers`
  - `POST /api/jacobs/podcast/file`
  - `GET /api/jacobs/podcast/audio/{filename}`
- Presentation:
  - `POST /api/jacobs/presentation/generate`
  - `POST /api/jacobs/presentation/generate-with-template`
  - `POST /api/jacobs/presentation/from-results`
  - `GET /api/jacobs/presentation/download/{filename}`
- Infographics: `POST /api/jacobs/infographics`
- Video proxy: `POST /api/jacobs/video/from-file`

### Web Parser

- `POST /api/jacobs/parser/parse`
- `POST /api/jacobs/parser/ingest`

## Рабочий сценарий (curl)

### 1) Индексация документа

```bash
curl -X POST http://localhost:8000/api/jacobs/rag/ingest \
  -F "files=@./data/example.pdf" \
  -F "collection=docs_ci"
```

### 2) Вопрос к RAG

```bash
curl -X POST http://localhost:8000/api/jacobs/rag/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Сделай краткий вывод по документу",
    "collection": "docs_ci",
    "top_k": 5
  }'
```

### 3) Генерация презентации по шаблону

```bash
curl -X POST http://localhost:8000/api/jacobs/presentation/generate-with-template \
  -F "query=Собери презентацию по ключевым выводам" \
  -F "collection=docs_ci" \
  -F "top_k=8" \
  -F "max_slides=10" \
  -F "template=@./template.pptx"
```

### 4) Скачивание презентации

```bash
# Требуется jq
RESP=$(curl -s -X POST http://localhost:8000/api/jacobs/presentation/generate \
  -H "Content-Type: application/json" \
  -d '{"query":"Краткий обзор","top_k":5}')

URL=$(echo "$RESP" | jq -r '.download_url')
curl -L "http://localhost:8000${URL}" -o generated.pptx
```

## Презентации: стиль и шаблоны

Для `POST /api/jacobs/presentation/generate-with-template`:

- `template` должен быть `.pptx`.
- `style` передается как строка с валидным JSON-объектом (multipart поле).

Поддерживаемые ключи `style`:

- `title_background`
- `title_text`
- `slide_background`
- `accent`
- `slide_title_text`
- `bullet_text`
- `font_family`
- `title_font_size`
- `slide_title_font_size`
- `bullet_font_size`

Пример:

```bash
curl -X POST http://localhost:8000/api/jacobs/presentation/generate-with-template \
  -F "query=Сделай презентацию для руководства" \
  -F 'style={"font_family":"Montserrat","bullet_font_size":18,"slide_background":"#0B132B"}' \
  -F "template=@./template.pptx"
```

## Где лежат артефакты

- `data/vector_store` — локальное векторное хранилище RAG
- `data/uploads` — загруженные файлы
- `data/audio` — WAV подкастов
- `data/presentations` — сгенерированные PPTX
- `data/infographics` — выход инфографики (если сохраняется сервисом)

## Частые ошибки

- `VIDEO_API_KEY is not configured`
  - Заполните `VIDEO_API_KEY` для `/api/jacobs/video/from-file`.
- `Неподдерживаемый формат` / `Unsupported file type`
  - Основные модули принимают `.pdf`, `.docx`, `.txt`.
- `Поле style должно быть валидным JSON-объектом`
  - Проверьте JSON в multipart-поле `style` (двойные кавычки, без лишних запятых).
- `No relevant chunks found` / `RAG не нашёл релевантных документов`
  - Сначала выполните `/rag/ingest`, проверьте `collection`, затем повторите запрос.

## Тесты и качество

```bash
pytest -q
ruff check --config=config/pyproject.toml main.py backend/
```

CI-пайплайн: `.github/workflows/lint.yml`.

## Ограничения текущей версии

- Роуты генерации контента (кроме auth-операций) сейчас не требуют JWT на уровне backend.
- История сессий (`backend/routers/history`) есть в коде, но не подключена в `main.py`.
- Каталог `video/` содержит отдельный модуль, который не является текущим роутом `/api/jacobs/video/from-file`.

## Команда

- Данил: RAG, Mind Map, Flashcards, Web Parser
- Миша: Summary, Podcast + TTS, Table Extraction
- Яковкин: Презентации
