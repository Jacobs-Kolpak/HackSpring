# HackSpring — AI-платформа для исследований

> Альтернатива NotebookLM с RAG, mind maps, audio summaries.
> Хакатон Центр Инвест 2026.

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
├── utils/          # document_reader, chunker, embeddings
├── services/       # rag, mindmap (бизнес-логика)
├── routers/        # auth, rag, mindmap (HTTP-слой)
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

### RAG — `/api/rag`

| Метод | URL | Описание |
|-------|-----|----------|
| POST | `/api/rag/ingest` | Загрузка и индексация документов |
| POST | `/api/rag/retrieve` | Поиск релевантных чанков |
| POST | `/api/rag/ask` | Поиск + ответ LLM |

#### Индексация документов

```bash
curl -X POST http://localhost:8000/api/rag/ingest \
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
curl -X POST http://localhost:8000/api/rag/retrieve \
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
curl -X POST http://localhost:8000/api/rag/ask \
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

### Mind Map — `/api/mindmap`

| Метод | URL | Описание |
|-------|-----|----------|
| POST | `/api/mindmap/text` | Mind map из текста |
| POST | `/api/mindmap/file` | Mind map из файла |
| POST | `/api/mindmap/query` | Mind map из RAG-поиска |

#### Из текста

```bash
curl -X POST http://localhost:8000/api/mindmap/text \
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
curl -X POST http://localhost:8000/api/mindmap/file \
  -F "file=@document.pdf" \
  -F "top_concepts=20" \
  -F "min_freq=2" \
  -F "min_edge=1"
```

#### Из RAG-запроса

```bash
curl -X POST http://localhost:8000/api/mindmap/query \
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

### Summary — `/api/summary`

| Метод | URL | Описание |
|-------|-----|----------|
| POST | `/api/summary/text` | Суммаризация из текста |
| POST | `/api/summary/file` | Суммаризация из файла |
| POST | `/api/summary/query` | Суммаризация из RAG-поиска |

#### Из текста

```bash
curl -X POST http://localhost:8000/api/summary/text \
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
curl -X POST http://localhost:8000/api/summary/file \
  -F "file=@document.pdf" \
  -F "topic=Финансы" \
  -F "max_sentences=10"
```

#### Из RAG-запроса

```bash
curl -X POST http://localhost:8000/api/summary/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "инвестиционные программы",
    "collection": "docs_ci",
    "top_k": 8,
    "topic": "Инвестиции",
    "max_sentences": 10
  }'
```

---

### Podcast — `/api/podcast`

| Метод | URL | Описание |
|-------|-----|----------|
| POST | `/api/podcast/text` | Подкаст из текста |
| POST | `/api/podcast/file` | Подкаст из файла |
| POST | `/api/podcast/query` | Подкаст из RAG-поиска |
| GET | `/api/podcast/audio/{filename}` | Скачать аудиофайл |

#### Из текста

```bash
curl -X POST http://localhost:8000/api/podcast/text \
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
  "audio_url": "/api/podcast/audio/podcast_a1b2c3d4e5f6.wav",
  "meta": {}
}
```

#### Из файла

```bash
curl -X POST http://localhost:8000/api/podcast/file \
  -F "file=@document.pdf" \
  -F "topic=Финансы" \
  -F "tone=everyday" \
  -F "pace=normal"
```

#### Из RAG-запроса

```bash
curl -X POST http://localhost:8000/api/podcast/query \
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
curl -O http://localhost:8000/api/podcast/audio/podcast_a1b2c3d4e5f6.wav
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
curl -X POST http://localhost:8000/api/rag/ingest \
  -F "files=@your_document.pdf"

# 6. Поиск
curl -X POST http://localhost:8000/api/rag/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "ваш вопрос"}'

# 7. Вопрос-ответ
curl -X POST http://localhost:8000/api/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "ваш вопрос"}'

# 8. Mind map из текста
curl -X POST http://localhost:8000/api/mindmap/text \
  -H "Content-Type: application/json" \
  -d '{"text": "ваш текст для анализа..."}'

# 9. Mind map из файла
curl -X POST http://localhost:8000/api/mindmap/file \
  -F "file=@your_document.pdf"

# 10. Суммаризация текста
curl -X POST http://localhost:8000/api/summary/text \
  -H "Content-Type: application/json" \
  -d '{"text": "ваш текст...", "topic": "Тема", "max_sentences": 5}'

# 11. Подкаст из текста
curl -X POST http://localhost:8000/api/podcast/text \
  -H "Content-Type: application/json" \
  -d '{"text": "ваш текст...", "topic": "Тема", "tone": "scientific", "pace": "normal"}'
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
- **Podcast:** Генерация диалога + TTS (pyttsx3/espeak)
- **Infra:** Docker Compose, PostgreSQL, Qdrant
