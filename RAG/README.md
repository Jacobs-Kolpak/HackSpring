# RAG System with Qdrant

Прокачанная RAG-система на `Qdrant`:
- загрузка `PDF`, `DOCX`, `TXT`;
- улучшенное извлечение PDF (`PyMuPDF` + fallback `pypdf`);
- очистка PDF-артефактов (разбитые слова, переносы);
- sentence-aware чанкинг;
- retrieval из Qdrant;
- гибридный retrieval: `Qdrant vector + OpenSearch lexical`;
- фильтрация по метаданным (`--metadata key=value`);
- гибридный rerank (`dense + BM25`);
- генерация ответа через OpenAI (`ask`).

## 1) Установка

```bash
cd /Users/danilvlasuk/HackSpring/RAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Индексация в Qdrant

```bash
python src/rag_cli.py ingest \
  /path/to/file_or_folder \
  --db-path data/qdrant \
  --collection docs \
  --chunk-size 900 \
  --chunk-overlap 180 \
  --embedding-model intfloat/multilingual-e5-large \
  --embedding-cache data/embedding_cache
```

## 3) Retrieval top-k (с rerank)

```bash
python src/rag_cli.py retrieve \
  --query "Ключевые навыки кандидата" \
  --top-k 5 \
  --fetch-k 30 \
  --dense-weight 0.75 \
  --min-score 0.20 \
  --db-path data/qdrant \
  --collection docs \
  --embedding-model intfloat/multilingual-e5-large \
  --embedding-cache data/embedding_cache

# с фильтром по метаданным (точное совпадение)
python src/rag_cli.py retrieve \
  --query "Ключевые навыки кандидата" \
  --metadata "department=risk" \
  --metadata "lang=ru"
```

JSON-режим:

```bash
python src/rag_cli.py retrieve --query "..." --top-k 5 --json
```

## 4) Полный RAG-ответ (retrieve + generation)

```bash
export OPENAI_API_KEY="your_api_key"
```

```bash
python src/rag_cli.py ask \
  --query "Сформулируй цель проекта" \
  --top-k 5 \
  --fetch-k 30 \
  --dense-weight 0.75 \
  --min-score 0.20 \
  --db-path data/qdrant \
  --collection docs \
  --embedding-model intfloat/multilingual-e5-large \
  --embedding-cache data/embedding_cache \
  --model gpt-4.1-mini
```

## 4.1) Mindmap (интерактивный граф понятий)

Построение mindmap из retrieval-результатов:

```bash
python src/rag_cli.py mindmap \
  --query "Ключевые навыки кандидата кратко" \
  --top-k 8 \
  --fetch-k 60 \
  --dense-weight 0.6 \
  --rerank-blend 0.35 \
  --min-score 0.15 \
  --db-path data/qdrant \
  --collection docs_ci \
  --embedder-url "https://hackai.centrinvest.ru:6620" \
  --embedding-model "Qwen3-Embedding-0.6B" \
  --api-key "$HACKAI_API_KEY" \
  --output mindmap/output/candidate_skills_map.html \
  --graph-json mindmap/output/candidate_skills_map.json
```

Построение mindmap из локального текста:

```bash
python src/rag_cli.py mindmap \
  --text-file mindmap/sample.txt \
  --output mindmap/output/sample_map.html
```

Построение mindmap напрямую из загруженного файла (`.pdf/.docx/.txt`):

```bash
python src/rag_cli.py mindmap \
  --inputs "/path/to/uploaded_file.pdf" \
  --output mindmap/output/uploaded_file_map.html \
  --graph-json mindmap/output/uploaded_file_map.json
```

## 5) Использование hackai.centrinvest.ru

Если у тебя есть ключ доступа:

```bash
export HACKAI_API_KEY="your_token"
```

Индексация через внешний эмбеддер (`6620`) + локальный Qdrant:

```bash
python src/rag_cli.py ingest \
  "/path/to/file.pdf" \
  --db-path data/qdrant \
  --collection docs_ci \
  --embedder-url "https://hackai.centrinvest.ru:6620" \
  --embedding-model "BAAI/bge-m3" \
  --api-key "$HACKAI_API_KEY"
```

Retrieval с внешним rerank (`6620`):

```bash
python src/rag_cli.py retrieve \
  --query "Ключевые навыки кандидата" \
  --top-k 5 \
  --fetch-k 40 \
  --dense-weight 0.7 \
  --min-score 0.2 \
  --db-path data/qdrant \
  --collection docs_ci \
  --embedder-url "https://hackai.centrinvest.ru:6620" \
  --rerank-url "https://hackai.centrinvest.ru:6620" \
  --embedding-model "BAAI/bge-m3" \
  --api-key "$HACKAI_API_KEY"
```

Полный `ask` с LLM (`6630`):

```bash
python src/rag_cli.py ask \
  --query "Сформулируй ключевые навыки кандидата кратко" \
  --top-k 5 \
  --fetch-k 40 \
  --dense-weight 0.7 \
  --min-score 0.2 \
  --db-path data/qdrant \
  --collection docs_ci \
  --embedder-url "https://hackai.centrinvest.ru:6620" \
  --rerank-url "https://hackai.centrinvest.ru:6620" \
  --llm-url "https://hackai.centrinvest.ru:6630" \
  --embedding-model "BAAI/bge-m3" \
  --model "gpt-4.1-mini" \
  --api-key "$HACKAI_API_KEY"
```

## OpenSearch hybrid (vector + lexical + metadata)

Индексация одновременно в Qdrant и OpenSearch:

```bash
python src/rag_cli.py ingest \
  "/path/to/docs" \
  --db-path data/qdrant \
  --collection docs \
  --opensearch-url "https://localhost:9200" \
  --opensearch-index "docs_chunks" \
  --opensearch-user "admin" \
  --opensearch-password "admin" \
  --metadata "department=risk" \
  --metadata "lang=ru"
```

Гибридный retrieval: Qdrant + OpenSearch (объединение скоринга):

```bash
python src/rag_cli.py retrieve \
  --query "Ключевые навыки кандидата" \
  --top-k 5 \
  --fetch-k 40 \
  --dense-weight 0.7 \
  --db-path data/qdrant \
  --collection docs \
  --opensearch-url "https://localhost:9200" \
  --opensearch-index "docs_chunks" \
  --opensearch-user "admin" \
  --opensearch-password "admin" \
  --metadata "department=risk" \
  --metadata "lang=ru"
```

## Remote Qdrant (production)

Вместо `--db-path` можно использовать удаленный Qdrant:

```bash
python src/rag_cli.py retrieve \
  --query "..." \
  --qdrant-url "https://your-qdrant-host:6333" \
  --qdrant-api-key "..." \
  --collection docs
```

## Параметры

- `--chunk-size`, `--chunk-overlap`: параметры чанкинга
- `--embedding-model`, `--embedding-cache`: эмбеддинги FastEmbed
- `--top-k`: сколько чанков вернуть в ответ
- `--fetch-k`: сколько кандидатов забрать из Qdrant перед rerank
- `--dense-weight`: вес dense-части в hybrid rerank (`0..1`)
- `--min-score`: порог финального reranked score
- `--source-name`: фильтр по имени файла
- `--metadata key=value`: фильтр по метаданным (repeatable)
- `--db-path`: локальный storage Qdrant
- `--qdrant-url`, `--qdrant-api-key`: удаленный Qdrant
- `--opensearch-url`, `--opensearch-index`: OpenSearch для hybrid retrieval
