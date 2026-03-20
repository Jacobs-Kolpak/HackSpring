# Mindmap

Артефакты для интерактивной интеллект-карты.

- `output/` — HTML/JSON результаты команды `python src/rag_cli.py mindmap ...`

## API для фронта

Запуск API:

```bash
cd /Users/danilvlasuk/HackSpring/RAG
source .venv/bin/activate
uvicorn src.mindmap_api:app --host 0.0.0.0 --port 8010 --reload
```

Эндпоинты:

- `POST /api/mindmap/text` — JSON с полем `text`
- `POST /api/mindmap/file` — `multipart/form-data` с полем `file`
- `POST /api/mindmap/query` — JSON, строит graph через retrieval

Пример с фронта (загруженный файл):

```js
const formData = new FormData();
formData.append("file", fileInput.files[0]);
formData.append("top_concepts", "14");
formData.append("min_concept_freq", "2");
formData.append("min_edge_weight", "1");

const res = await fetch("http://localhost:8010/api/mindmap/file", {
  method: "POST",
  body: formData,
});
const graph = await res.json();
// graph.nodes, graph.edges -> отдаете в рендер (vis-network/cytoscape/sigma и т.д.)
```

Пример запуска:

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

Режим «пользователь загрузил файл» (напрямую из PDF/DOCX/TXT, без retrieval):

```bash
python src/rag_cli.py mindmap \
  --inputs "/path/to/uploaded_file.pdf" \
  --output mindmap/output/uploaded_file_map.html \
  --graph-json mindmap/output/uploaded_file_map.json
```

Режим без retrieval (из локального текста):

```bash
python src/rag_cli.py mindmap \
  --text-file ./mindmap/sample.txt \
  --output mindmap/output/sample_map.html
```
