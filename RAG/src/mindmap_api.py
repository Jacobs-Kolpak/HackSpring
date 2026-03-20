import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.mindmap.generator import build_graph_data
from src.rag_cli import read_document, retrieve_chunks

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class MindmapRequest(BaseModel):
    text: str = Field(..., min_length=1)
    top_concepts: int = 14
    min_concept_freq: int = 2
    min_edge_weight: int = 1


class MindmapQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    db_path: Optional[str] = "data/qdrant"
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    collection: str = "docs"
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_cache: str = "data/embedding_cache"
    embedder_url: Optional[str] = None
    embedder_api_key: Optional[str] = None
    top_k: int = 8
    fetch_k: int = 60
    min_score: float = 0.15
    dense_weight: float = 0.6
    source_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    opensearch_url: Optional[str] = None
    opensearch_index: Optional[str] = None
    opensearch_api_key: Optional[str] = None
    opensearch_user: Optional[str] = None
    opensearch_password: Optional[str] = None
    opensearch_insecure: bool = False
    rerank_blend: float = 0.35
    rerank_url: Optional[str] = None
    rerank_api_key: Optional[str] = None
    rerank_model: Optional[str] = None
    top_concepts: int = 14
    min_concept_freq: int = 2
    min_edge_weight: int = 1


class MindmapResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    meta: Dict[str, Any]


app = FastAPI(
    title="RAG Mindmap API",
    version="1.0.0",
    description="API for generating mindmap graph data for frontend rendering.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


def _build_response(
    graph: Dict[str, List[Dict[str, Any]]],
    source: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> MindmapResponse:
    meta = {
        "source": source,
        "nodes_count": len(graph["nodes"]),
        "edges_count": len(graph["edges"]),
    }
    if extra_meta:
        meta.update(extra_meta)
    return MindmapResponse(nodes=graph["nodes"], edges=graph["edges"], meta=meta)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/mindmap/text", response_model=MindmapResponse)
def mindmap_from_text(payload: MindmapRequest) -> MindmapResponse:
    graph = build_graph_data(
        text=payload.text,
        top_n_concepts=payload.top_concepts,
        min_concept_freq=payload.min_concept_freq,
        min_edge_weight=payload.min_edge_weight,
    )
    return _build_response(graph, source="text")


@app.post("/api/mindmap/file", response_model=MindmapResponse)
async def mindmap_from_file(
    file: UploadFile = File(...),
    top_concepts: int = Form(14),
    min_concept_freq: int = Form(2),
    min_edge_weight: int = Form(1),
) -> MindmapResponse:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file extension. Use .pdf/.docx/.txt")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        data = await file.read()
        tmp.write(data)
        temp_path = Path(tmp.name)

    try:
        text = read_document(temp_path)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass

    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in file")

    graph = build_graph_data(
        text=text,
        top_n_concepts=top_concepts,
        min_concept_freq=min_concept_freq,
        min_edge_weight=min_edge_weight,
    )
    return _build_response(graph, source="file", extra_meta={"filename": file.filename})


@app.post("/api/mindmap/query", response_model=MindmapResponse)
def mindmap_from_query(payload: MindmapQueryRequest) -> MindmapResponse:
    results = retrieve_chunks(
        query=payload.query,
        db_path=Path(payload.db_path).resolve() if payload.db_path else None,
        qdrant_url=payload.qdrant_url,
        qdrant_api_key=payload.qdrant_api_key,
        collection_name=payload.collection,
        embedding_model=payload.embedding_model,
        embedding_cache=Path(payload.embedding_cache).resolve(),
        embedder_url=payload.embedder_url,
        embedder_api_key=payload.embedder_api_key,
        top_k=payload.top_k,
        fetch_k=payload.fetch_k,
        min_score=payload.min_score,
        dense_weight=payload.dense_weight,
        source_name=payload.source_name,
        metadata=payload.metadata,
        opensearch_url=payload.opensearch_url,
        opensearch_index=payload.opensearch_index,
        opensearch_api_key=payload.opensearch_api_key,
        opensearch_user=payload.opensearch_user,
        opensearch_password=payload.opensearch_password,
        opensearch_insecure=payload.opensearch_insecure,
        rerank_blend=payload.rerank_blend,
        rerank_url=payload.rerank_url,
        rerank_api_key=payload.rerank_api_key,
        rerank_model=payload.rerank_model,
    )
    if not results:
        raise HTTPException(status_code=404, detail="No relevant chunks found")

    text = "\n\n".join(item["text"] for item in results if item.get("text"))
    graph = build_graph_data(
        text=text,
        top_n_concepts=payload.top_concepts,
        min_concept_freq=payload.min_concept_freq,
        min_edge_weight=payload.min_edge_weight,
    )
    return _build_response(
        graph,
        source="query",
        extra_meta={"query": payload.query, "used_chunks": len(results)},
    )
