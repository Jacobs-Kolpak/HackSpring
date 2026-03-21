from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from backend.services.mindmap import build_graph_data
from backend.utils.document_reader import SUPPORTED_EXTENSIONS, read_document

router = APIRouter(prefix="/api/jacobs/mindmap", tags=["Mindmap"])


class MindmapResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    meta: Dict[str, Any]


@router.post("/file", response_model=MindmapResponse)
async def from_file(
    file: UploadFile = File(...),
    top_concepts: int = Form(10),
    min_freq: int = Form(2),
    min_edge: int = Form(2),
) -> MindmapResponse:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        temp_path = Path(tmp.name)
    try:
        text = read_document(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text in file")

    graph = build_graph_data(
        text, top_n=top_concepts, min_freq=min_freq, min_edge=min_edge
    )
    return MindmapResponse(
        nodes=graph["nodes"],
        edges=graph["edges"],
        meta={
            "source": "file",
            "filename": file.filename,
            "nodes_count": len(graph["nodes"]),
            "edges_count": len(graph["edges"]),
            "summary": graph.get("meta", {}).get("summary", ""),
            "top_terms": graph.get("meta", {}).get("top_terms", []),
        },
    )
