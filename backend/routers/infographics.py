from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.services import infographic as infographic_service
from backend.services import rag as rag_service

router = APIRouter(prefix="/api/jacobs/infographics", tags=["Infographics"])


class InfographicRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=0, le=50)
    fetch_k: Optional[int] = Field(default=None, ge=0, le=200)
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    dense_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source_name: Optional[str] = None
    model: Optional[str] = None
    max_topics: Optional[int] = Field(default=None, ge=0, le=20)
    filename_prefix: Optional[str] = None


class InfographicPoint(BaseModel):
    id: str
    name: str
    value: float


class InfographicMetric(BaseModel):
    id: str
    name: str
    value: Any
    unit: str
    description: str


class InfographicBarChart(BaseModel):
    title: str
    x_axis_label: str
    y_axis_label: str
    legend: str
    data: List[InfographicPoint]


class InfographicPieChart(BaseModel):
    title: str
    legend: str
    data: List[InfographicPoint]


class InfographicLineChart(BaseModel):
    title: str
    x_axis_label: str
    y_axis_label: str
    legend: str
    data: List[InfographicPoint]


class InfographicDownloadHint(BaseModel):
    mime_type: str
    strategy: str
    hint: str


class InfographicResponse(BaseModel):
    query: str
    title: str
    subtitle: str
    metrics: List[InfographicMetric]
    bar_chart: InfographicBarChart
    pie_chart: InfographicPieChart
    line_chart: InfographicLineChart
    key_insights: List[str]
    used_chunks: int
    models_used: List[str]
    failed_models: List[str]
    download: InfographicDownloadHint


def _zero_to_none_int(value: Optional[int]) -> Optional[int]:
    if value is None or value > 0:
        return value
    return None


@router.post("", response_model=InfographicResponse)
async def create_infographic(payload: InfographicRequest) -> InfographicResponse:
    results = rag_service.retrieve(
        query=payload.query,
        collection=payload.collection,
        top_k=_zero_to_none_int(payload.top_k),
        fetch_k=_zero_to_none_int(payload.fetch_k),
        min_score=payload.min_score,
        dense_weight=payload.dense_weight,
        source_name=payload.source_name,
    )
    if not results:
        raise HTTPException(status_code=404, detail="No relevant chunks found")

    try:
        infographics_payload = infographic_service.generate_infographic_payload(
            query=payload.query,
            results=results,
            model=payload.model,
            max_topics=_zero_to_none_int(payload.max_topics),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return InfographicResponse(**infographics_payload)


@router.get("/download")
async def download_infographic(file_path: str) -> FileResponse:
    path = Path(file_path).resolve()
    from backend.core.config import settings  # pylint: disable=import-outside-toplevel

    allowed_root = Path(settings.infographic.output_dir).resolve()
    if allowed_root not in path.parents:
        raise HTTPException(status_code=403, detail="Access to file is forbidden")
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".png":
        raise HTTPException(status_code=404, detail="Infographic file not found")
    return FileResponse(path=str(path), media_type="image/png", filename=path.name)
