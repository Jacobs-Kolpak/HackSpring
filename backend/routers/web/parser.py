from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.web import parser as parser_service

router = APIRouter(prefix="/api/jacobs/parser", tags=["Parser"])


class ParseRequest(BaseModel):
    url: str


class ParseAndIngestRequest(BaseModel):
    url: str
    collection: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


@router.post("/parse")
async def parse_url(body: ParseRequest):
    try:
        return parser_service.parse_url(body.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ingest")
async def parse_and_ingest(body: ParseAndIngestRequest):
    try:
        return parser_service.parse_and_ingest(
            body.url,
            collection=body.collection,
            chunk_size=body.chunk_size,
            chunk_overlap=body.chunk_overlap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
