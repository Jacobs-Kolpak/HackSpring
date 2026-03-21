from __future__ import annotations

import logging
import shutil
import tempfile
from typing import Any, Dict, List, Optional

from backend.utils.web_parser import (
    ParsedPage,
    ParserConfig,
    parse_website,
    summarize_extractors,
    write_pages_to_txt,
)

log = logging.getLogger(__name__)


def parse_url(
    url: str,
    *,
    max_pages: int = 10,
    max_depth: int = 1,
    same_domain: bool = True,
    min_chars: int = 280,
) -> Dict[str, Any]:
    cfg = ParserConfig(
        max_pages=max_pages,
        max_depth=max_depth,
        same_domain=same_domain,
        min_chars=min_chars,
    )
    pages = parse_website(url, config=cfg)
    stats = summarize_extractors(pages)

    return {
        "pages": [
            {
                "url": p.url,
                "title": p.title,
                "text": p.text,
                "extractor": p.extractor,
                "quality_score": p.quality_score,
                "depth": p.depth,
            }
            for p in pages
        ],
        "stats": stats,
    }


def parse_and_ingest(
    url: str,
    *,
    max_pages: int = 10,
    max_depth: int = 1,
    same_domain: bool = True,
    min_chars: int = 280,
    collection: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> Dict[str, Any]:
    from backend.services.rag import service as rag_service

    cfg = ParserConfig(
        max_pages=max_pages,
        max_depth=max_depth,
        same_domain=same_domain,
        min_chars=min_chars,
    )
    pages = parse_website(url, config=cfg)
    stats = summarize_extractors(pages)

    if not pages:
        return {
            "parsed_pages": 0,
            "indexed_files": 0,
            "inserted_chunks": 0,
            "stats": stats,
        }

    tmp_dir = tempfile.mkdtemp(prefix="parser_")
    try:
        from pathlib import Path

        paths = write_pages_to_txt(pages, Path(tmp_dir))
        rag_result = rag_service.ingest(
            paths=[str(p) for p in paths],
            collection=collection,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "parsed_pages": len(pages),
        "indexed_files": rag_result.get("indexed_files", 0),
        "inserted_chunks": rag_result.get("inserted_chunks", 0),
        "collection": rag_result.get("collection", ""),
        "stats": stats,
    }
