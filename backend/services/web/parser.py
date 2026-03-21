from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from backend.utils.web_parser import ParsedPage, ParserConfig, parse_page

log = logging.getLogger(__name__)


def parse_url(url: str) -> Dict[str, Any]:
    """Parse a single URL: fetch HTML, clean with LLM, return structured text."""
    page = parse_page(url)
    return {
        "url": page.url,
        "title": page.title,
        "text": page.text,
        "meta": page.meta,
    }


def parse_and_ingest(
    url: str,
    *,
    collection: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> Dict[str, Any]:
    """Parse a single URL with LLM refinement, then index into vector DB."""
    from backend.services.rag import service as rag_service

    page = parse_page(url)

    if not page.text.strip():
        return {
            "url": page.url,
            "title": page.title,
            "indexed": False,
            "inserted_chunks": 0,
            "message": "No content extracted from page",
        }

    tmp_dir = tempfile.mkdtemp(prefix="parser_")
    try:
        file_path = Path(tmp_dir) / "page.txt"
        header = f"URL: {page.url}\nTITLE: {page.title}\n\n"
        file_path.write_text(header + page.text, encoding="utf-8")

        rag_result = rag_service.ingest(
            paths=[file_path],
            collection=collection,
            chunk_size=chunk_size or 0,
            chunk_overlap=chunk_overlap or 0,
            source_name_overrides={str(file_path): page.title or page.url},
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "url": page.url,
        "title": page.title,
        "indexed": True,
        "inserted_chunks": rag_result.get("inserted_chunks", 0),
        "collection": rag_result.get("collection", ""),
    }
