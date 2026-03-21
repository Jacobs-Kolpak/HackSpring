from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from urllib.parse import urldefrag, urlparse

import httpx

from backend.utils.llm import generate_text

log = logging.getLogger(__name__)

_LLM_SYSTEM_PROMPT = (
    "Ты — специалист по извлечению контента из веб-страниц. "
    "Тебе дан HTML-контент страницы, предварительно очищенный от скриптов и стилей. "
    "Твоя задача:\n"
    "1. Извлечь весь полезный текстовый контент страницы.\n"
    "2. Убрать навигацию, рекламу, футеры, хедеры, сайдбары и прочий шум.\n"
    "3. Структурировать текст: сохранить заголовки, списки, абзацы.\n"
    "4. Вернуть чистый структурированный текст в формате:\n"
    "   - Заголовки обозначай через '# ', '## ', '### '\n"
    "   - Списки через '- ' или '1. '\n"
    "   - Абзацы разделяй пустой строкой\n"
    "5. НЕ добавляй от себя никакой информации. Только то, что есть на странице.\n"
    "6. НЕ оборачивай ответ в markdown-блоки (``` и т.п.), верни просто текст.\n"
    "7. Если на странице нет полезного контента — верни пустую строку."
)


@dataclass
class ParserConfig:
    timeout_sec: float = 20.0
    retries: int = 2
    backoff_sec: float = 1.2
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )


@dataclass
class ParsedPage:
    url: str
    title: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


def _normalize_url(url: str) -> str:
    clean = url.strip()
    if not clean:
        raise ValueError("URL is empty")
    if not clean.startswith(("http://", "https://")):
        clean = f"https://{clean}"
    return urldefrag(clean)[0]


def _extract_title(html_text: str, fallback: str = "untitled") -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_text, "html.parser")
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = str(og_title.get("content")).strip()
            if title:
                return title
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
            if title:
                return title
    except Exception:
        pass
    return fallback


def _preclean_html(html_text: str) -> str:
    """Remove scripts, styles, and other non-content elements to reduce token count."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return html_text

    soup = BeautifulSoup(html_text, "html.parser")

    for tag_name in [
        "script", "style", "noscript", "svg", "iframe",
        "nav", "footer", "header", "aside",
        "link", "meta",
    ]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for tag in soup.find_all(True):
        tag.attrs = {}

    cleaned = soup.get_text("\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def _fetch_with_retry(
    client: httpx.Client, url: str, *, retries: int, backoff_sec: float,
) -> Optional[httpx.Response]:
    import time

    for attempt in range(retries + 1):
        try:
            resp = client.get(url)
            if resp.status_code in {429, 500, 502, 503, 504}:
                raise httpx.HTTPStatusError(
                    "transient", request=resp.request, response=resp,
                )
            resp.raise_for_status()
            return resp
        except Exception:
            if attempt < retries:
                time.sleep(backoff_sec * (2 ** attempt))
    return None


def _refine_with_llm(precleaned_text: str, url: str) -> str:
    """Send pre-cleaned HTML text to LLM for structuring."""
    max_chars = 60_000
    text_to_send = precleaned_text[:max_chars]

    prompt = (
        f"URL страницы: {url}\n\n"
        f"Содержимое страницы:\n\n{text_to_send}"
    )

    try:
        result = generate_text(
            prompt=prompt,
            system=_LLM_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=4096,
        )
        return result.strip()
    except Exception as exc:
        log.error("LLM refinement failed for %s: %s", url, exc)
        return precleaned_text


def parse_page(
    url: str,
    *,
    config: Optional[ParserConfig] = None,
) -> ParsedPage:
    """Fetch a single page, pre-clean HTML, refine with LLM, return structured text."""
    cfg = config or ParserConfig()
    normalized_url = _normalize_url(url)
    headers = {"User-Agent": cfg.user_agent}

    with httpx.Client(
        follow_redirects=True, timeout=cfg.timeout_sec, headers=headers,
    ) as client:
        response = _fetch_with_retry(
            client, normalized_url,
            retries=cfg.retries, backoff_sec=cfg.backoff_sec,
        )

    if response is None:
        raise ValueError(f"Failed to fetch URL: {normalized_url}")

    if response.status_code >= 400:
        raise ValueError(
            f"HTTP {response.status_code} for URL: {normalized_url}"
        )

    ctype = response.headers.get("content-type", "").lower()
    if "html" not in ctype:
        raise ValueError(f"Not an HTML page (content-type: {ctype})")

    html_text = response.text
    title = _extract_title(
        html_text,
        fallback=urlparse(normalized_url).path.strip("/") or "untitled",
    )

    precleaned = _preclean_html(html_text)

    structured_text = _refine_with_llm(precleaned, normalized_url)

    return ParsedPage(
        url=normalized_url,
        title=title,
        text=structured_text,
        meta={
            "status_code": response.status_code,
            "content_type": ctype,
        },
    )
