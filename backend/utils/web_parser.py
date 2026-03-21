from __future__ import annotations

import re
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

_BINARY_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".7z", ".tar", ".gz", ".exe", ".dmg",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".mp3", ".wav", ".ogg", ".mp4", ".avi", ".mov", ".mkv",
}
_NAV_WORDS = {
    "menu", "login", "sign in", "cookie", "privacy", "policy", "terms",
    "copyright", "войти", "меню", "cookie", "политик",
}
_NOISY_PATTERNS = (
    "перейти к навигации",
    "перейти к поиску",
    "материал из википедии",
    "this page was last edited",
)


@dataclass
class ParserConfig:
    max_pages: int = 10
    max_depth: int = 1
    same_domain: bool = True
    min_chars: int = 280
    timeout_sec: float = 20.0
    retries: int = 2
    backoff_sec: float = 1.2
    respect_robots: bool = True
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )


@dataclass
class ParsedPage:
    url: str
    title: str
    text: str
    extractor: str = "unknown"
    quality_score: float = 0.0
    depth: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _ExtractionResult:
    text: str
    extractor: str
    score: float


_ROBOTS_CACHE: Dict[str, Optional[RobotFileParser]] = {}


def _normalize_url(url: str) -> str:
    clean = url.strip()
    if not clean:
        raise ValueError("URL is empty")
    if not clean.startswith(("http://", "https://")):
        clean = f"https://{clean}"
    return urldefrag(clean)[0]


def _sanitize_filename(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9а-яА-ЯёЁ_-]+", "_", value).strip("_")
    return (slug[:70] or fallback).lower()


def _normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    return text.strip()


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


def _path_extension(url: str) -> str:
    return Path(urlparse(url).path).suffix.lower()


def _is_probably_html_url(url: str) -> bool:
    ext = _path_extension(url)
    if not ext:
        return True
    return ext not in _BINARY_EXTENSIONS


def _extract_links(html_text: str, base_url: str) -> List[str]:
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return []

    soup = BeautifulSoup(html_text, "html.parser")
    result: List[str] = []

    for a_tag in soup.find_all("a", href=True):
        href = str(a_tag.get("href", "")).strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        abs_url = urldefrag(urljoin(base_url, href))[0]
        parsed = urlparse(abs_url)
        if parsed.scheme not in {"http", "https"}:
            continue
        if not _is_probably_html_url(abs_url):
            continue
        result.append(abs_url)

    seen: Set[str] = set()
    uniq: List[str] = []
    for link in result:
        if link not in seen:
            seen.add(link)
            uniq.append(link)
    return uniq


def _quality_score(text: str) -> float:
    if not text.strip():
        return 0.0

    normalized = _normalize_text(text)
    chars = len(normalized)
    if chars == 0:
        return 0.0

    words = re.findall(r"[a-zа-яё0-9]+", normalized.lower())
    unique_ratio = len(set(words)) / max(1, len(words))
    sentence_count = len(re.findall(r"[.!?]", normalized))
    punctuation_ratio = len(re.findall(r"[,.!?;:]", normalized)) / max(1, chars)
    nav_hits = sum(1 for w in _NAV_WORDS if w in normalized.lower())
    noise_hits = sum(1 for p in _NOISY_PATTERNS if p in normalized.lower())

    score = 0.0
    score += min(chars / 5000.0, 1.0) * 0.45
    score += min(sentence_count / 45.0, 1.0) * 0.20
    score += min(unique_ratio / 0.75, 1.0) * 0.20
    score += min(punctuation_ratio / 0.03, 1.0) * 0.15
    score -= min(nav_hits / 10.0, 0.20)
    score -= min(noise_hits / 8.0, 0.16)

    return max(0.0, min(1.0, score))


def _extract_with_trafilatura(
    html_text: str, url: str, *, prefer_precision: bool = True,
) -> str:
    try:
        import trafilatura

        extracted = trafilatura.extract(
            html_text,
            url=url,
            output_format="txt",
            include_links=False,
            include_comments=False,
            favor_precision=prefer_precision,
        )
        return _normalize_text(extracted or "")
    except Exception:
        return ""


def _extract_with_readability(html_text: str) -> str:
    try:
        from readability import Document
        from bs4 import BeautifulSoup

        doc = Document(html_text)
        article_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(article_html, "html.parser")
        return _normalize_text(soup.get_text("\n"))
    except Exception:
        return ""


def _extract_with_bs4_main_content(html_text: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return ""

    try:
        soup = BeautifulSoup(html_text, "html.parser")
        for selector in [
            "script", "style", "noscript", "svg", "iframe",
            "nav", "footer", "header", "aside",
        ]:
            for tag in soup.select(selector):
                tag.decompose()

        candidates = [
            soup.select_one("main"),
            soup.select_one("article"),
            soup.select_one("[role='main']"),
            soup.select_one("#mw-content-text"),
            soup.select_one(".mw-parser-output"),
            soup.select_one("#content"),
            soup.body,
        ]
        for node in candidates:
            if node is None:
                continue
            text = _normalize_text(node.get_text("\n"))
            if len(text) >= 180:
                return text
        return ""
    except Exception:
        return ""


def _extract_with_bs4_raw(html_text: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
            tag.extract()
        return _normalize_text(soup.get_text("\n"))
    except Exception:
        return ""


def _pick_best_extraction(
    html_text: str, url: str,
) -> _ExtractionResult:
    candidates: List[_ExtractionResult] = []

    def _push(text: str, extractor: str, bonus: float = 0.0) -> None:
        cleaned = _normalize_text(text)
        if len(cleaned) < 80:
            return
        base_score = _quality_score(cleaned)
        candidates.append(_ExtractionResult(
            text=cleaned,
            extractor=extractor,
            score=max(0.0, min(1.0, base_score + bonus)),
        ))

    _push(_extract_with_trafilatura(html_text, url), "trafilatura")
    _push(_extract_with_readability(html_text), "readability")
    _push(_extract_with_bs4_main_content(html_text), "bs4-main", bonus=0.06)
    _push(_extract_with_bs4_raw(html_text), "bs4-raw")

    if not candidates:
        return _ExtractionResult(text="", extractor="none", score=0.0)

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[0]


def _fetch_with_retry(
    client: httpx.Client, url: str, *, retries: int, backoff_sec: float,
) -> Optional[httpx.Response]:
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


def _get_robot_parser(target_url: str) -> Optional[RobotFileParser]:
    parsed = urlparse(target_url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    if root in _ROBOTS_CACHE:
        return _ROBOTS_CACHE[root]

    robots_url = f"{root}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception:
        _ROBOTS_CACHE[root] = None
        return None
    _ROBOTS_CACHE[root] = rp
    return rp


def _allowed_by_robots(url: str, user_agent: str) -> bool:
    rp = _get_robot_parser(url)
    if rp is None:
        return True
    try:
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True


def parse_website(
    seed_url: str,
    *,
    config: Optional[ParserConfig] = None,
) -> List[ParsedPage]:
    cfg = config or ParserConfig()
    start_url = _normalize_url(seed_url)
    start_domain = urlparse(start_url).netloc.lower()

    queue: Deque[Tuple[str, int]] = deque([(start_url, 0)])
    visited: Set[str] = set()
    pages: List[ParsedPage] = []
    headers = {"User-Agent": cfg.user_agent}

    with httpx.Client(
        follow_redirects=True, timeout=cfg.timeout_sec, headers=headers,
    ) as client:
        while queue and len(pages) < cfg.max_pages:
            current_url, depth = queue.popleft()
            current_url = urldefrag(current_url)[0]

            if current_url in visited:
                continue
            visited.add(current_url)

            if cfg.respect_robots and not _allowed_by_robots(
                current_url, cfg.user_agent,
            ):
                continue

            response = _fetch_with_retry(
                client, current_url,
                retries=cfg.retries, backoff_sec=cfg.backoff_sec,
            )
            if response is None or response.status_code >= 400:
                continue

            ctype = response.headers.get("content-type", "").lower()
            if "html" not in ctype:
                continue

            html_text = response.text
            title = _extract_title(
                html_text,
                fallback=urlparse(current_url).path.strip("/") or "untitled",
            )

            picked = _pick_best_extraction(html_text, current_url)

            if len(picked.text) >= cfg.min_chars:
                pages.append(ParsedPage(
                    url=current_url,
                    title=title,
                    text=picked.text,
                    extractor=picked.extractor,
                    quality_score=round(picked.score, 4),
                    depth=depth,
                    meta={
                        "status_code": response.status_code,
                        "content_type": ctype,
                    },
                ))

            if depth < cfg.max_depth:
                links = _extract_links(html_text, current_url)
                for link in links:
                    if link not in visited:
                        if not cfg.same_domain or urlparse(
                            link,
                        ).netloc.lower() == start_domain:
                            queue.append((link, depth + 1))

    return pages


def write_pages_to_txt(pages: List[ParsedPage], folder: Path) -> List[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for idx, page in enumerate(pages, 1):
        fallback = f"page_{idx:03d}"
        name = _sanitize_filename(page.title, fallback=fallback)
        file_path = folder / f"{idx:03d}_{name}.txt"
        header = (
            f"URL: {page.url}\n"
            f"TITLE: {page.title}\n"
            f"EXTRACTOR: {page.extractor}\n"
            f"QUALITY_SCORE: {page.quality_score}\n"
            f"DEPTH: {page.depth}\n\n"
        )
        file_path.write_text(header + page.text + "\n", encoding="utf-8")
        paths.append(file_path)
    return paths


def summarize_extractors(
    pages: List[ParsedPage],
) -> Dict[str, Any]:
    total = len(pages)
    by_extractor = Counter(p.extractor for p in pages)
    avg_quality = (
        round(sum(p.quality_score for p in pages) / total, 4)
        if total else 0.0
    )
    return {
        "total_pages": total,
        "avg_quality": avg_quality,
        "extractors": dict(by_extractor),
    }
