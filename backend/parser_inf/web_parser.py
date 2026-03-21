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
    "copyright", "подпис", "войти", "меню", "cookie", "политик",
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
        "Mozilla/5.0 (compatible; HackSpringParserInf/2.0; "
        "+https://example.local/parser)"
    )
    enable_browser_fallback: bool = False
    browser_timeout_ms: int = 18000
    use_transformer_ranker: bool = False
    prefer_precision: bool = True


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


_ZERO_SHOT_PIPELINE: Any = None
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
    text = re.sub(r"«\s+", "«", text)
    text = re.sub(r"\s+»", "»", text)
    return text.strip()


def _extract_title(html_text: str, fallback: str = "untitled") -> str:
    try:
        from bs4 import BeautifulSoup  # pylint: disable=import-outside-toplevel

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
    except Exception:  # pylint: disable=broad-except
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
        from bs4 import BeautifulSoup  # pylint: disable=import-outside-toplevel
    except Exception:  # pylint: disable=broad-except
        return []

    soup = BeautifulSoup(html_text, "html.parser")
    result: List[str] = []

    canonical = soup.find("link", rel=lambda v: v and "canonical" in str(v).lower())
    if canonical and canonical.get("href"):
        abs_canonical = urldefrag(urljoin(base_url, str(canonical.get("href"))))[0]
        if abs_canonical:
            result.append(abs_canonical)

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
    line_count = max(1, len([ln for ln in normalized.splitlines() if ln.strip()]))

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

    if line_count > 300:
        score -= 0.05

    return max(0.0, min(1.0, score))


def _extract_with_trafilatura(
    html_text: str,
    url: str,
    *,
    prefer_precision: bool,
) -> str:
    try:
        import trafilatura  # pylint: disable=import-outside-toplevel

        extracted = trafilatura.extract(
            html_text,
            url=url,
            output_format="txt",
            include_links=False,
            include_comments=False,
            favor_precision=prefer_precision,
        )
        return _normalize_text(extracted or "")
    except Exception:  # pylint: disable=broad-except
        return ""


def _extract_with_readability(html_text: str) -> str:
    try:
        from readability import Document  # pylint: disable=import-outside-toplevel
        from bs4 import BeautifulSoup  # pylint: disable=import-outside-toplevel

        doc = Document(html_text)
        article_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(article_html, "html.parser")
        return _normalize_text(soup.get_text("\n"))
    except Exception:  # pylint: disable=broad-except
        return ""


def _extract_with_goose(html_text: str) -> str:
    try:
        from goose3 import Goose  # pylint: disable=import-outside-toplevel

        goose = Goose()
        article = goose.extract(raw_html=html_text)
        return _normalize_text(article.cleaned_text or "")
    except Exception:  # pylint: disable=broad-except
        return ""


def _extract_with_newspaper(html_text: str, url: str) -> str:
    try:
        from newspaper import Article  # pylint: disable=import-outside-toplevel

        article = Article(url=url)
        article.set_html(html_text)
        article.parse()
        return _normalize_text(article.text or "")
    except Exception:  # pylint: disable=broad-except
        return ""


def _extract_with_bs4_raw(html_text: str) -> str:
    try:
        from bs4 import BeautifulSoup  # pylint: disable=import-outside-toplevel

        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
            tag.extract()
        return _normalize_text(soup.get_text("\n"))
    except Exception:  # pylint: disable=broad-except
        return ""


def _extract_with_bs4_main_content(html_text: str) -> str:
    try:
        from bs4 import BeautifulSoup  # pylint: disable=import-outside-toplevel
    except Exception:  # pylint: disable=broad-except
        return ""

    try:
        soup = BeautifulSoup(html_text, "html.parser")

        # Remove common boilerplate blocks before taking text.
        for selector in [
            "script", "style", "noscript", "svg", "iframe",
            "nav", "footer", "header", "aside",
            ".toc", ".navbox", ".metadata", ".mw-editsection",
            ".mw-jump-link", ".reference", "sup.reference",
            ".reflist", ".portal", ".catlinks",
        ]:
            for tag in soup.select(selector):
                tag.decompose()

        # Prefer semantic content containers, then known wiki containers.
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
    except Exception:  # pylint: disable=broad-except
        return ""


def _extract_with_wikipedia_blocks(html_text: str, url: str) -> str:
    if "wikipedia.org/wiki/" not in url:
        return ""
    try:
        from bs4 import BeautifulSoup  # pylint: disable=import-outside-toplevel
    except Exception:  # pylint: disable=broad-except
        return ""

    try:
        soup = BeautifulSoup(html_text, "html.parser")
        for selector in [
            ".mw-editsection", "sup.reference", ".reference", ".reflist",
            ".thumb", ".hatnote", ".shortdescription", ".toc",
        ]:
            for tag in soup.select(selector):
                tag.decompose()

        paragraph_nodes = soup.select("#mw-content-text .mw-parser-output > p")
        parts: List[str] = []
        for node in paragraph_nodes:
            txt = _normalize_text(node.get_text(" "))
            if len(txt) >= 80:
                parts.append(txt)
        if not parts:
            return ""
        return _normalize_text("\n\n".join(parts))
    except Exception:  # pylint: disable=broad-except
        return ""


def _get_zero_shot_pipeline() -> Any:
    global _ZERO_SHOT_PIPELINE  # noqa: PLW0603
    if _ZERO_SHOT_PIPELINE is not None:
        return _ZERO_SHOT_PIPELINE

    try:
        from transformers import pipeline  # pylint: disable=import-outside-toplevel

        _ZERO_SHOT_PIPELINE = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        )
        return _ZERO_SHOT_PIPELINE
    except Exception:  # pylint: disable=broad-except
        return None


def _extract_with_transformer_ranker(html_text: str) -> str:
    try:
        from bs4 import BeautifulSoup  # pylint: disable=import-outside-toplevel
    except Exception:  # pylint: disable=broad-except
        return ""

    clf = _get_zero_shot_pipeline()
    if clf is None:
        return ""

    soup = BeautifulSoup(html_text, "html.parser")
    blocks: List[str] = []
    for tag in soup.find_all(["p", "li", "h2", "h3", "article", "section"]):
        txt = _normalize_text(tag.get_text(" "))
        if len(txt) >= 60:
            blocks.append(txt)

    if not blocks:
        return ""

    selected: List[Tuple[float, str]] = []
    labels = ["main content", "boilerplate", "navigation"]
    for block in blocks[:140]:
        try:
            res = clf(block[:1200], candidate_labels=labels)
            found = dict(zip(res.get("labels", []), res.get("scores", [])))
            score = float(found.get("main content", 0.0))
            if score >= 0.45:
                selected.append((score, block))
        except Exception:  # pylint: disable=broad-except
            continue

    if not selected:
        return ""

    selected.sort(key=lambda x: x[0], reverse=True)
    merged = "\n\n".join(text for _, text in selected[:50])
    return _normalize_text(merged)


def _render_with_playwright(url: str, timeout_ms: int) -> str:
    try:
        from playwright.sync_api import sync_playwright  # pylint: disable=import-outside-toplevel

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            html_text = page.content()
            browser.close()
            return html_text
    except Exception:  # pylint: disable=broad-except
        return ""


def _pick_best_extraction(
    html_text: str,
    url: str,
    *,
    prefer_precision: bool,
    use_transformer_ranker: bool,
) -> _ExtractionResult:
    candidates: List[_ExtractionResult] = []

    def _push(text: str, extractor: str, bonus: float = 0.0) -> None:
        cleaned = _normalize_text(text)
        if len(cleaned) < 80:
            return
        base_score = _quality_score(cleaned)
        candidates.append(
            _ExtractionResult(
                text=cleaned,
                extractor=extractor,
                score=max(0.0, min(1.0, base_score + bonus)),
            )
        )

    _push(_extract_with_trafilatura(html_text, url, prefer_precision=prefer_precision), "trafilatura")
    _push(_extract_with_wikipedia_blocks(html_text, url), "wikipedia-blocks", bonus=0.12)
    _push(_extract_with_readability(html_text), "readability")
    _push(_extract_with_goose(html_text), "goose3")
    _push(_extract_with_newspaper(html_text, url), "newspaper")
    _push(_extract_with_bs4_main_content(html_text), "bs4-main-content", bonus=0.06)
    if use_transformer_ranker:
        _push(_extract_with_transformer_ranker(html_text), "transformer-ranker")
    _push(_extract_with_bs4_raw(html_text), "bs4-raw")

    if not candidates:
        return _ExtractionResult(text="", extractor="none", score=0.0)

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[0]


def _fetch_with_retry(
    client: httpx.Client,
    url: str,
    *,
    retries: int,
    backoff_sec: float,
) -> Optional[httpx.Response]:
    last: Optional[httpx.Response] = None
    for attempt in range(retries + 1):
        try:
            resp = client.get(url)
            if resp.status_code in {429, 500, 502, 503, 504}:
                last = resp
                raise httpx.HTTPStatusError(
                    "transient status", request=resp.request, response=resp
                )
            resp.raise_for_status()
            return resp
        except Exception:  # pylint: disable=broad-except
            if attempt < retries:
                time.sleep(backoff_sec * (2 ** attempt))
            continue
    return last


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
    except Exception:  # pylint: disable=broad-except
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
    except Exception:  # pylint: disable=broad-except
        return True


def _enqueue_links(
    queue: Deque[Tuple[str, int]],
    links: Iterable[str],
    *,
    depth: int,
    visited: Set[str],
    same_domain: bool,
    start_domain: str,
) -> None:
    for link in links:
        if link in visited:
            continue
        if same_domain and urlparse(link).netloc.lower() != start_domain:
            continue
        queue.append((link, depth + 1))


def parse_website(  # pylint: disable=too-many-arguments,too-many-locals
    seed_url: str,
    *,
    max_pages: int = 10,
    max_depth: int = 1,
    same_domain: bool = True,
    min_chars: int = 280,
    timeout_sec: float = 20.0,
    retries: int = 2,
    backoff_sec: float = 1.2,
    respect_robots: bool = True,
    user_agent: Optional[str] = None,
    enable_browser_fallback: bool = False,
    browser_timeout_ms: int = 18000,
    use_transformer_ranker: bool = False,
    prefer_precision: bool = True,
    config: Optional[ParserConfig] = None,
) -> List[ParsedPage]:
    cfg = config or ParserConfig(
        max_pages=max_pages,
        max_depth=max_depth,
        same_domain=same_domain,
        min_chars=min_chars,
        timeout_sec=timeout_sec,
        retries=retries,
        backoff_sec=backoff_sec,
        respect_robots=respect_robots,
        user_agent=user_agent or ParserConfig.user_agent,
        enable_browser_fallback=enable_browser_fallback,
        browser_timeout_ms=browser_timeout_ms,
        use_transformer_ranker=use_transformer_ranker,
        prefer_precision=prefer_precision,
    )

    start_url = _normalize_url(seed_url)
    start_domain = urlparse(start_url).netloc.lower()

    queue: Deque[Tuple[str, int]] = deque([(start_url, 0)])
    visited: Set[str] = set()
    pages: List[ParsedPage] = []

    headers = {"User-Agent": cfg.user_agent}

    with httpx.Client(
        follow_redirects=True,
        timeout=cfg.timeout_sec,
        headers=headers,
    ) as client:
        while queue and len(pages) < cfg.max_pages:
            current_url, depth = queue.popleft()
            current_url = urldefrag(current_url)[0]

            if current_url in visited:
                continue
            visited.add(current_url)

            if cfg.respect_robots and not _allowed_by_robots(current_url, cfg.user_agent):
                continue

            response = _fetch_with_retry(
                client,
                current_url,
                retries=cfg.retries,
                backoff_sec=cfg.backoff_sec,
            )
            if response is None or response.status_code >= 400:
                continue

            ctype = response.headers.get("content-type", "").lower()
            if "html" not in ctype and "xml" in ctype:
                continue

            html_text = response.text
            title = _extract_title(
                html_text,
                fallback=urlparse(current_url).path.strip("/") or "untitled",
            )

            picked = _pick_best_extraction(
                html_text,
                current_url,
                prefer_precision=cfg.prefer_precision,
                use_transformer_ranker=cfg.use_transformer_ranker,
            )

            if (
                len(picked.text) < cfg.min_chars
                and cfg.enable_browser_fallback
                and depth <= cfg.max_depth
            ):
                rendered_html = _render_with_playwright(
                    current_url,
                    timeout_ms=cfg.browser_timeout_ms,
                )
                if rendered_html:
                    rendered_pick = _pick_best_extraction(
                        rendered_html,
                        current_url,
                        prefer_precision=cfg.prefer_precision,
                        use_transformer_ranker=cfg.use_transformer_ranker,
                    )
                    if rendered_pick.score > picked.score:
                        html_text = rendered_html
                        picked = rendered_pick

            if len(picked.text) >= cfg.min_chars:
                pages.append(
                    ParsedPage(
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
                    )
                )

            if depth >= cfg.max_depth:
                continue

            links = _extract_links(html_text, current_url)
            _enqueue_links(
                queue,
                links,
                depth=depth,
                visited=visited,
                same_domain=cfg.same_domain,
                start_domain=start_domain,
            )

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


def summarize_extractors(pages: List[ParsedPage]) -> Dict[str, Any]:
    total = len(pages)
    by_extractor = Counter(p.extractor for p in pages)
    avg_quality = (
        round(sum(p.quality_score for p in pages) / total, 4)
        if total
        else 0.0
    )
    return {
        "total_pages": total,
        "avg_quality": avg_quality,
        "extractors": dict(by_extractor),
    }
