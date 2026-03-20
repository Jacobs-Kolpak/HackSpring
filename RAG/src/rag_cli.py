import argparse
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from docx import Document
from pypdf import PdfReader
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
import httpx


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@dataclass
class Chunk:
    chunk_id: str
    source_path: str
    source_name: str
    chunk_index: int
    text: str


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fix_pdf_artifacts(text: str) -> str:
    text = re.sub(r"([A-Za-zА-Яа-яЁё])\-\n([A-Za-zА-Яа-яЁё])", r"\1\2", text)
    text = re.sub(
        r"(?:(?<=\s)|^)([A-Za-zА-Яа-яЁё](?:\s+[A-Za-zА-Яа-яЁё]){2,})(?=\s|$)",
        lambda m: m.group(1).replace(" ", ""),
        text,
    )
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def decode_substitution_cyrillic(text: str) -> str:
    # Some PDFs have broken Cyrillic cmap and come as sequences like "@57C;LB0B".
    # Decode only runs that strongly look like this substitution alphabet.
    mapping = {
        "\x10": "А",
        "\x11": "Б",
        "\x12": "В",
        "\x13": "Г",
        "\x14": "Д",
        "\x15": "Е",
        "\x16": "Ж",
        "\x17": "З",
        "\x18": "И",
        "\x19": "Й",
        "\x1a": "К",
        "\x1b": "Л",
        "\x1c": "М",
        "\x1d": "Н",
        "\x1e": "О",
        "\x1f": "П",
        "!": "Р",
        "\"": "С",
        "#": "Т",
        "$": "У",
        "%": "Ф",
        "&": "Х",
        "'": "Ц",
        "(": "Ч",
        ")": "Ш",
        "*": "Щ",
        "+": "Ъ",
        ",": "Ы",
        "-": "Ь",
        ".": "Э",
        "/": "Ю",
        "0": "а",
        "1": "б",
        "2": "в",
        "3": "г",
        "4": "д",
        "5": "е",
        "6": "ж",
        "7": "з",
        "8": "и",
        "9": "й",
        ":": "к",
        ";": "л",
        "<": "м",
        "=": "н",
        ">": "о",
        "?": "п",
        "@": "р",
        "A": "с",
        "B": "т",
        "C": "у",
        "D": "ф",
        "E": "х",
        "F": "ц",
        "G": "ч",
        "H": "ш",
        "I": "щ",
        "J": "ъ",
        "K": "ы",
        "L": "ь",
        "M": "э",
        "N": "ю",
        "O": "я",
    }

    mapped_chars = set(mapping.keys())

    def decode_token(match: re.Match) -> str:
        token = match.group(0)
        if "http" in token.lower() or "/" in token:
            return token
        hits = sum(1 for ch in token if ch in mapped_chars)
        if hits < 3:
            return token
        if hits / max(1, len(token)) < 0.55:
            return token
        return "".join(mapping.get(ch, ch) for ch in token)

    # Decode suspicious non-space tokens that mostly consist of substitution alphabet symbols.
    return re.sub(r"[^\s]+", decode_token, text)


def normalize_text(text: str) -> str:
    text = decode_substitution_cyrillic(text)
    text = fix_pdf_artifacts(text)
    text = normalize_whitespace(text)
    return text


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_pymupdf(path: Path) -> str:
    import fitz

    parts: List[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            blocks = page.get_text("blocks")
            blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))
            page_lines = []
            for block in blocks:
                block_text = (block[4] or "").strip()
                if block_text:
                    page_lines.append(block_text)
            parts.append("\n".join(page_lines))
    return "\n\n".join(parts)


def read_pdf_pypdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def read_pdf(path: Path) -> str:
    try:
        return read_pdf_pymupdf(path)
    except Exception:
        return read_pdf_pypdf(path)


def read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


def read_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt":
        raw = read_txt(path)
    elif ext == ".pdf":
        raw = read_pdf(path)
    elif ext == ".docx":
        raw = read_docx(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")
    return normalize_text(raw)


def collect_files(inputs: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for item in inputs:
        path = Path(item).expanduser().resolve()
        if not path.exists():
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
        elif path.is_dir():
            for file_path in path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(file_path.resolve())
    return sorted(set(files))


def split_sentences(text: str) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    sentences: List[str] = []
    for para in paragraphs:
        parts = re.split(r"(?<=[.!?])\s+", para)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def chunk_text_sentence_aware(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(sentence):
                piece = sentence[start : start + chunk_size].strip()
                if piece:
                    chunks.append(piece)
                start += max(1, chunk_size - chunk_overlap)
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            overlap_tail = current[-chunk_overlap:].strip()
            current = f"{overlap_tail} {sentence}".strip() if overlap_tail else sentence
        else:
            current = sentence

    if current:
        chunks.append(current.strip())

    return chunks


def make_chunk_id(path: Path, chunk_index: int, text: str) -> str:
    stable_key = f"{path}:{chunk_index}:{text}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))


def build_chunks(paths: List[Path], chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    result: List[Chunk] = []

    for path in paths:
        text = read_document(path)
        if not text:
            continue

        pieces = chunk_text_sentence_aware(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, piece in enumerate(pieces):
            result.append(
                Chunk(
                    chunk_id=make_chunk_id(path, idx, piece),
                    source_path=str(path),
                    source_name=path.name,
                    chunk_index=idx,
                    text=piece,
                )
            )

    return result


def batched(items: List, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def normalize_openai_base_url(url: str) -> str:
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def resolve_api_key(primary: Optional[str], fallback_envs: List[str]) -> Optional[str]:
    if primary:
        return primary
    for env_name in fallback_envs:
        value = os.getenv(env_name)
        if value:
            return value
    return None


def get_embedder(model_name: str, cache_dir: Path, embedder_url: Optional[str], embedder_api_key: Optional[str]):
    if embedder_url:
        from openai import OpenAI

        api_key = resolve_api_key(embedder_api_key, ["HACKAI_API_KEY", "OPENAI_API_KEY"])
        if not api_key:
            raise SystemExit("Set --embedder-api-key (or HACKAI_API_KEY/OPENAI_API_KEY) for remote embedder.")
        client = OpenAI(
            api_key=api_key,
            base_url=normalize_openai_base_url(embedder_url),
            timeout=120.0,
            max_retries=2,
        )
        return {"kind": "api", "client": client}

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["FASTEMBED_CACHE_PATH"] = str(cache_dir)
    from fastembed import TextEmbedding

    return {"kind": "local", "client": TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))}


def prepare_embedding_text(text: str, model_name: str, is_query: bool) -> str:
    lowered = model_name.lower()
    if "e5" in lowered:
        return f"{'query' if is_query else 'passage'}: {text}"
    return text


def embed_texts(embedder, texts: List[str], model_name: str, is_query: bool) -> List[List[float]]:
    prepared = [prepare_embedding_text(t, model_name=model_name, is_query=is_query) for t in texts]
    if embedder["kind"] == "api":
        last_err = None
        for attempt in range(4):
            try:
                resp = embedder["client"].embeddings.create(model=model_name, input=prepared)
                return [item.embedding for item in resp.data]
            except Exception as exc:
                last_err = exc
                if attempt < 3:
                    time.sleep(1.5 * (2**attempt))
        raise SystemExit(f"Embedder request failed after retries: {last_err}")

    vectors = []
    for vec in embedder["client"].embed(prepared):
        vectors.append(vec.tolist())
    return vectors


def tokenize_for_lexical(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"[a-zа-яё0-9]+", text)
    return [tok for tok in tokens if len(tok) > 1]


def minmax_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def get_client(
    db_path: Optional[Path] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
) -> QdrantClient:
    if qdrant_url:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    if not db_path:
        raise ValueError("db_path is required for local mode")
    db_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(db_path))


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    if client.collection_exists(collection_name=collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )


def remove_existing_for_sources(client: QdrantClient, collection_name: str, source_paths: List[str]) -> None:
    for src in source_paths:
        client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="source_path", match=models.MatchValue(value=src))]
                )
            ),
            wait=True,
        )


def cmd_ingest(args: argparse.Namespace) -> None:
    files = collect_files(args.inputs)
    if not files:
        raise SystemExit("No supported files found (.pdf, .docx, .txt)")

    chunks = build_chunks(files, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    if not chunks:
        raise SystemExit("No text chunks extracted. Check input files.")

    db_path = Path(args.db_path).resolve() if args.db_path else None
    cache_dir = Path(args.embedding_cache).resolve()

    client = get_client(db_path=db_path, qdrant_url=args.qdrant_url, qdrant_api_key=args.qdrant_api_key)
    embedder = get_embedder(
        model_name=args.embedding_model,
        cache_dir=cache_dir,
        embedder_url=args.embedder_url,
        embedder_api_key=args.embedder_api_key or args.api_key,
    )

    sample_vector = embed_texts(embedder, [chunks[0].text], model_name=args.embedding_model, is_query=False)[0]
    ensure_collection(client=client, collection_name=args.collection, vector_size=len(sample_vector))

    remove_existing_for_sources(client, args.collection, [str(p) for p in files])

    total_upserted = 0
    for batch in batched(chunks, args.batch_size):
        batch_texts = [item.text for item in batch]
        batch_vectors = embed_texts(embedder, batch_texts, model_name=args.embedding_model, is_query=False)
        points = []

        for item, vector in zip(batch, batch_vectors):
            points.append(
                models.PointStruct(
                    id=item.chunk_id,
                    vector=vector,
                    payload={
                        "source_path": item.source_path,
                        "source_name": item.source_name,
                        "chunk_index": item.chunk_index,
                        "chunk_id": item.chunk_id,
                        "text": item.text,
                    },
                )
            )

        client.upsert(collection_name=args.collection, points=points, wait=True)
        total_upserted += len(points)

    print(f"Indexed files: {len(files)}")
    print(f"Inserted chunks: {total_upserted}")
    if db_path:
        print(f"DB path: {db_path}")
    if args.qdrant_url:
        print(f"Qdrant URL: {args.qdrant_url}")
    print(f"Collection: {args.collection}")
    print(f"Embedding model: {args.embedding_model}")


def rerank_candidates(query: str, candidates: List[dict], dense_weight: float) -> List[dict]:
    dense_weight = min(1.0, max(0.0, dense_weight))
    bm25_weight = 1.0 - dense_weight

    tokenized_docs = [tokenize_for_lexical(c["text"]) for c in candidates]
    tokenized_query = tokenize_for_lexical(query)

    bm25_scores = [0.0 for _ in candidates]
    if tokenized_query and any(tokenized_docs):
        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = bm25.get_scores(tokenized_query).tolist()

    dense_scores = [c["dense_score_raw"] for c in candidates]
    dense_norm = minmax_normalize(dense_scores)
    bm25_norm = minmax_normalize(bm25_scores)

    reranked: List[dict] = []
    for c, d_norm, b_norm in zip(candidates, dense_norm, bm25_norm):
        final_score = dense_weight * d_norm + bm25_weight * b_norm
        item = dict(c)
        item["dense_score"] = d_norm
        item["bm25_score"] = b_norm
        item["score"] = final_score
        reranked.append(item)

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked


def api_rerank_scores(
    rerank_url: str,
    rerank_api_key: str,
    rerank_model: Optional[str],
    query: str,
    documents: List[str],
) -> Optional[List[float]]:
    url = f"{normalize_openai_base_url(rerank_url)}/rerank"
    headers = {"Authorization": f"Bearer {rerank_api_key}"}
    payload = {
        "query": query,
        "documents": documents,
        "top_n": len(documents),
    }
    if rerank_model:
        payload["model"] = rerank_model

    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=40.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    # Common formats: {"results":[{"index":i,"relevance_score":x}]} or {"data":[...]}
    rows = data.get("results") or data.get("data") or []
    scores = [0.0 for _ in documents]
    for row in rows:
        idx = row.get("index")
        if idx is None or not (0 <= int(idx) < len(documents)):
            continue
        score = row.get("relevance_score")
        if score is None:
            score = row.get("score", 0.0)
        scores[int(idx)] = float(score)
    return scores


def retrieve_chunks(
    query: str,
    db_path: Optional[Path],
    qdrant_url: Optional[str],
    qdrant_api_key: Optional[str],
    collection_name: str,
    embedding_model: str,
    embedding_cache: Path,
    embedder_url: Optional[str],
    embedder_api_key: Optional[str],
    top_k: int,
    fetch_k: int,
    min_score: float,
    dense_weight: float,
    source_name: Optional[str],
    rerank_url: Optional[str],
    rerank_api_key: Optional[str],
    rerank_model: Optional[str],
):
    fetch_k = max(fetch_k, top_k)
    client = get_client(db_path=db_path, qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key)
    embedder = get_embedder(
        model_name=embedding_model,
        cache_dir=embedding_cache,
        embedder_url=embedder_url,
        embedder_api_key=embedder_api_key,
    )
    query_vector = embed_texts(embedder, [query], model_name=embedding_model, is_query=True)[0]

    query_filter = None
    if source_name:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="source_name", match=models.MatchValue(value=source_name))]
        )

    hits_response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=fetch_k,
        query_filter=query_filter,
        with_payload=True,
        with_vectors=False,
    )
    hits = hits_response.points

    candidates = []
    for hit in hits:
        payload = hit.payload or {}
        text = str(payload.get("text", ""))
        if not text.strip():
            continue
        candidates.append(
            {
                "rank": 0,
                "dense_score_raw": float(hit.score),
                "chunk_id": payload.get("chunk_id", str(hit.id)),
                "source_name": payload.get("source_name", "unknown"),
                "source_path": payload.get("source_path", "unknown"),
                "chunk_index": payload.get("chunk_index", -1),
                "text": text,
            }
        )

    # Default path: no local rerank, return by dense similarity.
    reranked = []
    for c in sorted(candidates, key=lambda x: x["dense_score_raw"], reverse=True):
        item = dict(c)
        item["dense_score"] = float(c["dense_score_raw"])
        item["bm25_score"] = 0.0
        item["score"] = float(c["dense_score_raw"])
        reranked.append(item)

    # Best mode for this project: when external embedder is used, rerank via the same endpoint by default.
    resolved_rerank_url = rerank_url or embedder_url
    resolved_rerank_key = resolve_api_key(
        rerank_api_key,
        ["HACKAI_API_KEY", "OPENAI_API_KEY"],
    )
    if resolved_rerank_url and resolved_rerank_key and candidates:
        api_scores = api_rerank_scores(
            rerank_url=resolved_rerank_url,
            rerank_api_key=resolved_rerank_key,
            rerank_model=rerank_model,
            query=query,
            documents=[c["text"] for c in candidates],
        )
        if api_scores is not None:
            dense_norm = minmax_normalize([c["dense_score_raw"] for c in candidates])
            api_norm = minmax_normalize(api_scores)
            mixed = []
            for c, d_norm, a_norm in zip(candidates, dense_norm, api_norm):
                item = dict(c)
                item["dense_score"] = d_norm
                item["bm25_score"] = a_norm
                item["score"] = dense_weight * d_norm + (1.0 - dense_weight) * a_norm
                mixed.append(item)
            mixed.sort(key=lambda x: x["score"], reverse=True)
            reranked = mixed

    results = [item for item in reranked if item["score"] >= min_score][:top_k]
    for idx, item in enumerate(results, start=1):
        item["rank"] = idx
    return results


def cmd_retrieve(args: argparse.Namespace) -> None:
    results = retrieve_chunks(
        query=args.query,
        db_path=Path(args.db_path).resolve() if args.db_path else None,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        embedding_cache=Path(args.embedding_cache).resolve(),
        embedder_url=args.embedder_url,
        embedder_api_key=args.embedder_api_key or args.api_key,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        min_score=args.min_score,
        dense_weight=args.dense_weight,
        source_name=args.source_name,
        rerank_url=args.rerank_url,
        rerank_api_key=args.rerank_api_key or args.api_key,
        rerank_model=args.rerank_model,
    )

    output = {
        "query": args.query,
        "top_k": args.top_k,
        "returned": len(results),
        "min_score": args.min_score,
        "fetch_k": args.fetch_k,
        "dense_weight": args.dense_weight,
        "db_path": str(Path(args.db_path).resolve()) if args.db_path else None,
        "qdrant_url": args.qdrant_url,
        "collection": args.collection,
        "embedding_model": args.embedding_model,
        "embedder_url": args.embedder_url,
        "results": results,
    }

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    if args.raw:
        print(f"Query: {args.query}")
        print(f"Top-k requested: {args.top_k}")
        print(f"Returned: {len(results)}")
        print(f"Min score: {args.min_score}")
        print(f"Fetch-k: {args.fetch_k}")
        print(f"Dense weight: {args.dense_weight}\n")

        for item in results:
            print(
                f"[{item['rank']}] score={item['score']:.4f} "
                f"(dense={item['dense_score']:.4f}, bm25={item['bm25_score']:.4f}, raw_dense={item['dense_score_raw']:.4f}) "
                f"| {item['source_name']} | chunk={item['chunk_index']}"
            )
            print(item["text"])
            print("-" * 80)
        return

    resolved_llm_url = args.llm_url
    if not resolved_llm_url and args.embedder_url and ":6620" in args.embedder_url:
        resolved_llm_url = args.embedder_url.replace(":6620", ":6630")

    if not results:
        print("No relevant chunks found.")
        return

    if resolved_llm_url:
        context = build_context(results)
        answer = generate_answer_with_openai(
            query=args.query,
            context=context,
            model=args.summary_model,
            llm_url=resolved_llm_url,
            llm_api_key=args.llm_api_key or args.api_key,
        )
        print("Answer:")
        print(answer)
        print("\nSources:")
        for item in results:
            print(
                f"- {item['source_name']} chunk={item['chunk_index']} "
                f"score={item['score']:.4f}"
            )
        return

    print("No --llm-url provided. Use --raw for debug chunks or pass --llm-url for human-readable output.")


def build_context(results: List[dict]) -> str:
    blocks = []
    for item in results:
        blocks.append(
            f"[Source: {item['source_name']} | chunk={item['chunk_index']} | score={item['score']:.4f}]\n"
            f"{item['text']}"
        )
    return "\n\n".join(blocks)


def generate_answer_with_openai(
    query: str,
    context: str,
    model: str,
    llm_url: Optional[str],
    llm_api_key: Optional[str],
) -> str:
    api_key = resolve_api_key(llm_api_key, ["HACKAI_API_KEY", "OPENAI_API_KEY"])
    if not api_key:
        raise SystemExit("Set --llm-api-key (or HACKAI_API_KEY/OPENAI_API_KEY) for ask command.")

    from openai import OpenAI

    if llm_url:
        client = OpenAI(
            api_key=api_key,
            base_url=normalize_openai_base_url(llm_url),
            timeout=180.0,
            max_retries=2,
        )
    else:
        client = OpenAI(api_key=api_key, timeout=180.0, max_retries=2)

    system_prompt = (
        "Ты RAG-ассистент. Отвечай строго на основе переданного контекста. "
        "Если данных не хватает, прямо скажи об этом."
    )
    user_prompt = (
        f"Вопрос:\n{query}\n\n"
        f"Контекст:\n{context}\n\n"
        "Дай краткий и точный ответ на русском языке."
    )

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return response.output_text.strip()


def cmd_ask(args: argparse.Namespace) -> None:
    results = retrieve_chunks(
        query=args.query,
        db_path=Path(args.db_path).resolve() if args.db_path else None,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        embedding_cache=Path(args.embedding_cache).resolve(),
        embedder_url=args.embedder_url,
        embedder_api_key=args.embedder_api_key or args.api_key,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        min_score=args.min_score,
        dense_weight=args.dense_weight,
        source_name=args.source_name,
        rerank_url=args.rerank_url,
        rerank_api_key=args.rerank_api_key or args.api_key,
        rerank_model=args.rerank_model,
    )

    if not results:
        print("No relevant chunks found. Increase top-k or reduce min-score.")
        return

    context = build_context(results)
    answer = generate_answer_with_openai(
        query=args.query,
        context=context,
        model=args.model,
        llm_url=args.llm_url,
        llm_api_key=args.llm_api_key or args.api_key,
    )

    output = {
        "query": args.query,
        "answer": answer,
        "model": args.model,
        "used_chunks": len(results),
        "results": results,
    }

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    print("Answer:")
    print(answer)
    print("\nSources:")
    for item in results:
        print(
            f"- {item['source_name']} chunk={item['chunk_index']} "
            f"score={item['score']:.4f} ({item['source_path']})"
        )


def add_shared_retrieval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--api-key", default=None, help="Shared API key for embedder/rerank/llm")
    p.add_argument("--db-path", default="data/qdrant", help="Path to local Qdrant storage")
    p.add_argument("--qdrant-url", default=None, help="Remote Qdrant URL, e.g. http://localhost:6333")
    p.add_argument("--qdrant-api-key", default=None, help="Qdrant API key for remote mode")
    p.add_argument("--collection", default="docs", help="Collection name")
    p.add_argument("--embedder-url", default=None, help="OpenAI-compatible embeddings endpoint base URL")
    p.add_argument("--embedder-api-key", default=None, help="API key for embedder endpoint")
    p.add_argument("--rerank-url", default=None, help="OpenAI-compatible rerank endpoint base URL")
    p.add_argument("--rerank-api-key", default=None, help="API key for rerank endpoint")
    p.add_argument("--rerank-model", default=None, help="Optional rerank model name")
    p.add_argument("--embedding-model", default="intfloat/multilingual-e5-large", help="FastEmbed model")
    p.add_argument("--embedding-cache", default="data/embedding_cache", help="Path to embedding model cache")
    p.add_argument("--top-k", type=int, default=5, help="How many chunks to return")
    p.add_argument("--fetch-k", type=int, default=30, help="How many candidates to fetch before rerank")
    p.add_argument("--min-score", type=float, default=0.20, help="Min final reranked score (0..1)")
    p.add_argument("--dense-weight", type=float, default=0.75, help="Dense weight in hybrid rerank (0..1)")
    p.add_argument("--source-name", default=None, help="Optional source filename filter")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG CLI with Qdrant + Hybrid Rerank")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest files into Qdrant")
    ingest.add_argument("inputs", nargs="+", help="Files or directories with PDF/DOCX/TXT")
    ingest.add_argument("--db-path", default="data/qdrant", help="Path to local Qdrant storage")
    ingest.add_argument("--qdrant-url", default=None, help="Remote Qdrant URL")
    ingest.add_argument("--qdrant-api-key", default=None, help="Qdrant API key")
    ingest.add_argument("--api-key", default=None, help="Shared API key for embedder/rerank/llm")
    ingest.add_argument("--embedder-url", default=None, help="OpenAI-compatible embeddings endpoint base URL")
    ingest.add_argument("--embedder-api-key", default=None, help="API key for embedder endpoint")
    ingest.add_argument("--collection", default="docs", help="Collection name")
    ingest.add_argument("--chunk-size", type=int, default=900, help="Chunk size in chars")
    ingest.add_argument("--chunk-overlap", type=int, default=180, help="Chunk overlap in chars")
    ingest.add_argument("--batch-size", type=int, default=64, help="Upsert batch size")
    ingest.add_argument("--embedding-model", default="intfloat/multilingual-e5-large", help="FastEmbed model")
    ingest.add_argument("--embedding-cache", default="data/embedding_cache", help="Path to embedding model cache")
    ingest.set_defaults(func=cmd_ingest)

    retrieve = sub.add_parser("retrieve", help="Retrieve top-k chunks from Qdrant")
    retrieve.add_argument("--query", required=True, help="Question/query text")
    add_shared_retrieval_args(retrieve)
    retrieve.add_argument("--llm-url", default=None, help="OpenAI-compatible LLM endpoint base URL")
    retrieve.add_argument("--llm-api-key", default=None, help="API key for LLM endpoint")
    retrieve.add_argument("--summary-model", default="gpt-oss-20b", help="Model for human-readable retrieve output")
    retrieve.add_argument("--raw", action="store_true", help="Show raw chunks (debug mode)")
    retrieve.add_argument("--json", action="store_true", help="Output JSON")
    retrieve.set_defaults(func=cmd_retrieve)

    ask = sub.add_parser("ask", help="Retrieve + generate final answer with OpenAI")
    ask.add_argument("--query", required=True, help="Question/query text")
    add_shared_retrieval_args(ask)
    ask.add_argument("--llm-url", default=None, help="OpenAI-compatible LLM endpoint base URL")
    ask.add_argument("--llm-api-key", default=None, help="API key for LLM endpoint")
    ask.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model for answer generation")
    ask.add_argument("--json", action="store_true", help="Output JSON")
    ask.set_defaults(func=cmd_ask)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
