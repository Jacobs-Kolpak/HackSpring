"""
Сервис генерации mind map.

Строит граф концептов из текста на основе частотного анализа
и совместной встречаемости терминов в предложениях.
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# ── Константы ───────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-zа-яё0-9]+", flags=re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+", flags=re.MULTILINE)

STOPWORDS = {
    # Russian
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а",
    "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же",
    "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от",
    "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже",
    "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был",
    "него", "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там",
    "потом", "себя", "ничего", "ей", "может", "они", "тут", "где", "есть",
    "надо", "ней", "для", "мы", "тебя", "их", "чем", "была", "сам",
    "чтоб", "без", "будто", "человек", "чего", "раз", "тоже", "себе",
    "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти",
    "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем",
    "всех", "никогда", "можно", "при", "наконец", "два", "об", "другой",
    "хоть", "после", "над", "больше", "тот", "через", "эти", "нас",
    "про", "всего", "них", "какая", "много", "разве", "три", "эту",
    "моя", "впрочем", "хорошо", "свою", "этой", "перед", "иногда",
    "лучше", "чуть", "том", "нельзя", "такой", "им", "более", "всегда",
    "конечно", "всю", "между",
    # English
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for",
    "to", "of", "on", "in", "at", "by", "with", "from", "as", "is",
    "are", "was", "were", "be", "been", "being", "it", "this", "that",
    "these", "those", "we", "you", "they", "he", "she", "i", "my",
    "your", "their", "our", "not", "can", "could", "will", "would",
    "should", "about", "into", "out", "up", "down", "over", "under",
    "again", "more", "most",
}


# ── Внутренние функции ──────────────────────────────────────


def _tokenize(text: str) -> List[str]:
    tokens = [t.lower() for t in _WORD_RE.findall(text)]
    return [
        t for t in tokens
        if len(t) >= 3 and t not in STOPWORDS and not t.isdigit()
    ]


def _split_sentences(text: str) -> List[str]:
    return [
        part.strip()
        for part in _SENTENCE_RE.split(text)
        if part.strip()
    ]


def _term_frequencies(text: str) -> Counter:
    return Counter(_tokenize(text))


def _select_concepts(
    freq: Counter, top_n: int, min_freq: int
) -> List[str]:
    filtered = [
        (term, count) for term, count in freq.items() if count >= min_freq
    ]
    filtered.sort(key=lambda x: (-x[1], x[0]))
    return [term for term, _ in filtered[:top_n]]


def _cooccurrence(
    sentences: List[str], concepts: List[str]
) -> Counter:
    concept_set = set(concepts)
    edges: Counter = Counter()
    for sentence in sentences:
        found = sorted(
            set(tok for tok in _tokenize(sentence) if tok in concept_set)
        )
        for i in range(len(found)):
            for j in range(i + 1, len(found)):
                edges[(found[i], found[j])] += 1
    return edges


def _normalize_scores(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    min_v = min(values.values())
    max_v = max(values.values())
    if abs(max_v - min_v) < 1e-12:
        return {key: 1.0 for key in values}
    return {
        key: (val - min_v) / (max_v - min_v)
        for key, val in values.items()
    }


# ── Публичный API ───────────────────────────────────────────


def build_graph_data(
    text: str,
    top_n_concepts: int = 24,
    min_concept_freq: int = 2,
    min_edge_weight: int = 1,
) -> Dict[str, List[Dict]]:
    """
    Строит граф концептов из текста.

    Args:
        text: исходный текст.
        top_n_concepts: максимум узлов.
        min_concept_freq: минимальная частота термина.
        min_edge_weight: минимальный вес связи.

    Returns:
        ``{"nodes": [...], "edges": [...]}``.
    """
    freq = _term_frequencies(text)
    sentences = _split_sentences(text)
    if not freq:
        return {"nodes": [], "edges": []}

    # Берём широкий пул кандидатов, потом сужаем
    candidate_cap = max(top_n_concepts * 4, top_n_concepts)
    candidate_terms = _select_concepts(
        freq, top_n=candidate_cap, min_freq=min_concept_freq
    )
    candidate_edges = _cooccurrence(sentences, candidate_terms)

    # Считаем взвешенную степень
    degree_weight: Counter = Counter()
    for (src, dst), weight in candidate_edges.items():
        degree_weight[src] += weight
        degree_weight[dst] += weight

    # Комбинированная релевантность: частота + степень
    freq_scores = _normalize_scores(
        {term: float(freq[term]) for term in candidate_terms}
    )
    degree_scores = _normalize_scores(
        {term: float(degree_weight.get(term, 0.0)) for term in candidate_terms}
    )
    relevance: Dict[str, float] = {}
    for term in candidate_terms:
        relevance[term] = (
            0.62 * freq_scores.get(term, 0.0)
            + 0.38 * degree_scores.get(term, 0.0)
        )

    concepts = sorted(
        candidate_terms, key=lambda t: (-relevance[t], t)
    )[:top_n_concepts]
    edge_counts = _cooccurrence(sentences, concepts)

    nodes: List[Dict] = [
        {
            "id": concept,
            "label": concept,
            "title": f"{concept}: {freq[concept]}",
            "value": max(1, freq[concept]),
            "relevance": round(relevance.get(concept, 0.0), 4),
        }
        for concept in concepts
    ]

    edges: List[Dict] = [
        {
            "from": source,
            "to": target,
            "value": weight,
            "title": f"Совместные упоминания: {weight}",
        }
        for (source, target), weight in edge_counts.items()
        if weight >= min_edge_weight
    ]

    # Убираем изолированные узлы
    connected_ids = {e["from"] for e in edges} | {e["to"] for e in edges}
    if connected_ids:
        nodes = [n for n in nodes if n["id"] in connected_ids]

    edges.sort(key=lambda e: (-e["value"], e["from"], e["to"]))
    return {"nodes": nodes, "edges": edges}


def render_mindmap_html(
    text: str,
    output_html: Path,
    title: str,
    top_n_concepts: int = 24,
    min_concept_freq: int = 2,
    min_edge_weight: int = 1,
) -> Tuple[int, int]:
    """
    Генерирует интерактивный HTML mind map и сохраняет в файл.

    Returns:
        Кортеж (количество узлов, количество рёбер).
    """
    graph_data = build_graph_data(
        text=text,
        top_n_concepts=top_n_concepts,
        min_concept_freq=min_concept_freq,
        min_edge_weight=min_edge_weight,
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        _build_html(graph_data, title=title), encoding="utf-8"
    )
    return len(graph_data["nodes"]), len(graph_data["edges"])


def _build_html(graph_data: Dict[str, List[Dict]], title: str) -> str:
    """Генерирует полный HTML с vis-network для визуализации графа."""
    safe_title = html.escape(title)
    payload = json.dumps(graph_data, ensure_ascii=False)

    # HTML-шаблон вынесен как строка — слишком большой для inline
    return _HTML_TEMPLATE.format(title=safe_title, payload=payload)


# Шаблон HTML хранится отдельно для читаемости
_HTML_TEMPLATE = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <script src="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"></script>
  <style>
    body {{ margin:0; font-family:sans-serif; background:#11131a; color:#dbe7ff; }}
    #mindmap {{ width:100%; height:90vh; }}
  </style>
</head>
<body>
  <h1 style="padding:16px">{title}</h1>
  <div id="mindmap"></div>
  <script>
    const data = {payload};
    const nodes = new vis.DataSet(data.nodes.map(n => ({{
      ...n, shape:'dot', scaling:{{min:12,max:46}},
      color:{{ background:'#7c8cff', border:'#6578ff' }},
      font:{{ color:'#08101e' }}
    }})));
    const edges = new vis.DataSet(data.edges.map(e => ({{
      ...e, width:1+Math.min(5,e.value*0.55),
      color:{{ color:'#5b6d90' }}, smooth:{{ type:'continuous' }}
    }})));
    new vis.Network(document.getElementById('mindmap'),
      {{ nodes, edges }},
      {{ physics:{{ stabilization:{{ iterations:420 }},
         barnesHut:{{ gravitationalConstant:-8300 }} }} }}
    );
  </script>
</body>
</html>"""
