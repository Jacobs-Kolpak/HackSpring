from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

_WORD_RE = re.compile(r"[a-zа-яё0-9]+", flags=re.IGNORECASE)
_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+", flags=re.MULTILINE)

STOPWORDS = {
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
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for",
    "to", "of", "on", "in", "at", "by", "with", "from", "as", "is",
    "are", "was", "were", "be", "been", "being", "it", "this", "that",
    "these", "those", "we", "you", "they", "he", "she", "i", "my",
    "your", "their", "our", "not", "can", "could", "will", "would",
    "should", "about", "into", "out", "up", "down", "over", "under",
    "again", "more", "most",
}


def _tokenize(text: str) -> List[str]:
    return [
        t.lower() for t in _WORD_RE.findall(text)
        if len(t) >= 3 and t.lower() not in STOPWORDS and not t.isdigit()
    ]


def _sentences(text: str) -> List[str]:
    return [p.strip() for p in _SENT_RE.split(text) if p.strip()]


def _cooccurrence(sents: List[str], concepts: set) -> Counter:
    edges: Counter = Counter()
    for sent in sents:
        found = sorted(set(t for t in _tokenize(sent) if t in concepts))
        for i, src in enumerate(found):
            for dst in found[i + 1:]:
                edges[(src, dst)] += 1
    return edges


def _norm(vals: Dict[str, float]) -> Dict[str, float]:
    if not vals:
        return {}
    lo, hi = min(vals.values()), max(vals.values())
    if abs(hi - lo) < 1e-12:
        return {k: 1.0 for k in vals}
    return {k: (v - lo) / (hi - lo) for k, v in vals.items()}


def build_graph_data(  # pylint: disable=too-many-locals
    text: str,
    *,
    top_n: int = 24,
    min_freq: int = 2,
    min_edge: int = 1,
) -> Dict[str, List[Dict]]:
    freq = Counter(_tokenize(text))
    sents = _sentences(text)
    if not freq:
        return {"nodes": [], "edges": []}

    pool = sorted(
        ((t, c) for t, c in freq.items() if c >= min_freq),
        key=lambda x: (-x[1], x[0]),
    )[:top_n * 4]
    terms = [t for t, _ in pool]

    edges = _cooccurrence(sents, set(terms))
    degree: Counter = Counter()
    for (s, d), w in edges.items():
        degree[s] += w
        degree[d] += w

    freq_n = _norm({t: float(freq[t]) for t in terms})
    deg_n = _norm({t: float(degree.get(t, 0)) for t in terms})
    relevance = {t: 0.62 * freq_n.get(t, 0) + 0.38 * deg_n.get(t, 0) for t in terms}

    top = sorted(terms, key=lambda t: (-relevance[t], t))[:top_n]
    top_set = set(top)
    final_edges = _cooccurrence(sents, top_set)

    nodes = [
        {
            "id": t, "label": t,
            "title": f"{t}: {freq[t]}",
            "value": max(1, freq[t]),
            "relevance": round(relevance.get(t, 0), 4),
        }
        for t in top
    ]
    edge_list = [
        {"from": s, "to": d, "value": w, "title": f"Совместные упоминания: {w}"}
        for (s, d), w in final_edges.items() if w >= min_edge
    ]

    connected = {e["from"] for e in edge_list} | {e["to"] for e in edge_list}
    if connected:
        nodes = [n for n in nodes if n["id"] in connected]
    edge_list.sort(key=lambda e: (-e["value"], e["from"], e["to"]))

    return {"nodes": nodes, "edges": edge_list}


def render_html(
    text: str,
    output: Path,
    title: str,
    *,
    top_n: int = 24,
    min_freq: int = 2,
    min_edge: int = 1,
) -> Tuple[int, int]:
    graph = build_graph_data(text, top_n=top_n, min_freq=min_freq, min_edge=min_edge)
    output.parent.mkdir(parents=True, exist_ok=True)
    safe = html.escape(title)
    payload = json.dumps(graph, ensure_ascii=False)
    output.write_text(
        _HTML.format(title=safe, payload=payload), encoding="utf-8"
    )
    return len(graph["nodes"]), len(graph["edges"])


_HTML = """<!doctype html>
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
