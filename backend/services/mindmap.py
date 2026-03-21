from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

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

GENERIC_WORDS = {
    "это", "этот", "эта", "эти", "такой", "такая", "такое", "такие",
    "какой", "какая", "какое", "какие", "который", "которая", "которое",
    "которые", "потому", "поэтому", "также", "очень", "просто", "пример",
    "вопрос", "ответ", "данный", "данная", "данное", "данные", "тема",
    "раздел", "часть", "уровень", "случай", "модель", "система",
}

_RU_SUFFIXES = (
    "иями", "ями", "ами", "иях", "ях", "ием", "ем", "ом", "ой", "ий", "ый",
    "ого", "его", "ому", "ему", "ыми", "ими", "ую", "юю", "ая", "яя",
    "ое", "ее", "ые", "ие", "ах", "ях", "ам", "ям", "ов", "ев", "ей",
    "ы", "и", "а", "я", "е", "о", "у", "ю",
)
_EN_SUFFIXES = ("ing", "ed", "ly", "es", "s")


def _normalize_token(token: str) -> str:
    token = token.lower().replace("ё", "е")
    if len(token) >= 6:
        for suffix in _RU_SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                token = token[: -len(suffix)]
                break
        for suffix in _EN_SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                token = token[: -len(suffix)]
                break
    return token


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in _WORD_RE.findall(text):
        raw_l = raw.lower()
        if raw_l.isdigit() or len(raw_l) < 3:
            continue
        token = _normalize_token(raw_l)
        if (
            len(token) < 3
            or token in STOPWORDS
            or token in GENERIC_WORDS
            or token.isdigit()
        ):
            continue
        tokens.append(token)
    return tokens


def _sentences(text: str) -> List[str]:
    return [p.strip() for p in _SENT_RE.split(text) if p.strip()]


def _cooccurrence(sents: List[str], concepts: Set[str]) -> Counter:
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


def _sentence_contains(term: str, sent: str) -> bool:
    return term in set(_tokenize(sent))


def _shorten(text: str, max_len: int = 180) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3].rstrip() + "..."


def _build_node_summary(term: str, sentences: Iterable[str]) -> str:
    for sent in sentences:
        if _sentence_contains(term, sent):
            return _shorten(sent)
    return f"Термин «{term}» — один из ключевых элементов текста."


def _build_graph_summary(
    top_terms: List[str],
    edges: Counter,
    freq: Counter,
) -> str:
    if not top_terms:
        return "Ключевые понятия в тексте не определены."

    key_terms = ", ".join(top_terms[:5])
    strongest = sorted(edges.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))[:3]
    if not strongest:
        return (
            "Граф отражает ключевые понятия текста: "
            f"{key_terms}. Связи между терминами выражены слабо."
        )

    links = "; ".join(
        f"{a} <-> {b} ({w})" for (a, b), w in strongest
    )
    dominant = max(top_terms, key=lambda t: freq[t])
    return (
        "Граф показывает основные темы текста и их совместные упоминания. "
        f"Доминирующий фокус: «{dominant}». Ключевые термины: {key_terms}. "
        f"Самые сильные связи: {links}."
    )


def build_graph_data(  # pylint: disable=too-many-locals
    text: str,
    *,
    top_n: int = 16,
    min_freq: int = 2,
    min_edge: int = 2,
) -> Dict[str, List[Dict]]:
    sents = _sentences(text)
    tokenized_sents = [_tokenize(s) for s in sents]
    freq = Counter(t for sent in tokenized_sents for t in sent)
    if not freq:
        return {"nodes": [], "edges": [], "meta": {"summary": ""}}

    sentence_count = max(1, len(tokenized_sents))
    sentence_df = Counter()
    for sent_tokens in tokenized_sents:
        sentence_df.update(set(sent_tokens))

    pool = sorted(
        (
            (t, c)
            for t, c in freq.items()
            if c >= min_freq
            and sentence_df[t] / sentence_count <= 0.65
            and t not in GENERIC_WORDS
        ),
        key=lambda x: (
            -x[1],
            -(sentence_df[x[0]] / sentence_count),
            x[0],
        ),
    )[: max(top_n * 4, 40)]
    terms = [t for t, _ in pool]

    edges = _cooccurrence(sents, set(terms))
    degree: Counter = Counter()
    for (s, d), w in edges.items():
        degree[s] += w
        degree[d] += w

    freq_n = _norm({t: float(freq[t]) for t in terms})
    deg_n = _norm({t: float(degree.get(t, 0)) for t in terms})
    relevance = {t: 0.62 * freq_n.get(t, 0) + 0.38 * deg_n.get(t, 0) for t in terms}

    top = sorted(
        terms,
        key=lambda t: (
            -relevance[t],
            -freq[t],
            sentence_df[t],
            t,
        ),
    )[:top_n]
    top_set = set(top)
    final_edges = _cooccurrence(sents, top_set)

    nodes = [
        {
            "id": t, "label": t,
            "title": f"{t}: {freq[t]}",
            "value": max(1, freq[t]),
            "relevance": round(relevance.get(t, 0), 4),
            "summary": _build_node_summary(t, sents),
        }
        for t in top
    ]
    edge_list = [
        {
            "from": s,
            "to": d,
            "value": w,
            "title": f"Совместные упоминания: {w}",
        }
        for (s, d), w in final_edges.items() if w >= min_edge
    ]

    connected = {e["from"] for e in edge_list} | {e["to"] for e in edge_list}
    if connected:
        nodes = [n for n in nodes if n["id"] in connected]
    edge_list.sort(key=lambda e: (-e["value"], e["from"], e["to"]))

    top_terms = [n["id"] for n in nodes[:6]]
    summary = _build_graph_summary(top_terms, final_edges, freq)
    return {
        "nodes": nodes,
        "edges": edge_list,
        "meta": {
            "summary": summary,
            "top_terms": top_terms,
        },
    }


def render_html(
    text: str,
    output: Path,
    title: str,
    *,
    top_n: int = 16,
    min_freq: int = 2,
    min_edge: int = 2,
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
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  <script src="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"></script>
  <style>
    :root {{
      --bg-1: #f7faf8;
      --bg-2: #e5f0ea;
      --card: rgba(255, 255, 255, 0.84);
      --text-main: #133228;
      --text-soft: #44655a;
      --accent: #1f8c6b;
      --accent-soft: #53b598;
      --border: rgba(19, 50, 40, 0.14);
      --shadow: 0 20px 48px rgba(19, 50, 40, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "Manrope", "Segoe UI", -apple-system, sans-serif;
      color: var(--text-main);
      background:
        radial-gradient(1100px 620px at 10% -10%, #d7efe3 0%, transparent 55%),
        radial-gradient(980px 520px at 100% 0%, #d5ebe4 0%, transparent 52%),
        linear-gradient(180deg, var(--bg-1), var(--bg-2));
    }}
    .page {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 16px;
      padding: 18px;
    }}
    .canvas {{
      border-radius: 20px;
      overflow: hidden;
      border: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.7);
      box-shadow: var(--shadow);
      min-height: 78vh;
    }}
    #mindmap {{ width: 100%; height: 78vh; }}
    .panel {{
      border-radius: 20px;
      border: 1px solid var(--border);
      background: var(--card);
      box-shadow: var(--shadow);
      padding: 18px 16px;
      backdrop-filter: blur(3px);
    }}
    .kicker {{
      margin: 0;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
      font-weight: 700;
    }}
    h1 {{
      margin: 8px 0 12px;
      font-size: clamp(22px, 2.2vw, 30px);
      line-height: 1.14;
    }}
    .muted {{
      margin: 0;
      color: var(--text-soft);
      line-height: 1.45;
      font-size: 14px;
    }}
    .node-card {{
      margin-top: 16px;
      border-top: 1px dashed var(--border);
      padding-top: 14px;
    }}
    .node-title {{
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
      color: var(--accent);
    }}
    .stats {{
      margin: 8px 0 10px;
      font-size: 13px;
      color: var(--text-soft);
    }}
    @media (max-width: 980px) {{
      .page {{
        grid-template-columns: 1fr;
      }}
      #mindmap, .canvas {{
        min-height: 64vh;
        height: 64vh;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="canvas">
      <div id="mindmap"></div>
    </div>
    <aside class="panel">
      <p class="kicker">Mind Map</p>
      <h1>{title}</h1>
      <p id="graphSummary" class="muted"></p>
      <div class="node-card">
        <h2 id="nodeTitle" class="node-title">Выберите узел</h2>
        <p id="nodeStats" class="stats">Кликните на вершину графа, чтобы увидеть краткое описание.</p>
        <p id="nodeSummary" class="muted">Здесь появится объяснение выбранного элемента.</p>
      </div>
    </aside>
  </div>
  <script>
    const data = {payload};
    const nodes = new vis.DataSet(data.nodes.map((n) => ({{
      ...n,
      shape: "dot",
      scaling: {{ min: 14, max: 42 }},
      color: {{
        background: "#57b597",
        border: "#1f8c6b",
        highlight: {{ background: "#23a27d", border: "#0f6d52" }},
      }},
      font: {{ color: "#10372b", size: 15, face: "Manrope" }},
      borderWidth: 1.6,
    }})));
    const edges = new vis.DataSet(data.edges.map(e => ({{
      ...e,
      width: 1.1 + Math.min(5.4, e.value * 0.5),
      color: {{ color: "rgba(27,111,84,0.40)", highlight: "#1f8c6b" }},
      smooth: {{ type: "dynamic" }},
    }})));
    const graphSummaryEl = document.getElementById("graphSummary");
    const nodeTitleEl = document.getElementById("nodeTitle");
    const nodeStatsEl = document.getElementById("nodeStats");
    const nodeSummaryEl = document.getElementById("nodeSummary");

    graphSummaryEl.textContent = data?.meta?.summary || "Краткое описание графа недоступно.";

    const network = new vis.Network(
      document.getElementById("mindmap"),
      {{ nodes, edges }},
      {{
        interaction: {{ hover: true, tooltipDelay: 120 }},
        physics: {{
          stabilization: {{ iterations: 380 }},
          barnesHut: {{
            gravitationalConstant: -9000,
            centralGravity: 0.21,
            springLength: 132,
            springConstant: 0.028,
            damping: 0.75,
          }},
        }},
      }}
    );

    function setNodeDetails(node) {{
      if (!node) {{
        nodeTitleEl.textContent = "Выберите узел";
        nodeStatsEl.textContent = "Кликните на вершину графа, чтобы увидеть краткое описание.";
        nodeSummaryEl.textContent = "Здесь появится объяснение выбранного элемента.";
        return;
      }}
      const connected = network.getConnectedNodes(node.id).length;
      nodeTitleEl.textContent = node.label || node.id;
      nodeStatsEl.textContent =
        `Упоминаний: ${{node.value || 0}} | Связей: ${{connected}} | Релевантность: ${{node.relevance ?? "n/a"}}`;
      nodeSummaryEl.textContent = node.summary || "Краткое описание не найдено.";
    }}

    network.on("click", (params) => {{
      if (params.nodes && params.nodes.length > 0) {{
        const node = nodes.get(params.nodes[0]);
        setNodeDetails(node);
        return;
      }}
      if (params.edges && params.edges.length > 0) {{
        const edge = edges.get(params.edges[0]);
        nodeTitleEl.textContent = `${{edge.from}} <-> ${{edge.to}}`;
        nodeStatsEl.textContent = `Сила связи: ${{edge.value || 0}}`;
        nodeSummaryEl.textContent = "Эти понятия часто встречаются вместе в одном контексте.";
        return;
      }}
      setNodeDetails(null);
    }});
  </script>
</body>
</html>"""
