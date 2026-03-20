import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

WORD_RE = re.compile(r"[a-zа-яё0-9]+", flags=re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+", flags=re.MULTILINE)

STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она", "так",
    "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот",
    "от", "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если",
    "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там",
    "потом", "себя", "ничего", "ей", "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "человек", "чего", "раз", "тоже", "себе", "под", "будет",
    "ж", "тогда", "кто", "этот", "того", "потому", "этого", "какой", "совсем", "ним", "здесь", "этом", "один",
    "почти", "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех", "никогда", "можно", "при",
    "наконец", "два", "об", "другой", "хоть", "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего",
    "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой", "перед", "иногда",
    "лучше", "чуть", "том", "нельзя", "такой", "им", "более", "всегда", "конечно", "всю", "между",
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "to", "of", "on", "in", "at", "by",
    "with", "from", "as", "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these",
    "those", "we", "you", "they", "he", "she", "i", "my", "your", "their", "our", "not", "can", "could",
    "will", "would", "should", "about", "into", "out", "up", "down", "over", "under", "again", "more", "most",
}


def _tokenize(text: str) -> List[str]:
    tokens = [t.lower() for t in WORD_RE.findall(text)]
    return [t for t in tokens if len(t) >= 3 and t not in STOPWORDS and not t.isdigit()]


def _split_sentences(text: str) -> List[str]:
    return [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]


def _term_frequencies(text: str) -> Counter:
    return Counter(_tokenize(text))


def _select_concepts(freq: Counter, top_n: int, min_freq: int) -> List[str]:
    filtered = [(term, count) for term, count in freq.items() if count >= min_freq]
    filtered.sort(key=lambda x: (-x[1], x[0]))
    return [term for term, _ in filtered[:top_n]]


def _cooccurrence(sentences: List[str], concepts: List[str]) -> Counter:
    concept_set = set(concepts)
    edges: Counter = Counter()

    for sentence in sentences:
        found = sorted(set(tok for tok in _tokenize(sentence) if tok in concept_set))
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
    return {key: (val - min_v) / (max_v - min_v) for key, val in values.items()}


def build_graph_data(
    text: str,
    top_n_concepts: int = 24,
    min_concept_freq: int = 2,
    min_edge_weight: int = 1,
) -> Dict[str, List[Dict]]:
    freq = _term_frequencies(text)
    sentences = _split_sentences(text)
    if not freq:
        return {"nodes": [], "edges": []}

    # Start from a wider pool and then keep only conceptually central terms.
    candidate_cap = max(top_n_concepts * 4, top_n_concepts)
    candidate_terms = _select_concepts(freq, top_n=candidate_cap, min_freq=min_concept_freq)
    candidate_edges = _cooccurrence(sentences, candidate_terms)

    degree_weight: Counter = Counter()
    for (src, dst), weight in candidate_edges.items():
        degree_weight[src] += weight
        degree_weight[dst] += weight

    freq_scores = _normalize_scores({term: float(freq[term]) for term in candidate_terms})
    degree_scores = _normalize_scores({term: float(degree_weight.get(term, 0.0)) for term in candidate_terms})
    relevance_scores: Dict[str, float] = {}
    for term in candidate_terms:
        relevance_scores[term] = 0.62 * freq_scores.get(term, 0.0) + 0.38 * degree_scores.get(term, 0.0)

    concepts = sorted(candidate_terms, key=lambda t: (-relevance_scores[t], t))[:top_n_concepts]
    edge_counts = _cooccurrence(sentences, concepts)

    nodes: List[Dict] = []
    for concept in concepts:
        count = freq[concept]
        nodes.append(
            {
                "id": concept,
                "label": concept,
                "title": f"{concept}: {count}",
                "value": max(1, count),
                "relevance": round(relevance_scores.get(concept, 0.0), 4),
            }
        )

    edges: List[Dict] = []
    for (source, target), weight in edge_counts.items():
        if weight < min_edge_weight:
            continue
        edges.append(
            {
                "from": source,
                "to": target,
                "value": weight,
                "title": f"Совместные упоминания: {weight}",
            }
        )

    # Remove isolated nodes to reduce visual noise.
    connected_ids = {edge["from"] for edge in edges} | {edge["to"] for edge in edges}
    if connected_ids:
        nodes = [node for node in nodes if node["id"] in connected_ids]

    edges.sort(key=lambda e: (-e["value"], e["from"], e["to"]))
    return {"nodes": nodes, "edges": edges}


def _build_html(graph_data: Dict[str, List[Dict]], title: str) -> str:
    safe_title = html.escape(title)
    payload = json.dumps(graph_data, ensure_ascii=False)

    return f"""<!doctype html>
<html lang=\"ru\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{safe_title}</title>
  <script src=\"https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js\"></script>
  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
  <link href=\"https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap\" rel=\"stylesheet\">
  <style>
    :root {{
      --bg: #11131a;
      --card: rgba(23, 27, 36, 0.84);
      --text: #dbe7ff;
      --muted: #95a4c7;
      --line: #506180;
      --brand: #7c8cff;
      --brand-2: #5cc8ff;
      --warn: #ca8a04;
    }}
    body {{
      margin: 0;
      font-family: 'Manrope', 'Segoe UI', sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 8% 16%, rgba(92, 200, 255, 0.13) 0%, transparent 34%),
        radial-gradient(circle at 92% 12%, rgba(124, 140, 255, 0.18) 0%, transparent 32%),
        linear-gradient(180deg, #0e1016 0%, #141823 100%);
    }}
    .wrap {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 18px 18px 22px;
    }}
    .header {{
      background: var(--card);
      border-radius: 16px;
      border: 1px solid rgba(147, 163, 198, 0.25);
      backdrop-filter: blur(7px);
      padding: 16px 18px;
      box-shadow: 0 16px 40px rgba(3, 6, 15, 0.42);
      margin-bottom: 12px;
    }}
    h1 {{ margin: 0; font-size: 24px; letter-spacing: -0.02em; }}
    p {{ margin: 6px 0 0; color: var(--muted); font-size: 14px; }}
    .toolbar {{
      display: grid;
      grid-template-columns: 1.6fr 1fr auto auto auto;
      gap: 10px;
      margin-bottom: 12px;
      align-items: center;
      background: var(--card);
      border-radius: 14px;
      border: 1px solid rgba(147, 163, 198, 0.22);
      padding: 12px;
      backdrop-filter: blur(7px);
      box-shadow: 0 10px 30px rgba(3, 6, 15, 0.38);
    }}
    .toolbar label {{
      color: var(--muted);
      font-size: 13px;
      display: block;
      margin-bottom: 4px;
    }}
    .toolbar input[type="text"],
    .toolbar input[type="range"] {{
      width: 100%;
      box-sizing: border-box;
    }}
    .toolbar input[type="text"] {{
      border-radius: 10px;
      border: 1px solid #39465f;
      padding: 9px 10px;
      background: rgba(15, 19, 29, 0.95);
      color: #dbe7ff;
    }}
    .btn {{
      border: 0;
      border-radius: 10px;
      padding: 10px 12px;
      background: linear-gradient(135deg, var(--brand) 0%, var(--brand-2) 100%);
      color: #08101e;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 8px 20px rgba(13, 148, 136, 0.3);
    }}
    .btn.secondary {{
      background: #2a3347;
      color: #dbe7ff;
      box-shadow: none;
    }}
    .btn.warning {{
      background: #5a4a22;
      color: #fde68a;
      box-shadow: none;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 1fr 320px;
      gap: 12px;
    }}
    .canvas-wrap {{
      position: relative;
    }}
    #mindmap {{
      width: 100%;
      height: 78vh;
      background: var(--card);
      border-radius: 16px;
      border: 1px solid rgba(147, 163, 198, 0.28);
      box-shadow: inset 0 0 0 1px rgba(124, 140, 255, 0.08), 0 20px 45px rgba(3, 6, 15, 0.45);
      backdrop-filter: blur(4px);
    }}
    .legend {{
      position: absolute;
      left: 14px;
      bottom: 12px;
      display: flex;
      gap: 8px;
      background: rgba(17, 23, 35, 0.88);
      border-radius: 999px;
      padding: 7px 10px;
      border: 1px solid #39465f;
      font-size: 12px;
      color: #c7d5f5;
      flex-wrap: wrap;
    }}
    .legend .dot {{
      width: 9px;
      height: 9px;
      border-radius: 999px;
      display: inline-block;
    }}
    .side {{
      background: var(--card);
      border-radius: 16px;
      border: 1px solid rgba(147, 163, 198, 0.25);
      box-shadow: 0 16px 36px rgba(3, 6, 15, 0.4);
      padding: 12px;
      overflow: auto;
      max-height: 78vh;
    }}
    .side h3 {{
      margin: 8px 0 8px;
      font-size: 14px;
      color: #a9bae0;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .stat {{
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .chip {{
      background: #263248;
      color: #dbe7ff;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 12px;
      cursor: pointer;
      user-select: none;
    }}
    .chip:hover {{
      background: #324463;
    }}
    .meta {{
      font-size: 12px;
      color: #9eb0d8;
      line-height: 1.4;
      margin-top: 8px;
      border-top: 1px dashed #364862;
      padding-top: 8px;
    }}
    .concept-list {{
      display: grid;
      gap: 6px;
    }}
    .concept-btn {{
      border: 1px solid #3a4b68;
      border-radius: 9px;
      background: rgba(17, 24, 39, 0.8);
      color: #dbe7ff;
      text-align: left;
      padding: 8px 10px;
      font-size: 12px;
      cursor: pointer;
      font-family: inherit;
    }}
    .concept-btn:hover {{
      border-color: #7c8cff;
      background: #1f2a40;
    }}
    @media (max-width: 900px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .toolbar {{
        grid-template-columns: 1fr;
      }}
      .side {{
        max-height: none;
      }}
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"header\">
      <h1>{safe_title}</h1>
      <p>Перетаскивайте узлы, масштабируйте граф колесом мыши, наводите на связи для веса.</p>
    </div>
    <div class=\"toolbar\">
      <div>
        <label for=\"search\">Поиск понятия</label>
        <input id=\"search\" type=\"text\" placeholder=\"Например: python\" />
      </div>
      <div>
        <label for=\"edgeRange\">Мин. вес связи: <span id=\"edgeValue\">1</span></label>
        <input id=\"edgeRange\" type=\"range\" min=\"1\" max=\"10\" value=\"1\" />
      </div>
      <button id=\"resetView\" class=\"btn\">Сбросить вид</button>
      <button id=\"togglePhysics\" class=\"btn secondary\">Physics: ON</button>
      <button id=\"exportPng\" class=\"btn warning\">Export PNG</button>
    </div>
    <div class=\"layout\">
      <div class=\"canvas-wrap\">
        <div id=\"mindmap\"></div>
        <div class=\"legend\">
          <span><span class=\"dot\" style=\"background:#7c8cff\"></span> частые понятия</span>
          <span><span class=\"dot\" style=\"background:#5cc8ff\"></span> средняя частота</span>
          <span><span class=\"dot\" style=\"background:#4ade80\"></span> редкие понятия</span>
        </div>
      </div>
      <aside class=\"side\">
        <h3>Информация</h3>
        <div class=\"stat\" id=\"stats\"></div>
        <div class=\"meta\" id=\"selectionMeta\">Выберите узел на графе, чтобы увидеть детали.</div>
        <h3>Топ-концепты</h3>
        <div class=\"concept-list\" id=\"topConcepts\"></div>
        <h3>Связанные понятия</h3>
        <div class=\"chips\" id=\"neighbors\"></div>
      </aside>
    </div>
  </div>

  <script>
    const data = {payload};
    const rawNodes = [...data.nodes];
    const valueList = rawNodes.map((n) => Number(n.value || 1));
    const minVal = Math.min(...valueList);
    const maxVal = Math.max(...valueList);
    const denom = Math.max(1e-9, maxVal - minVal);

    function colorForValue(value) {{
      const norm = (value - minVal) / denom;
      if (norm >= 0.66) {{
        return {{ background: '#7c8cff', border: '#6578ff', font: '#08101e' }};
      }}
      if (norm >= 0.33) {{
        return {{ background: '#5cc8ff', border: '#28aeea', font: '#08101e' }};
      }}
      return {{ background: '#4ade80', border: '#22c55e', font: '#052014' }};
    }}

    const baseNodes = rawNodes.map((n) => {{
      const color = colorForValue(Number(n.value || 1));
      return {{
        ...n,
        shape: 'dot',
        scaling: {{ min: 12, max: 46 }},
        borderWidth: 1.4,
        color: {{
          background: color.background,
          border: color.border,
          highlight: {{ background: '#f8fafc', border: '#7c8cff' }}
        }},
        font: {{ color: color.font, face: 'Manrope', strokeWidth: 0 }}
      }};
    }});
    const baseEdges = data.edges.map((e) => ({{
      ...e,
      width: 1 + Math.min(5, Number(e.value || 1) * 0.55),
      color: {{ color: '#5b6d90', highlight: '#c7d2fe' }},
      smooth: {{ type: 'continuous' }}
    }}));

    const nodes = new vis.DataSet(baseNodes);
    const edges = new vis.DataSet(baseEdges);

    const network = new vis.Network(
      document.getElementById('mindmap'),
      {{ nodes, edges }},
      {{
        autoResize: true,
        interaction: {{ hover: true, tooltipDelay: 90 }},
        physics: {{
          enabled: true,
          stabilization: {{ iterations: 420 }},
          barnesHut: {{ gravitationalConstant: -8300, springLength: 160, springConstant: 0.025 }}
        }}
      }}
    );

    const edgeRange = document.getElementById('edgeRange');
    const edgeValue = document.getElementById('edgeValue');
    const search = document.getElementById('search');
    const stats = document.getElementById('stats');
    const selectionMeta = document.getElementById('selectionMeta');
    const topConcepts = document.getElementById('topConcepts');
    const neighbors = document.getElementById('neighbors');
    const resetViewBtn = document.getElementById('resetView');
    const togglePhysicsBtn = document.getElementById('togglePhysics');
    const exportPngBtn = document.getElementById('exportPng');

    let physicsEnabled = true;
    const maxEdgeValue = Math.max(1, ...baseEdges.map((e) => Number(e.value || 1)));
    edgeRange.max = String(maxEdgeValue);

    function updateStats() {{
      stats.textContent = `Узлов: ${{nodes.length}} | Связей: ${{edges.length}}`;
    }}

    function filterEdges(minWeight) {{
      const nextEdges = baseEdges.filter((e) => Number(e.value || 1) >= minWeight);
      edges.clear();
      edges.add(nextEdges);
      edgeValue.textContent = String(minWeight);
      updateStats();
    }}

    function showNeighbors(nodeId) {{
      neighbors.innerHTML = '';
      if (!nodeId) {{
        selectionMeta.textContent = 'Выберите узел на графе, чтобы увидеть детали.';
        return;
      }}
      const nodeObj = nodes.get(nodeId);
      const deg = network.getConnectedNodes(nodeId).length;
      selectionMeta.textContent = `Понятие: ${{nodeObj.label}} | Частота: ${{nodeObj.value}} | Связей: ${{deg}}`;
      if (nodeObj.relevance !== undefined) {{
        selectionMeta.textContent += ` | Релевантность: ${{nodeObj.relevance}}`;
      }}

      const connected = network.getConnectedNodes(nodeId);
      if (!connected.length) {{
        neighbors.innerHTML = '<span class="chip">Нет связей</span>';
        return;
      }}
      connected.slice(0, 30).forEach((id) => {{
        const el = document.createElement('span');
        el.className = 'chip';
        el.textContent = String(id);
        el.addEventListener('click', () => focusNodeByLabel(String(id)));
        neighbors.appendChild(el);
      }});
    }}

    function focusNodeByLabel(query) {{
      const normalized = query.trim().toLowerCase();
      if (!normalized) {{
        nodes.clear();
        nodes.add(baseNodes);
        return;
      }}
      const found = baseNodes.find((n) => String(n.label).toLowerCase() === normalized);
      if (!found) {{
        return;
      }}
      const updated = baseNodes.map((n) => {{
        if (n.id === found.id) {{
          return {{ ...n, borderWidth: 4 }};
        }}
        return {{ ...n, opacity: 0.23 }};
      }});
      nodes.clear();
      nodes.add(updated);
      network.selectNodes([found.id]);
      network.focus(found.id, {{ scale: 1.15, animation: true }});
      showNeighbors(found.id);
    }}

    function restoreNodes() {{
      nodes.clear();
      nodes.add(baseNodes);
    }}

    edgeRange.addEventListener('input', (e) => {{
      filterEdges(Number(e.target.value));
    }});

    search.addEventListener('keydown', (e) => {{
      if (e.key === 'Enter') {{
        focusNodeByLabel(search.value);
      }}
    }});

    resetViewBtn.addEventListener('click', () => {{
      search.value = '';
      edgeRange.value = '1';
      filterEdges(1);
      restoreNodes();
      network.fit({{ animation: true }});
      network.unselectAll();
      showNeighbors(null);
    }});

    togglePhysicsBtn.addEventListener('click', () => {{
      physicsEnabled = !physicsEnabled;
      network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
      togglePhysicsBtn.textContent = `Physics: ${{physicsEnabled ? 'ON' : 'OFF'}}`;
    }});

    network.on('click', (params) => {{
      const nodeId = params.nodes[0] || null;
      if (!nodeId) {{
        restoreNodes();
      }}
      showNeighbors(nodeId);
    }});

    exportPngBtn.addEventListener('click', () => {{
      const canvas = document.querySelector('#mindmap canvas');
      if (!canvas) {{
        return;
      }}
      const link = document.createElement('a');
      link.download = 'mindmap.png';
      link.href = canvas.toDataURL('image/png');
      link.click();
    }});

    function mountTopConcepts() {{
      const sorted = [...baseNodes].sort((a, b) => Number(b.value || 1) - Number(a.value || 1)).slice(0, 18);
      topConcepts.innerHTML = '';
      sorted.forEach((item) => {{
        const btn = document.createElement('button');
        btn.className = 'concept-btn';
        btn.textContent = `${{item.label}} (${{item.value}})`;
        btn.addEventListener('click', () => focusNodeByLabel(String(item.label)));
        topConcepts.appendChild(btn);
      }});
    }}

    network.once('stabilizationIterationsDone', () => {{
      network.setOptions({{ physics: false }});
      physicsEnabled = false;
      togglePhysicsBtn.textContent = 'Physics: OFF';
    }});

    filterEdges(1);
    updateStats();
    mountTopConcepts();
  </script>
</body>
</html>
"""


def render_mindmap_html(
    text: str,
    output_html: Path,
    title: str,
    top_n_concepts: int = 24,
    min_concept_freq: int = 2,
    min_edge_weight: int = 1,
) -> Tuple[int, int]:
    graph_data = build_graph_data(
        text=text,
        top_n_concepts=top_n_concepts,
        min_concept_freq=min_concept_freq,
        min_edge_weight=min_edge_weight,
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(_build_html(graph_data, title=title), encoding="utf-8")
    return len(graph_data["nodes"]), len(graph_data["edges"])
