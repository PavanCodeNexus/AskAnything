"""Unique Handwritten Mind Map and Knowledge Graph Generator with full concept explanations."""
import html
import json
import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from ingest.base_loader import DocumentChunk
from utils.logger import setup_logger

logger = setup_logger("mindmap_generator")

HANDWRITTEN_MINDMAP_PROMPT = """You are an expert mind map architect and visual educator.
Given a user query, synthesized answer, and source excerpts, create an educational, hand-drawn concept map that explains everything clearly.

Return a valid JSON object matching this schema:
{
  "nodes": [
    {
      "id": "c1",
      "label": "Short Name (2-4 words)",
      "group": "Topic" | "Component" | "Metric" | "Organization" | "Method" | "Insight",
      "explanation": "Clear 2-sentence handwritten explanation explaining what this is, how it works, and its importance.",
      "takeaway": "Key note or citation fact (e.g. 'Cites Page 4' or 'Key advantage')"
    }
  ],
  "edges": [
    {
      "from": "c1",
      "to": "c2",
      "label": "relationship (e.g. connects to, verified by, produces)"
    }
  ]
}

Rules:
1. The first node MUST be the central root topic (group="Topic") summarizing the user query.
2. Extract 5 to 10 meaningful, distinct concept nodes.
3. Crucial: The "explanation" must thoroughly explain the concept in plain English as if written in a study notebook.
4. Edge labels must be brief active verbs.
5. Return ONLY raw JSON. No markdown code blocks, backticks, or intro text.
"""


class MindMapGenerator:
    """Extracts concept entities and renders a unique handwritten-style SVG Mind Map with explanations."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.llm = None
        if self.api_key and self.api_key != "your_groq_api_key_here":
            for cand in [self.model_name, "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]:
                try:
                    self.llm = ChatGroq(
                        groq_api_key=self.api_key,
                        model_name=cand,
                        temperature=0.2,
                        max_tokens=950,
                    )
                    self.model_name = cand
                    break
                except Exception as e:
                    logger.warning("Could not initialize Groq for MindMapGenerator on %s: %s", cand, e)

    def generate_graph_data(self, query: str, answer: str, chunks: List[DocumentChunk]) -> Dict[str, List[Dict[str, str]]]:
        """Extracts concept nodes and handwritten explanations from the Q&A context."""
        if self.llm is not None:
            try:
                excerpt_text = "\n".join([f"- {c.text[:220]}" for c in chunks[:5]])
                prompt = (
                    f"User Query: {query}\n\n"
                    f"Synthesized Answer: {answer[:500]}\n\n"
                    f"Source Excerpts:\n{excerpt_text}"
                )
                response = self.llm.invoke([
                    SystemMessage(content=HANDWRITTEN_MINDMAP_PROMPT),
                    HumanMessage(content=prompt),
                ])
                raw = response.content.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?", "", raw)
                    raw = re.sub(r"```$", "", raw).strip()

                data = json.loads(raw)
                if "nodes" in data and "edges" in data and len(data["nodes"]) >= 2:
                    return data
            except Exception as e:
                logger.warning("LLM handwritten concept extraction failed: %s. Using heuristic graph.", e)

        return self._heuristic_graph(query, answer, chunks)

    def _heuristic_graph(self, query: str, answer: str, chunks: List[DocumentChunk]) -> Dict[str, List[Dict[str, str]]]:
        """Creates an educational concept graph with handwritten explanations when LLM is unavailable."""
        center_label = query.strip("? .")[:26]
        nodes = [{
            "id": "root",
            "label": center_label,
            "group": "Topic",
            "explanation": f"Central topic investigated: {query}. Synthesizes multi-source findings into grounded insights.",
            "takeaway": "Core Inquiry Focus",
        }]
        edges = []

        words = re.findall(r"\b[A-Z][a-z0-9]{3,}\b", answer + " " + query)
        unique_terms = []
        for w in words:
            if w.lower() not in {center_label.lower(), "what", "which", "this", "that", "from", "with", "have", "when"} and w not in unique_terms:
                unique_terms.append(w)

        groups_cycle = [
            ("Component", "Core system building block responsible for modular task execution.", "System Architecture"),
            ("Method", "Operational procedure applied to process text, vectors, and citations.", "Workflow Step"),
            ("Metric", "Quantitative measurement evaluating precision, recall, and reliability.", "Evaluation Standard"),
            ("Insight", "Critical finding uncovered across corroborated documentation.", "Key Finding"),
        ]

        for idx, term in enumerate(unique_terms[:6]):
            node_id = f"node_{idx+1}"
            grp, default_expl, default_note = groups_cycle[idx % len(groups_cycle)]
            nodes.append({
                "id": node_id,
                "label": term,
                "group": grp,
                "explanation": f"{term}: {default_expl}",
                "takeaway": default_note,
            })
            edges.append({"from": "root", "to": node_id, "label": "relates to"})

        for c_idx, c in enumerate(chunks[:2]):
            doc_label = c.filename or c.title or c.source_type or f"Source {c_idx+1}"
            doc_id = f"src_{c_idx+1}"
            clean_excerpt = c.text[:95].replace("\n", " ").strip()
            nodes.append({
                "id": doc_id,
                "label": doc_label[:22],
                "group": "Organization",
                "explanation": f"Primary grounding source ({c.source_type}): '{clean_excerpt}...'",
                "takeaway": f"Verified Citation [{c.source_type}]",
            })
            edges.append({"from": "root", "to": doc_id, "label": "cites"})

        return {"nodes": nodes, "edges": edges}

    def generate_html(self, query: str, answer: str, chunks: List[DocumentChunk], height: str = "600px") -> str:
        """Constructs an authentic handwritten, sketchy SVG Mind Map with notes and clear controls."""
        graph_data = self.generate_graph_data(query, answer, chunks)
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        if not nodes:
            return "<div style='color:#94a3b8;padding:20px;text-align:center;'>No mind map data available.</div>"

        # Handwritten Sticky Chalkboard Themes
        group_meta = {
            "Topic": {
                "border": "#E879F9",
                "bg": "#2A1744",
                "pill_bg": "#581C87",
                "text": "#FDF4FF",
                "icon": "🎯",
                "badge": "Central Topic",
            },
            "Component": {
                "border": "#38BDF8",
                "bg": "#0B2545",
                "pill_bg": "#0369A1",
                "text": "#F0F9FF",
                "icon": "🧩",
                "badge": "Component",
            },
            "Metric": {
                "border": "#34D399",
                "bg": "#063725",
                "pill_bg": "#047857",
                "text": "#ECFDF5",
                "icon": "📊",
                "badge": "Metric & Data",
            },
            "Organization": {
                "border": "#FBBF24",
                "bg": "#362204",
                "pill_bg": "#B45309",
                "text": "#FFFBEB",
                "icon": "🏢",
                "badge": "Source Citation",
            },
            "Method": {
                "border": "#FB7185",
                "bg": "#381023",
                "pill_bg": "#BE123C",
                "text": "#FFF1F2",
                "icon": "⚙️",
                "badge": "Methodology",
            },
            "Insight": {
                "border": "#C084FC",
                "bg": "#25124A",
                "pill_bg": "#7E22CE",
                "text": "#FAF5FF",
                "icon": "💡",
                "badge": "Key Takeaway",
            },
        }

        root_node = nodes[0]
        for n in nodes:
            if n.get("group") == "Topic" or n.get("id") == "root":
                root_node = n
                break

        other_nodes = [n for n in nodes if n != root_node]

        # Calculate coordinates: center root at (700, 400)
        root_x, root_y = 700, 400
        half = (len(other_nodes) + 1) // 2
        right_nodes = other_nodes[:half]
        left_nodes = other_nodes[half:]

        coords: Dict[str, Dict[str, Any]] = {
            root_node["id"]: {"x": root_x, "y": root_y, "side": "center", "node": root_node, "angle": 0}
        }

        def layout_side(side_nodes: List[Dict[str, Any]], side: str):
            count = len(side_nodes)
            if count == 0:
                return
            x_offset = 380 if side == "right" else -380
            target_x = root_x + x_offset
            
            spacing_y = max(180, min(220, 700 // max(count, 1)))
            start_y = root_y - ((count - 1) * spacing_y) / 2

            angles = [-1.4, 1.2, -0.9, 1.5, -1.8, 1.1]
            for i, n in enumerate(side_nodes):
                ny = start_y + (i * spacing_y)
                angle = angles[i % len(angles)]
                coords[n["id"]] = {"x": target_x, "y": ny, "side": side, "node": n, "angle": angle}

        layout_side(right_nodes, "right")
        layout_side(left_nodes, "left")

        # Map edges
        rendered_edges = []
        for e in edges:
            u, v = e.get("from"), e.get("to")
            if u in coords and v in coords:
                rendered_edges.append((coords[u], coords[v], e.get("label", "connects")))
            elif v in coords and root_node["id"] in coords:
                rendered_edges.append((coords[root_node["id"]], coords[v], e.get("label", "relates")))

        if not rendered_edges:
            for n in other_nodes:
                if n["id"] in coords:
                    rendered_edges.append((coords[root_node["id"]], coords[n["id"]], "relates"))

        # Build wavy organic hand-drawn SVG lines
        path_elements = []
        for s_idx, (source, target, elabel) in enumerate(rendered_edges):
            x1, y1 = source["x"], source["y"]
            x2, y2 = target["x"], target["y"]

            # Slight hand-drawn wavy offsets
            wobble = 14 if (s_idx % 2 == 0) else -14
            dx = (x2 - x1) * 0.5
            cx1 = x1 + dx + wobble
            cy1 = y1 + (wobble * 0.5)
            cx2 = x2 - dx - wobble
            cy2 = y2 - (wobble * 0.5)

            t_grp = target["node"].get("group", "Component")
            meta = group_meta.get(t_grp, group_meta["Component"])
            stroke_color = meta["border"]

            path_d = f"M {x1:.1f} {y1:.1f} C {cx1:.1f} {cy1:.1f}, {cx2:.1f} {cy2:.1f}, {x2:.1f} {y2:.1f}"

            mid_x = (x1 + x2) / 2 + (wobble * 0.6)
            mid_y = (y1 + y2) / 2

            path_elements.append(f"""
                <g class="hand-edge-group">
                    <path d="{path_d}" fill="none" stroke="{stroke_color}" stroke-width="2.6" stroke-linecap="round" stroke-dasharray="7,4" stroke-opacity="0.8" />
                    <!-- Handwritten relationship pill -->
                    <rect x="{mid_x - 48:.1f}" y="{mid_y - 12:.1f}" width="96" height="24" rx="12" fill="#141923" stroke="{stroke_color}" stroke-width="1.4" stroke-dasharray="4,2" />
                    <text x="{mid_x:.1f}" y="{mid_y + 4:.1f}" fill="#E2E8F0" font-size="13" font-family="'Caveat', cursive" text-anchor="middle" font-weight="700">✎ {html.escape(elabel[:18])}</text>
                </g>
            """)

        # Build Handwritten Cards with full explanations using foreignObject
        card_elements = []
        for nid, cdata in coords.items():
            nx, ny = cdata["x"], cdata["y"]
            node = cdata["node"]
            angle = cdata["angle"]
            group = node.get("group", "Component")
            meta = group_meta.get(group, group_meta["Component"])
            is_root = (nid == root_node["id"])

            label = html.escape(node.get("label", nid))
            expl = html.escape(node.get("explanation", "Explains key grounded concepts extracted from documentation."))
            takeaway = html.escape(node.get("takeaway", "Key Observation"))
            icon = meta["icon"]
            bg = meta["bg"]
            border = meta["border"]
            pill_bg = meta["pill_bg"]
            text_color = meta["text"]

            w = 280 if not is_root else 310
            h = 165 if not is_root else 175

            card_elements.append(f"""
                <foreignObject x="{nx - w/2:.1f}" y="{ny - h/2:.1f}" width="{w}" height="{h}" class="hand-card-fo">
                    <div xmlns="http://www.w3.org/1999/xhtml" class="hand-card {'hand-root-card' if is_root else ''}" style="
                        background: {bg};
                        border: 2px dashed {border};
                        box-shadow: 4px 10px 20px rgba(0,0,0,0.5);
                        transform: rotate({angle}deg);
                    ">
                        <!-- Scotch tape strip -->
                        <div class="tape-strip"></div>
                        
                        <!-- Header & Tag -->
                        <div class="card-header">
                            <span class="card-title" style="color: {text_color};">
                                {icon} {label}
                            </span>
                            <span class="badge-tag" style="background: {pill_bg}; color: {text_color}; border: 1px dashed {border};">
                                {meta['badge']}
                            </span>
                        </div>

                        <!-- Handwritten Explanation -->
                        <div class="card-explanation">
                            "{expl}"
                        </div>

                        <!-- Handwritten Takeaway Footer -->
                        <div class="card-footer" style="color: {border};">
                            ★ <em>{takeaway}</em>
                        </div>
                    </div>
                </foreignObject>
            """)

        all_paths_str = "\n".join(path_elements)
        all_cards_str = "\n".join(card_elements)

        html_code = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Caveat:wght@600;700&family=Kalam:wght@700&family=Inter:wght@500;600&display=swap');

    body {{
        margin: 0;
        padding: 0;
        background: #12161E;
        font-family: 'Caveat', cursive;
        overflow: hidden;
        user-select: none;
    }}
    .mindmap-board {{
        position: relative;
        width: 100%;
        height: {height};
        /* Chalkboard / blueprint grid texture */
        background-color: #131722;
        background-image: 
            linear-gradient(rgba(255, 255, 255, 0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.04) 1px, transparent 1px);
        background-size: 28px 28px;
        border: 2px dashed rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: inset 0 0 60px rgba(0, 0, 0, 0.6);
    }}
    svg {{
        width: 100%;
        height: 100%;
        cursor: grab;
    }}
    svg:active {{
        cursor: grabbing;
    }}
    .tape-strip {{
        width: 52px;
        height: 13px;
        background: rgba(255, 255, 255, 0.22);
        margin: -8px auto 6px;
        transform: rotate(-2deg);
        border: 1px dashed rgba(255, 255, 255, 0.35);
        box-shadow: 0 2px 4px rgba(0,0,0,0.25);
    }}
    .hand-card {{
        width: 92%;
        height: 84%;
        box-sizing: border-box;
        padding: 10px 14px;
        border-radius: 255px 15px 225px 15px/15px 225px 15px 255px;
        transition: transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.2s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .hand-card:hover {{
        transform: scale(1.04) rotate(0deg) !important;
        box-shadow: 0 16px 30px rgba(0, 0, 0, 0.7) !important;
        z-index: 100;
    }}
    .hand-root-card {{
        border-width: 3px !important;
    }}
    .card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }}
    .card-title {{
        font-family: 'Kalam', cursive;
        font-size: 16px;
        font-weight: 700;
        line-height: 1.15;
    }}
    .badge-tag {{
        font-family: 'Caveat', cursive;
        font-size: 12px;
        font-weight: 700;
        padding: 1px 7px;
        border-radius: 999px;
        white-space: nowrap;
    }}
    .card-explanation {{
        font-family: 'Caveat', cursive;
        font-size: 14.5px;
        font-weight: 600;
        color: #E2E8F0;
        line-height: 1.25;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
    }}
    .card-footer {{
        font-family: 'Caveat', cursive;
        font-size: 12.5px;
        font-weight: 700;
        text-align: right;
        margin-top: 2px;
    }}
    /* Top Floating Toolbar */
    .board-toolbar {{
        position: absolute;
        top: 14px;
        right: 14px;
        display: flex;
        align-items: center;
        gap: 6px;
        background: rgba(18, 22, 30, 0.92);
        backdrop-filter: blur(10px);
        padding: 6px 12px;
        border-radius: 9999px;
        border: 1.5px dashed rgba(255, 255, 255, 0.25);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.5);
        z-index: 20;
    }}
    .board-btn {{
        background: transparent;
        border: none;
        color: #F8FAFC;
        font-family: 'Caveat', cursive;
        font-size: 16px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 4px;
        transition: background 0.15s, color 0.15s;
    }}
    .board-btn:hover {{
        background: rgba(255, 255, 255, 0.12);
        color: #FDE047;
    }}
    .clear-btn {{
        color: #F87171 !important;
        border: 1px dashed #F87171 !important;
        padding: 2px 10px;
        border-radius: 999px;
    }}
    .clear-btn:hover {{
        background: rgba(239, 68, 68, 0.2) !important;
        color: #FCA5A5 !important;
    }}
    /* Dismissed notice */
    #dismissNotice {{
        display: none;
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        color: #94A3B8;
        font-family: 'Caveat', cursive;
        font-size: 22px;
    }}
</style>
</head>
<body>
<div class="mindmap-board" id="board">
    <div class="board-toolbar">
        <button class="board-btn" onclick="zoomIn()" title="Zoom In">➕ Zoom In</button>
        <button class="board-btn" onclick="zoomOut()" title="Zoom Out">➖ Zoom Out</button>
        <button class="board-btn" onclick="resetZoom()" title="Reset View">🔄 Reset</button>
        <span style="color:rgba(255,255,255,0.25);padding:0 2px;">|</span>
        <button class="board-btn clear-btn" onclick="clearMindMap()" title="Clear & Hide Mind Map">✕ Clear Mind Map</button>
    </div>

    <div id="dismissNotice">
        ✎ Mind map dismissed.<br>
        <button class="board-btn" onclick="restoreMindMap()" style="margin-top:10px; border:1px dashed #38BDF8; color:#38BDF8;">⟳ Restore View</button>
    </div>

    <svg id="handSvg" viewBox="0 0 1400 800">
        <g id="handGroup" transform="translate(0, 0) scale(1)">
            <!-- Connectors -->
            {all_paths_str}
            <!-- Handwritten Cards -->
            {all_cards_str}
        </g>
    </svg>
</div>

<script>
    let scale = 1;
    let pointX = 0;
    let pointY = 0;
    let startX = 0;
    let startY = 0;
    let isPanning = false;

    const svg = document.getElementById('handSvg');
    const group = document.getElementById('handGroup');
    const dismissNotice = document.getElementById('dismissNotice');

    function updateTransform() {{
        group.setAttribute('transform', `translate(${{pointX}}, ${{pointY}}) scale(${{scale}})`);
    }}

    function zoomIn() {{
        scale = Math.min(2.2, scale * 1.2);
        updateTransform();
    }}

    function zoomOut() {{
        scale = Math.max(0.4, scale / 1.2);
        updateTransform();
    }}

    function resetZoom() {{
        scale = 1;
        pointX = 0;
        pointY = 0;
        updateTransform();
    }}

    function clearMindMap() {{
        svg.style.display = 'none';
        dismissNotice.style.display = 'block';
    }}

    function restoreMindMap() {{
        svg.style.display = 'block';
        dismissNotice.style.display = 'none';
        resetZoom();
    }}

    svg.addEventListener('wheel', function(e) {{
        e.preventDefault();
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        const newScale = Math.min(2.2, Math.max(0.4, scale * delta));
        scale = newScale;
        updateTransform();
    }});

    svg.addEventListener('mousedown', function(e) {{
        isPanning = true;
        startX = e.clientX - pointX;
        startY = e.clientY - pointY;
    }});

    window.addEventListener('mousemove', function(e) {{
        if (!isPanning) return;
        pointX = e.clientX - startX;
        pointY = e.clientY - startY;
        updateTransform();
    }});

    window.addEventListener('mouseup', function() {{
        isPanning = false;
    }});
</script>
</body>
</html>
"""
        return html_code
