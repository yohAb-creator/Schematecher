from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import statistics

import numpy as np
from pypdf import PdfReader
from tqdm import tqdm

from .embeddings import Embedder
from .graph import GraphEdge, GraphNode, KnowledgeGraph
from .index import ChunkMeta, GlobalChunkIndex, load_concept_map, save_concept_map
from .storage import Chunk, VectorStore

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional dependency
    pdfplumber = None
try:
    from transformers import pipeline
except ImportError:  # pragma: no cover - optional dependency
    pipeline = None


@dataclass
class Section:
    section_id: str
    title: str
    text: str
    definitions: List[str] = field(default_factory=list)


def _sections_from_outline(
    pdf_path: pathlib.Path,
    page_texts: List[str],
    pages: Optional[List[object]] = None,
) -> List[Section]:
    """
    Use PDF bookmarks/outlines to build sections that mirror the document structure,
    including nested subsections. Section IDs are generated based on outline depth.
    """
    try:
        reader = PdfReader(str(pdf_path))
        outlines = getattr(reader, "outlines", None) or getattr(reader, "outline", None)
    except Exception:
        return []
    if not outlines:
        return []

    entries: List[tuple[int, str, int, int]] = []  # (page_num, title, level, order)
    order = 0

    def _walk(items, level: int) -> None:
        nonlocal order
        for item in items:
            if isinstance(item, list):
                _walk(item, level + 1)
                continue
            try:
                title = str(getattr(item, "title", "") or getattr(item, "name", "")).strip()
                page_num = reader.get_destination_page_number(item)
                entries.append((page_num, title, level, order))
                order += 1
            except Exception:
                continue

    _walk(outlines, level=1)
    if not entries:
        return []

    # Keep outline order but sort by page to bound spans
    entries.sort(key=lambda x: (x[0], x[3]))

    # Generate hierarchical section ids (e.g., 1, 1.1, 1.1.1)
    sections: List[Section] = []
    counters: List[int] = []
    for idx, (start_page, title, level, _) in enumerate(entries):
        while len(counters) < level:
            counters.append(0)
        counters = counters[:level]
        counters[level - 1] += 1
        section_id = ".".join(str(c) for c in counters)
        end_page = entries[idx + 1][0] if idx + 1 < len(entries) else len(page_texts)
        text = "\n".join(page_texts[start_page:end_page]).strip()
        if not text:
            continue
        defs: List[str] = []
        if pages:
            defs = _extract_definitions_from_pages(pages[start_page:end_page])
        if not defs:
            defs = _extract_definitions_from_text(text)
        sections.append(Section(section_id=section_id, title=title or f"Section {section_id}", text=text, definitions=defs))
    return sections


def _extract_definitions_from_text(text: str) -> List[str]:
    defs: List[str] = []
    for match in re.finditer(r"(Definition\\s+\\d+(?:\\.\\d+)*\\.?\\s+.+?)(?=\\n\\s*(Definition|Lemma|Theorem|Proposition|Corollary|Remark|Example|Exercise|Proof|Chapter|Section)\\b|\\Z)", text, re.S):
        snippet = " ".join(match.group(1).split())
        defs.append(snippet)
    return defs


def _extract_definitions_from_pages(pages: List[object]) -> List[str]:
    defs: List[str] = []
    if not pages:
        return defs
    lines: List[Tuple[str, float]] = []
    sizes: List[float] = []
    for page in pages:
        try:
            words = page.extract_words(keep_blank_chars=False, extra_attrs=["size", "top", "x0"])
        except Exception:
            continue
        words.sort(key=lambda w: (w.get("top", 0), w.get("x0", 0)))
        current_top = None
        current_words = []
        for w in words:
            top = w.get("top", 0)
            if current_top is None or abs(top - current_top) <= 2.0:
                current_top = top if current_top is None else current_top
                current_words.append(w)
            else:
                line_text = " ".join(item["text"] for item in current_words if item.get("text"))
                if line_text.strip():
                    line_sizes = [float(item.get("size", 0)) for item in current_words if item.get("size")]
                    if line_sizes:
                        lines.append((line_text, statistics.median(line_sizes)))
                        sizes.extend(line_sizes)
                current_words = [w]
                current_top = top
        if current_words:
            line_text = " ".join(item["text"] for item in current_words if item.get("text"))
            if line_text.strip():
                line_sizes = [float(item.get("size", 0)) for item in current_words if item.get("size")]
                if line_sizes:
                    lines.append((line_text, statistics.median(line_sizes)))
                    sizes.extend(line_sizes)
    if not sizes:
        return defs
    body_size = statistics.median(sizes)
    stop_re = re.compile(
        r"^(Definition\\s+\\d|Lemma\\s+\\d|Theorem\\s+\\d|Proposition\\s+\\d|Corollary\\s+\\d|Remark\\s+\\d|Example\\s+\\d|Exercise\\s+\\d|Proof\\b|Chapter\\s+\\d|Section\\s+\\d)"
    )
    collecting = False
    current: List[str] = []
    for line_text, line_size in lines:
        clean = line_text.strip()
        if not clean:
            continue
        if re.match(r"^Definition\\s+\\d", clean):
            if collecting and current:
                defs.append(" ".join(current).strip())
            collecting = True
            current = [clean]
            continue
        if collecting:
            if stop_re.match(clean):
                defs.append(" ".join(current).strip())
                collecting = False
                current = []
                continue
            if line_size > body_size + 1.5 and re.match(r"^\\d+(?:\\.\\d+)*\\s+", clean):
                defs.append(" ".join(current).strip())
                collecting = False
                current = []
                continue
            current.append(clean)
    if collecting and current:
        defs.append(" ".join(current).strip())
    return defs


def _normalize_hf_model(model: str) -> str:
    if model.startswith("flan-t5-"):
        return f"google/{model}"
    return model


def _sections_from_llm_pages(page_texts: List[str], model: str, max_pages: int = 12) -> List[Section]:
    if pipeline is None:
        return []
    trimmed_pages = []
    for idx, text in enumerate(page_texts[:max_pages], start=1):
        snippet = " ".join(text.split())[:800]
        trimmed_pages.append(f"Page {idx}: {snippet}")
    prompt = (
        "You are extracting section structure from a textbook. "
        "Given page snippets, return JSON with an array 'sections', where each item has "
        "id (numbered like 1, 1.1, 1.2), title, start_page, end_page (inclusive). "
        "Do not include figures or tables. Example: "
        '{"sections":[{"id":"1","title":"Introduction","start_page":1,"end_page":2},{"id":"1.1","title":"Definition","start_page":3,"end_page":3}]}\n'
        "Pages:\n" + "\n".join(trimmed_pages) + "\nJSON:"
    )
    gen = pipeline("text2text-generation", model=_normalize_hf_model(model))
    out = gen(prompt, max_new_tokens=256, do_sample=False, truncation=True)
    text = out[0]["generated_text"]
    try:
        data = json.loads(text)
        sections_data = data.get("sections", [])
    except Exception:
        return []
    sections: List[Section] = []
    for item in sections_data:
        try:
            sid = str(item["id"])
            title = str(item.get("title", "")).strip()
            start = int(item.get("start_page", 0))
            end = int(item.get("end_page", start))
            if start <= 0 or end < start:
                continue
            slice_text = "\n".join(page_texts[start - 1 : end]).strip()
            if not slice_text:
                continue
            defs = _extract_definitions_from_text(slice_text)
            sections.append(Section(section_id=sid, title=title or f"Section {sid}", text=slice_text, definitions=defs))
        except Exception:
            continue
    return sections


def extract_sections(pdf_path: pathlib.Path, use_llm_sections: bool = False, llm_section_model: Optional[str] = None) -> List[Section]:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found or not a file: {pdf_path}")
    raw_text: List[str] = []
    page_count = 0
    reader = None
    plumber_pages: List[object] = []
    if pdfplumber is not None:
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                plumber_pages = list(pdf.pages)
                page_count = len(plumber_pages)
                for page in plumber_pages:
                    raw_text.append(page.extract_text(layout=True) or "")
        except Exception:
            raw_text = []
            plumber_pages = []
    if not raw_text:  # fallback to pypdf
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
        for page in reader.pages:
            try:
                raw_text.append(page.extract_text() or "")
            except Exception:
                raw_text.append("")

    outline_sections = _sections_from_outline(pdf_path, raw_text, pages=plumber_pages if plumber_pages else None)
    if outline_sections:
        return outline_sections

    full_text = "\n".join(raw_text).replace("\xa0", " ")

    def _detect_numbered_sections(text: str) -> List[Section]:
        primary = re.compile(r"^(?P<num>\d+(?:\.\d+)*)(?:\s+|\.)?(?P<title>.+)$", re.MULTILINE)
        matches = list(primary.finditer(text))
        if len(matches) < 2:
            alt = re.compile(r"^(Chapter|Section)\s+(?P<num>[\dIVXLC]+)[\.:\\s]+(?P<title>.+)$", re.MULTILINE)
            matches = list(alt.finditer(text))
        filtered = []
        for m in matches:
            title = m.groupdict().get("title", "").strip().lower()
            if title.startswith(("figure", "fig.", "fig ", "table", "tbl")):
                continue
            filtered.append(m)
        matches = filtered
        sections_local: List[Section] = []
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            section_num = match.group("num")
            title = match.groupdict().get("title", "").strip()
            body = text[start:end].strip()
            if not body:
                continue
            defs = _extract_definitions_from_text(body)
            sections_local.append(Section(section_id=section_num, title=title or f"Section {section_num}", text=body, definitions=defs))
        return sections_local

    # Detect numbered sections across full text.
    detected = _detect_numbered_sections(full_text)
    if detected:
        return detected

    if use_llm_sections and llm_section_model:
        llm_sections = _sections_from_llm_pages(raw_text, llm_section_model)
        if llm_sections:
            return llm_sections

    # Fallback: treat each page as a section to avoid single-node graph
    if page_count > 1:
        sections = []
        for idx, page_text in enumerate(raw_text, start=1):
            defs = []
            if plumber_pages:
                defs = _extract_definitions_from_pages([plumber_pages[idx - 1]])
            if not defs:
                defs = _extract_definitions_from_text(page_text)
            sections.append(Section(section_id=str(idx), title=f"Page {idx}", text=page_text.strip(), definitions=defs))
        return sections

    # Final fallback: single full document
    defs = _extract_definitions_from_text(full_text)
    return [Section(section_id="1", title="Full Document", text=full_text.strip(), definitions=defs)]


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 60) -> List[str]:
    tokens = text.split()
    if not tokens:
        return []
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i : i + chunk_size]
        chunks.append(" ".join(chunk_tokens))
        i += chunk_size - overlap
    return chunks


def build_vector_store(section: Section, embedder: Embedder, node_dir: pathlib.Path, chunk_size: int, overlap: int) -> Tuple[List[float], VectorStore]:
    chunk_texts = chunk_text(section.text, chunk_size=chunk_size, overlap=overlap)
    if not chunk_texts:
        return [], VectorStore(node_dir)
    embeddings = embedder.embed(chunk_texts)
    vs = VectorStore(node_dir)
    chunks = []
    for idx, text in enumerate(chunk_texts):
        chunks.append(Chunk(chunk_id=f"{section.section_id}-{idx}", text=text, rank=idx, metadata={"section": section.section_id}))
    vs.set_embeddings(np.stack(embeddings, axis=0), chunks)
    centroid = np.mean(np.stack(embeddings, axis=0), axis=0).tolist()
    return centroid, vs


def extract_concepts(text: str) -> List[str]:
    concepts: List[str] = []
    def_pattern = re.compile(
        r"Definition\\s+\\d+(?:\\.\\d+)*\\.?\\s+(?:A|An|The)\\s+([A-Za-z0-9\\-\\s\\(\\)]+?)\\s+(?:is|are)\\b",
        re.I,
    )
    for match in def_pattern.finditer(text):
        term = match.group(1)
        if term:
            cleaned = re.sub(r"\\s+", " ", term).strip(" .,:;")
            if cleaned:
                concepts.append(cleaned)
    return list(dict.fromkeys(concepts))


def infer_prereq(section_id: str) -> List[str]:
    parts = section_id.split(".")
    prereqs = []
    if len(parts) > 1:
        prereqs.append(".".join(parts[:-1]))
    elif parts and parts[0].isdigit():
        prev_num = int(parts[0]) - 1
        if prev_num > 0:
            prereqs.append(str(prev_num))
    return list(dict.fromkeys(prereqs))


def ingest_pdf(
    pdf_path: pathlib.Path,
    pdf_id: str,
    output_graph: pathlib.Path,
    nodes_root: pathlib.Path,
    embedder: Embedder,
    chunk_size: int = 600,
    overlap: int = 60,
    use_llm_sections: bool = False,
    llm_section_model: Optional[str] = None,
) -> KnowledgeGraph:
    kg = KnowledgeGraph.load(output_graph)
    global_index = GlobalChunkIndex.load(output_graph.parent)
    concept_map = load_concept_map(output_graph.parent)
    sections = extract_sections(pdf_path, use_llm_sections=use_llm_sections, llm_section_model=llm_section_model)
    current_node_ids: List[str] = []
    new_meta: List[ChunkMeta] = []
    new_embeddings: List[np.ndarray] = []
    for section in tqdm(sections, desc="Sections"):
        node_dir = nodes_root / pdf_id / section.section_id
        centroid, vs = build_vector_store(section, embedder, node_dir=node_dir, chunk_size=chunk_size, overlap=overlap)
        node_dir.mkdir(parents=True, exist_ok=True)
        if centroid == []:
            continue
        if vs.embeddings is not None and len(vs.chunks) > 0:
            vs.save()
        node_id = f"{pdf_id}:{section.section_id}"
        concepts = extract_concepts(section.text)
        for term in concepts:
            key = term.lower()
            concept_map.setdefault(key, [])
            if node_id not in concept_map[key]:
                concept_map[key].append(node_id)
        manifest = {
            "section_id": section.section_id,
            "title": section.title,
            "summary": section.text[:400],
            "topics": [],
            "centroid": centroid,
            "prerequisites": infer_prereq(section.section_id),
            "concepts": concepts,
            "definitions": section.definitions,
        }
        (node_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        current_node_ids.append(node_id)
        kg.add_node(
            GraphNode(
                node_id=node_id,
                pdf_id=pdf_id,
                section_id=section.section_id,
                title=section.title,
                summary=manifest["summary"],
                topics=[],
                centroid=centroid,
                prerequisites=manifest["prerequisites"],
                path=str(node_dir),
            )
        )

        if vs.embeddings is not None and vs.chunks:
            new_embeddings.append(vs.embeddings)
            for chunk in vs.chunks:
                new_meta.append(
                    ChunkMeta(
                        node_id=node_id,
                        pdf_id=pdf_id,
                        section_id=section.section_id,
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                    )
                )
    # Rebuild prerequisite edges for this PDF to avoid dangling targets.
    current_nodes = [kg.nodes[nid] for nid in current_node_ids if nid in kg.nodes]
    section_map = {node.section_id: node for node in current_nodes}
    grouped: dict[str, list[GraphNode]] = {}
    for node in current_nodes:
        major = node.section_id.split(".")[0]
        grouped.setdefault(major, []).append(node)

    def _section_key(section_id: str) -> tuple:
        parts = section_id.split(".")
        key = []
        for part in parts:
            if part.isdigit():
                key.append(int(part))
            else:
                key.append(part)
        return tuple(key)

    kg.edges = [edge for edge in kg.edges if not edge.source.startswith(f"{pdf_id}:")]
    existing_edges = {(edge.source, edge.target, edge.edge_type) for edge in kg.edges}

    for _, nodes in grouped.items():
        nodes.sort(key=lambda n: _section_key(n.section_id))
        chapter_head = nodes[0] if nodes else None
        last_seen_by_depth: dict[int, GraphNode] = {}
        for node in nodes:
            parts = node.section_id.split(".")
            depth = len(parts)
            prereqs: List[str] = []
            if depth > 1:
                parent_id = ".".join(parts[:-1])
                if parent_id in section_map:
                    prereqs.append(parent_id)
            if depth in last_seen_by_depth:
                prereqs.append(last_seen_by_depth[depth].section_id)
            if chapter_head and chapter_head.section_id != node.section_id:
                prereqs.append(chapter_head.section_id)

            node.prerequisites = list(dict.fromkeys(prereqs))
            manifest_path = pathlib.Path(node.path) / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["prerequisites"] = node.prerequisites
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            for prereq in node.prerequisites:
                target_id = f"{pdf_id}:{prereq}"
                edge_key = (node.node_id, target_id, "prereq")
                if edge_key in existing_edges or target_id not in kg.nodes:
                    continue
                kg.add_edge(GraphEdge(edge_type="prereq", source=node.node_id, target=target_id))
                existing_edges.add(edge_key)

            last_seen_by_depth[depth] = node
    if new_embeddings:
        global_index.append(np.vstack(new_embeddings), new_meta)
        global_index.save(output_graph.parent)
    save_concept_map(output_graph.parent, concept_map)
    kg.save(output_graph)
    return kg
