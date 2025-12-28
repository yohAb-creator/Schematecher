from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from pypdf import PdfReader
from tqdm import tqdm

from .embeddings import Embedder
from .graph import GraphEdge, GraphNode, KnowledgeGraph
from .storage import Chunk, VectorStore

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional dependency
    pdfplumber = None


@dataclass
class Section:
    section_id: str
    title: str
    text: str


def _sections_from_outline(pdf_path: pathlib.Path, page_texts: List[str]) -> List[Section]:
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
        sections.append(Section(section_id=section_id, title=title or f"Section {section_id}", text=text))
    return sections


def extract_sections(pdf_path: pathlib.Path) -> List[Section]:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found or not a file: {pdf_path}")
    raw_text: List[str] = []
    page_count = 0
    reader = None
    if pdfplumber is not None:
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    raw_text.append(page.extract_text(layout=True) or "")
        except Exception:
            raw_text = []
    if not raw_text:  # fallback to pypdf
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
        for page in reader.pages:
            try:
                raw_text.append(page.extract_text() or "")
            except Exception:
                raw_text.append("")

    outline_sections = _sections_from_outline(pdf_path, raw_text)
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
            sections_local.append(Section(section_id=section_num, title=title or f"Section {section_num}", text=body))
        return sections_local

    # Detect numbered sections across full text.
    detected = _detect_numbered_sections(full_text)
    if detected:
        return detected

    # Fallback: treat each page as a section to avoid single-node graph
    if page_count > 1:
        return [Section(section_id=str(idx), title=f"Page {idx}", text=page_text.strip()) for idx, page_text in enumerate(raw_text, start=1)]

    # Final fallback: single full document
    return [Section(section_id="1", title="Full Document", text=full_text.strip())]


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


def infer_prereq(section_id: str) -> List[str]:
    parts = section_id.split(".")
    prereqs = []
    if len(parts) > 1:
        prereqs.append(parts[0])
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
) -> KnowledgeGraph:
    kg = KnowledgeGraph.load(output_graph)
    sections = extract_sections(pdf_path)
    for section in tqdm(sections, desc="Sections"):
        node_dir = nodes_root / pdf_id / section.section_id
        centroid, vs = build_vector_store(section, embedder, node_dir=node_dir, chunk_size=chunk_size, overlap=overlap)
        node_dir.mkdir(parents=True, exist_ok=True)
        if centroid == []:
            continue
        if vs.embeddings is not None and len(vs.chunks) > 0:
            vs.save()
        manifest = {
            "section_id": section.section_id,
            "title": section.title,
            "summary": section.text[:400],
            "topics": [],
            "centroid": centroid,
            "prerequisites": infer_prereq(section.section_id),
        }
        (node_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        node_id = f"{pdf_id}:{section.section_id}"
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
        for prereq in manifest["prerequisites"]:
            kg.add_edge(GraphEdge(edge_type="prereq", source=node_id, target=f"{pdf_id}:{prereq}"))
    kg.save(output_graph)
    return kg

