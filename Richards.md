## Richards.md

I am Reed Richards, and I will describe our system in precise, causal terms. Consider this a lab notebook entry from the Baxter Building: clear hypotheses, measured signals, and engineered constraints.

### What We Built (Chronological Summary)
- A hybrid graph + vector RAG system that treats each PDF section as a node with its own vector store.
- A resilient ingestion pipeline that favors PDF outlines and falls back to headings, pages, or optional LLM sectioning.
- A front end imported from v0, wired to a real backend via a FastAPI `/query` endpoint.
- An interactive knowledge graph view (both static and web) with visibility fixes for labels and edges.
- A routing stack upgraded from title matching to content-based routing with dense, lexical, and concept backstops.

### The System (SchemaTeach)
SchemaTeach is a retrieval architecture optimized for mathematical proofs and structured explanations. It is a graph of sections, not a flat vector heap.

1) Ingestion
- Extract sections using PDF outlines/bookmarks (the true LaTeX structure). If outlines are missing, fall back to regex headings or page-level sections.
- Chunk each section, embed the chunks, and store them per node in `nodes/<pdf>/<section>/`.
- Compute a node embedding as the centroid of its chunk embeddings.
- Build global artifacts: `graph/graph.jsonl`, `graph/chunks_index.npz`, `graph/chunks_meta.jsonl`, and `graph/concepts.json`.
- Extract definition blocks pre-chunking using layout signals (font size and line structure) and save them in each `manifest.json`.

2) Routing (the actual intelligence)
- Dense signal: compare query embedding to node centroids (content-based, not title-based).
- Evidence signal: global chunk search -> node voting (robust when titles are vague).
- Sparse signal: lexical overlap on section summaries.
- Concept signal: boost nodes linked to extracted definitions or labeled terms.
The router answers: "Which content distribution most likely explains this query?"

3) Retrieval + Expansion
- Retrieve top chunks within the routed nodes.
- Expand along prerequisite edges to supply missing context.
- Deduplicate chunks so one section does not drown the rest.

4) Synthesis
- Truncate and filter evidence to avoid small-model failure.
- For definition queries, extract the definition directly (no paraphrase if not needed).
- For general queries, use an LLM prompt with citations and repetition penalties.

5) Frontend Integration
- Next.js app in `frontend/` calls `/api/search`, which proxies to the backend.
- Graph view uses `vis-network` with fixed imports, visible edges, and legible nodes.
- Evidence cards are interactive, expand in place, and are keyboard accessible.

### What We Just Fixed
- Build errors in the graph visualization (vis-network import paths).
- Missing edges (API alias for `from`, and edge generation aligned to real nodes).
- LLM truncation issues and repetition loops in small models.
- Definition visibility in evidence cards (definition extraction at ingestion time).
- Frontend wiring to the backend (real query path instead of placeholder data).
- OpenAI model routing and UI model selection (provider switching, allowed models).
- Reasoning paths surfaced in UI (graph traversal exposed to users).
- Multi-hop prerequisite traversal (`prereq_paths`) feeds explicit reasoning chains into synthesis.
- Definition-first query handling and keyword-focused evidence selection.
- Layout-aware definition extraction persisted to section manifests.
- Health endpoint for verifying OpenAI key visibility in the backend.

### Files of Record
- Backend: `schemateach/server.py`, `schemateach/router.py`, `schemateach/ingest.py`, `schemateach/synthesizer.py`
- Graph artifacts: `graph/graph.jsonl`, `graph/chunks_index.npz`, `graph/concepts.json`
- Frontend: `frontend/app/page.tsx`, `frontend/app/api/search/route.ts`, `frontend/components/knowledge-graph.tsx`, `frontend/components/evidence-cards.tsx`
 - Ops: `schemateach/server.py` (`/health`), `config.yaml` (routing + LLM settings)

### Next Rational Experiments
- Add a small evaluation harness with proof-focused queries to tune routing weights.
- Add section summaries generated at ingest time to improve centroid quality.
- Add a "Definitions" panel in the UI that lists all extracted definitions for the active section.

The system now behaves like a research assistant: it routes by content, retrieves by evidence, and speaks in grounded citations. This is the correct direction.
