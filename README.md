## SchemaTeach Study Assistant (Graph + Vector RAG)

Hybrid retrieval architecture for studying mathematical proofs using PDFs as a knowledge graph of section-level vector databases.

### Goals
- Treat each PDF section as a node containing its own vector index; edges encode prerequisites and related concepts.
- Route queries to relevant nodes, retrieve passages locally, expand context via graph neighbors, and synthesize grounded answers with citations.
- Evaluate whether structured, prerequisite-aware retrieval improves proof understanding over baseline RAG/LLM.

### Quickstart
1) Install deps (includes local HF models and visualization):
   ```
   pip install -r requirements.txt
   ```
2) Configure models/paths in `config.yaml` (defaults: MiniLM embeddings, flan-t5-small LLM).
3) Ingest a PDF into per-section indices + graph:
   ```
   python ingest_pdfs.py --pdf path/to/book.pdf --pdf-id book1
   # Optional cleanup flags:
   #   --clear-graph   removes graph/graph.jsonl before ingest
   #   --clear-nodes   removes nodes/<pdf-id>/ before ingest
   ```
4) Run a query (override LLM model at runtime if desired):
   ```
   python -m schemateach.main --query "What is the fundamental group?"
   python -m schemateach.main --query "What is the fundamental group?" --llm-model google/flan-t5-large
   ```
   - Default config runs locally: `embedding.provider: hf` with `sentence-transformers/all-MiniLM-L6-v2`, and `llm.provider: hf` with `google/flan-t5-small`. Switch to OpenAI by editing `config.yaml` and setting `OPENAI_API_KEY`.
5) Visualize the knowledge graph (requires `graph/graph.jsonl` from ingestion):
   ```
   python -m schemateach.knowledge_graph_display --graph graph/graph.jsonl --out graph.png --node-size 1600
   ```
   For an interactive web view with zoom/pan/tooltips:
   ```
   python -m schemateach.knowledge_graph_web --graph graph/graph.jsonl --out graph.html --max-depth 3
   ```
   Open `graph.html` in a browser; use `--max-depth` to filter subsections and `--no-physics` to disable force layout.

### High-Level Architecture
- **Ingestion/Graph Builder**
  - Parse PDFs into structured sections (via PDF outlines/bookmarks when available; fallback to heading/page heuristics).
  - Chunk within section (semantic or sentence-level) and embed with math-aware model (default MiniLM; can swap).
  - Store per-section vectors in local stores; naming `nodes/<pdf>/<section>/`.
  - Build graph metadata: nodes `{pdf_id, section_id, title, prerequisites, topics, centroid}`; edges typed (`prereq`, etc).
- **Query Pipeline**
  1) Query analysis/embedding.
  2) Node routing (centroid similarity + prereq bonus).
  3) Local retrieval inside nodes; dedupe chunks.
  4) Graph expansion along prerequisites (configurable hops).
  5) LLM synthesis with citations and truncation to fit model limits.
- **Data + Config**
  - `graph/graph.jsonl`: node + edge metadata.
  - `nodes/<pdf>/<section>/index.npz`, `chunks.jsonl`, `manifest.json`.
  - `config.yaml`: embedding model, chunking params, routing weights, expansion budgets.

### Evaluation Sketch
- Build proof-related queries (definition, theorem restatement, missing step).
- Compare baseline single-DB RAG vs graph+prereq expansion on answer quality, citation correctness, and hallucination rate.
