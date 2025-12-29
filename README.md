<img width="819" height="497" alt="image" src="https://github.com/user-attachments/assets/c277e588-b28a-4421-aa4f-a54a94aac614" />

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
   # Optional LLM-assisted section extraction when outlines/headings are missing:
   #   --llm-section-extract --llm-section-model google/flan-t5-large
   ```
4) Run a query (override LLM model at runtime if desired):
   ```
   python -m schemateach.main --query "What is the fundamental group?"
   python -m schemateach.main --query "What is the fundamental group?" --llm-model google/flan-t5-large
   python -m schemateach.main --query "What is the fundamental group?" --llm-model openai:gpt-3.5-turbo
   python -m schemateach.main --query "What is the fundamental group?" --llm-model openai:gpt-4o
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

### Frontend (v0 Study Assistant)
The UI lives in `frontend/` (Next.js). To run it locally:
```
cd frontend
pnpm install
pnpm dev
```
If you prefer npm or yarn, swap the package manager. This frontend is currently standalone; wire it to the backend by adding an API route or proxy once you decide on a serving layer (e.g., FastAPI/Flask).

### Backend API (FastAPI)
Run the backend server (serves `/query` for the frontend):
```
python -m uvicorn schemateach.server:app --host 127.0.0.1 --port 8000
```
The frontend proxies through `frontend/app/api/search/route.ts`. You can override the backend URL:
```
SCHEMATEACH_BACKEND_URL=http://127.0.0.1:8000 pnpm dev
```
