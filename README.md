## SchemaTeach Study Assistant (Graph + Vector RAG)

Hybrid retrieval architecture for studying mathematical proofs using PDFs as a knowledge graph of section-level vector databases.

### Goals
- Treat each PDF section as a node containing its own vector index; edges encode prerequisites and related concepts.
- Route queries to relevant nodes, retrieve passages locally, expand context via graph neighbors, and synthesize grounded answers with citations.
- Evaluate whether structured, prerequisite-aware retrieval improves proof understanding over baseline RAG/LLM.

### Quickstart
1) Install deps (optionally skip `openai` if you only need the fallback deterministic embedder/LLM):
   ```
   pip install -r requirements.txt
   ```
2) Configure models/paths in `config.yaml`.
3) Ingest a PDF into per-section indices + graph:
   ```
   python ingest_pdfs.py --pdf path/to/book.pdf --pdf-id book1
   # Optional cleanup flags:
   #   --clear-graph   removes graph/graph.jsonl before ingest
   #   --clear-nodes   removes nodes/<pdf-id>/ before ingest
   ```
4) Run a query:
   ```
   python -m schemateach.main --query "Outline the proof of the dominated convergence theorem."
   # Override LLM model (e.g., flan-t5-large) at runtime:
   python -m schemateach.main --query "What is the fundamental group?" --llm-model google/flan-t5-large
   ```
   - Default config now runs locally: `embedding.provider: hf` with `all-MiniLM-L6-v2`, and `llm.provider: hf` with `flan-t5-small`. Switch to OpenAI by editing `config.yaml` and setting `OPENAI_API_KEY`.
5) Visualize the knowledge graph (requires `graph/graph.jsonl` from ingestion):
   ```
   python -m schemateach.knowledge_graph_display --graph graph/graph.jsonl --out graph.png --node-size 1600
   ```
   Omit `--out` to display an interactive window.
   For an interactive web view with zoom/pan/tooltips:
   ```
   python -m schemateach.knowledge_graph_web --graph graph/graph.jsonl --out graph.html --max-depth 3
   ```
   Open `graph.html` in a browser; use `--max-depth` to filter subsections and `--no-physics` to disable force layout.

### High-Level Architecture
- **Ingestion/Graph Builder**
  - Parse PDFs into structured sections (title, number, text, math blocks, figures).
  - Chunk within section (semantic or sentence-level) and embed with math-aware model (e.g., `text-embedding-3-large`, `bge-m3`, `nomic-embed-text`, or domain math model).
  - Store per-section vectors in lightweight local DBs (e.g., Chroma/Qdrant/HNSWlib per node; naming `pdf_id/section_id.index`).
  - Build graph metadata: nodes `{pdf_id, section_id, title, prerequisites, topics, embedding_stats}`; edges typed (`prereq`, `refers_to`, `same_topic`).
- **Query Pipeline**
  1) **Query analysis**: classify intent (definition, theorem proof, example) and extract key entities; embed query.
  2) **Node routing**: score query vs section summaries/prereq topics to pick top-K nodes; optionally include ancestor prerequisites if query complexity high.
  3) **Local retrieval**: run vector search inside chosen nodes; apply MMR and positional boosts (statements > examples when proving).
  4) **Graph expansion**: if coverage low, expand along edges (prereq -> ancestors; refers_to -> cited lemmas) with tight budget.
  5) **LLM synthesis**: craft answer grounded in retrieved chunks; include labeled citations `[(pdf, section, chunk_id)]`; include missing-context callouts.
  6) **Feedback loop**: capture relevance judgments and proof-correctness checks to refine routing weights and chunking.
- **Data + Config**
  - `graph/graph.jsonl`: node + edge metadata.
  - `nodes/<pdf>/<section>/index/`: vector DB files.
  - `nodes/<pdf>/<section>/manifest.json`: section summary, centroid embedding, topics, quality scores.
  - `config.yaml`: embedding model, chunking params, routing weights, expansion budgets.

### Ingestion Pipeline (deterministic script)
- `ingest_pdfs.py --pdf path --out graph/ --chunk-size 300 --overlap 60`
  - Extract structure (PyMuPDF/pdfplumber + regex for headings).
  - Detect math blocks (LaTeX, TeX delimiters, unicode math) and keep atomic.
  - Generate section summary + topics with small LLM; store in manifest.
  - Build per-section vector DB; write centroid embedding to graph node.
  - Infer edges: 
    - Prereq via heading numbers (e.g., 2.1 -> 2) + keyword heuristics (definition → theorem).
    - Cited labels (`Lemma X`, `Theorem Y`); map to sections when possible.
    - Topic similarity between section summaries to create `related` edges.

### Query Routing Details
- Node score = `w_sem * cosine(query_emb, node_centroid) + w_prereq * prereq_match + w_recent * recency`.
- Always include explicit prerequisites for top nodes when intent involves proofs/derivations.
- Stop routing expansion when marginal gain < threshold or budget hit (time or token).

### Retrieval + Synthesis Prompts (sketch)
- **Retriever**: `select top chunks maximizing relevance + diversity; prefer statements/lemmas if query asks for proof`.
- **LLM prompt** template:
  ```
  You are a study assistant for math proofs. Use only the provided evidence.
  Cite as [pdf:section:chunk]. If missing steps, state them explicitly before answering.
  ```

### Evaluation Plan
- Build eval set of proof tasks: restate theorem, outline proof, fill missing step, explain prerequisite concept.
- Compare:
  - **Baseline**: vanilla RAG over whole corpus (single vector DB).
  - **Hybrid**: graph + per-section nodes + prereq expansion.
- Metrics: answer quality (expert/LLM grading), citation correctness, completeness, hallucination rate, time cost.

### Implementation Roadmap
1) Scaffold repo: `ingest_pdfs.py`, `router.py`, `retriever.py`, `synthesizer.py`, `config.yaml`, `graph/`.
2) Implement ingestion with one PDF; emit graph manifests + per-section indices (Chroma/Qdrant local).
3) Implement router using node centroids + prerequisites expansion.
4) Implement retrieval + synthesis (LLM wrapper; streaming answer with citations).
5) Add evaluations + CLI (`python main.py --query "..."`).
6) Iterate chunking/edge heuristics and routing weights based on eval.

### Next Steps
- Confirm preferred embedding + vector DB (Chroma vs Qdrant) and LLM provider.
- Provide a sample PDF to run ingestion and test end-to-end.
- Decide on edge schema details (`prereq`, `refers_to`, `related`) and budgets for expansion.
