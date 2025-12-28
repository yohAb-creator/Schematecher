## SchemaTeach Demo Script

### 0:00 - 0:15 Intro
"Hi everyone. This is SchemaTeach: a hybrid graph + vector RAG system for studying math. Instead of one giant index, each PDF section is a node with its own vector store, and edges encode prerequisites."

### 0:15 - 0:35 Why It Matters
"Standard RAG gives you relevant chunks, but it often misses structure. For proofs and definitions, structure is the point. SchemaTeach routes by section content, expands along prerequisites, and synthesizes a grounded answer with citations."

### 0:35 - 1:05 Show The UI
Action: Open the frontend.
"Here is the UI. I can choose the model, ask a question, and get three things back: the answer, the evidence, and the graph view."

### 1:05 - 1:40 Demo Query (Definition)
Action: Query "What is a topological space?"
"The router picks the right section based on content embeddings, not titles. The system then retrieves local chunks and extracts the formal definition first, so the answer is self-contained and cited."

### 1:40 - 2:10 Demo Query (Reasoning Path)
Action: Toggle graph and show reasoning paths.
"Now I show the reasoning path. These are the prerequisite chains used to expand context. This makes the model's context selection explicit and debuggable."

### 2:10 - 2:35 Evidence Cards
Action: Click an evidence card.
"Each card is a chunk with its source section. This is the provenance layer: the model can only say what the evidence supports."

### 2:35 - 3:00 Close
"The goal is stronger explanations than a base LLM, especially for proofs. The graph gives structure, the vectors give recall, and the synthesis stays grounded. That's SchemaTeach."
