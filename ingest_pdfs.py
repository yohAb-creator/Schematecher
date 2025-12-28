from __future__ import annotations

import argparse
import pathlib

from schemateach.config import Config
from schemateach.embeddings import Embedder
from schemateach.ingest import ingest_pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDF into graph + per-section vector stores")
    parser.add_argument("--pdf", required=True, help="Path to PDF")
    parser.add_argument("--pdf-id", required=True, help="Identifier for the PDF (used in node paths)")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--chunk-size", type=int, default=600, help="Chunk size in tokens (rough)")
    parser.add_argument("--overlap", type=int, default=60, help="Chunk overlap")
    parser.add_argument("--clear-graph", action="store_true", help="Remove existing graph file before ingestion")
    parser.add_argument("--clear-nodes", action="store_true", help="Remove existing nodes directory for this PDF before ingestion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_file(args.config)
    embedder = Embedder(cfg.embedding_provider, cfg.embedding_model, cfg.embedding_dims)
    pdf_path = pathlib.Path(args.pdf)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found or not a file: {pdf_path}")
    graph_path = cfg.graph_dir / "graph.jsonl"
    if args.clear_graph and graph_path.exists():
        graph_path.unlink()
        print(f"Cleared graph file: {graph_path}")
    if args.clear_nodes:
        pdf_nodes_dir = cfg.nodes_dir / args.pdf_id
        if pdf_nodes_dir.exists():
            import shutil

            shutil.rmtree(pdf_nodes_dir)
            print(f"Cleared node embeddings: {pdf_nodes_dir}")
    ingest_pdf(
        pdf_path=pdf_path,
        pdf_id=args.pdf_id,
        output_graph=graph_path,
        nodes_root=cfg.nodes_dir,
        embedder=embedder,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(f"Ingested {pdf_path} -> {graph_path}")


if __name__ == "__main__":
    main()
