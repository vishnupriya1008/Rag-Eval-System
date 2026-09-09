"""
End-to-end pipeline: index the corpus once, then answer queries.

Run modes:
  python pipeline.py --index          # build/rebuild the search indices
  python pipeline.py --query "..."    # answer a single question
  python pipeline.py                  # interactive REPL

BM25 works immediately (no downloads needed). Dense/hybrid search and
LLM generation require running `python embeddings.py` first, which
needs internet access to Hugging Face (works locally, not in a
network-restricted sandbox - see embeddings.py's docstring).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from ingest import load_corpus
from chunking import heading_aware_chunk
from bm25_retriever import BM25Retriever

CORPUS_DIR = str(Path(__file__).parent.parent / "data/raw/fastapi_docs")
EMBEDDINGS_PATH = Path(__file__).parent.parent / "data/processed/chunk_embeddings.npy"


class RAGPipeline:
    def __init__(self):
        print("Loading corpus and building BM25 index...")
        self.docs = load_corpus(CORPUS_DIR)
        self.chunks = heading_aware_chunk(self.docs)
        self.bm25 = BM25Retriever(self.chunks)

        self.vector_store = None
        self.embedder = None
        self.hybrid = None
        self.reranking_retriever = None
        self.generator = None

        if EMBEDDINGS_PATH.exists():
            self._load_dense_components()
            self._try_load_reranker()
        else:
            print("  (dense embeddings not found - run `python embeddings.py` for hybrid search)")

        print(f"Indexed {len(self.chunks)} chunks from {len(self.docs)} documents.\n")

    def _load_dense_components(self):
        from vector_store import VectorStore
        from embeddings import Embedder
        from hybrid_retriever import HybridRetriever

        print("  Loading dense embeddings and hybrid retriever...")
        chunk_embeddings = np.load(EMBEDDINGS_PATH)
        self.embedder = Embedder()
        self.vector_store = VectorStore.from_normalized(self.chunks, chunk_embeddings)
        self.hybrid = HybridRetriever(self.bm25, self.vector_store, self.embedder)

    def _try_load_reranker(self):
        try:
            from reranker import CrossEncoderReranker, RerankingRetriever
            print("  Loading cross-encoder reranker...")
            reranker = CrossEncoderReranker()
            self.reranking_retriever = RerankingRetriever(self.hybrid, reranker, candidate_k=20)
        except Exception as e:
            print(f"  (reranker not available: {e})")

    def _load_generator(self):
        if self.generator is None:
            from generator import LocalGenerator
            print("Loading local LLM (first run downloads the model)...")
            self.generator = LocalGenerator()

    def retrieve(self, query: str, top_k: int = 5, method: str = "auto"):
        # "auto" picks the best available method: reranked > hybrid > bm25-only
        if method in ("auto", "rerank") and self.reranking_retriever is not None:
            return [c for c, _ in self.reranking_retriever.search(query, top_k)]
        if method in ("auto", "hybrid") and self.hybrid is not None:
            return [c for c, _ in self.hybrid.search(query, top_k)]
        return [c for c, _ in self.bm25.search(query, top_k)]  # falls back to BM25-only

    def answer(self, query: str, top_k: int = 5, generate: bool = True) -> dict:
        if self.reranking_retriever is not None:
            method = "rerank"
        elif self.hybrid is not None:
            method = "hybrid"
        else:
            method = "bm25"
        retrieved = self.retrieve(query, top_k=top_k, method=method)

        result = {
            "query": query,
            "method": method,
            "sources": [{"doc_id": c.doc_id, "heading": c.heading_path} for c in retrieved],
        }

        if generate:
            self._load_generator()
            result["answer"] = self.generator.generate(query, retrieved)

        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Answer a single question and exit")
    parser.add_argument("--no-generate", action="store_true", help="Retrieval only, skip LLM generation")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    pipeline = RAGPipeline()

    def run_one(q):
        result = pipeline.answer(q, top_k=args.top_k, generate=not args.no_generate)
        print(f"\nMethod: {result['method']}")
        print("Sources:")
        for s in result["sources"]:
            print(f"  - {s['doc_id']}" + (f" ({s['heading']})" if s["heading"] else ""))
        if "answer" in result:
            print(f"\nAnswer:\n{result['answer']}")

    if args.query:
        run_one(args.query)
    else:
        print("Interactive mode - type a question, or 'quit' to exit.\n")
        while True:
            q = input("> ").strip()
            if q.lower() in ("quit", "exit", ""):
                break
            run_one(q)


if __name__ == "__main__":
    main()
