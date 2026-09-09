"""
Runs the eval harness against whichever retrievers are available and
prints a results table in the same shape as the README's results
section. This is the script whose output you paste into your resume
bullets and README - so its numbers should always come from an actual
run, never be hand-typed.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingest import load_corpus
from chunking import fixed_size_chunk, heading_aware_chunk
from bm25_retriever import BM25Retriever
from metrics import evaluate_retriever

CORPUS_DIR = str(Path(__file__).parent.parent / "data/raw/fastapi_docs")
EVAL_SET_PATH = Path(__file__).parent / "eval_dataset.json"


def load_eval_set():
    with open(EVAL_SET_PATH) as f:
        return json.load(f)


def run_all_configs(k: int = 5):
    docs = load_corpus(CORPUS_DIR)
    eval_set = load_eval_set()
    results = {}

    # --- Config 1: BM25 over fixed-size chunks (baseline) ---
    fixed_chunks = fixed_size_chunk(docs)
    bm25_fixed = BM25Retriever(fixed_chunks)
    results["BM25 + fixed-size chunks"] = evaluate_retriever(
        lambda q, top_k: [c for c, _ in bm25_fixed.search(q, top_k)],
        eval_set, k=k,
    )

    # --- Config 2: BM25 over heading-aware chunks ---
    heading_chunks = heading_aware_chunk(docs)
    bm25_heading = BM25Retriever(heading_chunks)
    results["BM25 + heading-aware chunks"] = evaluate_retriever(
        lambda q, top_k: [c for c, _ in bm25_heading.search(q, top_k)],
        eval_set, k=k,
    )

    # --- Config 3+: dense and hybrid, only if embeddings have been
    # generated locally (they won't exist in the build sandbox - see
    # embeddings.py's docstring for why) ---
    embeddings_path = Path(__file__).parent.parent / "data/processed/chunk_embeddings.npy"
    if embeddings_path.exists():
        import numpy as np
        from vector_store import VectorStore
        from embeddings import Embedder
        from hybrid_retriever import HybridRetriever

        chunk_embeddings = np.load(embeddings_path)
        embedder = Embedder()
        store = VectorStore.from_normalized(heading_chunks, chunk_embeddings)

        def dense_search(query, top_k):
            query_emb = embedder.embed_query(query)
            return [c for c, _ in store.search(query_emb, top_k)]

        results["Dense only + heading-aware chunks"] = evaluate_retriever(dense_search, eval_set, k=k)

        hybrid = HybridRetriever(bm25_heading, store, embedder)
        results["Hybrid (BM25 + dense) + heading-aware chunks"] = evaluate_retriever(
            lambda q, top_k: [c for c, _ in hybrid.search(q, top_k)],
            eval_set, k=k,
        )

        # --- Config 5: hybrid retrieval + cross-encoder reranking ---
        # Only attempted if sentence-transformers' CrossEncoder loads successfully
        # (same network dependency as the embedding model above).
        try:
            from reranker import CrossEncoderReranker, RerankingRetriever

            reranker = CrossEncoderReranker()
            reranking_retriever = RerankingRetriever(hybrid, reranker, candidate_k=20)
            results["Hybrid + cross-encoder reranker"] = evaluate_retriever(
                lambda q, top_k: [c for c, _ in reranking_retriever.search(q, top_k)],
                eval_set, k=k,
            )
        except Exception as e:
            print(f"NOTE: reranker config skipped ({e})\n")
    else:
        print("NOTE: data/processed/chunk_embeddings.npy not found - skipping dense "
              "and hybrid configs. Run `python src/embeddings.py` locally first "
              "(needs internet access to download the embedding model).\n")

    return results


def print_results_table(results: dict, k: int):
    print(f"{'Configuration':<45} {'Recall@' + str(k):<12} {'MRR':<8}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<45} {r['recall_at_k']:<12.3f} {r['mrr']:<8.3f}")


if __name__ == "__main__":
    K = 5
    results = run_all_configs(k=K)
    print_results_table(results, k=K)

    # Save full results (including per-query breakdown) for later inspection
    out_path = Path(__file__).parent / "eval_results.json"
    serializable = {
        name: {k: v for k, v in r.items()}
        for name, r in results.items()
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nFull per-query results saved to {out_path}")
