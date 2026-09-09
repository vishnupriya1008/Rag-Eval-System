"""
Combines BM25 (sparse/keyword) and dense (semantic) retrieval using
Reciprocal Rank Fusion (RRF).

Why RRF instead of averaging raw scores: BM25 scores and cosine
similarities live on completely different, incomparable scales (BM25
is roughly 0-30+ unbounded, cosine similarity is -1 to 1). Averaging
them directly would let whichever method happens to produce larger
numbers dominate, for no principled reason. RRF sidesteps this by only
using each method's *rank* (1st, 2nd, 3rd...), which is scale-free.

RRF formula: score(doc) = sum over each ranker of  1 / (k + rank)
k=60 is the standard constant from the original RRF paper (Cormack
et al.) - it's not very sensitive to this value, but it's worth
naming that it's a knob rather than something derived from the data.
"""
from collections import defaultdict

from chunking import Chunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[Chunk]],
    k: int = 60,
    top_k: int = 5,
) -> list[tuple[Chunk, float]]:
    """
    ranked_lists: one list per retrieval method, each already sorted
    best-to-worst (score values are ignored - only rank position is used).
    Returns fused [(chunk, rrf_score), ...] sorted descending.
    """
    scores: dict[str, float] = defaultdict(float)
    chunk_by_id: dict[str, Chunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.chunk_id] += 1.0 / (k + rank)
            chunk_by_id[chunk.chunk_id] = chunk

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(chunk_by_id[chunk_id], score) for chunk_id, score in fused]


class HybridRetriever:
    """
    Ties a BM25Retriever and a dense VectorStore (+ Embedder for the
    query) together behind one .search() call.
    """
    def __init__(self, bm25_retriever, vector_store=None, embedder=None, candidate_k: int = 20):
        self.bm25 = bm25_retriever
        self.vector_store = vector_store
        self.embedder = embedder
        self.candidate_k = candidate_k  # how many candidates each method contributes before fusion

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        bm25_results = [c for c, _ in self.bm25.search(query, top_k=self.candidate_k)]

        ranked_lists = [bm25_results]

        if self.vector_store is not None and self.embedder is not None:
            query_emb = self.embedder.embed_query(query)
            dense_results = [c for c, _ in self.vector_store.search(query_emb, top_k=self.candidate_k)]
            ranked_lists.append(dense_results)

        return reciprocal_rank_fusion(ranked_lists, top_k=top_k)


if __name__ == "__main__":
    # Test RRF fusion logic with two synthetic ranked lists sharing some
    # overlap - proves the fusion math without needing real embeddings.
    from chunking import Chunk

    def make_chunk(cid):
        return Chunk(chunk_id=cid, doc_id=cid, section="test", heading_path="", text="")

    list_a = [make_chunk(c) for c in ["x", "y", "z", "w"]]       # ranks: x=1, y=2, z=3, w=4
    list_b = [make_chunk(c) for c in ["y", "x", "v", "w"]]        # ranks: y=1, x=2, v=3, w=4

    fused = reciprocal_rank_fusion([list_a, list_b], k=60, top_k=4)

    print("Fused ranking (x and y both appear near-top in both lists, should win):")
    for chunk, score in fused:
        print(f"  [{score:.5f}] {chunk.chunk_id}")

    # x: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03251
    # y: 1/(60+2) + 1/(60+1) = same as x by symmetry = 0.03251
    # w: 1/(60+4) + 1/(60+4) = 0.03125
    # v: 1/(60+3) = 0.01587 (only in list_b)
    # z: 1/(60+3) = 0.01587 (only in list_a)
    assert fused[0][0].chunk_id in ("x", "y"), "FAILED: expected x or y to rank highest"
    print("\nPASSED: items ranked highly by both retrievers correctly win fusion")
