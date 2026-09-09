"""
Cross-encoder reranking: a second-stage refinement over a first-stage
retriever's (BM25 or hybrid) top candidates.

Why a separate stage instead of just using the cross-encoder for
everything: a cross-encoder scores a query and a document *together*
in one forward pass (full attention across both), which is much more
accurate than comparing two independently-computed embeddings - but
it means you can't precompute anything, so scoring the entire corpus
per query would be far too slow. The standard pattern is exactly what's
implemented here: use a cheap first-stage retriever (BM25/dense) to
narrow ~1,200 chunks down to ~20 candidates, then spend the expensive
cross-encoder pass only on those 20.

NOT EXECUTED END-TO-END IN THE BUILD SANDBOX - loading the cross-encoder
model requires downloading from Hugging Face, which this sandbox's
network allowlist blocks (same constraint as embeddings.py and
generator.py). The sorting/truncation logic below (`_rank_by_scores`)
is pure and IS unit-tested with fake scores, so the only untested part
is the model call itself, not the logic around it.
"""
from chunking import Chunk

# Small, fast, well-established for query-passage relevance - trained
# on MS MARCO passage ranking, a good general-purpose default that
# doesn't need domain-specific fine-tuning to be useful here.
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _rank_by_scores(candidates: list[Chunk], scores: list[float], top_k: int) -> list[tuple[Chunk, float]]:
    """
    Pure sort/truncate logic, deliberately separated from the model call
    so it can be unit-tested without loading any model - see
    tests/test_core.py::test_reranker_sorts_by_score_descending.
    """
    assert len(candidates) == len(scores), "candidates and scores must align 1:1"
    paired = list(zip(candidates, scores))
    paired.sort(key=lambda x: x[1], reverse=True)
    return paired[:top_k]


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import CrossEncoder
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[Chunk], top_k: int = 5) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        pairs = [(query, chunk.text) for chunk in candidates]
        scores = self.model.predict(pairs)
        return _rank_by_scores(candidates, list(scores), top_k)


class RerankingRetriever:
    """
    Wraps any first-stage retriever (BM25Retriever, HybridRetriever) with
    a reranking pass. Same .search(query, top_k) interface as the other
    retrievers, so it drops into eval/run_eval.py and pipeline.py without
    changes to either.
    """
    def __init__(self, base_retriever, reranker: CrossEncoderReranker, candidate_k: int = 20):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.candidate_k = candidate_k  # how many first-stage results to feed the reranker

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        first_stage = self.base_retriever.search(query, top_k=self.candidate_k)
        candidates = [c for c, _ in first_stage]
        return self.reranker.rerank(query, candidates, top_k=top_k)


if __name__ == "__main__":
    # Proves _rank_by_scores is correct without needing the actual model
    def make_chunk(cid):
        return Chunk(chunk_id=cid, doc_id=cid, section="test", heading_path="", text=f"content for {cid}")

    candidates = [make_chunk(c) for c in ["a", "b", "c", "d"]]
    # Deliberately scramble scores relative to input order
    fake_scores = [0.2, 0.9, 0.5, 0.1]  # b should win, then c, then a, then d

    ranked = _rank_by_scores(candidates, fake_scores, top_k=3)
    ranked_ids = [c.chunk_id for c, _ in ranked]

    print(f"Input order: {[c.chunk_id for c in candidates]}")
    print(f"Fake scores: {fake_scores}")
    print(f"Reranked (top 3): {ranked_ids}")

    assert ranked_ids == ["b", "c", "a"], f"FAILED: expected ['b', 'c', 'a'], got {ranked_ids}"
    print("PASSED: reranker correctly sorts by score descending and truncates to top_k")
