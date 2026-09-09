"""
Dense vector store for semantic search.

Design decision: brute-force cosine similarity with numpy, not FAISS.
At ~1,200 chunks, an exact brute-force search over a (1200, 384) matrix
is a few milliseconds - well within any reasonable latency budget, and
it avoids adding a C++ dependency for zero benefit. FAISS (or a proper
ANN index) starts to matter once you're past ~100K-1M vectors, where
exact search stops being fast enough. Document this tradeoff explicitly
if asked in an interview - "I used the right tool for the scale" is a
better answer than "I used FAISS because it's what people use."
"""
import numpy as np

from chunking import Chunk


class VectorStore:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray):
        """
        embeddings: shape (n_chunks, embedding_dim), expected to already
        be L2-normalized so dot product == cosine similarity.
        """
        assert len(chunks) == embeddings.shape[0], "chunks and embeddings must align 1:1"
        self.chunks = chunks
        self.embeddings = embeddings

    @classmethod
    def from_normalized(cls, chunks: list[Chunk], embeddings: np.ndarray) -> "VectorStore":
        """Build a store, L2-normalizing embeddings if they aren't already."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8  # avoid divide-by-zero on a zero vector
        normalized = embeddings / norms
        return cls(chunks, normalized)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """
        query_embedding: shape (embedding_dim,), should be L2-normalized
        the same way the stored embeddings are.
        Returns [(chunk, cosine_similarity), ...] sorted descending.
        """
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        # Cosine similarity via dot product (both sides normalized)
        scores = self.embeddings @ query_embedding

        top_idx = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]


if __name__ == "__main__":
    # Prove the search/ranking logic is correct using synthetic embeddings
    # (no real model needed - this validates the algorithm, not semantic quality)
    from chunking import Chunk

    np.random.seed(42)
    fake_chunks = [
        Chunk(chunk_id=f"doc{i}::0", doc_id=f"doc{i}", section="test", heading_path="", text=f"content {i}")
        for i in range(100)
    ]
    fake_embeddings = np.random.randn(100, 384).astype(np.float32)

    # Make chunk 7's embedding deliberately close to a known query vector
    query = np.random.randn(384).astype(np.float32)
    fake_embeddings[7] = query + np.random.randn(384) * 0.01  # near-identical + tiny noise

    store = VectorStore.from_normalized(fake_chunks, fake_embeddings)
    results = store.search(query, top_k=3)

    print("Sanity check: chunk 7 should rank #1 (it's near-identical to the query)")
    for chunk, score in results:
        print(f"  [{score:.4f}] {chunk.chunk_id}")
    assert results[0][0].chunk_id == "doc7::0", "FAILED: nearest-neighbor search is not returning the closest vector"
    print("PASSED: vector store correctly ranks nearest neighbor first")
