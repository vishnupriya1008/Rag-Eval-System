"""
BM25 keyword retrieval over chunks. This is the "sparse" half of the
hybrid search - good at exact term matches (function names, error
strings) which dense embeddings sometimes miss.
"""
import re

from rank_bm25 import BM25Okapi

from chunking import Chunk


def _tokenize(text: str) -> list[str]:
    """
    Simple tokenizer: lowercase, split on non-alphanumeric. Keeps
    underscores so things like `background_tasks` stay one token,
    which matters a lot for a docs corpus full of code identifiers.
    """
    text = text.lower()
    return re.findall(r"[a-z0-9_]+", text)


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Returns [(chunk, score), ...] sorted by score descending."""
        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_idx]


if __name__ == "__main__":
    from pathlib import Path

    from ingest import load_corpus
    from chunking import heading_aware_chunk

    corpus_dir = Path(__file__).parent.parent / "data" / "raw" / "fastapi_docs"
    docs = load_corpus(str(corpus_dir))
    chunks = heading_aware_chunk(docs)
    retriever = BM25Retriever(chunks)

    test_queries = [
        "how do I add a background task",
        "dependency injection with yield",
        "how to handle CORS",
    ]

    for q in test_queries:
        print(f"\nQuery: {q!r}")
        results = retriever.search(q, top_k=3)
        for chunk, score in results:
            print(f"  [{score:.2f}] {chunk.doc_id} | {chunk.heading_path[:60]}")
