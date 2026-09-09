"""
Wraps a local sentence-transformers model for embedding chunks and
queries. Runs fully offline once the model weights are cached locally
(first run downloads ~90MB from the Hugging Face hub, then it's cached
in ~/.cache/huggingface/ and works with no internet after that).

NOTE ON THIS SANDBOX: this file was written but NOT executed in the
build sandbox - the sandbox's network allowlist doesn't include
huggingface.co, so the model download fails there. Run this on your
own machine, where it should work with a normal internet connection.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from chunking import Chunk

# bge-small is a strong, small (33M param) embedding model - good
# accuracy-per-dollar for a portfolio project running on a laptop CPU.
# Swap to a bigger model (e.g. bge-base or bge-large) if you have GPU
# access and want to push retrieval quality further as an ablation.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: list[Chunk], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """Returns (n_chunks, embedding_dim) float32 array, L2-normalized."""
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # so cosine similarity == dot product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        bge models are trained with an instruction prefix for queries
        (asymmetric retrieval - queries and documents are embedded
        slightly differently). Skipping this measurably hurts recall,
        which is worth calling out if you ablate it.
        """
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        embedding = self.model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
        return embedding.astype(np.float32)


if __name__ == "__main__":
    from pathlib import Path

    from ingest import load_corpus
    from chunking import heading_aware_chunk

    # Resolve paths relative to THIS FILE's location, not the current
    # working directory - so this works whether you run it as
    # `python embeddings.py` from src/, or `python src/embeddings.py`
    # from the project root, or from anywhere else.
    project_root = Path(__file__).parent.parent
    corpus_dir = project_root / "data" / "raw" / "fastapi_docs"
    output_path = project_root / "data" / "processed" / "chunk_embeddings.npy"

    docs = load_corpus(str(corpus_dir))
    if not docs:
        raise FileNotFoundError(
            f"No documents found at {corpus_dir}. Check that the data/raw/fastapi_docs "
            f"folder exists and contains .md files."
        )
    chunks = heading_aware_chunk(docs)

    embedder = Embedder()
    print(f"Embedding {len(chunks)} chunks with {embedder.model_name}...")
    chunk_embeddings = embedder.embed_chunks(chunks)
    print(f"Shape: {chunk_embeddings.shape}")

    # Save to disk so later steps don't need to re-embed every time
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, chunk_embeddings)
    print(f"Saved to {output_path}")

    # Quick sanity check
    query_emb = embedder.embed_query("how do I add a background task")
    sims = chunk_embeddings @ query_emb
    top_idx = np.argsort(-sims)[:3]
    print("\nTop matches for 'how do I add a background task':")
    for i in top_idx:
        print(f"  [{sims[i]:.3f}] {chunks[i].doc_id} | {chunks[i].heading_path[:50]}")
