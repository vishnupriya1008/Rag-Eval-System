"""
Generates an answer from retrieved chunks using a local LLM - no API
keys, runs offline once the model is downloaded.

NOT EXECUTED IN THE BUILD SANDBOX (same reason as embeddings.py: no
internet access to download model weights there). Written correctly
for you to run locally.

Uses llama-cpp-python with a small GGUF-quantized model, which runs
on CPU at reasonable speed - no GPU required. Any instruction-tuned
small model works; Qwen2.5-1.5B-Instruct is a solid default (fast,
coherent, small download).
"""
from pathlib import Path

from chunking import Chunk

SYSTEM_PROMPT = """You are a documentation assistant. Answer the user's \
question using ONLY the provided context. If the context doesn't contain \
the answer, say so explicitly instead of guessing. Cite which document \
you drew each part of your answer from."""


def build_prompt(query: str, retrieved_chunks: list[Chunk]) -> str:
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        label = f"[{i}] {chunk.doc_id}" + (f" ({chunk.heading_path})" if chunk.heading_path else "")
        context_blocks.append(f"{label}\n{chunk.text}")

    context = "\n\n---\n\n".join(context_blocks)

    return f"""Context:
{context}

Question: {query}

Answer using only the context above. Cite sources like [1], [2]."""


class LocalGenerator:
    def __init__(self, model_path: str = None, model_repo: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                 model_file: str = "qwen2.5-1.5b-instruct-q4_k_m.gguf"):
        """
        model_path: path to an already-downloaded .gguf file, OR
        model_repo/model_file: auto-downloads from Hugging Face on first run
        (needs internet - won't work in this sandbox, will work on your machine).
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Run: pip install llama-cpp-python"
            )

        if model_path and Path(model_path).exists():
            self.llm = Llama(model_path=model_path, n_ctx=4096, verbose=False)
        else:
            self.llm = Llama.from_pretrained(
                repo_id=model_repo,
                filename=model_file,
                n_ctx=4096,
                verbose=False,
            )

    def generate(self, query: str, retrieved_chunks: list[Chunk], max_tokens: int = 400) -> str:
        prompt = build_prompt(query, retrieved_chunks)

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.1,  # low temperature - this is a factual Q&A task, not creative writing
        )
        return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    # This block requires llama-cpp-python installed AND internet access
    # to download the model - run locally, not in the build sandbox.
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).parent))

    from ingest import load_corpus
    from chunking import heading_aware_chunk
    from bm25_retriever import BM25Retriever

    corpus_dir = P(__file__).parent.parent / "data" / "raw" / "fastapi_docs"
    docs = load_corpus(str(corpus_dir))
    chunks = heading_aware_chunk(docs)
    retriever = BM25Retriever(chunks)

    query = "how do I add a background task"
    retrieved = [c for c, _ in retriever.search(query, top_k=3)]

    print("Retrieved context:")
    for c in retrieved:
        print(f"  - {c.doc_id} | {c.heading_path[:50]}")

    print("\nGenerating answer (this downloads a ~1GB model on first run)...")
    generator = LocalGenerator()
    answer = generator.generate(query, retrieved)
    print(f"\nAnswer:\n{answer}")
