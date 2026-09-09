"""
Loads markdown documents from disk and attaches metadata (source path,
section) needed later for evaluation and citation.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    doc_id: str          # stable id, e.g. "tutorial/body.md"
    source_path: str      # full path on disk
    section: str           # top-level folder, e.g. "tutorial", "advanced"
    text: str               # raw markdown content


def load_corpus(corpus_dir: str) -> list[Document]:
    """
    Walk `corpus_dir` recursively and load every .md file into a Document.
    `doc_id` is the path relative to corpus_dir (stable across runs, used
    as the ground-truth identifier in the eval set).
    """
    root = Path(corpus_dir)
    docs = []

    for md_path in sorted(root.rglob("*.md")):
        if md_path.name.startswith("_"):
            continue  # skip internal/test fixture files, not real doc content
        rel_path = md_path.relative_to(root)
        doc_id = str(rel_path)
        # section = top-level folder, or "root" for files directly in corpus_dir
        section = rel_path.parts[0] if len(rel_path.parts) > 1 else "root"
        text = md_path.read_text(encoding="utf-8", errors="ignore")

        docs.append(Document(
            doc_id=doc_id,
            source_path=str(md_path),
            section=section,
            text=text,
        ))

    return docs


if __name__ == "__main__":
    from pathlib import Path

    corpus_dir = Path(__file__).parent.parent / "data" / "raw" / "fastapi_docs"
    docs = load_corpus(str(corpus_dir))
    print(f"Loaded {len(docs)} documents")
    print(f"Sections: {sorted(set(d.section for d in docs))}")
    print(f"\nExample doc_id: {docs[0].doc_id}")
    print(f"Example text (first 200 chars):\n{docs[0].text[:200]}")
