"""
Two chunking strategies, deliberately kept comparable so the eval harness
can isolate "did chunking strategy change retrieval quality" from other
variables:

1. fixed_size_chunk   - naive sliding window over raw text (baseline)
2. heading_aware_chunk - splits on markdown headings first, then sub-chunks
                          any section that's still too long. Preserves the
                          heading path as context, which should help
                          retrieval on docs (this is the hypothesis the
                          eval harness will actually test, not assume).
"""
import re
from dataclasses import dataclass, field

from ingest import Document


@dataclass
class Chunk:
    chunk_id: str            # f"{doc_id}::{index}"
    doc_id: str               # which document this came from (for eval)
    section: str                # top-level doc section, inherited
    heading_path: str            # "" for fixed-size; "## Title > ### Sub" for heading-aware
    text: str                      # the chunk content given to the retriever
    word_count: int = field(init=False)

    def __post_init__(self):
        self.word_count = len(self.text.split())


def fixed_size_chunk(docs: list[Document], chunk_size: int = 200, overlap: int = 40) -> list[Chunk]:
    """
    Naive sliding-window chunking over raw markdown text, ignoring
    structure. This is the baseline every other strategy should beat.

    chunk_size / overlap are in words, not tokens (tokens vary by model;
    words is a stable enough proxy for comparing strategies fairly).
    """
    chunks = []
    for doc in docs:
        words = doc.text.split()
        if not words:
            continue

        start = 0
        idx = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}::{idx}",
                doc_id=doc.doc_id,
                section=doc.section,
                heading_path="",
                text=chunk_text,
            ))
            idx += 1
            if end == len(words):
                break
            start = end - overlap  # slide with overlap

    return chunks


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def heading_aware_chunk(docs: list[Document], max_chunk_size: int = 200, overlap: int = 40) -> list[Chunk]:
    """
    Splits each document at markdown headings first, so a chunk never
    straddles two unrelated sections. Each chunk is prefixed with its
    heading path (e.g. "Body > Request Body") for extra retrieval signal.

    Any section still longer than max_chunk_size gets sub-chunked with
    fixed_size_chunk's sliding window, so long sections don't become one
    giant unsearchable blob.
    """
    chunks = []

    for doc in docs:
        # Find all heading positions to split the doc into sections
        headings = list(_HEADING_RE.finditer(doc.text))

        if not headings:
            # No headings at all -> treat whole doc as one section
            sections = [("", doc.text)]
        else:
            sections = []
            heading_stack = []  # tracks nested heading path, e.g. ["Body", "Request Body"]

            for i, match in enumerate(headings):
                level = len(match.group(1))
                title = match.group(2).strip()

                # Trim stack to current heading's level, then append
                heading_stack = heading_stack[:level - 1]
                heading_stack.append(title)
                heading_path = " > ".join(heading_stack)

                start = match.end()
                end = headings[i + 1].start() if i + 1 < len(headings) else len(doc.text)
                section_text = doc.text[start:end].strip()

                if section_text:
                    sections.append((heading_path, section_text))

        idx = 0
        for heading_path, section_text in sections:
            words = section_text.split()
            if not words:
                continue

            if len(words) <= max_chunk_size:
                prefixed_text = f"{heading_path}\n{section_text}" if heading_path else section_text
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}::{idx}",
                    doc_id=doc.doc_id,
                    section=doc.section,
                    heading_path=heading_path,
                    text=prefixed_text,
                ))
                idx += 1
            else:
                # Section too long -> sub-chunk it, still tagged with heading_path
                start = 0
                while start < len(words):
                    end = min(start + max_chunk_size, len(words))
                    sub_text = " ".join(words[start:end])
                    prefixed_text = f"{heading_path}\n{sub_text}" if heading_path else sub_text
                    chunks.append(Chunk(
                        chunk_id=f"{doc.doc_id}::{idx}",
                        doc_id=doc.doc_id,
                        section=doc.section,
                        heading_path=heading_path,
                        text=prefixed_text,
                    ))
                    idx += 1
                    if end == len(words):
                        break
                    start = end - overlap

    return chunks


if __name__ == "__main__":
    from pathlib import Path

    from ingest import load_corpus

    corpus_dir = Path(__file__).parent.parent / "data" / "raw" / "fastapi_docs"
    docs = load_corpus(str(corpus_dir))

    fixed = fixed_size_chunk(docs)
    heading = heading_aware_chunk(docs)

    print(f"Documents: {len(docs)}")
    print(f"Fixed-size chunks: {len(fixed)} (avg {sum(c.word_count for c in fixed) / len(fixed):.0f} words)")
    print(f"Heading-aware chunks: {len(heading)} (avg {sum(c.word_count for c in heading) / len(heading):.0f} words)")

    print("\n--- Example fixed-size chunk ---")
    print(fixed[10].text[:300])

    print("\n--- Example heading-aware chunk ---")
    example = next(c for c in heading if c.heading_path)
    print(f"heading_path: {example.heading_path}")
    print(example.text[:300])
