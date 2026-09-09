"""
Formal test suite. The individual modules' __main__ blocks already run
these as sanity checks during development - this file consolidates them
into pytest format for CI/discoverability (`pytest tests/` from repo root).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

from chunking import Chunk, fixed_size_chunk, heading_aware_chunk
from ingest import Document
from metrics import recall_at_k, reciprocal_rank, evaluate_retriever
from hybrid_retriever import reciprocal_rank_fusion
from reranker import _rank_by_scores


# --- chunking.py ---

def test_fixed_size_chunk_respects_size_and_overlap():
    doc = Document(doc_id="d1", source_path="", section="test", text=" ".join(f"word{i}" for i in range(500)))
    chunks = fixed_size_chunk([doc], chunk_size=200, overlap=40)

    assert all(c.word_count <= 200 for c in chunks)
    # with 500 words, chunk_size=200, overlap=40: stride=160, so chunks start at 0, 160, 320
    assert len(chunks) == 3


def test_heading_aware_chunk_splits_on_headings():
    text = "# Title\nintro text\n## Section A\ncontent A\n## Section B\ncontent B"
    doc = Document(doc_id="d1", source_path="", section="test", text=text)
    chunks = heading_aware_chunk([doc], max_chunk_size=200)

    heading_paths = [c.heading_path for c in chunks]
    assert "Title" in heading_paths
    assert "Title > Section A" in heading_paths
    assert "Title > Section B" in heading_paths


def test_heading_aware_chunk_subsplits_long_sections():
    long_content = " ".join(f"word{i}" for i in range(500))
    text = f"# Title\n{long_content}"
    doc = Document(doc_id="d1", source_path="", section="test", text=text)
    chunks = heading_aware_chunk([doc], max_chunk_size=200, overlap=40)

    # a 500-word section under one heading should still get sub-chunked
    assert len(chunks) > 1
    assert all(c.heading_path == "Title" for c in chunks)


# --- eval/metrics.py ---

def test_recall_at_k_full_match():
    assert recall_at_k(["a", "b", "c"], {"a"}, k=3) == 1.0


def test_recall_at_k_partial_match():
    assert recall_at_k(["a", "b"], {"a", "z"}, k=2) == 0.5


def test_recall_at_k_outside_k_scores_zero():
    assert recall_at_k(["a", "b", "c", "d"], {"c"}, k=2) == 0.0


def test_reciprocal_rank_position_matters():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_evaluate_retriever_integration():
    class FakeChunk:
        def __init__(self, doc_id):
            self.doc_id = doc_id

    def fake_search(query, top_k):
        return [FakeChunk("wrong.md"), FakeChunk("correct.md")][:top_k]

    eval_set = [{"query": "test", "relevant_doc_ids": ["correct.md"]}]
    result = evaluate_retriever(fake_search, eval_set, k=2)

    assert result["recall_at_k"] == 1.0
    assert result["mrr"] == 0.5


# --- hybrid_retriever.py ---

def test_rrf_favors_items_ranked_highly_in_both_lists():
    def make_chunk(cid):
        return Chunk(chunk_id=cid, doc_id=cid, section="test", heading_path="", text="")

    list_a = [make_chunk(c) for c in ["x", "y", "z"]]
    list_b = [make_chunk(c) for c in ["y", "x", "v"]]

    fused = reciprocal_rank_fusion([list_a, list_b], k=60, top_k=3)
    top_ids = {fused[0][0].chunk_id, fused[1][0].chunk_id}

    assert top_ids == {"x", "y"}, "x and y appear near-top in both lists, should rank above z/v"


def test_rrf_item_only_in_one_list_still_included():
    def make_chunk(cid):
        return Chunk(chunk_id=cid, doc_id=cid, section="test", heading_path="", text="")

    list_a = [make_chunk("only_in_a")]
    fused = reciprocal_rank_fusion([list_a, []], k=60, top_k=5)

    assert len(fused) == 1
    assert fused[0][0].chunk_id == "only_in_a"


# --- reranker.py ---

def test_reranker_sorts_by_score_descending():
    def make_chunk(cid):
        return Chunk(chunk_id=cid, doc_id=cid, section="test", heading_path="", text="")

    candidates = [make_chunk(c) for c in ["a", "b", "c", "d"]]
    fake_scores = [0.2, 0.9, 0.5, 0.1]  # b > c > a > d

    ranked = _rank_by_scores(candidates, fake_scores, top_k=3)
    ranked_ids = [c.chunk_id for c, _ in ranked]

    assert ranked_ids == ["b", "c", "a"]


def test_reranker_respects_top_k_truncation():
    def make_chunk(cid):
        return Chunk(chunk_id=cid, doc_id=cid, section="test", heading_path="", text="")

    candidates = [make_chunk(c) for c in ["a", "b", "c"]]
    scores = [0.5, 0.5, 0.5]

    ranked = _rank_by_scores(candidates, scores, top_k=1)
    assert len(ranked) == 1


def test_reranker_mismatched_lengths_raises():
    def make_chunk(cid):
        return Chunk(chunk_id=cid, doc_id=cid, section="test", heading_path="", text="")

    candidates = [make_chunk("a"), make_chunk("b")]
    scores = [0.5]  # deliberately wrong length

    try:
        _rank_by_scores(candidates, scores, top_k=2)
        assert False, "expected AssertionError for mismatched lengths"
    except AssertionError as e:
        assert "align" in str(e)
