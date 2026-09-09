"""
Standard retrieval metrics. Kept dependency-free and simple enough to
hand-verify - an evaluation harness whose own math you can't verify is
worse than useless.
"""


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    """
    Of all relevant docs for this query, what fraction appeared in the
    top-k retrieved? 1.0 = every relevant doc was found in the top k.
    """
    if not relevant_doc_ids:
        return 0.0
    top_k = set(retrieved_doc_ids[:k])
    found = top_k & relevant_doc_ids
    return len(found) / len(relevant_doc_ids)


def reciprocal_rank(retrieved_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    """
    1 / (rank of the first relevant doc), or 0 if none found.
    Rewards ranking a correct answer at position 1 much more than at
    position 10 - unlike recall@k, order within the top-k matters.
    """
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retriever(retriever_search_fn, eval_set: list[dict], k: int = 5) -> dict:
    """
    retriever_search_fn: callable(query: str, top_k: int) -> list[Chunk]
        (already sorted best-to-worst; only .doc_id is used for scoring
        since we evaluate at the *document* level - a chunk from the
        right doc counts as a hit, even if it's not the exact same chunk
        the eval set was labeled against)
    eval_set: list of {"query": str, "relevant_doc_ids": list[str]}

    Returns per-query results plus averaged recall@k and MRR.
    """
    per_query = []

    for item in eval_set:
        query = item["query"]
        relevant = set(item["relevant_doc_ids"])

        results = retriever_search_fn(query, k)
        retrieved_doc_ids = [chunk.doc_id for chunk in results]

        per_query.append({
            "query": query,
            "recall_at_k": recall_at_k(retrieved_doc_ids, relevant, k),
            "reciprocal_rank": reciprocal_rank(retrieved_doc_ids, relevant),
            "retrieved": retrieved_doc_ids,
            "relevant": list(relevant),
        })

    avg_recall = sum(r["recall_at_k"] for r in per_query) / len(per_query)
    avg_mrr = sum(r["reciprocal_rank"] for r in per_query) / len(per_query)

    return {
        "recall_at_k": avg_recall,
        "mrr": avg_mrr,
        "k": k,
        "n_queries": len(eval_set),
        "per_query": per_query,
    }


if __name__ == "__main__":
    # Hand-verified test cases - work these out on paper and confirm
    # the function agrees, so the eval harness itself is trustworthy.

    # recall@k tests
    assert recall_at_k(["a", "b", "c"], {"a"}, k=3) == 1.0, "single relevant doc found -> recall 1.0"
    assert recall_at_k(["a", "b", "c"], {"z"}, k=3) == 0.0, "relevant doc not retrieved -> recall 0.0"
    assert recall_at_k(["a", "b"], {"a", "z"}, k=2) == 0.5, "found 1 of 2 relevant -> recall 0.5"
    assert recall_at_k(["a", "b", "c", "d"], {"c"}, k=2) == 0.0, "relevant doc outside top-k -> recall 0.0"
    print("recall_at_k: all assertions passed")

    # reciprocal_rank tests
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0, "relevant doc at rank 1 -> RR 1.0"
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3, "relevant doc at rank 3 -> RR 1/3"
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0, "relevant doc missing -> RR 0.0"
    print("reciprocal_rank: all assertions passed")

    # evaluate_retriever integration test with a fake retriever
    class FakeChunk:
        def __init__(self, doc_id):
            self.doc_id = doc_id

    def fake_search(query, top_k):
        # deliberately returns the "wrong" doc first, correct doc second
        return [FakeChunk("wrong.md"), FakeChunk("correct.md")][:top_k]

    eval_set = [{"query": "test query", "relevant_doc_ids": ["correct.md"]}]
    result = evaluate_retriever(fake_search, eval_set, k=2)
    assert result["recall_at_k"] == 1.0, "correct.md is within top-2 -> recall 1.0"
    assert result["mrr"] == 0.5, "correct.md at rank 2 -> MRR 0.5"
    print("evaluate_retriever: integration test passed")

    print("\nAll metric sanity checks passed - the eval harness's math is verified correct.")
