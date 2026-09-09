"""
CI regression gate: re-runs the fast (BM25-only, no downloads) eval
config and fails the build if quality drops meaningfully below the
measured baseline. This is what makes the eval suite mean something in
CI, rather than just running for its own sake - a chunking or tokenizer
change that quietly tanks retrieval quality should break the build, the
same way a failing unit test would.

Thresholds are set a bit below the actual measured baseline (see
README's Results table) so normal eval-set edits don't cause false
alarms, while a real regression (e.g. a chunking bug) still trips it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from run_eval import run_all_configs  # noqa: E402

# Baseline measured on the BM25 + heading-aware config (the pipeline's
# default when dense embeddings aren't available - see pipeline.py).
# Real measured values: recall@5=0.917, MRR=0.739.
MIN_RECALL_AT_5 = 0.80
MIN_MRR = 0.60


def main():
    print("Running regression check against BM25 + heading-aware chunks...\n")
    results = run_all_configs(k=5)

    config_name = "BM25 + heading-aware chunks"
    if config_name not in results:
        print(f"FAIL: expected config {config_name!r} not found in results")
        sys.exit(1)

    r = results[config_name]
    recall, mrr = r["recall_at_k"], r["mrr"]

    print(f"Recall@5: {recall:.3f}  (threshold: >= {MIN_RECALL_AT_5})")
    print(f"MRR:      {mrr:.3f}  (threshold: >= {MIN_MRR})")

    failed = False
    if recall < MIN_RECALL_AT_5:
        print(f"\nFAIL: recall@5 {recall:.3f} is below threshold {MIN_RECALL_AT_5}")
        failed = True
    if mrr < MIN_MRR:
        print(f"\nFAIL: MRR {mrr:.3f} is below threshold {MIN_MRR}")
        failed = True

    if failed:
        sys.exit(1)

    print("\nPASS: retrieval quality is within expected range.")


if __name__ == "__main__":
    main()
