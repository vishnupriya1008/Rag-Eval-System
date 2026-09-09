# FastAPI Docs RAG — with a Real Retrieval Evaluation Harness

![Eval Suite](https://github.com/<your-username>/<your-repo>/actions/workflows/eval.yml/badge.svg)

A retrieval-augmented Q&A system over FastAPI's official documentation, built to answer one question rigorously: **does each design decision actually improve retrieval quality, or does it just feel like it should?**

Most RAG side projects show a demo and stop. This one measures — and CI re-checks those measurements on every push, so the numbers in this README can't silently go stale.

## Why I built this

I wanted to go past "call an LLM with some retrieved context" and actually test the retrieval decisions that get hand-waved in most RAG tutorials — chunking strategy, keyword vs. semantic search, hybrid fusion — against a real, verifiable corpus with hand-labeled ground truth.

## Architecture

```
FastAPI GitHub repo (docs/en/docs, 152 markdown files)
        │
        ▼
   ingest.py  ──────────────────────────  loads + tags each doc
        │
        ▼
  chunking.py ── two strategies, compared head-to-head:
        │          • fixed_size_chunk    (naive sliding window, baseline)
        │          • heading_aware_chunk (splits on markdown headings first)
        │
        ├──────────────────────┬─────────────────────┐
        ▼                      ▼                      │
 bm25_retriever.py      embeddings.py                 │
 (keyword search,       + vector_store.py             │
  rank_bm25)             (dense search,                │
        │                 sentence-transformers)       │
        └──────────┬───────────┘                       │
                    ▼                                   │
          hybrid_retriever.py                           │
          (Reciprocal Rank Fusion)                       │
                    │                                     │
                    ▼                                     │
            reranker.py                                   │
     (cross-encoder, 2nd-stage refinement                  │
      over top-20 hybrid candidates)                       │
                    │                                       │
                    ▼                                       │
             generator.py ◄───────────────────────────────┘
        (local LLM, llama-cpp-python)
                    │
                    ▼
              pipeline.py  (ties it all together, CLI)

  eval/metrics.py + eval/run_eval.py + eval/eval_dataset.json
  (30 hand-verified queries, run against every retriever config above)
  eval/check_thresholds.py  (CI regression gate - see Continuous Integration below)
```

## Key Design Decisions

| Decision | Options considered | What I chose & why |
|---|---|---|
| Corpus | Wikipedia / arXiv / docs | FastAPI's real docs — code-heavy text is a harder, more realistic retrieval problem than plain prose, and results are independently verifiable (anyone can check the actual page) |
| Chunking | Fixed-size vs. heading-aware | Built and evaluated both — see Results below, the answer turned out to be "it depends on the metric," not a clean win |
| Vector search | FAISS vs. brute-force numpy | Brute-force cosine similarity. At ~1,200 chunks an exact numpy search is milliseconds; FAISS's approximate-search speedup only matters past ~100K+ vectors. Using it here would be complexity with no benefit |
| Fusion method | Weighted score average vs. Reciprocal Rank Fusion | RRF — BM25 scores and cosine similarities live on incomparable scales, so averaging them raw would let whichever number happens to be bigger dominate for no principled reason. RRF only uses rank position, which is scale-free |
| Reranking | Rerank the whole corpus vs. rerank top-N candidates only | Rerank only the hybrid retriever's top 20 — a cross-encoder scores query+doc together in one pass (more accurate, no precomputation possible), so it's too slow to run over all ~1,200 chunks per query. Narrowing with a cheap first-stage retriever first is the standard pattern |
| Generation | OpenAI/Anthropic API vs. local model | Local (llama-cpp-python + Qwen2.5-1.5B-Instruct GGUF) — zero cost, zero API keys, fully reproducible by anyone who clones the repo |
| CI eval gate | Run eval only locally vs. gate CI on it | Gated — `eval/check_thresholds.py` fails the build if recall@5 or MRR drops below baseline, so a chunking/tokenizer regression gets caught the same way a failing unit test would, not discovered later by chance |

## Evaluation Methodology

- **Eval set:** 30 hand-written queries, each manually checked against the actual doc content before being labeled (not generated and trusted blindly — see `eval/eval_dataset.json`). Mix of exact-terminology queries ("how to handle CORS") and paraphrased/conceptual ones ("understand async def versus regular def"), because those stress retrieval differently.
- **Metrics:** Recall@5 and MRR (`eval/metrics.py`), each with hand-verified unit tests (`python eval/metrics.py`) — a metrics function you can't verify is worse than no metrics function.
- **Process:** Each retriever configuration runs against the identical 30-query set via `eval/run_eval.py`, so comparisons are apples-to-apples.

## Results

Real output from `python eval/run_eval.py` against the actual FastAPI docs corpus:

| Configuration | Recall@5 | MRR |
|---|---|---|
| BM25 + fixed-size chunks | 0.900 | 0.744 |
| BM25 + heading-aware chunks | 0.917 | 0.739 |
| Dense only + heading-aware chunks | *[run locally — see Setup]* | |
| Hybrid (BM25 + dense) + heading-aware chunks | *[run locally — see Setup]* | |
| Hybrid + cross-encoder reranker | *[run locally — see Setup]* | |

The three dense/hybrid/reranker rows above fill in automatically on a machine with Hugging Face access — either your own machine, or the `full-pipeline` CI job (see below), which uploads `eval_results.json` as a build artifact you can download.

**This is not a clean "everything got better" story, on purpose.** Heading-aware chunking improved recall@5 (found the right doc more often) but very slightly hurt MRR (the right doc landed a rank position or two lower on average, likely from more numerous small chunks slightly diluting exact-term BM25 scores). That tradeoff — recall vs. rank quality — is a real finding, not a rounding error, and it's the kind of nuance a resume bullet claiming "improved retrieval by X%" usually hides.

### Concrete failure cases (from `eval/eval_results.json`)

- **"serve static files like images or CSS"** → both BM25 configs return `reference/staticfiles.md` (API reference) instead of the correct `tutorial/static-files.md` (conceptual tutorial). The two pages share too much vocabulary for keyword matching to distinguish intent — a strong candidate for where dense/semantic search or a reranker should win, and worth actually testing once you run the dense config locally.
- **"how to handle CORS"** (development note, not in final eval set) → BM25 ranked the real `tutorial/cors.md` page 4th, not top-3, because the query paraphrases ("handle") don't literally appear on the page, while `advanced/middleware.md` scores higher on generic term overlap. Searching the literal term "CORS" alone finds it instantly — a clean illustration of BM25's core weakness: it matches words, not intent.

## Continuous Integration

Two GitHub Actions jobs (`.github/workflows/eval.yml`):

- **`test-and-eval`** — runs on every push/PR. Unit tests (`pytest tests/`) plus the BM25-only eval, gated by `eval/check_thresholds.py` (fails the build if recall@5 drops below 0.80 or MRR below 0.60 — well under the measured 0.917/0.739 baseline, so it catches real regressions without flagging normal noise). No model downloads, finishes in under a minute — this is the check that should never be slow enough to ignore.
- **`full-pipeline`** — runs weekly (and on manual trigger) rather than every commit, since it downloads real models and runs the full dense/hybrid/reranker pipeline. GitHub's runners have full internet access, unlike the sandbox this project was originally built in, so this job is what actually exercises the parts that couldn't be tested during initial development. Results upload as a build artifact.

This structure — fast required check on every commit, slower comprehensive check on a schedule — is a standard tradeoff for any test suite with both cheap and expensive tests, and it's worth being able to explain that tradeoff rather than just having a green checkmark.

## Setup

```bash
git clone <this-repo>
cd rag-eval-system
pip install -r requirements.txt

# BM25 works immediately, no downloads needed:
python eval/run_eval.py

# For dense + hybrid + reranked search and LLM generation, run once
# (downloads ~130MB embedding model + ~1GB local LLM + a small
# cross-encoder reranker model, all from Hugging Face):
python src/embeddings.py    # generates data/processed/chunk_embeddings.npy
python eval/run_eval.py     # re-run — now includes dense + hybrid + reranker rows

# Or just check retrieval quality hasn't regressed (what CI runs):
python eval/check_thresholds.py

# Ask a question end to end:
python src/pipeline.py --query "how do I add a background task"

# Run the test suite:
pytest tests/ -v
```

## Limitations & What I'd Do Differently

- Eval set is 30 queries — enough to see real signal, but a larger set (100+) would make the MRR differences more statistically trustworthy rather than close to noise.
- Eval is document-level (a chunk from the right doc counts as a hit). A stricter, chunk-level eval would be more demanding but requires labeling which *exact chunk* answers each query, not just which document.
- Dense, hybrid, and reranker rows in the Results table need to be run on a machine with Hugging Face access — I built and unit-tested every component's logic (sorting, fusion, ranking) in a network-restricted sandbox using synthetic vectors/scores, but the real numbers still need to be generated for the table above to be complete. The weekly CI job (see Continuous Integration) now does this automatically, which is the actual fix for "I built it but never ran it."
- The reranker always reranks the hybrid retriever's top 20 candidates — an ablation worth running is whether reranking BM25-only candidates (skipping dense retrieval entirely) gets most of the benefit for less complexity.
- CI regression thresholds (0.80 recall / 0.60 MRR) were chosen by eyeballing a margin below the measured baseline, not by any statistical process — with more eval queries, a principled confidence interval would be a better gate than a hand-picked number.

## Tech Stack

`Python` · `rank-bm25` · `sentence-transformers` (BAAI/bge-small-en-v1.5 embeddings + cross-encoder/ms-marco-MiniLM-L-6-v2 reranker) · `numpy` · `llama-cpp-python` (Qwen2.5-1.5B-Instruct) · `pytest` · `GitHub Actions`

## What I Learned

The biggest surprise was that heading-aware chunking didn't uniformly win — it traded MRR for recall. Going in, I expected "smarter chunking" to just be strictly better, and having to explain *why* it wasn't taught me more about BM25's scoring behavior (term frequency dilution across more, smaller chunks) than a clean win would have.

The static-files.md vs. reference/staticfiles.md confusion was also a genuinely useful failure to find by actually measuring instead of eyeballing a demo — it's exactly the kind of case that motivates reranking in the first place, and it's now sitting in CI as a live test of whether the reranker actually fixes it, rather than a claim I'm taking on faith.

Setting up the CI regression gate also forced a decision I hadn't thought about going in: a retrieval eval isn't pass/fail the way a unit test is, it's a threshold judgment call. Picking 0.80/0.60 as the gate values made me realize "did quality regress" and "is quality perfect" are different questions, and conflating them would make the gate either too strict (breaks on noise) or too loose (misses real regressions) — worth naming explicitly rather than leaving implicit.
