# exercise-03: MTEB Evaluation Suite

**Estimated effort:** 3 hours

## Objective

Run the Massive Text Embedding Benchmark on a slate of public encoders — and on the two bi-encoders you trained in exercises 01 and 02 — across all seven MTEB task categories. Produce a category-broken-out comparison table, wire task-specific prefixes correctly, and defend a model-selection recommendation for a stated target use case. Chapter 08 is the source material; this exercise is the "actually run it and read the output" version.

## Prerequisites

- Chapter [08](../08-mteb-evaluation-suite.md).
- Exercises 01 and 02 (bi-encoder checkpoints).
- Python 3.10+; `mteb>=1.10`, `sentence-transformers>=3.0`, `torch`, `pandas`, `numpy`, `matplotlib` (optional, for the summary chart).
- A GPU is strongly recommended. The subset defined below runs in ~1–2 hours per model on a single mid-range GPU; the full English benchmark is a multi-day run and is *not* required for this exercise.

## Models under test

Six models minimum. Add more if your GPU budget allows.

- `sentence-transformers/all-MiniLM-L6-v2` — the cheap English default.
- `sentence-transformers/all-mpnet-base-v2` — the quality English default of 2021.
- `intfloat/e5-large-v2` — requires `"query: "` / `"passage: "` prefixes.
- `BAAI/bge-large-en-v1.5` — requires the query-instruction prefix from its model card.
- `nomic-ai/nomic-embed-text-v1.5` — requires task-specific prefixes (`search_document:`, `search_query:`, `classification:`, `clustering:`). Trust remote code with `trust_remote_code=True`.
- **Your exercise-01 bi-encoder** (MNR-trained).
- **Your exercise-02 bi-encoder** (hard-negative + denoise-trained).

Optionally: a hosted API (`text-embedding-3-small`, `voyage-3`, `cohere/embed-english-v3.0`) — costs pennies at this scale and is a valuable contrast.

## MTEB task slate

The full English benchmark is out of scope. Run the following one-task-per-category slate — chapter 08's shape-check subset:

| Category            | Task                          | Approximate runtime per model (GPU) |
|---------------------|-------------------------------|-------------------------------------|
| Retrieval           | `NFCorpus`                    | ~5 min                              |
| Retrieval           | `SciFact`                     | ~5 min                              |
| Reranking           | `AskUbuntuDupQuestions`       | ~5 min                              |
| STS                 | `STSBenchmark`                | <1 min                              |
| STS                 | `STS17` (English subset)      | <1 min                              |
| Classification      | `Banking77Classification`     | ~10 min                             |
| Clustering          | `ArxivClusteringS2S`          | ~10 min                             |
| Pair Classification | `SprintDuplicateQuestions`    | ~5 min                              |
| Summarization       | `SummEvalSummarization`       | ~5 min                              |

Ten tasks × six models ≈ 6–10 hours of GPU wall-clock, batchable across models with `screen`/`tmux` or an overnight run.

## Problem statement

### Part A — Encoder wrapper with correct prefix dispatch

Implement one `mteb.Encoder` subclass that dispatches the correct prefix per model and per task. Chapter 08 is explicit: skipping the prefix costs 3–8 points on E5/BGE/Nomic retrieval and you will not reproduce leaderboard numbers.

```python
class PrefixedEncoder(mteb.Encoder):
    def __init__(self, model_id: str, prefix_style: str):
        self.model = SentenceTransformer(model_id, trust_remote_code=True)
        self.prefix_style = prefix_style   # "none" | "e5" | "bge" | "nomic"

    def encode(self, sentences, prompt_name=None, task_name=None, **kwargs):
        prompt = self._resolve_prompt(task_name, prompt_name)
        prefixed = [f"{prompt}{s}" for s in sentences]
        return self.model.encode(prefixed, normalize_embeddings=True, **kwargs)
```

Table your prefix rules per model (E5: `"query: "` on queries, `"passage: "` on documents; BGE: `"Represent this sentence for searching relevant passages: "` on queries, no prefix on documents; Nomic: task-scoped prefixes) and cite each rule against its model card in a comment.

Save as `encoder.py`.

### Part B — Run the slate

Run the full 10-task slate for every model. Persist raw MTEB output JSON to `results/{model_slug}/{task_name}/`. `mteb` handles the layout for you.

Sanity check: for `all-MiniLM-L6-v2` you should reproduce (within ±0.5 points) the numbers on the [public MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard). If your MiniLM number is 3+ points off, your wrapper is wrong. Chapter 08's "leaderboard reproducibility" section is the debug guide.

Save invocation as `run_mteb.py`.

### Part C — Category-broken-out results table

Aggregate the per-task results into a Markdown table with one row per model and one column per MTEB category:

|                   | Retrieval (mean nDCG@10) | Reranking (MAP) | STS (Spearman ρ) | Classification (Acc) | Clustering (V-measure) | PairClass (AP) | Summarization | **Overall** |
|-------------------|--------------------------|-----------------|------------------|----------------------|------------------------|----------------|---------------|-------------|
| all-MiniLM-L6-v2  |                          |                 |                  |                      |                        |                |               |             |
| all-mpnet-base-v2 |                          |                 |                  |                      |                        |                |               |             |
| e5-large-v2       |                          |                 |                  |                      |                        |                |               |             |
| bge-large-en-v1.5 |                          |                 |                  |                      |                        |                |               |             |
| nomic-embed-v1.5  |                          |                 |                  |                      |                        |                |               |             |
| your-ex01         |                          |                 |                  |                      |                        |                |               |             |
| your-ex02         |                          |                 |                  |                      |                        |                |               |             |

For each row, bold the *highest* category score in the panel. Chapter 08's rule: never average categories away — highlight the specialisations.

Save as `results.md`.

### Part D — Selection defence

State a target use case in one sentence, e.g. "customer-support RAG over a 500 k FAQ corpus with a bi-encoder → cross-encoder pipeline serving 20 QPS on a single T4."

Then apply chapter 08's five-step selection procedure:

1. List the MTEB categories your target task depends on. (For the FAQ RAG example: retrieval + pair classification.)
2. Rank the seven models by the arithmetic mean of those category scores only.
3. Look at cost for the top-5 (params, output dimension, inference cost per 1k tokens — cite each).
4. Pick the smallest / cheapest of the top-5. Justify.
5. Note that you still need a *domain* evaluation (exercise-04) before shipping.

Write the recommendation as a 300–400 word section in `results.md` titled "Selection for {use case}."

### Part E — Failure-mode audit

Deliberately reproduce two of the pitfalls chapter 08 lists. Pick two of these:

- **Missing prefix.** Run E5-large-v2 on `NFCorpus` without the `"query: "` / `"passage: "` prefix. Report the delta vs. the correct run.
- **Wrong normalisation.** Load `all-mpnet-base-v2`, disable `normalize_embeddings`, run `STSBenchmark`. Report the delta.
- **Version-mismatched comparison.** Compare your MiniLM number against a 2022 paper's reported MiniLM MTEB number and describe what you cannot conclude from the comparison.
- **Task truncation.** Log the token-length distribution and truncation rate for one long-document retrieval task (`NFCorpus` documents can exceed 512 tokens for MiniLM). Report how much was silently truncated.

Save as `failure_audit.md`.

### Part F — Write-up

400–600 word `report.md` covering:

- The seven-model, ten-task panel (Parts B, C).
- Where each model wins and loses (Part C's bolded highs).
- Your selection defence for the stated target use case (Part D).
- The failure-mode audit (Part E) and what it implies about trusting model-card numbers.
- One "what next": what domain-specific evaluation you would build before shipping (exercise-04).

## Starter guidance

- **`mteb` handles task loading and result serialisation.** Do not hand-roll — you will get the metric definitions wrong. `mteb.get_tasks(tasks=["NFCorpus", ...])` + `MTEB(tasks=tasks).run(model, output_folder=...)`.
- **The prefix contract is per-model, not per-family.** BGE's prefix is different from E5's, and BGE's changed between v1.0 and v1.5. Always read the current model card; do not copy prefixes from a blog post.
- **Nomic requires `trust_remote_code=True`.** Read its `modelling_hf_nomic_bert.py` before enabling; it is public and reviewable. Chapter 08 discusses this trade-off.
- **`normalize_embeddings=True` is a `SentenceTransformer.encode` argument.** Every model in this exercise expects unit-norm cosine. If you evaluate with `normalize_embeddings=False`, STS will drop 5+ points silently.
- **Persist raw results.** `mteb` caches per-task results and will not re-run if the same output folder exists. Use that: iterate on your wrapper, delete only the affected folder, re-run.
- **Do not report "MTEB average" from a 10-task subset as if it were the leaderboard average.** Call it "10-task subset average" everywhere. Chapter 08 is explicit about not conflating the two.
- **Hosted encoders cost money.** `text-embedding-3-small` on this slate is a few cents; still keep an eye on the counter.

## Acceptance criteria

- [ ] `encoder.py` wraps at least six models with correctly-dispatched task prefixes, cited per model card.
- [ ] `run_mteb.py` executes the 10-task slate across all six (or more) models; raw MTEB JSON is persisted per model per task.
- [ ] `results.md` presents a category-broken-out comparison table with per-row category highs bolded.
- [ ] MiniLM reproduces the public MTEB numbers within ±0.5 points on at least 6/10 tasks (documented in `results.md`).
- [ ] A "Selection for {use case}" section applies chapter 08's five-step procedure and defends a pick.
- [ ] `failure_audit.md` documents two reproduced pitfalls from chapter 08 with the measured deltas.
- [ ] `report.md` (400–600 words) covers the panel, the selection, the failures, and one next step.

## Stretch goals

- **Domain sub-leaderboard.** Add `MTEB(law)`, `MTEB(medical)`, or `CoIR` on top of the English slate. Report where each general-purpose model breaks down. Chapter 08.
- **Multilingual slate.** Add `MTEB(fra)` or `MTEB(cmn, v1)` with `multilingual-e5-large` and `bge-m3` as additional models. Compare against the English models on the same subset (they will all lose). Chapter 10 previews this.
- **Cost-quality frontier plot.** Scatter models by "target-category mean" (y) vs. inference cost per 1k tokens (x). Circle the Pareto frontier. Better than any table for defending a selection.
- **Confidence intervals.** For the four retrieval tasks, bootstrap 95 % CIs per model per task (1 000 resamples over queries). Report which model-pair differences are actually significant.
- **Build your own MTEB task.** Take 500 real query-document pairs from any domain you have access to and register a `mteb.AbsTaskRetrieval` subclass. Add it to the slate. Chapter 08's "Building your own MTEB-shaped task" section.
- **Long-context evaluation.** Add a long-document retrieval task (LongDoc / MLDR English subset) and observe how models with 8k context (`nomic-embed-v1.5`, `bge-m3`) compare with 512-token models. Chapter 10.

## Deliverables

Ship as a directory with:

```
encoder.py
run_mteb.py
results/                    # per-model per-task raw MTEB JSON
results.md
failure_audit.md
report.md
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
