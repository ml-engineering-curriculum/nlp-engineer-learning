# MTEB: The Massive Text Embedding Benchmark

## Motivation

Before MTEB, evaluating a sentence encoder meant "run it on STS-B and one or two BEIR retrieval datasets and call it a day." The result was a field where the best model on STS was routinely bad at retrieval, the best retriever was often mediocre at clustering, and there was no shared vocabulary for the trade-offs.

**MTEB** (Muennighoff, Tazi, Magne & Reimers, ["MTEB: Massive Text Embedding Benchmark"](https://arxiv.org/abs/2210.07316), *EACL 2023*) is the fix. It aggregates 50+ text embedding evaluation tasks across seven categories, runs them on a frozen encoder with standardised protocols, and publishes a leaderboard that has become the industry reference. If a model card says "MTEB average 65.5," it means something specific and reproducible. If it doesn't cite MTEB at all, that itself is a signal.

This chapter is how to run MTEB, how to *read* its numbers, and — most importantly — how to avoid the common misinterpretations that lead teams to pick the wrong encoder for their actual use case.

## What MTEB is measuring

MTEB is structured around seven task categories, each with its own metric family. As of the current published version:

| Category           | Task type                                              | Metric                | Example datasets                              |
|--------------------|--------------------------------------------------------|-----------------------|-----------------------------------------------|
| **Retrieval**      | Query → passage ranking (BEIR-style)                   | nDCG@10               | MSMARCO, NQ, HotpotQA, FiQA, TREC-COVID       |
| **Reranking**      | Reranking a pre-retrieved candidate list                | MAP                   | SciDocs, MindSmall, StackOverflow, AskUbuntu  |
| **STS**            | Semantic Textual Similarity — sentence pair correlation | Spearman ρ on cosine  | STS12–17, STS-B, SICK-R, BIOSSES              |
| **Classification** | Linear probe / logistic regression on frozen embeddings | Accuracy              | Banking77, AmazonReviews, TweetSentiment, IMDb |
| **Clustering**     | Unsupervised clustering label recovery                  | V-measure             | ArxivClustering, RedditClustering, TwentyNews  |
| **Pair classification** | Binary "same or different" over pairs             | AP (Average Precision) | SprintDuplicate, TwitterSemEval, OpusparcusPC |
| **Bitext mining**  | Cross-lingual sentence pair alignment                    | F1                   | Tatoeba, BUCC (multilingual only)             |

The default "MTEB average" reported on the leaderboard is the *arithmetic mean of task means*, computed over the English MTEB subset. Multilingual MTEB averages are computed separately per language and reported alongside.

## The one insight to internalise

Models specialise. A model can win on retrieval and lose on STS, and this is not a bug — it is a training-data signal. Specifically:

- **Retrieval winners** were trained on query-passage pair data (MS MARCO, NQ, BEIR-like). They excel at asymmetric matching.
- **STS winners** were trained on paraphrase and NLI data. They excel at *symmetric* similarity on similarly shaped inputs.
- **Classification winners** were trained on data that spans many topics, so their embeddings are linearly separable by topic. Often overlap with STS winners.
- **Clustering winners** need vectors with good *cluster structure*. Often correlate with classification but not with retrieval.

The leaderboard-topping models (BGE, E5, GTE, `text-embedding-3`, Cohere v3) are the ones that score well *across all seven* — not because there is a magic architecture, but because they were trained on data covering all seven distributions.

**Practical rule.** When picking a model, look at the *category* score for your target use case, not the overall average. A 71.5 overall model that scores 55 on retrieval is worse for your RAG stack than a 65.0 overall model that scores 60 on retrieval.

## Running MTEB

The `mteb` Python package is the reference implementation:

```bash
pip install mteb
```

Running the full English benchmark on a `SentenceTransformer`:

```python
import mteb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
tasks = mteb.get_benchmark("MTEB(eng, v1)")            # 56 English tasks
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder="./mteb-results")
```

That will take multiple hours to days on a single GPU for the full English benchmark. Reasonable subsets for iteration:

```python
# ~5 retrieval-only tasks, ~30 min total
tasks = mteb.get_tasks(tasks=["NFCorpus", "SciFact", "TRECCOVID", "FiQA2018", "SCIDOCS"])

# ~5 STS-only tasks, minutes
tasks = mteb.get_tasks(tasks=["STS12", "STS13", "STS14", "STS15", "STSBenchmark"])
```

For a fast smoke test, running one task per category (`NFCorpus`, `STSBenchmark`, `Banking77Classification`, `ArxivClusteringS2S`, `SprintDuplicateQuestions`, `AskUbuntuDupQuestions`) takes ~30 min and gives you a shape-check across all seven categories.

## Bringing your own model

Any callable that produces embeddings can be wrapped as an `mteb.Encoder`:

```python
import mteb
import numpy as np

class MyEncoder(mteb.Encoder):
    def __init__(self, hf_model_id: str):
        self.model = load_your_model(hf_model_id)

    def encode(self, sentences, prompt_name=None, task_name=None, **kwargs) -> np.ndarray:
        # apply the correct task-specific prefix based on task_name
        return self.model.encode(sentences, normalize_embeddings=True, **kwargs)
```

The `task_name` argument is important. Modern models (E5, BGE, Nomic) require *different prefixes for queries vs. passages* (chapter 03). `mteb.Encoder` passes `task_name` and `prompt_name` so you can dispatch — the `MTEB` framework knows which sentences are queries vs. passages for each task.

Every published encoder that ships MTEB numbers has done this correctly. If you don't, your scores will be silently 2–8 points lower than the leaderboard's for the same weights, and you will confuse yourself trying to reproduce them.

## Multilingual MTEB

The MTEB benchmark has been extended to multiple language collections since the original English release. In `mteb`:

```python
# Multilingual retrieval + classification + clustering + bitext
tasks_ml = mteb.get_benchmark("MTEB(Multilingual, v1)")

# Language-specific
tasks_fr = mteb.get_benchmark("MTEB(fra)")
tasks_zh = mteb.get_benchmark("MTEB(cmn, v1)")
```

Cross-lingual bitext-mining tasks (Tatoeba, BUCC) live specifically in the multilingual subset. If you are evaluating a multilingual model, run the multilingual benchmark, not the English one — the English MTEB has no bitext-mining category.

Chapter 10 covers what "good multilingual embedding" looks like on these tasks.

## Domain-specific MTEB extensions

Since the original release, MTEB has grown per-domain sub-leaderboards. As of 2025:

- **MTEB(law)** — Chalkidis et al., ["LEXTREME: A Multi-Lingual and Multi-Task Benchmark for the Legal Domain"](https://arxiv.org/abs/2301.13126) and follow-ups; legal-domain tasks.
- **MTEB(medical)** — Xiong et al., ["MTEB(medical)"](https://arxiv.org/abs/2404.01617); clinical and biomedical embedding evaluation.
- **MTEB(code)** — CoIR benchmark; code search and reranking.
- **`ChemTEB`** — chemistry-domain embedding benchmark.

Running domain sub-leaderboards:

```python
tasks_law     = mteb.get_benchmark("MTEB(law)")
tasks_medical = mteb.get_benchmark("MTEB(medical)")
tasks_code    = mteb.get_benchmark("CoIR")
```

Chapter 09 uses one of these as the evaluation set for the domain-adaptation exercise.

## Reading a leaderboard result correctly

A typical MTEB report line looks like:

```
Model: BAAI/bge-large-en-v1.5
Avg 64.23   Retrieval 54.29   Reranking 60.03   STS 83.11
Classification 75.97   Clustering 46.08   PairClassification 87.12   Summarization 31.61
```

Six things to notice:

- **The average hides retrieval.** Retrieval numbers are usually the lowest — nDCG@10 is a hard metric. A 54 on retrieval is *good*.
- **STS numbers are the highest.** STS on the standard datasets is close to saturated. A 5-point STS lift is a much bigger deal than the same absolute lift on retrieval.
- **Classification and pair classification are close to their upper bound.** These tasks favour any model that has seen enough diverse text.
- **Clustering is uniquely sensitive to representation geometry.** Cluster-friendly models spread topics apart. Models trained purely for retrieval sometimes cluster poorly.
- **Reranking measures a *reranker's* quality when applied to a *pre-retrieved candidate list*.** It is not measuring the model as a first-stage retriever.
- **Summarization is a bespoke MTEB task** — evaluate summary/document embedding similarity — not a general-purpose summarisation metric.

The point: do not average away the specialisations. Read the categories.

## Common evaluation pitfalls

- **Missing task prefix.** E5 and BGE need `"query: "` / `"passage: "` (or equivalent). Skipping them costs 3–8 points on retrieval. `mteb.Encoder`'s `task_name` argument is how you dispatch.
- **Wrong pooling.** If you loaded a `SentenceTransformer` bundle, you are fine. If you loaded a raw HF encoder, you need to know whether to CLS-pool or mean-pool (chapter 03).
- **Wrong normalisation.** Cosine similarity in MTEB assumes unit-norm vectors. Not normalising cuts STS numbers by 5+ points.
- **Truncated inputs.** Long documents on a 512-token encoder get silently truncated. Log the token distribution and truncation rate before believing any MTEB number for long-document tasks.
- **Comparing across MTEB versions.** MTEB has revved its task selection. A "70.1 MTEB average" from 2022 is not comparable to a "70.1 MTEB average" from 2024. Check the version tag (`MTEB(eng, v1)`, `MTEB(eng, v2)`).
- **Cherry-picking datasets in a paper.** A paper reporting only STS-B and one retrieval dataset is *avoiding* the MTEB discipline. Distrust.

## Interpreting the leaderboard for model selection

The right decision procedure:

1. **List your target tasks.** "We use the encoder for RAG retrieval + a dedup pass on the ingest side" → retrieval + pair classification are your target categories.
2. **Rank models by the arithmetic mean of your target category scores only.** Ignore the overall average.
3. **For the top-5 by your target rank, look at cost.** Params, output dimension, inference cost per 1k tokens.
4. **Pick the smallest / cheapest of the top 5.** The last 1–2 MTEB points rarely justify a 10× cost increase in production.
5. **Then run *your own* domain evaluation.** Chapter 09.

MTEB narrows the field from "hundreds of models" to "5–10 credible options." Your own domain evaluation picks the winner from those.

## Building your own MTEB-shaped task

For any target domain that is not well-covered by public MTEB tasks, you build one. The template:

- **Retrieval task.** A set of queries, a corpus of documents, and a relevance judgement file (`qrels`). ~1000 queries and ~50k documents is a workable minimum. `mteb.AbsTaskRetrieval` handles the plumbing.
- **STS task.** Sentence pairs with a 0–5 similarity label. ~500 pairs is enough for a stable Spearman ρ.
- **Classification task.** Sentences with class labels. ~5000 examples per class minimum for stable linear-probe results.

The `mteb` docs have templates for each abstract task type; add your task locally and it runs through the same harness. This is how you turn "the encoder works on our data" from a vibes-check into a reproducible metric that a future team can retrain against.

## Chapter summary

- MTEB is the reference evaluation for text embeddings — seven task categories, 50+ tasks, standardised protocols, public leaderboard.
- Never look at only the average. Look at the category score for your target use case; models specialise heavily.
- Retrieval scores are systematically the lowest (hard metric); STS is close to saturated (easy metric); the difference between two models is only meaningful within a category.
- `mteb` Python package runs the whole benchmark; for iteration, run one task per category first.
- Task prefixes (`"query: "`, `"passage: "`) matter for E5, BGE, Nomic and others. Skip them and your numbers are silently wrong.
- Multilingual, legal, medical, and code sub-leaderboards exist and are the right benchmark when your target domain matches.
- Selection rule: rank by target-category mean, pick the cheapest among the top 5, then run your own domain evaluation to break ties.
- Building your own MTEB-shaped task for a bespoke domain is straightforward and worth doing once — it converts "gut feel" into a reproducible number that survives team changes.
