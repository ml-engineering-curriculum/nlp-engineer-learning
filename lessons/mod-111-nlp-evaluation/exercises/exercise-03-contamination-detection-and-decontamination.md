# exercise-03: Contamination Detection and Decontamination

**Estimated effort:** 3 hours

## Objective

Detect **pretraining-data contamination** for a public benchmark against a public open-corpus model, decontaminate a training-data subset against that benchmark, and design a **contamination-resistant evaluation** that will still be honest one model generation from now. Deliver a contamination audit that a model card could publish, plus a small decontamination pipeline, plus a time-boxed eval spec.

## Prerequisites

- Chapter [08](../08-contamination-detection-and-decontamination.md).
- Python 3.10+; `datasets`, `datasketch` (MinHash / LSH), `transformers`, `torch`, `numpy`, `tqdm`.
- Access to a small slice of an open pretraining corpus — recommended: the first shard of `EleutherAI/pile-uncopyrighted` (or a stand-in of your choice — RedPajama, C4-en subset, whatever your bandwidth allows). Full corpora are hundreds of GB; you will work on a fixed slice.
- One open-weight LM whose pretraining corpus you have access to. Recommended: `EleutherAI/pythia-160m` (or up to `pythia-1.4b` if compute allows) — Pythia's pretraining is Pile-only.

## Problem statement

### Part A — N-gram contamination check

Pick **three public benchmark test sets** you might report on:

- **MMLU** (Hendrycks et al., ["Measuring Massive Multitask Language Understanding"](https://arxiv.org/abs/2009.03300)) — a `cais/mmlu` subset of one subject (e.g., `high_school_us_history`).
- **HellaSwag** (`Rowan/hellaswag`) — 200-example dev slice.
- **A dataset of your choice** with a plausible contamination story — SQuAD dev, ARC-Challenge, GSM8K test.

For each benchmark:

1. **Extract the input text** per test item (question + choices for MMLU / HellaSwag; question + first 200 chars of context for SQuAD).
2. **Build a MinHash-LSH index over the pretraining slice.** Use 13-token word shingles, 128 permutations, Jaccard threshold 0.5 for the LSH bucket. Persist the index to disk after building — this is the expensive step.
3. **Query each test item.** Record which items have any pretraining-doc hit above the threshold and, for a random sample of 10 hits, the actual pretraining-doc excerpts (for evidence).
4. **Report the contamination rate** per benchmark: `|hits| / |total|`.

Save as `a_ngram_check/`: `build_lsh.py`, `query_benchmarks.py`, `contamination_report.md` with the three rates + sample hits.

### Part B — Clean vs. dirty subset metrics

For each of the three benchmarks in Part A, use Pythia-160m (or your chosen model) via `lm-evaluation-harness` (or a hand-rolled evaluator) to score:

- **Overall accuracy** on the benchmark.
- **Accuracy on the *clean* subset** (items with no LSH hit).
- **Accuracy on the *dirty* subset** (items with an LSH hit).
- **Bootstrap 95 % CI** on each.

The clean/dirty delta is the contamination-attributable inflation. Report as a table plus a one-paragraph interpretation per benchmark. Where the dirty subset is very small (< 20 items), state the caveat — the CI will be wide and the delta unreliable.

Save as `b_clean_vs_dirty/`: `evaluate.py`, `report.md`.

### Part C — Behavioural contamination probes

For **one** of the three benchmarks (pick whichever showed the highest contamination rate), run two behavioural probes that work even if you did *not* have the pretraining corpus:

**C1: Verbatim continuation.** For 20 test items, feed the first half of the input text to the model and check whether the completion reproduces the second half verbatim (measure: normalised edit distance or top-1 exact match). Compare against a **control** set of 20 similar-format inputs that were *published after* the model's training cutoff (find recent-arXiv or recent-news items of similar length and structure). The model should complete the seen items more accurately than the control.

**C2: Min-K% Prob.** Following Shi et al. (["Detecting Pretraining Data from Large Language Models"](https://arxiv.org/abs/2310.16789)), score each of the 20 seen items and the 20 controls with the model's per-token log-probability. Compute Min-K% Prob (mean log-p of the lowest K% of tokens; K = 20 % is standard). Compare distributions between seen and control sets — expect seen items to have *higher* (less negative) Min-K% Prob.

Report both probes as `c_behavioural/`: `verbatim.py`, `mink.py`, `report.md` with the two comparisons and a statement of "consistent with contamination" or "no evidence." Include a scatter plot (matplotlib is fine).

### Part D — Decontaminate a training-data subset

Take a **100k-document sample** of the pretraining slice (or the whole slice if it is small enough — Pythia's Pile is huge, sample it). Decontaminate against the three benchmarks:

1. **Extract 13-token shingles** from every test-item input across the three benchmarks; put them in a fast set (bloom filter or dict — hash the shingle).
2. **Stream through the training sample.** For each document, extract its shingles and check whether any is in the eval-shingle set. If yes, drop the document (LLaMA 2 approach) or mask the offending span (OLMo approach) — pick one and document.
3. **Report the drop rate** — fraction of the sample removed by decontamination against the three benchmarks. Report as a table with per-benchmark contribution (how many docs did each benchmark trigger) and the total dedupe-adjusted final count.

Save as `d_decontaminate/`: `decontaminate.py`, `report.md` with the drop rates, one example dropped document (excerpt + which benchmark shingles it matched), and a discussion of false-positive risk (common phrases that will accidentally match).

### Part E — Design a contamination-resistant eval

Design a **rolling, time-boxed evaluation** for one of the following task families (pick one):

- News-article classification (fine-grained topic / stance / sentiment).
- Recent-arXiv abstract summarisation.
- QA over recent Wikipedia edits.
- Code-understanding on recently-created GitHub repositories.

Produce `e_resistant_eval/eval_spec.md` (400–600 words) with:

- **Task definition.** Input, output, primary metric, secondary metric, per-slice breakdowns (chapter 01 rubric applies).
- **Time-boxing recipe.** Source(s), cutoff-postdating window (e.g., "arXiv papers submitted after 2025-06-01"), how many items per rolling batch, how often to rotate.
- **Canary provision.** A canary string embedded in the eval and instructions to model providers on how to exclude it from training. Publish a specimen canary in the spec.
- **Contamination-check protocol.** How the next reviewer of this eval should re-verify contamination against the model they are evaluating (which corpus to check against; if closed-model, which behavioural probes).
- **Known weaknesses.** State them — e.g., "model may re-scrape and retrain between our releases; time-boxing is a partial defence."

Optionally sketch (but do not implement in full) a minimal generation script for the eval batches.

## Starter guidance

- **MinHash-LSH is memory-heavy on large corpora.** For Part A, budget the corpus slice down until the index fits in a few GB. `datasketch` is fine; if you want speed, `datasketch` + on-disk backend or `mrjob`-style parallel processing.
- **13-tokens is the community-standard n-gram length** for text contamination checks. Character-level 50-gram is an alternative for code-heavy content.
- **Report the exact corpus slice** you queried against — `pile-uncopyrighted` shard 00, N documents, byte count. Reproducibility depends on it.
- **Watch out for the "everything is contaminated" trap.** With low thresholds and common shingles, you can decontaminate away the whole training set. Aim for high precision on rare 13-grams; document your threshold choice.
- **Behavioural probes give evidence, not proof.** Report a comparison, not a categorical claim. "Seen items have systematically higher Min-K% Prob than controls (p < 0.01 by Mann-Whitney)" is the right kind of statement.
- **The rolling eval spec is the highest-leverage deliverable.** Contamination detection is remediation after the fact; contamination-resistant design prevents the problem for the next model.

## Acceptance criteria

- [ ] `a_ngram_check/` — MinHash-LSH index built on a documented pretraining slice; contamination rates for three benchmarks; sample hit excerpts for verification.
- [ ] `b_clean_vs_dirty/` — overall / clean / dirty accuracy per benchmark with 95 % bootstrap CIs, plus interpretation.
- [ ] `c_behavioural/` — verbatim-continuation and Min-K% Prob probes on one benchmark's items vs. a time-boxed control set, with statistical comparison.
- [ ] `d_decontaminate/` — decontamination script with per-benchmark drop rates, example dropped document, false-positive discussion.
- [ ] `e_resistant_eval/eval_spec.md` — full spec for a time-boxed, canary-instrumented, contamination-check-documented rolling eval.
- [ ] Every corpus / benchmark / model reference names the exact version, shard, or checkpoint used.
- [ ] Where a claim cannot be verified (e.g., closed-model behavioural probes without pretraining-corpus access), the report says so explicitly.

## Stretch goals

- **Substring exact match.** Implement the exact substring contamination check from Marone & Van Durme (["Data Contamination Through the Lens of Time"](https://arxiv.org/abs/2310.10628)) — sort suffix array, exact substring lookup — and compare its findings to your MinHash-LSH results.
- **Semantic decontamination.** For Part D, add a semantic-similarity pass (embed test items and training docs with a sentence encoder; drop docs within cosine < 0.9). Compare drop-rate delta vs. plain n-gram; identify one paraphrase case caught semantically but missed by n-gram.
- **Cross-model probe.** Run the Part C behavioural probes on a *closed-weight* model (via API) as well. Report which probe is more discriminative in the API-only setting.
- **Canary detection audit.** Take the BIG-bench canary string and check whether any downstream model's outputs (any open-weight instruct model you can run) regurgitate it. Report and cite.
- **Publish the resistant eval.** Generate one full batch of Part E's rolling eval (10–50 items) and score two frontier LMs on it. Include the batch and the scores as an appendix.

## Deliverables

```
a_ngram_check/          build_lsh.py  query_benchmarks.py  contamination_report.md  index/
b_clean_vs_dirty/       evaluate.py  report.md  scores.json
c_behavioural/          verbatim.py  mink.py  report.md  plots/
d_decontaminate/        decontaminate.py  report.md  dropped_examples.md
e_resistant_eval/       eval_spec.md
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
