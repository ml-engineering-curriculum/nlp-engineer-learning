# exercise-01: Task-Aware Metric Selection Rubric

**Estimated effort:** 3 hours

## Objective

Build a **task-aware metric selection rubric** — a decision aid that, given a proposed NLP task and its constraints, prescribes the metric panel (primary metric, secondary metric of a different shape, task-specific structural metric, per-slice breakdowns, and significance test) and produces a *worked example* on real data for each of five NLP task families. Deliver both the rubric itself and evidence that it produces the right metric panel for each family — including a case where the "obvious" choice is wrong.

## Prerequisites

- Chapters [01](../01-metric-taxonomy-and-task-fit.md), [02](../02-classification-and-sequence-labelling-metrics.md), [03](../03-extractive-qa-metrics.md), [04](../04-generation-metrics-bleu-rouge-meteor-bertscore-comet.md), [05](../05-perplexity-and-language-model-evaluation.md).
- Python 3.10+; `datasets`, `evaluate`, `sacrebleu`, `unbabel-comet`, `seqeval`, `scikit-learn`, `bert-score`, `rouge-score`, `numpy`.
- A machine with a GPU (COMET / BERTScore) or willingness to run those on CPU (slower but works).

## Problem statement

### Part A — Write the rubric

Produce `metric_rubric.md` — a short (500–800 words) decision aid organised as a table plus a short explainer for the reader:

- **A "task family → metric panel" table.** Rows: `classification-balanced`, `classification-imbalanced`, `multi-label classification`, `span-tagging (NER)`, `extractive QA`, `abstractive QA`, `machine translation (high-resource pair)`, `machine translation (low-resource pair)`, `summarisation (news)`, `summarisation (long-doc)`, `language modelling (same tokeniser as baseline)`, `language modelling (different tokeniser)`, `text embedding (retrieval)`, `text embedding (STS)`, `open-ended chat generation`.
- **Columns:** primary metric, secondary metric (different *shape* — chapter 01 taxonomy), task-specific structural metric (or "n/a"), required per-slice breakdowns, significance test to run, one-line note on the top failure mode.
- **The explainer** must (a) name the four metric shapes from chapter 01, (b) state the panel rule ("primary + at least one different-shape secondary + significance + per-slice"), and (c) give three worked cases where the naive choice would be wrong (e.g., accuracy on imbalanced data, BLEU alone on Chinese MT, ROUGE alone on abstractive summarisation).

Format the table so a reader can copy a row into their own eval spec and understand what to run. Keep it opinionated; do not enumerate every possible metric — pick the panel *you* would recommend.

### Part B — Five worked task examples

For each of the following five task families, produce a small worked example that instantiates your rubric with real data and real numbers. Each example goes in its own subdirectory:

**B1: Classification (imbalanced).** Use `datasets`' `imdb` split truncated / re-sampled to a 90/10 imbalance (majority = negative). Fine-tune a light classifier (or use the `sentiment` zero-shot pipeline for speed) and produce two systems whose predictions differ. Report the metric panel from your rubric row for `classification-imbalanced`: macro-F1 with per-class breakdown, MCC, ECE for a probability-emitting model, bootstrap CIs, per-class confusion, and paired bootstrap between the two systems.

Save as `b1_classification/`: `run.py`, `report.md`, `predictions_a.csv`, `predictions_b.csv`.

**B2: Span tagging (NER).** Use `conll2003` (or `wikiann` en if compute is tight); pick two off-the-shelf NER models (e.g., `dslim/bert-base-NER` and `Jean-Baptiste/roberta-large-ner-english`). Score with `seqeval` in `strict` mode with `scheme=IOB2`. Report micro/macro entity-F1, per-type F1, and paired bootstrap on entity-F1 deltas. Include one example where a per-token accuracy report would give a very different picture from the entity-F1 report — and explain why in `report.md`.

Save as `b2_ner/`.

**B3: Extractive QA (SQuAD-family).** Use `squad_v2` dev. Pick two off-the-shelf extractive QA models (e.g., `deepset/roberta-base-squad2` and `deepset/deberta-v3-large-squad2`). Score with `evaluate.load("squad_v2")` — report overall EM/F1, HasAns / NoAns split, best-threshold F1, per-question-type F1 slices. Show the delta between `best_f1` and `f1` for each model, and explain what it tells you about calibration.

Save as `b3_squad/`.

**B4: MT (chrF / COMET disagreement).** Use FLORES-200 devtest on any one high-resource pair for which you can run two models — e.g., `Helsinki-NLP/opus-mt-en-de` vs. `facebook/nllb-200-distilled-600M` on eng→deu. Score with SacreBLEU + chrF (SacreBLEU CLI or `sacrebleu` Python), COMET (`Unbabel/wmt22-comet-da`), and BERTScore. Find at least *five* sentences where SacreBLEU and COMET disagree on which system is better, categorise the disagreement (paraphrase-friendly / hallucination / BLEU-noise), and put your rubric's recommendation for shipping decisions (chrF and COMET as primary, SacreBLEU as legacy pin).

Save as `b4_mt/`.

**B5: Language modelling (cross-tokeniser).** Score two open-weight LMs with different tokenisers on the same held-out text (e.g., `gpt2` and `EleutherAI/pythia-160m` on the first 1MB of WikiText-103 test). Report **perplexity** and **bits-per-byte** for each; show that the PPL comparison gives a misleading answer while BPB gives the right one. This example is the one where your rubric's `language modelling (different tokeniser)` row saves the reader from a mistake.

Save as `b5_lm/`.

### Part C — Anti-pattern gallery

`anti_patterns.md` — enumerate five specific reporting mistakes your rubric prevents, one per family. For each: describe the mistake in ≤ 3 sentences, cite where in the rubric it is caught, and give a citation or example (with page or table reference where possible) from the literature or a well-known model card where the mistake has actually been made. Do not invent citations; if you cannot find a real example for one of the five, replace it with a different mistake you *can* attribute or write `<!-- needs-research: candidate mistake — no real citation found -->`.

### Part D — Rubric self-test

At the top of `metric_rubric.md`, provide a **10-question self-test** the reader can use to check whether they applied the rubric correctly to a novel task. Questions of the form "What is the primary metric for a class-imbalanced multi-class classifier?" / "You are comparing two LMs with different tokenisers; what do you report?" / "You are shipping a summariser and only report ROUGE-L; what is missing?" — with the answers immediately below (or in an `answers` collapsible / footer).

## Starter guidance

- **Do not reinvent the metric implementations.** Use `evaluate.load(...)`, `sacrebleu`, `seqeval`, `unbabel-comet`, `bert-score`, `sklearn.metrics`, `rouge_score`. The whole point of the rubric is that it names the canonical implementation.
- **Signature strings and checkpoints go in `report.md` for each B*/`.** No signature = not reproducible.
- **Bootstrap CIs everywhere.** 1 000 resamples, 95 % CI. Reuse the same helper across all five examples.
- **Paired bootstrap for two-system deltas.** Same test-set indices resampled for both systems each iteration.
- **B5 is the pedagogical highlight** — do the PPL vs. BPB comparison carefully; it is the clearest demonstration of the rubric preventing a real mistake.
- **Keep the rubric opinionated.** A rubric that enumerates 40 candidate metrics per task is not a rubric; it is a metric catalogue. Pick three or four per row and defend them.
- **Do not exceed the disk budget on model downloads.** For B3 / B4 use the smaller variants first (`base` before `large`); scale up only if compute allows.

## Acceptance criteria

- [ ] `metric_rubric.md` (500–800 words) with the full task-family × metric-panel table, four-shape explainer, three worked mistakes, and 10-question self-test with answers.
- [ ] Five worked example directories (`b1_classification/` through `b5_lm/`), each containing runnable `run.py`, a `report.md` with the panel results, and any intermediate CSV / JSON artifacts.
- [ ] Every score in every `report.md` has a bootstrap CI; every two-system delta has a paired significance test.
- [ ] Every metric result names the canonical implementation and the signature / checkpoint / config.
- [ ] B4 identifies ≥ 5 SacreBLEU / COMET disagreement examples with category labels.
- [ ] B5 shows the PPL-vs-BPB flip: the two systems rank differently under PPL than under BPB (or, if they rank the same, `report.md` documents why and shows the numeric gap).
- [ ] `anti_patterns.md` documents 5 real-world reporting mistakes the rubric catches, each with a citation (or a `needs-research` marker).

## Stretch goals

- **Multi-label row.** Extend the rubric with a full worked example on Reuters-21578 or GoEmotions.
- **Cross-lingual QA.** Add a MLQA or TyDi example that exercises the "use the benchmark's own scorer" rule from chapter 03.
- **Embedding model panel.** Instantiate the `text embedding (retrieval)` row with a small MTEB subset (BEIR NF-Corpus + SciFact) and report nDCG@10 per task, plus paired bootstrap.
- **LLM-as-judge sanity check.** For B4, add a GEMBA-style LLM-judge column and correlate against COMET on your 5 disagreement cases.
- **Publish the rubric as a decision tree.** Instead of (or in addition to) a table, produce a Mermaid flowchart the reader can follow: `is the task classification? → balanced? → yes: accuracy + confusion → no: macro-F1 + MCC + per-class + ECE`.

## Deliverables

```
metric_rubric.md
anti_patterns.md
b1_classification/    run.py  report.md  predictions_a.csv  predictions_b.csv
b2_ner/               run.py  report.md
b3_squad/             run.py  report.md
b4_mt/                run.py  report.md  disagreements.md
b5_lm/                run.py  report.md
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
