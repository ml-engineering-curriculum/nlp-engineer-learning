# The NLP Metric Catalogue and Task Fit

## Motivation

Every NLP task has a small stable of metrics the community agrees are "the ones to report" — and a much larger cast of metrics that look reasonable but silently mismeasure the thing you care about. Classification papers report macro-F1 when the label distribution is skewed; MT papers report SacreBLEU, chrF, and COMET; summarisation papers still lead with ROUGE but pair it with a faithfulness metric because ROUGE happily rewards fluent hallucinations. QA papers report SQuAD EM/F1 with the official normaliser. Perplexity is language-modelling's default but is not comparable across tokenisers.

Picking the wrong metric is not a small mistake. A team that ships on macro-F1 when precision at low recall is what matters will over-value majority-class performance; a team that ships on BLEU without COMET will over-value surface-form similarity; a team that ships on ROUGE without a faithfulness check will ship hallucinating summarisers. This chapter is the map: for each task, what metric shape is right, what it measures, what it does not, and what to pair it with.

The rest of the module then zooms in — chapters 02–05 walk the concrete metric implementations per task family, chapter 06 covers picking the right benchmark suite, chapters 07–08 give you statistical rigour and contamination controls, and chapter 09 covers human evaluation.

## The metric taxonomy

Every automatic NLP metric is one of four shapes. Naming them explicitly helps you reason about limitations before you pick.

**1. Exact-match / structured comparison.** Compare the prediction to the reference under a normaliser and either match or not. Examples: SQuAD EM, span-level entity F1 (`seqeval`), accuracy on classification. Cheap, deterministic, comparable across papers when the normaliser is standardised. Blind to paraphrase and to partial correctness beyond what the normaliser tolerates.

**2. Surface-form n-gram overlap.** Score how much of the prediction's n-gram content overlaps with a reference. Examples: BLEU (word n-grams, precision-flavoured), ROUGE-N / ROUGE-L (recall-flavoured), chrF (character n-grams), METEOR (adds stemming + synonyms). Cheap and reproducible with tokeniser-pinned implementations. Blind to paraphrase; unfair to morphologically rich or non-space-segmented languages when word-based; still the leaderboard defaults for MT and summarisation.

**3. Learned neural metrics.** A model — usually a fine-tuned multilingual encoder — takes `(source, hypothesis[, reference])` and emits a scalar quality score trained on human judgements. Examples: COMET, BLEURT, BERTScore, GEMBA (LLM-as-judge). Correlate much better with human ratings; require a GPU (or a paid API) and a named checkpoint to be reproducible. Susceptible to their own systematic biases — most were trained on high-resource languages and inherit those distributions.

**4. Task-specific structured metrics.** Metrics whose shape encodes the task's structural constraints: perplexity for language modelling, seqeval entity-F1 for span-level tagging, MRR / nDCG for retrieval-flavoured tasks in MTEB, SARI for text simplification, translation edit rate (TER) for post-editing effort. These are correct by construction for the task; they are usually not portable to other tasks.

A well-designed evaluation report combines at least one metric from a different shape than the others. On MT, that means BLEU (surface) + COMET (learned). On summarisation, ROUGE (surface) + BERTScore (learned) + a faithfulness metric (task-specific). On QA, SQuAD F1 (exact-ish) + optionally an answer-equivalence classifier for open-ended cases.

## Task-to-metric quick reference

| Task family | Primary metric | Secondary / pair with | Chapter |
|---|---|---|---|
| Binary / multi-class classification | Macro-F1 (imbalanced) or accuracy (balanced) | Per-class F1, MCC, calibration (ECE) | 02 |
| Multi-label classification | Micro-F1 or subset accuracy | Per-label F1, coverage, hamming loss | 02 |
| Token-level tagging (POS) | Per-tag accuracy | Macro-F1 over tag set | 02 |
| Span-level tagging (NER, chunking) | `seqeval` entity-level F1 | Per-type F1, boundary vs. type errors | 02 |
| Extractive QA (SQuAD-style) | SQuAD EM + F1 (official normaliser) | Answerable / unanswerable split (SQuAD 2.0), per-question-type F1 | 03 |
| Machine translation | chrF and/or COMET (`Unbabel/wmt22-comet-da`) | SacreBLEU with signature; COMET-Kiwi for reference-free | 04 (this module); mod-107 ch10 |
| Summarisation | ROUGE-1/2/L | BERTScore, QA-based / NLI-based faithfulness | 04 (this module); mod-106 ch10-11 |
| Language modelling | Perplexity on a held-out set with same tokeniser | Bits-per-byte for cross-tokeniser comparison | 05 |
| Sentence retrieval / STS | Spearman correlation on STS; nDCG / MRR on retrieval | MTEB per-task means (report each) | 06 |
| Open-ended generation | LLM-as-judge with rubric + human eval | Reference-based (if refs exist), calibration | 09 |

The pattern is invariant: **primary metric + at least one metric of a different shape + statistical significance + per-slice breakdown**. Report a single number and a reader cannot judge whether the delta matters. Report a panel and they can.

## What "correct per task" actually requires

Applying a metric correctly is not just picking the right one — it is a small stack of decisions that determine reproducibility.

- **The implementation must be canonical.** Use SacreBLEU, not `nltk.translate.bleu_score`. Use `seqeval`, not homegrown span alignment. Use the official SQuAD `evaluate.py` (or `evaluate.load("squad")`), not a re-implementation of the normaliser. Every canonical implementation has an idiosyncrasy your homegrown version will get wrong; the whole point of adopting the community metric is that everyone else got the same idiosyncrasy.
- **The signature must be reported.** SacreBLEU emits a signature string (`nrefs:1|case:mixed|tok:13a|smooth:exp|version:2.4.0`). COMET has a named checkpoint (`Unbabel/wmt22-comet-da`). BERTScore has a `model_type`, `num_layers`, `rescale_with_baseline` triple. If the reader cannot reproduce the number from the signature, the number is not reproducible.
- **The reference set matters.** SQuAD provides multiple references per question and you take `max` over them. WMT test sets provide one reference. Some MT datasets (WMT'23 multi-reference) provide multiple. Silently using one when many were given inflates or deflates the number.
- **Aggregation direction.** Macro-average over classes, macro-average over questions, micro-average over tokens — pick deliberately, document explicitly, do not silently switch when it flatters your numbers.
- **Confidence intervals.** No serious metric report ships without a bootstrap CI. A metric-only number invites the reader to imagine a precision that is not there.

## Common failure modes, catalogued once

Each of these is a metric-reporting bug that has appeared in production and in top-tier papers. Chapter 02 onward catches them per task; here they are as one list to internalise.

- **Wrong averaging.** Reporting micro-F1 on a class-imbalanced dataset lets the majority class swamp the number. Report macro-F1 with per-class F1, or subgroup metrics with the smallest subgroup called out.
- **Wrong tokeniser.** `nltk.translate.bleu_score` on pre-tokenised text produces a different number than SacreBLEU on raw text on the same predictions. Use SacreBLEU.
- **Ignoring the normaliser.** Case-sensitive comparison on SQuAD, or accent-sensitive comparison on French QA, silently drops F1 that the official normaliser would have granted.
- **Perplexity across tokenisers.** Two LMs with different tokenisers cannot be compared on perplexity — bits-per-byte or bits-per-character normalise for tokenisation.
- **One reference where many exist.** Common on SQuAD dev, XNLI, and multi-reference MT sets.
- **Averaging across languages.** A single "multilingual F1" hides which languages are broken. Report per-language, then a macro-average, and call out the worst.
- **Silent contamination.** The pretraining data of a modern LLM has plausibly seen your test set. Chapter 08 shows how to check.
- **No significance test.** A 0.3-BLEU or 0.5-F1 improvement is almost never significant on a 1 000-example test set. Chapter 07 shows how to prove it.
- **No human eval on high-stakes deployments.** Automatic metrics are proxies. For consequential shipping decisions, back the automatic panel with a human protocol (chapter 09).

## What this module builds toward

By the end of the module you should be able to:

- **Compose a metric panel** for any NLP task in your track — pick the primary metric, the paired metric of a different shape, and the task-specific slicing (chapters 02–05).
- **Choose the right benchmark suite** — GLUE, SuperGLUE, XTREME, FLORES, MTEB, or lm-evaluation-harness — knowing what each measures and where each falls short (chapter 06).
- **Prove or disprove a system-vs-system improvement** with bootstrap resampling or paired permutation, correctly (chapter 07).
- **Detect and mitigate pretraining-data contamination** of your evaluation set, and design contamination-resistant held-out evaluations (chapter 08).
- **Design a human-evaluation protocol** with calibrated rubrics, position-bias controls, and demographic / dialect slicing that produces numbers a stakeholder can act on (chapter 09).

These are the moves that separate an evaluation you can ship on from an evaluation that looks convincing until someone re-runs it with a different seed.

## Chapter summary

- Every automatic NLP metric is one of four shapes: exact-match / structured, surface-form n-gram overlap, learned neural, or task-specific structured. A serious evaluation combines at least two different shapes.
- Metric choice is task-first. Classification → macro-F1 + calibration; span tagging → `seqeval` entity-F1; extractive QA → SQuAD EM/F1; MT → chrF / COMET; summarisation → ROUGE + BERTScore + faithfulness; LM → perplexity with same tokeniser.
- Correct application requires the canonical implementation, a reported signature (SacreBLEU / COMET checkpoint / BERTScore config), the full reference set, deliberate aggregation, and bootstrap confidence intervals.
- The most common failure modes are silent: wrong averaging, wrong tokeniser, one-reference-of-many, cross-tokeniser perplexity, silent contamination, missing significance test.
- The rest of this module makes each of those failure modes recoverable: task-specific metrics (02–05), benchmark selection (06), statistical rigour (07), contamination controls (08), and human evaluation (09).
