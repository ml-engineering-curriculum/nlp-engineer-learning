# Why Text Classification Is Still the Workhorse Task

## Motivation

Ask a foundation-model engineer what production NLP looks like and you may hear "prompt an LLM." Look inside any large product and you will find dozens of dedicated text classifiers ahead of, alongside, and downstream of every LLM: spam and abuse triage, intent routing, sentiment monitoring, PII and policy detection, ticket categorisation, review moderation, ad quality, search query classification, click-log labelling. These jobs are millions to billions of decisions per day where a 5 ms model beats a 500 ms LLM on cost, latency, and — with rare exceptions — quality.

This module is about that layer. It builds the ladder every serious NLP engineer needs to climb before defaulting to fine-tuning or prompting:

1. **A linear baseline** — TF-IDF plus logistic regression or linear SVM. Trained in seconds, memory-mapped in kilobytes, still competitive on many product tasks.
2. **A hashed-embedding linear classifier** — Meta's fastText. Character n-grams handle morphology and OOV words, hierarchical softmax scales to thousands of labels.
3. **A fine-tuned encoder** — BERT, RoBERTa, DeBERTa, XLM-R. The default when the baselines cap out, and the substrate the rest of the module leans on.
4. **The task-shape decisions** — binary, multi-class, multi-label, hierarchical — each with its own head, loss, and metric.
5. **The correctness-under-imbalance stack** — reweighting, resampling, focal loss, and threshold tuning.
6. **The calibration and decision layer** — turning logits into probabilities you can trust, and probabilities into decisions that respect an actual cost function.
7. **The multilingual layer** — XLM-R plus language-aware sampling for products that ship in more than one language.

If module 101 was tokens and module 102 was classical NLP, this module is the first place you assemble those pieces into a shippable neural component.

## The ladder, and why you climb it in order

Every classification project has a *right rung*. The failure mode is skipping straight to fine-tuning, spending a week on Trainer configs, and discovering that a 200-line `sklearn` pipeline is within 1 F1 point on a real business metric — but 100× cheaper to serve.

The ladder:

| Rung | Model | When it wins | When it caps out |
| --- | --- | --- | --- |
| 0 | Regex / rule list | Tiny label set, unambiguous surface cues, need auditability | Every case with paraphrasing, long tail, or novel wording |
| 1 | TF-IDF + logistic regression / linear SVM | Balanced datasets ≥ ~1k examples per class, English or well-tokenised languages, tabular-like documents | Semantics beyond bag-of-n-grams (negation, entailment, sarcasm) |
| 2 | fastText | Millions of documents, thousands of labels, morphologically rich languages, need CPU-only inference | Subtle compositional semantics, tasks where subword-averaging washes out context |
| 3 | Fine-tuned encoder (BERT / RoBERTa / DeBERTa) | You have a GPU budget and ≥ ~500 labelled examples per class; task requires paraphrase invariance or entailment | Very low-resource labelling, tasks where zero-shot or few-shot from an LLM is competitive and cheaper end-to-end |
| 4 | Multilingual encoder (XLM-R, mDeBERTa) | Same as rung 3 but across many languages, with a lopsided data mix | Same limits as rung 3, plus the "curse of multilinguality" for high-resource languages |
| 5 | LLM zero/few-shot | Very small labelled sets, exploratory taxonomies, or tasks where the label set changes weekly | Latency, cost, determinism, auditability, sustained throughput |

You climb this ladder by measuring: build the cheaper rung, hold out a real evaluation set, then decide whether the delta is worth the operational cost of the next rung. Chapter 02 through 04 build rungs 1–4 concretely.

## What "text classification" actually covers

The five task shapes you will build heads for in chapter 05:

- **Binary.** One decision boundary. Spam vs. ham, toxic vs. non-toxic, refund vs. no-refund. Sigmoid + BCE.
- **Multi-class, single-label.** Exactly one of `K` classes. News topic, review star rating, intent slot. Softmax + cross-entropy.
- **Multi-label.** Any subset of `K` labels can be true simultaneously. Toxic-comment tags (`toxic`, `insult`, `threat`), movie genres, medical codes. `K` independent sigmoids + BCE.
- **Hierarchical.** Labels form a taxonomy (tree or DAG). Product category (`Home > Kitchen > Cookware`), legal document type, medical ontology (ICD, MeSH). Requires either a flat classifier over leaves plus post-hoc consistency, or a structured loss that respects the hierarchy. See Silla & Freitas, ["A Survey of Hierarchical Classification across Different Application Domains"](https://link.springer.com/article/10.1007/s10618-010-0175-9), *Data Mining and Knowledge Discovery*, 2011, for the canonical taxonomy of these approaches.
- **Ranking / relevance classification.** Effectively binary or ordinal, but scored by ranking metrics (nDCG, MAP). Learning-to-rank belongs in another track; this module covers the classification perspective on it.

Each shape has a *matching* loss, output head, and evaluation metric. Mixing them up (softmax on a multi-label problem, accuracy on an imbalanced binary problem) is the most common source of "the model looks great but the product regresses" incidents.

## What makes classifiers hard in production

Every rung of the ladder shares a common set of failure modes. This module builds the vocabulary for each:

- **Class imbalance.** Real label distributions are power-law. A 99 %-negative binary problem trained with plain cross-entropy converges to "always predict negative." Chapter 06 covers the weighted-loss, resampling, focal-loss, and threshold-tuning fixes.
- **Miscalibrated probabilities.** Fine-tuned transformers systematically overconfident on their softmax outputs (Guo et al., ["On Calibration of Modern Neural Networks"](https://arxiv.org/abs/1706.04599), *ICML 2017*). A downstream service reading `p(class) > 0.9` is trusting a number the model did not earn. Chapter 07 covers temperature scaling and isotonic regression.
- **The wrong metric.** Accuracy hides class imbalance. Micro-F1 hides tail-class regressions. Macro-F1 rewards tail classes equally, which may or may not match your product. Chapter 05 walks the choices.
- **Decision-threshold drift.** A model trained six months ago at `p > 0.5` may need `p > 0.62` today because the base rate shifted. Chapter 07 covers cost-sensitive threshold tuning as an ongoing operational task, not a one-shot.
- **Language mix.** A single-language classifier fed multilingual traffic fails silently, usually as false negatives on a language you don't speak. Chapter 08 covers XLM-R and language-aware sampling.
- **Label noise.** Human annotators disagree. Kappa below 0.7 caps model quality regardless of architecture. This is a data-engineering problem covered in mod-110, but classifiers surface it: unstable macro-F1 across seeds is almost always a labelling problem, not a modelling one.

## Where LLMs fit — and where they do not

A reasonable reader in 2026 asks: why train a classifier at all when I can prompt an LLM?

For prototypes with 10 labelled examples and a taxonomy that changes weekly, LLMs are the right choice. Once the label set stabilises, the input distribution stabilises, and traffic exceeds a few thousand requests per day, dedicated classifiers become cheaper by two-to-three orders of magnitude and more predictable in tail latency. The industry pattern is:

- **LLM to bootstrap labels.** Use an LLM to label a seed set, review with humans, distil into a small classifier.
- **LLM to fill the tail.** Route the head of the distribution to a fast classifier, escalate uncertain cases to an LLM.
- **Classifier as the guardrail on an LLM.** A dedicated toxicity/PII/policy classifier gates every LLM output — this is a classification task, not a generation task, and needs a calibrated probability.

None of these patterns retire text classification. They change *what* you classify.

## How to read the rest of the module

- Chapters **02** and **03** build the linear baselines (TF-IDF + logreg/SVM, then fastText). You will finish able to justify why either is your first bake-off entry.
- Chapter **04** covers the fine-tuning stack: which encoder to pick, how to configure a run, how to avoid the reproducibility pitfalls.
- Chapter **05** enumerates task shapes with their heads, losses, and metrics — the reference chart you will come back to.
- Chapter **06** is the imbalance chapter: how to diagnose it, and the ordered toolbox of fixes.
- Chapter **07** is the probability-calibration and cost-sensitive decision chapter.
- Chapter **08** closes with multilingual classification and language-aware training.

By the end you should be able to walk into a new classification project, pick the right rung, train it, calibrate it, and defend the threshold you shipped.

## Chapter summary

- Text classification is the highest-volume production NLP task, and the ladder from tf-idf up to fine-tuned encoders and LLMs each has a rung that wins on some real product.
- Task shape (binary, multi-class, multi-label, hierarchical) determines head, loss, and metric — mixing them up is the top source of shipped-but-broken classifiers.
- The recurring failure modes — class imbalance, miscalibrated probabilities, wrong metric, decision-threshold drift, language mix, label noise — are what the rest of the module addresses systematically.
- LLMs bootstrap and escalate; they rarely replace dedicated classifiers in high-throughput production paths.
