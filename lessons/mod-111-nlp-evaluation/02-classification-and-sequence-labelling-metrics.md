# Classification and Sequence-Labelling Metrics

## Motivation

Classification and sequence-labelling metrics look easy — accuracy, precision, recall, F1, and you're done — and are wrong more often than any other family of NLP metrics. The failure modes are all in the aggregation: micro vs. macro vs. weighted, subset vs. sample vs. label F1, `sklearn.metrics.f1_score` vs. `seqeval.metrics.f1_score`. A model that "beat SOTA on entity F1" often loses by 3–5 points when you re-score with the community-standard tool.

This chapter fixes the ambiguity. It walks through the classification metric zoo (accuracy, F1 with all averaging modes, MCC, calibration), then covers sequence-labelling — token-level accuracy for POS-style tagging and span-level entity F1 via `seqeval` for NER. The rule that unifies them: the metric you report must match the granularity of the correct-answer signal your users care about.

## Classification: the metric zoo

### Accuracy

$$\text{Accuracy} = \frac{1}{N} \sum_{i} \mathbb{1}[\hat{y}_i = y_i]$$

Simple, interpretable, and only trustworthy when the class distribution is roughly balanced. On an imbalanced binary dataset (90 % negative, 10 % positive), the always-negative classifier scores 0.9. Accuracy is the default only for balanced multi-class tasks with roughly equal per-class costs — MNIST, SST-2 in GLUE, RTE, PAWS. For anything with class imbalance or asymmetric costs, prefer F1 with a stated averaging mode.

### Precision, recall, and F1 per class

Per-class definitions, with `TP`, `FP`, `FN` counted against class $c$:

$$P_c = \frac{TP_c}{TP_c + FP_c}, \quad R_c = \frac{TP_c}{TP_c + FN_c}, \quad F1_c = \frac{2 P_c R_c}{P_c + R_c}$$

Always report per-class F1 alongside any averaged number. A "0.87 F1" without per-class breakdown hides which classes the model fails on — usually the low-prevalence ones, which are usually the ones you care about.

### Micro, macro, weighted, samples

The averaging mode changes what "the F1 score" means. `sklearn.metrics.f1_score` takes an `average=` argument that you should always pass deliberately.

- **`average="micro"`.** Sum `TP`, `FP`, `FN` across classes, compute F1 once. For multi-class single-label problems, micro-F1 equals accuracy. Its useful home is multi-label classification, where you want a global F1 over (example, label) pairs.
- **`average="macro"`.** Compute per-class F1 first, then arithmetic mean. Weighs every class equally regardless of prevalence — the right choice when you care about tail classes as much as head classes.
- **`average="weighted"`.** Per-class F1 weighted by class support. Reweights toward the head; halfway between micro and macro. Useful when class prevalence in test genuinely reflects prevalence you care about in production.
- **`average="samples"`.** For multi-label only: compute F1 per example (over its label set), then mean. Rewards per-example label-set correctness.
- **`average=None`.** Returns per-class F1. Always compute this even if you only report the average — it is the diagnostic.

The macro/micro asymmetry is the source of most reporting bugs. GLUE, SuperGLUE, and most academic multi-class benchmarks report accuracy or macro-F1; production imbalanced classifiers should default to macro-F1 with per-class breakout.

```python
from sklearn.metrics import (
    classification_report, f1_score, matthews_corrcoef, confusion_matrix,
)

print(classification_report(y_true, y_pred, digits=4))   # per-class P/R/F1 + macro/weighted
macro_f1 = f1_score(y_true, y_pred, average="macro")
mcc      = matthews_corrcoef(y_true, y_pred)
cm       = confusion_matrix(y_true, y_pred, labels=class_names)
```

### Matthews correlation coefficient (MCC)

$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

MCC ∈ [−1, 1]; 0 is chance, 1 is perfect. Unlike F1, MCC uses all four cells of the confusion matrix and behaves robustly under class imbalance. GLUE reports MCC as the primary metric for CoLA (Warstadt et al., ["Neural Network Acceptability Judgments"](https://arxiv.org/abs/1805.12471), *TACL 2019*) precisely because the acceptability task is skewed and MCC is stable there. Report MCC alongside macro-F1 when the balance is bad or costs are asymmetric.

### Calibration (ECE, reliability diagrams)

For probability-emitting classifiers (basically all fine-tuned Transformers), the metric shape you should also report is *calibration*: does a predicted 0.8 probability actually correspond to an 80 % empirical accuracy over the bin of predictions with that probability?

**Expected Calibration Error (ECE)** (Guo et al., ["On Calibration of Modern Neural Networks"](https://arxiv.org/abs/1706.04599), *ICML 2017*) is the standard summary — bin predictions by predicted confidence, compute `|accuracy_in_bin − mean_confidence_in_bin|`, take the support-weighted mean over bins:

$$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{N} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$

Modern deep classifiers are systematically over-confident; ECE quantifies the gap. Report ECE (typically 15 bins) any time downstream decisions use the predicted probability directly — routing thresholds, abstain-below-p rules, ensemble weights.

```python
def ece(probs, labels, n_bins=15):
    import numpy as np
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc  = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(acc[mask].mean() - conf[mask].mean())
    return ece
```

For calibration *fixes* (temperature scaling, Platt scaling, isotonic regression) see the same Guo et al. paper.

## Sequence labelling: token-level vs. span-level

Sequence-labelling tasks come in two flavours and each takes a different metric.

**Token-level tasks.** POS tagging, morphological tagging, dependency-arc labelling — every token gets a label, the label is fine-grained, and there is no notion of "an entity that spans multiple tokens." The right metric is per-token accuracy or per-tag macro-F1. Use `sklearn.metrics` directly on the flattened list of `(gold_tag, pred_tag)` pairs.

**Span-level tasks.** NER, chunking, event-trigger detection, keyphrase extraction — the correctness unit is the *span*, not the token. A prediction that gets the type right but the boundary wrong ("Barack Obama" vs. "Barack") should score zero. A prediction that gets the boundary right but the type wrong (PER vs. ORG) should also score zero. Per-token accuracy over BIO tags is *not* the right metric — it gives partial credit for near-misses in ways the downstream user cannot cash in.

## `seqeval`: the community-standard span-level scorer

`seqeval` (Nakayama, [seqeval](https://github.com/chakki-works/seqeval)) is the canonical span-level scorer. It takes lists of BIO/BIOES/IOB2/IOBES tag sequences, extracts the entity spans, and scores exact span-and-type matches:

$$P = \frac{|\text{predicted spans} \cap \text{gold spans}|}{|\text{predicted spans}|}, \quad R = \frac{|\text{predicted spans} \cap \text{gold spans}|}{|\text{gold spans}|}$$

where the intersection is over `(start, end, type)` triples. Any published NER F1 number is expected to be `seqeval` micro-F1 unless stated otherwise — CoNLL-2003, OntoNotes, WNUT-17, MultiCoNER all use `seqeval`.

```python
from seqeval.metrics import (
    classification_report, f1_score, precision_score, recall_score,
)
from seqeval.scheme import IOB2

y_true = [["B-PER", "I-PER", "O", "B-LOC"], ["O", "B-ORG", "O"]]
y_pred = [["B-PER", "I-PER", "O", "B-LOC"], ["O", "B-ORG", "O"]]

print(f1_score(y_true, y_pred, mode="strict", scheme=IOB2))
print(classification_report(y_true, y_pred, mode="strict", scheme=IOB2, digits=4))
```

Two configuration decisions matter:

- **`mode="strict"` and `scheme=IOB2` (or `IOBES` etc.).** Without these, `seqeval` uses its legacy default which is more permissive on malformed tag sequences and can inflate F1 by 1–3 points on models that emit BIO violations. Pass them explicitly and match the scheme your data uses. See Nakayama's discussion in the repo and Nam & Nakayama's clarification for strict mode.
- **Averaging.** `seqeval` defaults to `average="micro"` — sum spans across all entity types then compute one P/R/F1. For per-type reporting use `classification_report` or `average=None`. For imbalanced entity type distributions (LOC dominates PER dominates ORG dominates MISC in a typical news set), report both micro and macro.

### Common `seqeval` pitfalls

- **Mismatched tag schemes.** Your data is IOB2 (`B-` starts a new entity, `I-` continues; `O` outside) but the model emits IOB1 (`B-` only when a same-type entity follows another same-type entity). `seqeval` in `strict` mode will flag this; in default mode it silently coerces and produces wrong spans.
- **BIO violations in output.** A model that emits `I-PER` without a preceding `B-PER` produces an invalid span sequence. `mode="strict"` counts this as no-span; the default mode may still extract a span. The strict number is the honest one.
- **Sub-token vs. word-level scoring.** Transformers emit per-subword tags. You must align back to word-level (usually taking the tag of the *first* subword of each word) *before* passing to `seqeval`. Scoring at the subword level over-counts entities and gives a different number.
- **Sentence boundaries.** `seqeval` treats each list as one document. Cross-sentence entities need consistent handling (usually: split at sentence boundaries so no entity crosses).
- **Nested entities.** `seqeval` cannot score nested spans — an "Apple Store in Palo Alto" with a nested LOC "Palo Alto" and an ORG "Apple Store in Palo Alto" requires a nested-NER-aware scorer. See mod-104 chapter 05 on nested NER.

## Multi-label classification

For tasks where each example may carry zero, one, or many labels (topic tagging, ICD coding, hate-speech attribute tagging), the metric surface expands.

- **Micro-F1 over (example, label) pairs.** Rewards precision and recall globally; the standard multi-label default.
- **Macro-F1 over labels.** Per-label F1, then mean. Necessary when the label distribution is long-tailed (ICD-10 has tens of thousands of codes) — micro-F1 will be dominated by the head labels.
- **Subset accuracy (a.k.a. exact-match ratio).** The fraction of examples whose full predicted label set exactly equals the gold set. Harsh — one missed label is a full miss. Useful when the label set as a whole is the actionable unit.
- **Hamming loss.** Fraction of (example, label) pairs where prediction disagrees with gold. Cheap complement to micro-F1.
- **Coverage / label ranking metrics.** For probability-emitting multi-label models, report Precision@k, Recall@k, and label-ranking average precision (LRAP) — same shape as retrieval.

## Statistical significance for classification

Two classifiers with a 1-point macro-F1 delta on a 2 000-example dev set may or may not be significantly different. The standard tests:

- **Bootstrap resample** the test set 1 000 times with replacement; compute the metric on each resample; take the 2.5th and 97.5th percentiles as the 95 % CI. If the CIs overlap substantially, the difference is not significant.
- **Paired bootstrap** for two systems on the same test set — resample the *same* indices for both systems each iteration, compute `metric_A - metric_B`, take the CI of that difference. If the CI on the difference crosses zero, not significant.
- **McNemar's test** for two binary classifiers on the same test set — the 2×2 disagreement contingency table gives a chi-square statistic (Dietterich, ["Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms"](https://direct.mit.edu/neco/article/10/7/1895/6224), *Neural Computation 1998*). Cheap and appropriate for paired binary predictions.
- **Approximate randomisation (paired permutation)** for arbitrary metrics — swap predictions between systems with probability 0.5 many times, count how often the shuffled difference exceeds the observed. Chapter 07 covers the details.

Report the confidence interval on the metric and on the delta, not just the point estimate. The point estimate is a summary statistic on a sample; the CI tells the reader how much the sample constrains the true value.

## Per-slice reporting

Aggregate F1 hides where the classifier is broken. Standard slices worth reporting:

- **Per class.** The default diagnostic — always.
- **Per input length bucket.** Short-text classifiers often behave differently from long-text on the same task.
- **Per input source.** If your test set mixes tweet-length social text with news paragraphs, slice by source.
- **Per demographic / dialect group** where available (chapter 09 formalises this). A hate-speech classifier that scores 0.87 F1 overall but 0.62 on tweets in African American English is not a 0.87-F1 classifier for its full audience.
- **Per confidence bin.** For probability-emitting models, per-bin accuracy (feeds into the reliability diagram) shows where the model is over-confident.
- **Per confusion pair.** The top-5 confused class pairs in the confusion matrix — often surfaces annotation-ambiguity in the labels themselves.

## Chapter summary

- Classification: report per-class F1 plus one averaged number, always. Micro = accuracy for multi-class single-label; macro treats all classes equally; weighted reweights by support. Pass `average=` deliberately.
- MCC is the robust-to-imbalance summary; report alongside macro-F1 for skewed data or asymmetric costs. GLUE's CoLA reports MCC as primary.
- For probability-emitting classifiers, report ECE (Expected Calibration Error) — modern deep classifiers are systematically over-confident and downstream thresholds break silently.
- Token-level tasks (POS) use per-token accuracy or macro-F1. Span-level tasks (NER, chunking) require **span-level** scoring — `seqeval` with `mode="strict"` and the explicit `scheme=` for your data. Never report per-token F1 as "entity F1."
- Multi-label: micro-F1 + macro-F1 + (sometimes) subset accuracy. Micro dominates by head labels; macro rewards tail labels; subset accuracy is the strictest.
- Every metric report needs bootstrap CIs (or paired bootstrap for system-vs-system deltas), per-class breakdowns, and per-slice breakdowns. A single macro-F1 with no CI hides everything worth acting on.
