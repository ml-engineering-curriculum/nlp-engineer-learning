# Unanswerability and Abstention with SQuAD 2.0

## Motivation

Every QA system built in the previous chapters shares a common structural weakness: it *always* answers. On SQuAD 1.1, that is fine — every question in the dataset is answerable by construction. In production, it is a disaster. Users ask questions that are outside the corpus, phrased ambiguously, based on outdated assumptions, or plain unanswerable. A model that confidently emits *some* span for every one of them is worse than a model that says "I don't know" — because a wrong-but-confident answer erodes user trust and, in regulated domains, exposes the operator to liability.

Unanswerability is the calibration problem of QA. This chapter formalises it (SQuAD 2.0), covers the training and decoding changes required, and gives a practical protocol for tuning the abstention threshold.

## SQuAD 2.0 in one paragraph

Rajpurkar, Jia and Liang (["Know What You Don't Know: Unanswerable Questions for SQuAD"](https://arxiv.org/abs/1806.03822), *ACL 2018*) added ~50k unanswerable questions to SQuAD 1.1. The unanswerable questions were crowd-authored to look answerable — they reference entities and phrasings that appear in the paragraph, but no valid answer span exists. The model must produce an empty string for these. Evaluation reports EM and F1 (per chapter 04), with an empty predicted string counted as correct when the gold answer set is empty.

The formulation forces the model to distinguish "the answer is *this* span" from "no span is a valid answer". That is exactly the calibration signal a production system needs.

## The training-time change: null spans on the CLS token

The extractive-QA recipe (chapters 02–03) already has the machinery: sliding-window chunks that do not contain the answer are labelled `start_positions = end_positions = 0`, the position of `[CLS]`. On SQuAD 1.1 this only fires accidentally at chunk boundaries; on SQuAD 2.0 it fires deliberately for every unanswerable question.

At training time, this means:

- Every unanswerable question contributes `start = end = 0` for *every* chunk.
- The model learns to concentrate probability mass on the `[CLS]` position when no valid span exists.
- The training loss is unchanged — it is still averaged cross-entropy on start and end.

For encoder-decoder abstractive QA (chapter 05), the equivalent is a special target string like `"unanswerable"` or an empty string with a `<no_answer>` token, chosen consistently at training and inference. FLAN-T5 and other instruction-tuned models often already know these conventions from their pretraining mixture.

For decoder-only closed-book QA (chapter 06), the equivalent is a licensed abstention phrase in the prompt ("If you do not know, say 'I don't know.'") and possibly a fine-tuning pass where you show the model examples of correct abstention.

## The decoding-time change: null-score threshold

At inference on SQuAD 2.0 the reader produces two scores per chunk:

- `best_non_null_score = max over valid (start, end) of start_logit + end_logit` — as in chapter 03.
- `null_score = start_logits[CLS] + end_logits[CLS]` — the score the model assigns to "no answer".

The decision rule:

```
if best_non_null_score - null_score < threshold:
    predict ""      # abstain
else:
    predict best_non_null_span
```

`threshold = 0` means "predict a span whenever the model prefers *any* span to CLS". Positive `threshold` biases toward abstention (predict a span only when the model *strongly* prefers it). Negative `threshold` biases toward answering. This one hyperparameter is what you tune on the dev set.

## Tuning the threshold — the sweep protocol

`evaluate.load("squad_v2")` gives you a `best_f1_thresh` and a `best_exact_thresh` — the thresholds that maximise the corresponding metric on the supplied predictions. But those are dev-set-optimal by construction. For a defensible production pipeline:

1. Split your dev set into `dev-tune` (2/3) and `dev-holdout` (1/3), stratifying by answerable / unanswerable.
2. Run the reader on `dev-tune` and record `(best_non_null_score - null_score)` and the correct answer for every question.
3. Sweep `threshold` over a grid (e.g., `linspace(-10, 10, 201)`), compute the SQuAD 2.0 F1 at each value, pick the maximiser.
4. Fix that threshold. Evaluate once on `dev-holdout` to get an honest estimate. If the two numbers differ by more than 1 F1, your threshold is over-fit; widen `dev-tune` or coarsen the grid.
5. Report both `dev-holdout` F1 and (separately) answerable-F1 and unanswerable-accuracy.

Skipping the split — sweeping on the whole dev set and reporting `best_f1` — inflates numbers by 1–3 F1 and is a common mistake in shipping-team writeups.

## Metrics that surface the calibration story

The single SQuAD 2.0 F1 number hides the trade-off. Always report the following alongside it:

- **Answerable F1** (F1 on questions with a gold span, over predictions that are also non-empty).
- **Unanswerable accuracy** (fraction of gold-unanswerable questions where the model predicted the empty string).
- **Precision on non-empty predictions** (of the times the model answered, how often was it right?).
- **Coverage** (fraction of questions the model chose to answer).
- **Risk-coverage curve** (Geifman & El-Yaniv, ["Selective Classification for Deep Neural Networks"](https://arxiv.org/abs/1705.08500), *NeurIPS 2017*). Plot error rate as a function of coverage as you sweep the threshold. Report AUC of this curve as a threshold-independent calibration metric.

A model that scores 82 SQuAD 2.0 F1 with 60 % coverage and 95 % precision on non-empty answers is a very different product from one that scores 82 F1 with 100 % coverage and 82 % precision. The first is deployable to a compliance-sensitive domain; the second is not.

## Calibration beyond a fixed threshold

The null-score threshold is a global scalar. Real production traffic has *heterogeneous* calibration needs — some question types are always answerable (unit conversions), others rarely so (private-corpus queries against a public LLM). Techniques to reach for once the basic threshold is in place:

- **Temperature scaling** (Guo et al., ["On Calibration of Modern Neural Networks"](https://arxiv.org/abs/1706.04599), *ICML 2017*). Fit a single scalar $T$ on the dev set to divide the logits before softmax. Free 1–2 % ECE improvement, no metric loss.
- **Per-slice thresholds.** Cluster questions by embedding or type and set a per-cluster threshold. Requires more dev data but tracks user intent better than a global threshold.
- **Ensemble null-scores.** Run $k$ readers with different seeds; average their null-scores. Ensembling helps calibration more than it helps EM.
- **Verifier / auditor model.** A second model — sometimes an NLI model, sometimes a small LLM — reads `(question, context, predicted_answer)` and predicts *is this a correct answer?*. Threshold on that probability instead of the raw null-score. Marries with the faithfulness metric from chapter 05.
- **Calibrator on top of features.** A logistic regression on `[null_score_diff, answer_length, question_length, embedding_of_question]` fit on `dev-tune` and applied at inference. Common in production teams that can afford the extra plumbing.

## Beyond SQuAD 2.0: when is *this specific question* answerable?

SQuAD 2.0 is a training and evaluation *simulation* of unanswerability. In practice, the reasons a question is unanswerable in the wild are broader and each has a different fix:

- **Not in the corpus.** The user asked about a document that does not exist. Fix: retrieval abstention (retriever returns empty; reader must respect that).
- **Ambiguous.** The question has multiple valid answers. Fix: return all of them, or ask a clarifying question.
- **Requires reasoning the model cannot do.** Complex arithmetic, chains longer than the model handles. Fix: delegate to tool use (chapter 08 for multi-hop patterns), or abstain.
- **Refused by policy.** Legal, medical, or safety-sensitive question. Fix: policy layer above the reader that rewrites the response.
- **Requires up-to-date information the model does not have.** Fix: retrieval-augmented, and abstain when retrieval returns empty.

A production QA system almost always has *layers* of abstention — retrieval abstains, reader abstains, policy abstains — and treats abstention as a first-class response type with its own evaluation.

## Chapter summary

- SQuAD 2.0 formalises unanswerability by adding adversarially crafted unanswerable questions. Training uses `start = end = 0` on `[CLS]`; decoding thresholds `best_non_null_score - null_score` against a tuned constant.
- Tune the threshold on a `dev-tune` split, evaluate once on `dev-holdout`. Sweeping on the whole dev set and reporting `best_f1` is a common inflated-numbers mistake.
- Report answerable F1, unanswerable accuracy, precision on non-empty predictions, coverage, and the risk-coverage AUC — not just the aggregate SQuAD 2.0 F1.
- Beyond a global threshold, add temperature scaling, per-slice thresholds, or a verifier model. Beyond SQuAD 2.0, treat abstention as a first-class output that can originate at the retriever, the reader, or a policy layer.
