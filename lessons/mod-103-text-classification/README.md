# mod-103 · Text Classification End-to-End

Baselines, encoder fine-tuning, imbalance handling, calibration, and multilingual classification — the workhorse NLP task from a production standpoint.

**Estimated effort:** 12 hours

## Learning objectives

- Train logistic-regression, fastText, and fine-tuned encoder (BERT / RoBERTa / DeBERTa / XLM-R) text classifiers.
- Handle binary, multi-class, multi-label, and hierarchical classification with the right loss, head, and metric per case.
- Diagnose and mitigate class imbalance with weighting, sampling, focal loss, and threshold tuning.
- Calibrate probabilistic outputs (temperature scaling, isotonic regression) and pick thresholds for cost-sensitive deployment.
- Train and evaluate multilingual classification with XLM-R and language-aware sampling.

## Chapters

1. [Why text classification is still the workhorse task](01-why-text-classification-is-still-the-workhorse-task.md)
2. [TF-IDF plus logistic regression and linear SVM baselines](02-tfidf-logistic-regression-and-linear-svm-baselines.md)
3. [fastText: linear classifier with subword embeddings](03-fasttext-linear-classifier-with-subword-embeddings.md)
4. [Fine-tuning encoders: BERT, RoBERTa, DeBERTa, XLM-R](04-fine-tuning-encoders-bert-roberta-deberta-xlmr.md)
5. [Heads, losses, and metrics for binary, multi-class, multi-label, and hierarchical tasks](05-heads-losses-and-metrics-per-task-shape.md)
6. [Class imbalance: weighting, sampling, and focal loss](06-class-imbalance-weighting-sampling-and-focal-loss.md)
7. [Calibration and threshold tuning for cost-sensitive deployment](07-calibration-and-threshold-tuning.md)
8. [Multilingual classification with XLM-R and language-aware sampling](08-multilingual-classification-with-xlm-r.md)

## Exercises

- [exercise-01 · Baseline bake-off: logreg, fastText, encoder](exercises/exercise-01-baseline-bake-off-logreg-fasttext-encoder.md)
- [exercise-02 · Multi-label and hierarchical classification](exercises/exercise-02-multi-label-and-hierarchical-classification.md)
- [exercise-03 · Calibration and threshold tuning](exercises/exercise-03-calibration-and-threshold-tuning.md)
- [exercise-04 · Multilingual classification with XLM-R](exercises/exercise-04-multilingual-classification-with-xlm-r.md)

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.

## Also in this module

- `labs/` — long-form hands-on labs (added on a later authoring cycle).
- `quizzes/` — knowledge checks (added on a later authoring cycle).
- [`resources.md`](resources.md) — primary sources, standards, and further reading.
