# Heads, Losses, and Metrics for Binary, Multi-class, Multi-label, and Hierarchical Tasks

## Motivation

Nine times in ten, the wrong loss and the wrong metric explain more of a broken classifier than the wrong model. A multi-label problem trained with softmax cross-entropy will look like it works — it will emit exactly one label per example — and it will silently drop the rest. A skewed binary problem evaluated on accuracy will read as 99 % right while missing every positive.

This chapter is the reference chart for the four task shapes you will meet: binary, multi-class, multi-label, and hierarchical. Each has a matching head, loss, and family of metrics. Learn to pattern-match — most of the failure modes above come from choosing the wrong pattern in the first place.

## Binary classification

**One label. One decision boundary.** Spam vs. ham, toxic vs. non-toxic, refund vs. no-refund.

### Head and loss

- **Head**: `nn.Linear(hidden_size, 1)`. `AutoModelForSequenceClassification` will use this shape when `num_labels=1` and `problem_type="single_label_classification"` is not overridden — or you can set `num_labels=2` and get a softmax over two classes. Both are valid; the sigmoid+BCE variant plays nicer with threshold tuning (chapter 07).
- **Loss**: `BCEWithLogitsLoss` — binary cross-entropy applied to raw logits. Numerically stable, handles class weights via `pos_weight`:

  ```python
  loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.0]))  # up-weight positives 9×
  ```

- **Alternative**: softmax over 2 classes with `CrossEntropyLoss`. Equivalent in expressiveness, easier to compare against multi-class heads.

### Metrics

Accuracy is meaningful only for balanced binary tasks. The default reporting set:

- **Precision, recall, F1** at your chosen threshold (usually 0.5, tuned per chapter 07).
- **AUROC** — area under the ROC curve, threshold-independent, decent for balanced data.
- **AUPRC** (average precision) — threshold-independent, the right choice for imbalanced binary problems (Saito & Rehmsmeier, ["The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets"](https://doi.org/10.1371/journal.pone.0118432), *PLoS ONE 2015*).
- **Confusion matrix** — always. Reporting only F1 without the confusion matrix hides which side you erred on.

### The 0.5 threshold trap

`predict_proba > 0.5` is a *convention*, not a decision rule. For any imbalanced or cost-sensitive problem, the right threshold is derived from a business cost matrix; chapter 07 covers this. Never accept 0.5 as a default without justifying it.

## Multi-class, single-label

**Exactly one of `K` classes per example.** News topic, review star rating, intent slot.

### Head and loss

- **Head**: `nn.Linear(hidden_size, K)`.
- **Loss**: `CrossEntropyLoss` — softmax + negative log-likelihood in one numerically-stable operator.
- **Optional**: `label_smoothing=0.1` for regularisation on tasks with high-quality labels but noisy input distribution (Szegedy et al., ["Rethinking the Inception Architecture for Computer Vision"](https://arxiv.org/abs/1512.00567), *CVPR 2016*; adopted into NLP by Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), *NeurIPS 2017*). Trades a small amount of calibration quality for a small F1 gain; disable if you plan to run temperature scaling in chapter 07.

### Metrics

- **Accuracy** — meaningful when classes are balanced or you specifically care about overall correct-rate.
- **Top-`k` accuracy** — for taxonomies where "second guess is right" counts (image tagging, product categorisation).
- **Macro-F1** — arithmetic mean of per-class F1. Weights tail classes equally, so a single-class regression hurts the number even if that class is rare.
- **Micro-F1** — pooled TP/FP/FN across classes. For single-label multi-class, micro-F1 equals accuracy. Reporting micro-F1 for single-label multi-class problems is common but not informative.
- **Weighted-F1** — per-class F1 weighted by support. A middle-ground between macro and micro; useful when you care about tail classes proportionally.

Pick macro-F1 for the "hold up under any class regressing" story, weighted-F1 for the "matches overall correct-rate but with per-class visibility" story. Never report a single number without also reporting the per-class breakdown from `sklearn.metrics.classification_report`.

### Confusion matrices actually help

Off-diagonal mass concentrated in one cell usually indicates a labelling policy problem or a taxonomy collapse (e.g., `bug` vs. `defect`). Interpret them as feedback to the *dataset*, not just to the model.

## Multi-label classification

**Any subset of `K` labels can be true.** Toxic-comment tags, movie genres, medical codes, ArXiv paper categories.

### Head and loss

- **Head**: `nn.Linear(hidden_size, K)` — same shape as multi-class, different loss.
- **Loss**: `BCEWithLogitsLoss` — `K` independent binary decisions on the same shared representation. In Hugging Face:

  ```python
  model = AutoModelForSequenceClassification.from_pretrained(
      MODEL, num_labels=K,
      problem_type="multi_label_classification",
  )
  ```

  The `problem_type` string switches the loss to `BCEWithLogitsLoss` and expects `labels` as float tensors of shape `(batch, K)` with `0.` / `1.` entries.

- **The mistake**: using `CrossEntropyLoss` on multi-label data. The softmax forces the probabilities to sum to 1, so predictions are always exactly one label. This is one of the most common silent bugs in text-classification code.

### Metrics

Multi-label evaluation is a family of choices; report several because they measure different things:

- **Subset accuracy (exact match)** — fraction of examples where the predicted label set exactly equals the true label set. Extremely harsh; treat as a floor.
- **Micro-F1** — pool TP/FP/FN across all `(example, label)` pairs. Emphasises frequent labels.
- **Macro-F1** — per-label F1 averaged uniformly. Emphasises tail labels equally.
- **Sample-averaged F1** — per-example F1 averaged across examples. Emphasises examples that got most labels right.
- **Hamming loss** — fraction of `(example, label)` pairs predicted wrong. A gentler metric than subset accuracy.
- **mAP / AUPRC per label** — the threshold-independent multi-label answer. Compute AUPRC per label, then macro/micro-average.

Rule: report macro-F1 for tail-class fairness, micro-F1 for head-class throughput, and one threshold-independent number (mAP) so calibration and threshold choices can be evaluated separately (chapter 07).

### Label correlation and independence

`BCEWithLogitsLoss` treats labels as independent conditional on the input representation, which is *not the same* as treating them as independent. The shared encoder does learn label correlations. When correlations are strong and you need them respected exactly, structured approaches help:

- **Classifier chains** (Read et al., ["Classifier Chains for Multi-label Classification"](https://link.springer.com/chapter/10.1007/978-3-642-04174-7_17), *ECML-PKDD 2009*) — predict label `k` conditioned on labels `< k`.
- **Label-embedding attention** (Xiao et al., ["Label-Specific Document Representation for Multi-Label Text Classification"](https://aclanthology.org/D19-1044/), *EMNLP-IJCNLP 2019*).
- **Structured prediction** on the label graph — rare in NLP, common in image tagging.

For most text-classification multi-label problems, the shared-encoder + `K` sigmoids setup is the pragmatic default.

### Threshold selection is now `K` thresholds

Multi-label deployment picks one threshold per label, not a single global one. Each label has its own base rate and cost profile. Chapter 07 covers per-label threshold tuning.

## Hierarchical classification

**Labels form a taxonomy** — a tree or DAG. Product category (`Home > Kitchen > Cookware > Non-stick pans`), medical ontology (ICD-10, MeSH), Amazon Browse Nodes, Web-of-Science topics.

The canonical taxonomy of approaches is Silla & Freitas, ["A Survey of Hierarchical Classification across Different Application Domains"](https://link.springer.com/article/10.1007/s10618-010-0175-9), *Data Mining and Knowledge Discovery*, 2011. The three families:

### Flat classification with post-hoc consistency

Train a single flat multi-class or multi-label classifier over all leaves (or all nodes), ignore the hierarchy at training time, and enforce hierarchy constraints at inference:

- Predict leaves, propagate up to ancestors.
- Reject predictions that violate parent-child rules.
- Cheap, easy, and often the right first cut.

Cost: doesn't leverage hierarchical structure to share signal, wastes tail-class capacity.

### Local classifiers

Train a separate classifier at each internal node (or per level, or per parent). At inference, walk the tree top-down. Variants:

- **Local classifier per node (LCN)** — one binary classifier per node.
- **Local classifier per parent node (LCPN)** — one multi-class classifier per parent, choosing among its children.
- **Local classifier per level (LCL)** — one multi-class classifier per depth level.

Trade-off: more models to train and serve, but each model has a simpler decision, and error propagation down the tree is a real concern.

### Global classifiers

A single model whose output structure respects the hierarchy. Modern implementations:

- **Hierarchical softmax** in fastText (chapter 03) is the simplest example — a tree of binary decisions.
- **Hierarchical text classification with attention over label taxonomies** — Zhou et al., ["Hierarchy-Aware Global Model for Hierarchical Text Classification"](https://aclanthology.org/2020.acl-main.104/), *ACL 2020* (HiAGM), and follow-ups.
- **Loss functions that penalise hierarchy violations** — e.g., using shortest-path distance in the taxonomy as a weighting term (Kosmopoulos et al., ["Evaluation Measures for Hierarchical Classification: a unified view and novel approaches"](https://arxiv.org/abs/1306.6802), *Data Mining and Knowledge Discovery 2015*).

### Metrics for hierarchical tasks

Plain flat F1 over leaves under-counts: predicting a *sibling* of the true leaf is worse than predicting the true leaf, but predicting the true *parent* is better than predicting an unrelated leaf. The standard family:

- **Hierarchical precision / recall / F1** — Kiritchenko, Matwin & Famili's construction: treat a prediction as the set of all ancestors of the predicted node, ground-truth as the set of all ancestors of the true node, then compute precision, recall, F1 over these sets. Reference: Kiritchenko, Matwin, Nock & Famili, ["Learning and Evaluation in the Presence of Class Hierarchies: Application to Text Categorization"](https://link.springer.com/chapter/10.1007/11766247_34), *AI 2006*. Kosmopoulos et al. (above) generalises this and provides Python code.
- **Distance-based metrics** — e.g., the taxonomic distance between predicted and true nodes, aggregated across examples.
- **Consistency-constrained accuracy** — only count as correct if the predicted path is a valid root-to-node walk *and* matches the true path.

Report at least one hierarchy-aware metric alongside flat F1; without it, you cannot see the difference between "close miss" and "wrong subtree."

## Ordinal classification: the special case

Ordinal targets — a 1–5 star rating, a severity score, an age band — sit between multi-class and regression. They have an ordering, so predicting `4` when the answer is `5` is better than predicting `1`.

Options in order of increasing sophistication:

- **Multi-class softmax + macro-F1.** Simplest, ignores ordering. Baseline.
- **Regression + rounding.** Treat the target as continuous, train with MSE, round to nearest integer at inference. Wins when the ordering is fine-grained.
- **Cumulative link / ordinal loss.** `K-1` binary heads predicting `P(y ≥ k)` (Frank & Hall, 2001). The right choice when ordering carries most of the signal.

For evaluation, add **quadratic weighted kappa** or **mean absolute error** to the reporting so you can see the "how far off" story that F1 discards.

## The reference chart

| Task shape | Head | Loss | Primary metric | Secondary metrics |
| --- | --- | --- | --- | --- |
| Binary | `Linear(H, 1)` | `BCEWithLogitsLoss` | F1 at tuned threshold | AUROC, AUPRC, confusion matrix |
| Multi-class single-label | `Linear(H, K)` | `CrossEntropyLoss` | Macro-F1 | Accuracy, per-class F1, confusion matrix |
| Multi-label | `Linear(H, K)` | `BCEWithLogitsLoss` (`problem_type="multi_label_classification"`) | Micro-F1 and macro-F1 | mAP, Hamming loss, per-label PR curves |
| Hierarchical | Flat / local / global (see above) | CE / BCE / structured | Hierarchical F1 | Flat macro-F1, consistency rate |
| Ordinal | `Linear(H, K-1)` cumulative | Cumulative link | Quadratic weighted kappa | MAE, macro-F1 |

Reach for this chart before writing the training loop, not after.

## Chapter summary

- Task shape determines head, loss, and metric — mixing them up (softmax on multi-label, accuracy on imbalanced binary, flat F1 on hierarchical) is the single most common source of shipped-but-broken classifiers.
- Binary tasks need BCE + a threshold tuned to a real cost function (chapter 07), not 0.5 by convention.
- Multi-class tasks need `CrossEntropyLoss` and a macro-vs-micro-F1 decision framed by whether tail classes matter.
- Multi-label tasks need `BCEWithLogitsLoss` and per-label thresholds; report at least one threshold-independent metric (mAP).
- Hierarchical tasks need a decision among flat / local / global architectures and *must* be evaluated with a hierarchy-aware metric; Silla & Freitas 2011 is the canonical reference.
- Chapter 06 addresses what to do when the class distribution is not what the loss above implicitly assumes.
