# exercise-02: Multi-Label and Hierarchical Classification

**Estimated effort:** 3 hours

## Objective

Build one multi-label classifier and one hierarchical classifier, evaluate each with the metrics that actually respect the task shape, and diagnose the failure modes each family introduces. By the end you should be able to tell someone why softmax was wrong for the multi-label task and why flat macro-F1 was wrong for the hierarchical one — with concrete numbers.

## Prerequisites

- Chapters [04](../04-fine-tuning-encoders-bert-roberta-deberta-xlmr.md) and [05](../05-heads-losses-and-metrics-per-task-shape.md).
- Python 3.10+; `transformers`, `datasets`, `evaluate`, `scikit-learn`, `torch`, `numpy`.
- A GPU is strongly recommended for the encoder fine-tunes.

## Part A — Multi-label classification

### Dataset

Pick one of:

- **`go_emotions`** (Demszky et al., ["GoEmotions: A Dataset of Fine-Grained Emotions"](https://arxiv.org/abs/2005.00547), *ACL 2020*) — 27 emotion labels + neutral, multi-label. `datasets.load_dataset("go_emotions", "raw")` returns raw labels; use the `"simplified"` config for the reduced-label version.
- **`reuters21578`** — the classic multi-label news dataset. Available via `nltk.corpus.reuters` or `datasets.load_dataset("reuters21578")`.
- **`jigsaw_toxicity_pred`** (the toxic comment classification dataset from the Jigsaw Kaggle competition) — 6 toxicity labels, multi-label.
- Any comparable dataset with per-example label counts averaging ≥ 1.2 (i.e., a real multi-label distribution, not disguised multi-class).

### Requirements

1. **Fine-tune a base-size encoder** (`distilroberta-base` or `roberta-base`). Set `problem_type="multi_label_classification"` on `from_pretrained` — verify by printing that `model.config.problem_type` is `"multi_label_classification"` and that the loss is BCE.
2. **Train with the right label shape**: `float32` labels of shape `(batch, K)`.
3. **Also train a broken baseline** with `CrossEntropyLoss` on the same data (arg-max the multi-hot label vector to force one class) to observe the failure mode. Report predictions per example (always exactly one label) vs. the correct model (variable number of labels).
4. **Evaluate the correct multi-label model** on the test set with:
   - Micro-F1, macro-F1, sample-averaged F1.
   - Subset (exact-match) accuracy.
   - Hamming loss.
   - Per-label AUPRC (`sklearn.metrics.average_precision_score` with `average=None`).
5. **Threshold selection per label.** Do not use `0.5` globally. Fit a per-label threshold on the validation split that maximises per-label F1 (see chapter 05's snippet). Report both the pre- and post-threshold-tuning micro/macro-F1.
6. **Confusion analysis**: pick the label with worst macro-F1 and describe what's happening — is it low-support, label-noise, or overlap with a semantically similar label?

### Deliverables

- Two model directories (correct multi-label + broken softmax baseline).
- A metrics table: pre-tuning and post-tuning F1 (micro/macro/sample), Hamming, subset accuracy, per-label AUPRC.
- A short paragraph on the softmax-baseline failure — quote the label-count statistic ("always predicts exactly N labels per example").
- A per-label threshold JSON (`{"label_name": 0.42, ...}`) saved to disk, loadable by an inference script.

## Part B — Hierarchical classification

### Dataset

Pick one of:

- **Web of Science (WoS-46985)** — Kowsari et al., ["HDLTex: Hierarchical Deep Learning for Text Classification"](https://arxiv.org/abs/1709.08267), *ICMLA 2017*. Two-level taxonomy: 7 root categories × ~130 sub-categories.
- **Blurb Genre Collection (BGC)** — Aly, Remus & Biemann, ["Hierarchical Multi-label Classification of Text with Capsule Networks"](https://aclanthology.org/P19-2045/), *ACL SRW 2019*. Book blurbs with a hierarchical genre taxonomy.
- **DBpedia hierarchical** — DBpedia14 grouped by supercategory (Wikipedia's actual taxonomy) provides a two-level hierarchy.
- **Reuters RCV1-v2** — Lewis et al., ["RCV1: A New Benchmark Collection for Text Categorization Research"](https://www.jmlr.org/papers/v5/lewis04a.html), *JMLR 2004*. Available via `sklearn.datasets.fetch_rcv1`. The topic taxonomy is a real DAG.

### Requirements

Build **two** hierarchical classifiers and compare them on hierarchy-aware metrics.

1. **Flat leaf classifier**: fine-tune an encoder to predict the leaf label directly, ignoring hierarchy at training time. Enforce hierarchy at inference by propagating the predicted leaf to all its ancestors.
2. **Local classifier per parent node (LCPN)**: for each internal node, train a small multi-class head over its children. At inference, walk the tree top-down, picking the highest-probability child at each level.
   - You can use one shared encoder (frozen after Part A's fine-tune, or fine-tuned again from the pretrained checkpoint) with per-node linear heads, or one full model per parent for simplicity.
3. Evaluate **both** models with:
   - **Flat macro-F1 over leaves** — the "wrong" metric, included as a baseline.
   - **Hierarchical F1** — Kiritchenko et al. 2006 / Kosmopoulos et al. 2015 construction: treat predicted-node's ancestor set and true-node's ancestor set as bags, compute precision/recall/F1 over them.
   - **Consistency rate**: fraction of predictions that form a valid root-to-leaf path in the taxonomy.
   - **Level-wise accuracy**: accuracy at each depth (level 1 root, level 2 subcategory, etc.). Shows where error propagation dominates.
4. Which model wins on each metric? Where does flat leaf F1 disagree with hierarchical F1? Give a specific example — a prediction that is "wrong" flat but "close" hierarchical.

### Deliverables

- Model artefacts for both approaches.
- A metrics table with flat macro-F1, hierarchical F1, consistency rate, level-wise accuracy for both.
- A concrete example (one row from the test set) illustrating the flat-vs-hierarchical disagreement.
- A recommendation: which model would you ship, and under what deployment constraint would the other one win?

## Starter guidance

- Multi-label pitfall: when you have a rare label (say, 20 positive examples in 100k), the model may never predict it above any threshold. This is imbalance (chapter 06 material). For this exercise, just document it — the fix comes in exercise-03.
- For LCPN, do not train `N` full encoder fine-tunes if `N > 10`. Share the encoder and stack per-parent linear heads; iterate over parents in a single training loop.
- The hierarchical-F1 code from Kosmopoulos et al. is available at <https://github.com/AAI-Kosmopoulos/hierarchical-evaluation-measures>. You can also implement it yourself in ~30 lines — a good exercise.
- Do not confuse hierarchical *classification* with hierarchical *softmax* (chapter 03). The latter is an efficiency technique for flat multi-class over huge label sets; the former is a task-shape choice.
- If your hierarchical dataset also has multi-label leaves (RCV1 does), pick one interpretation (single-path or multi-path) and stick with it. Document the choice.

## Acceptance criteria

- [ ] Part A: correct multi-label encoder, broken softmax baseline, per-label thresholds JSON, metrics table with pre/post-threshold tuning numbers.
- [ ] Part A: paragraph diagnosing the softmax failure with concrete evidence (label-count distribution of the broken model's predictions).
- [ ] Part B: flat leaf classifier and LCPN, both trained and evaluated.
- [ ] Part B: metrics table with flat macro-F1, hierarchical F1, consistency rate, level-wise accuracy.
- [ ] Part B: at least one concrete example where flat and hierarchical metrics disagree, with explanation.
- [ ] A 200–400 word `README.md` summarising both parts, including the label-shape and metric-shape lessons.

## Stretch goals

- **Global hierarchical classifier**. Implement a simple hierarchical loss on the flat model (e.g., add a penalty proportional to taxonomic distance between predicted and true node). Does it beat both the flat and LCPN baselines?
- **HiAGM-style label attention.** Reproduce a simplified version of Zhou et al., ["Hierarchy-Aware Global Model for Hierarchical Text Classification"](https://aclanthology.org/2020.acl-main.104/), *ACL 2020*. Compare on the same test set.
- **Classifier chains** for multi-label. Implement Read et al. 2009's classifier chain on top of your encoder (predict labels in a fixed order, conditioning each on the previous). Compare macro-F1 vs. the independent BCE baseline.
- **Multi-label + hierarchical combined.** For datasets like RCV1, the true task is multi-label *and* hierarchical. Design and evaluate one model that respects both.
- **Ordinal classification comparison.** If your dataset has an ordinal label (star ratings), also fit an ordinal head (chapter 05) and report quadratic weighted kappa alongside macro-F1.
