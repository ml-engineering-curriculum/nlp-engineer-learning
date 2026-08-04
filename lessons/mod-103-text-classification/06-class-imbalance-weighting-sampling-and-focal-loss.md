# Class Imbalance: Weighting, Sampling, and Focal Loss

## Motivation

Real label distributions are power-law. Fraud vs. legitimate transactions is 1:1000. Toxic vs. safe comments is 1:100 to 1:10 000. The 200th intent in an assistant's taxonomy has 30 examples where the top intent has 300 000. Trained naïvely with plain cross-entropy, the classifier learns "always predict the majority class" — a state that has a *lower* training loss than any actually-useful decision boundary.

This chapter is the ordered toolbox for the imbalance problem. It goes in the order you should try things: diagnose, then reweight, then resample, then reach for focal loss, and finally acknowledge that most of the "imbalance fix" is a threshold-tuning problem covered in chapter 07.

## Diagnose first

Before applying any fix, get honest numbers:

1. **Class-distribution histogram of your training set.** Print `Counter(y_train)`. Look at the ratio of majority to minority.
2. **Class-distribution histogram of your validation and test sets.** These should reflect the *deployment* distribution, not necessarily the training one. Never resample the eval sets.
3. **Per-class F1 and support from `classification_report`.** Not accuracy. Not a single number.
4. **The base-rate baseline.** What F1 does "always predict majority" get? A model that beats accuracy but not per-class F1 is not doing useful work.
5. **A confusion matrix.** Look at *where* the model fails.

If your minority-class recall is near zero and precision is undefined ("no predictions"), the loss surface has collapsed to trivial. That is imbalance biting. If minority-class precision is high and recall low, the model has learned the class but is conservative — a threshold-tuning problem, not necessarily a loss-function problem.

## The three levers, in the order to try them

### Lever 1: class weights in the loss

Cheapest fix, no data change, one line of code.

**Logistic regression (sklearn):**

```python
LogisticRegression(class_weight="balanced")
# equivalent to weight[k] = n / (K · n_k) for class k
```

**PyTorch cross-entropy:**

```python
weights = torch.tensor([n / (K * count[k]) for k in range(K)], dtype=torch.float32)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))
```

**Hugging Face Trainer:** subclass and override `compute_loss`:

```python
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kw):
        super().__init__(*args, **kw)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(
            logits, labels, weight=self.class_weights.to(logits.device)
        )
        return (loss, outputs) if return_outputs else loss
```

**Binary BCE:** use `pos_weight`:

```python
pos_weight = torch.tensor([n_neg / n_pos])
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

Two variants worth knowing:

- **Inverse frequency** — `w_k = 1 / n_k`, normalised. The simple `class_weight="balanced"` equivalent.
- **Effective number of samples** — Cui, Jia, Lin, Song & Belongie, ["Class-Balanced Loss Based on Effective Number of Samples"](https://arxiv.org/abs/1901.05555), *CVPR 2019*. Weight each class by `(1 - β) / (1 - β^n_k)` where `β ∈ [0, 1)`. Empirically better than pure inverse-frequency on very long-tailed distributions because it approaches a per-class cap as `n_k` grows. `β = 0.9999` is a common starting point for `n_k > 10 000`.

### Lever 2: resampling

Change the empirical distribution the model sees.

- **Random oversampling of the minority class.** Duplicate minority examples until the training distribution is balanced (or nearly so). Simple; increases epoch length; can overfit specific minority examples if you go too far.
- **Random undersampling of the majority class.** Drop majority examples. Faster training, throws away data. Combine with ensembling (train `k` models on different undersamples, average) to recover some of the signal.
- **Class-balanced sampling** (`WeightedRandomSampler` in PyTorch): sample examples with probability inversely proportional to their class frequency, without duplication. The default for imbalanced classification with modern DataLoaders.

**PyTorch class-balanced sampling:**

```python
from torch.utils.data import WeightedRandomSampler

class_counts = np.bincount(y_train)
sample_weights = 1.0 / class_counts[y_train]
sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(y_train), replacement=True,
)
DataLoader(dataset, sampler=sampler, batch_size=32)
```

**A caveat**: SMOTE (Chawla et al., ["SMOTE: Synthetic Minority Over-sampling Technique"](https://arxiv.org/abs/1106.1813), *JAIR 2002*) synthesises minority examples by interpolating in feature space. This makes sense on tabular data. It does *not* make sense in the input space of text — you cannot interpolate between two documents. If you use SMOTE on text, do it in encoder-embedding space (after freezing an encoder), and validate that the synthetic points do not corrupt the decision boundary.

### Lever 3: focal loss

Focal loss, from Lin, Goyal, Girshick, He & Dollár, ["Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002), *ICCV 2017*, modifies cross-entropy to down-weight examples the model already classifies confidently:

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```

where `p_t` is the model's probability for the correct class, `γ` is a focusing parameter (0 recovers cross-entropy; 1 to 2 is normal), and `α_t` is an optional class weight.

Practical notes:

- `γ = 2` is the most common default and often works out of the box.
- Focal loss and class weights can be combined; `α` behaves like a class weight.
- Focal loss shines when the class imbalance is extreme (< 1 % minority) *and* many majority-class examples are "easy" — the loss cares less about them. On text-classification benchmarks with moderate imbalance (say, 10:1), plain `CrossEntropyLoss` with class weights is often equivalent.
- Focal loss deforms the loss surface enough to make calibration worse (chapter 07). Plan on calibrating post-hoc.

A minimal PyTorch focal loss for multi-class:

```python
import torch
import torch.nn.functional as F

def focal_loss(logits, targets, gamma=2.0, alpha=None):
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    focal_factor = (1.0 - p_t) ** gamma
    if alpha is not None:
        alpha_t = alpha[targets]
        return -(alpha_t * focal_factor * log_p_t).mean()
    return -(focal_factor * log_p_t).mean()
```

For multi-label problems, apply the same formula on `sigmoid` outputs, per label — the same relative weighting logic applies.

### Lever 4: threshold tuning

Often the cheapest and most powerful fix. If your minority class has usable precision but low recall at `p > 0.5`, the model *knows* about the class and is just being conservative because the base rate is low. Move the threshold to `argmax_{τ} F1_minority(τ)` or the cost-weighted equivalent on validation. This is chapter 07's territory.

Rule of thumb: threshold tuning gives you a `p > τ` that recovers the majority of the "we're missing minority instances" complaint at zero training cost. Always try it before rerunning training.

## Diagnosis-driven order of operations

1. Compute per-class F1 and confusion matrix at the default threshold.
2. If minority-class *precision is undefined* (zero predictions): apply lever 1 (class weights or `pos_weight`) and retrain.
3. If minority-class *precision high, recall low*: try lever 4 (threshold tuning). Retrain only if that isn't enough.
4. If lever 1 helps but minority F1 is still poor: try lever 2 (resampling) or increase weights and monitor validation loss.
5. If imbalance is extreme (< 1 % minority) and many easy majority examples exist: try lever 3 (focal loss).
6. If none of the above moves the number materially: the issue is probably labelling noise, not imbalance. See mod-110.

## Metrics that respect imbalance

Report and threshold-tune against metrics that don't dissolve under class imbalance:

- **Macro-F1** — equal weight to each class regardless of support.
- **AUPRC** (average precision) — the threshold-independent binary answer for imbalanced data.
- **F1 at the minority class** — often the only number your product actually cares about.
- **Balanced accuracy** — the average of per-class recall (`sklearn.metrics.balanced_accuracy_score`).
- **Matthews correlation coefficient (MCC)** — a single balanced metric for binary problems that penalises both false positives and false negatives symmetrically; robust to imbalance (Chicco & Jurman, ["The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation"](https://doi.org/10.1186/s12864-019-6413-7), *BMC Genomics 2020*).

Never report accuracy alone on an imbalanced problem.

## Common failure modes and how to spot them

- **"The model got worse after I added class weights."** You likely inflated the minority weight past where the model can generalise — validation loss on the majority went up more than validation F1 on the minority. Try smaller weights, or use `class_weight="balanced"` first.
- **"Focal loss dropped my macro-F1."** Focal loss on already-well-calibrated tasks can hurt because it robs signal from confident-and-correct examples. Verify by ablating with `γ = 0`.
- **"My oversampled model overfits."** Duplicating minority examples produces a training set that's easy to memorise. Use `WeightedRandomSampler` instead of static duplication so different epochs see different subsets.
- **"My eval numbers went up but the product regressed."** You resampled the eval set or reported micro-F1 on an imbalanced problem. Fix the reporting, then talk about model changes.
- **"Class weights collide with mixed precision."** Very large weights (say, `1000×`) can produce loss values that overflow `fp16`. Use `bf16`, or reduce weights and rely more on threshold tuning.

## When imbalance is not the real problem

Not every low-F1 minority class is caused by imbalance. Other suspects:

- **Label noise.** Annotators disagreeing on the minority class. Check inter-annotator agreement.
- **Label ambiguity.** The minority class is a genuinely fuzzy category that even humans can't reliably tag.
- **Feature paucity.** The minority class needs a signal your representation doesn't capture — an entity type your tokenizer splits badly, a language your encoder wasn't pretrained on.
- **Concept drift.** Recent minority examples look nothing like historical ones. Retrain on recent data, or add temporal weighting.

If lever 1 + threshold tuning doesn't move the number, do not throw more losses at the problem. Investigate the data.

## Chapter summary

- Diagnose imbalance before treating it: per-class F1, confusion matrix, and the base-rate baseline.
- Apply the four levers in order: class weights, resampling, focal loss, threshold tuning. Threshold tuning is often the cheapest and most effective; try it before rerunning training.
- Use `class_weight="balanced"` or Cui et al.'s effective-number-of-samples weighting; use `pos_weight` for binary BCE; use `WeightedRandomSampler` for class-balanced sampling. Skip SMOTE on raw text.
- Focal loss (Lin et al. 2017) helps when imbalance is extreme and many majority examples are easy; it worsens calibration so plan for chapter 07's fixes.
- Never evaluate on accuracy alone under imbalance; use macro-F1, AUPRC, minority-class F1, balanced accuracy, or MCC.
- If none of the levers move the number, the problem is likely not imbalance — investigate labelling, ambiguity, features, or drift.
