# TF-IDF plus Logistic Regression and Linear SVM Baselines

## Motivation

The first classifier you train on a new dataset should be a linear model over TF-IDF features. Not because it will win — sometimes it will, sometimes it won't — but because it establishes the number every later architecture has to beat. Skipping this step is how projects end up spending a week on a fine-tune that a five-minute logistic regression could have equalled.

Beyond the baseline role, tf-idf + linear ships in real products where interpretability, latency, or CPU-only inference matter more than the last two F1 points. Email spam filters, log-line classifiers, first-tier support routing, and countless internal tools run on a sparse linear model behind an API that costs microseconds per document.

## The three vectorizers you should know

`scikit-learn` ships three text vectorizers, and the differences matter more in production than in a Kaggle notebook.

### `CountVectorizer` — raw term counts

```python
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    token_pattern=r"(?u)\b\w\w+\b",
)
X = vec.fit_transform(corpus)
```

`min_df` prunes hapax legomena that only add noise; `max_df` drops corpus-wide function words. `ngram_range=(1,2)` captures short phrases and typically beats unigrams alone by 1–3 F1 points on tasks where phrase-level signals matter (sentiment, topic).

### `TfidfVectorizer` — TF-IDF with `sublinear_tf`

TF-IDF weights down common words and up discriminative rare ones:

```
tfidf(t, d) = tf(t, d) · idf(t)
idf(t)      = log( (1 + n) / (1 + df(t)) ) + 1        # sklearn's smoothed variant
```

with `sublinear_tf=True` replacing `tf` with `1 + log(tf)`. This dampens frequent-term explosion (Manning, Raghavan, Schütze, *Introduction to Information Retrieval*, §6.4, [free online](https://nlp.stanford.edu/IR-book/)):

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    norm="l2",
)
```

`norm="l2"` unit-normalises each document vector — required for well-behaved linear classifier optimisation.

### `HashingVectorizer` — feature hashing for streaming and distributed training

`TfidfVectorizer` builds a vocabulary, which means `fit` must see all data at once and the vocabulary is memory-resident. `HashingVectorizer` hashes each token to an index in `[0, n_features)`, avoiding vocabulary storage and enabling out-of-core / distributed training. The reference is Weinberger et al., ["Feature Hashing for Large Scale Multitask Learning"](https://arxiv.org/abs/0902.2206), *ICML 2009*.

```python
from sklearn.feature_extraction.text import HashingVectorizer

vec = HashingVectorizer(
    n_features=2**20,
    alternate_sign=True,
    norm="l2",
    ngram_range=(1, 2),
)
```

Trade-off: hash collisions cost a fraction of an F1 point at `n_features=2**18` or larger, and you lose the ability to interrogate model coefficients by feature name. In exchange, you gain the ability to train on a stream you never fully materialise.

## Logistic regression: the specific case you want to reach for

For multi-class classification, `sklearn`'s `LogisticRegression(solver="saga", penalty="l2")` with `class_weight="balanced"` covers ~90 % of your linear-baseline needs. The math:

- **Multinomial log-loss** (a.k.a. softmax cross-entropy) over `K` classes, `L2` regularisation by default:
  ```
  L(θ) = -1/N · Σᵢ log p(yᵢ | xᵢ; θ) + λ · ‖θ‖²
  ```
  where `p(k | x; θ) = softmax(θₖ · x)`.
- **`saga`** solver handles sparse feature matrices and is the only sklearn solver that supports elastic net (`penalty="elasticnet"`, `l1_ratio=…`) on multinomial loss.
- **`class_weight="balanced"`** rescales the per-class weight to `n_samples / (n_classes · n_samplesᵥ)` — a cheap first swing at class imbalance (chapter 06 covers the more nuanced fixes).

A concrete pipeline:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                              sublinear_tf=True, strip_accents="unicode")),
    ("logreg", LogisticRegression(
        solver="saga", penalty="l2", C=1.0, max_iter=200,
        class_weight="balanced", n_jobs=-1)),
])
clf.fit(X_train, y_train)
```

`C` is the inverse of `λ` and is the single most impactful hyperparameter. A short grid — `C ∈ {0.1, 1.0, 10.0}` — on a validation set almost always brackets the optimum. Larger `C` = weaker regularisation = more variance; smaller `C` = stronger regularisation = more bias.

## Linear SVM: hinge loss, when it beats log loss

`LinearSVC` optimises squared hinge loss instead of log loss. Practically:

- Historically, linear SVM slightly beats logistic regression on well-cleaned text classification benchmarks (Joachims, ["Text Categorization with Support Vector Machines"](https://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf), *ECML 1998*).
- The margin structure makes it robust to noisy overlapping features.
- `LinearSVC` does **not** produce probabilities. Wrap it in `CalibratedClassifierCV` if you need `predict_proba` (this is a Platt-scaling wrapper; see chapter 07).
- Prefer `LogisticRegression` when you need calibrated probabilities out of the box or plan to fine-tune the decision threshold.

```python
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

svm = CalibratedClassifierCV(
    LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=2000),
    method="isotonic",
    cv=5,
)
```

For binary classification with extreme imbalance, `LinearSVC(class_weight={0: 1, 1: 20})` with a manually chosen class weight often outperforms `class_weight="balanced"` — the ratio is a hyperparameter you tune on validation macro-F1 or on a cost-weighted metric.

## Preprocessing: the choices that actually move the number

You will read tutorials that lower-case, remove stopwords, and stem. Do the first, be careful with the second, and generally skip the third:

- **Lower-casing.** Always for English. For German (nouns are cased), for Turkish (dotless vs. dotted i), and for morphologically case-marked scripts, revisit — but a `lower()` upstream is usually fine because the tf-idf vectorizer is not deriving morphology.
- **Stopword removal.** Do it for information retrieval (BM25), not for classification. Stopwords carry sentiment and negation signal (`not`, `no`, `only`, `just`); dropping them regresses F1 on sentiment and stance tasks. If you must, use `min_df` / `max_df` pruning, which is data-driven.
- **Stemming / lemmatisation.** Rarely helps for classification once you have char n-grams or subword features. The overhead cost is nonzero; the benefit usually is not.
- **Character n-grams (`analyzer="char_wb"`, `ngram_range=(3, 5)`).** Small but consistent gains on morphologically rich languages, misspelled corpora, and short-text tasks. Often 2× the model size for 0.5–1.5 F1 points.
- **Accent stripping (`strip_accents="unicode"`).** Consider carefully. For French / Spanish it flattens `é → e`, `ñ → n` — usually a win because you halve vocabulary sparsity, occasionally a loss because `pena` ≠ `peña`.
- **Language-specific tokenisation.** For CJK, always segment first (jieba, MeCab, ICU) — a whitespace-only vectorizer over Chinese is a null model.

## Metrics that match this rung

Because you'll compare this to fancier models in chapter 04, log the same metrics you plan to use later:

- **Macro / micro / weighted F1** — chapter 05 covers when to prefer each.
- **Accuracy** — only for balanced problems.
- **AUROC and AUPRC** — for binary tasks and for retrieving decision thresholds later.
- **Log-loss** — the loss you actually optimised; report it so calibration work in chapter 07 has a baseline.

Compute them with `sklearn.metrics.classification_report` and `sklearn.metrics.log_loss`.

## Model interpretation: the free advantage

Linear models give you feature-level explanations for free:

```python
import numpy as np

vec: TfidfVectorizer = clf.named_steps["tfidf"]
logreg = clf.named_steps["logreg"]
feature_names = np.array(vec.get_feature_names_out())

for class_idx, class_name in enumerate(logreg.classes_):
    coefs = logreg.coef_[class_idx]
    top_pos = feature_names[np.argsort(coefs)[-15:][::-1]]
    print(class_name, top_pos)
```

Two production uses beyond debugging:

1. **Label-taxonomy sanity checks.** If the top positive features for `refund` include `login` and `password`, the labelling scheme (or the taxonomy) has a problem — the classifier is telling you where the confusion is.
2. **Rule extraction.** Some regulated domains ship rules distilled from top features, keeping the linear model as a research aid rather than the production artefact.

## When tf-idf + linear beats fine-tuning

This is not a rare list:

- **Domain jargon dominates the signal.** Legal citation strings, log lines, SKU codes — the discriminative signal is literal token match, not semantic composition.
- **Tiny label counts per class.** Under ~200 examples per class, encoder fine-tuning overfits fast; linear models with strong regularisation degrade gracefully.
- **Latency budget under 5 ms on CPU.** A hashed 2^20-dim linear model dot-products against a document in microseconds. A DistilBERT forward pass on CPU is milliseconds even at short sequences.
- **Interpretability is a shipping requirement.** Regulated domains where every decision must be explained; linear coefficients are a compliance-friendly explanation surface.
- **Enormous label sets.** With hundreds of thousands of classes, one-vs-rest linear SVMs still scale (see `sklearn`'s `OneVsRestClassifier` + `LinearSVC` or dedicated libraries like `Vowpal Wabbit`). Encoder heads over that many labels blow up memory.
- **Cold-start iteration.** You can retrain a linear model in seconds after every label-set change during taxonomy design. Fine-tunes take hours per iteration.

## When tf-idf + linear caps out

- **Paraphrase-heavy tasks.** Two sentences that mean the same thing share no tokens. Semantic tasks (entailment, paraphrase detection, stance) demand contextual embeddings.
- **Long-range compositional cues.** Negation scope, sarcasm, quantifier binding.
- **Multilingual mixed input.** A single tf-idf vocabulary can't share signal across languages the way a multilingual encoder can.
- **Very short input, low-token-count classification.** Titles or queries under 4–5 tokens rarely give a linear model enough features; contextual embeddings win.

Chapter 03 shows how fastText covers some of the same ground with subword n-grams; chapter 04 covers the encoder fine-tune that starts winning once these ceilings hit.

## Chapter summary

- Every classification project should start with `TfidfVectorizer` + `LogisticRegression` (or `LinearSVC`) as the number-to-beat baseline. Cost is minutes; upside is a defensible sanity check.
- `CountVectorizer`, `TfidfVectorizer`, and `HashingVectorizer` cover the vocabulary vs. streaming trade-off; `min_df`/`max_df`/`sublinear_tf` are the levers that matter.
- Logistic regression gives calibrated-ish probabilities and an interpretable coefficient view; linear SVM slightly beats it on some cleaned benchmarks but needs Platt / isotonic wrappers to produce probabilities.
- Preprocessing choices that move the number: casing policy, `min_df`/`max_df`, character n-grams for morphology-heavy or noisy text. Choices that usually don't: stemming, aggressive stopword lists.
- The tf-idf + linear rung ships in production when interpretability, CPU-only latency, cold-start iteration, or huge label sets constrain you. It caps out on paraphrase-heavy, semantically compositional, or short-input tasks — the ceiling that motivates chapters 03 and 04.
