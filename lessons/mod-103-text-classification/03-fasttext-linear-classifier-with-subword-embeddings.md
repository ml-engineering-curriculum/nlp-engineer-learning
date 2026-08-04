# fastText: Linear Classifier with Subword Embeddings

## Motivation

fastText, released by Meta (then Facebook AI Research) in 2016, is the middle rung of the classification ladder. It is a *linear* classifier in the same family as chapter 02's logistic regression — same softmax head, same log loss — but the features are averaged word embeddings enriched with character n-grams, learned end-to-end with the classifier. That combination lets it:

- Handle out-of-vocabulary and misspelled tokens gracefully via character n-grams.
- Compete with much heavier neural models on document classification while training in minutes on CPU.
- Scale to hundreds of thousands of labels via hierarchical softmax.
- Quantise to under 10 MB while keeping most of the accuracy.

Two papers frame it: Bojanowski, Grave, Joulin & Mikolov, ["Enriching Word Vectors with Subword Information"](https://arxiv.org/abs/1607.04606), *TACL 2017* (the embedding side), and Joulin, Grave, Bojanowski & Mikolov, ["Bag of Tricks for Efficient Text Classification"](https://arxiv.org/abs/1607.01759), *EACL 2017* (the classifier side). Read both — they are short and directly describe what the library does.

## Architecture, in one paragraph

Each word `w` is represented as the sum of its own vector plus its character n-gram vectors (typically n = 3 to 6, plus `<w>` boundary markers). A document is the average of its word vectors. That single vector is fed through a linear layer to `K` classes and softmax-cross-entropy trains the whole stack — embeddings and classifier — jointly. In pseudo-notation:

```
u_w        = v_w + Σ_{g ∈ Gₙ(w)} v_g            # word vector = self + char n-grams
d          = mean_{w ∈ doc}(u_w)                # doc vector = mean word vector
p(y | doc) = softmax(W · d + b)
```

The full model is `|V_word + V_ngram| × dim` embedding parameters plus `dim × K` classifier parameters — a couple of hundred MB uncompressed at typical settings, small after quantisation.

## Training a classifier: the shortest useful example

fastText ships a Python binding (`pip install fasttext`) and a CLI. Both consume the same file format: one document per line, prefixed by `__label__<class>` tokens.

Training data file:

```
__label__spam  buy now cheap watches -- exclusive offer
__label__ham   meeting at 3pm in the small conference room
__label__spam  claim your free gift card today only
```

Training call:

```python
import fasttext

model = fasttext.train_supervised(
    input="train.txt",
    epoch=25,
    lr=0.5,
    wordNgrams=2,       # phrase-level features
    minn=3, maxn=6,     # character n-grams
    dim=100,
    loss="softmax",     # or "ova" for multi-label, "hs" for hierarchical softmax
    thread=8,
)
model.save_model("classifier.bin")
```

Evaluation:

```python
p, r, _ = model.test("test.txt", k=1)
print(f"P@1 = {p:.3f}, R@1 = {r:.3f}")
```

And prediction:

```python
label, prob = model.predict("your text here", k=1)
```

## The dials that matter

- **`wordNgrams`** — usually `1` or `2`. Above `2` you rarely gain and often overfit small corpora. `wordNgrams=2` typically buys a small but consistent gain, mirroring the tf-idf bigram lift from chapter 02.
- **`minn` / `maxn`** — character n-gram range. Set to `0` if you have a clean, well-tokenised, low-OOV corpus (native English news, for instance). Leave at `(3, 6)` for social media, code, biomedical text, or any morphologically rich language. Character n-grams are what let fastText represent `undecidable` when it has only seen `decidable`.
- **`dim`** — embedding dimension. 100 is the default; 200–300 helps on large corpora with many labels; below 50 loses accuracy on all but toy problems.
- **`epoch`** — how many passes over the data. fastText is trained with SGD, so more epochs help until they overfit. 5–25 for large corpora, up to 100 for small ones.
- **`lr`** — learning rate. 0.1–1.0 is normal for classification (much higher than deep-learning defaults because the model is shallow). Sweep `{0.1, 0.5, 1.0}`.
- **`loss`** — `softmax` (multi-class), `ns` (negative sampling, mostly for the embedding-training mode), `hs` (hierarchical softmax, for large label sets), `ova` (one-vs-all, for multi-label).
- **`bucket`** — the character-n-gram hash table size. Default 2M. Reduce for smaller models; increase if character-n-gram collisions are visibly hurting a large-vocabulary language.

## Hierarchical softmax: how fastText scales to enormous label sets

Vanilla softmax computes `p(y | x) ∝ exp(w_y · d)`, which requires iterating over all `K` classes at every training step. At `K = 100 000` this is prohibitive. Hierarchical softmax, introduced by Morin & Bengio, ["Hierarchical Probabilistic Neural Network Language Model"](https://proceedings.mlr.press/r5/morin05a.html), *AISTATS 2005*, and refined by Mikolov et al., ["Distributed Representations of Words and Phrases and their Compositionality"](https://papers.nips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html), *NeurIPS 2013*, arranges the labels as a Huffman tree. Predicting a label is now a sequence of `log₂ K` binary decisions:

```
p(y | x) = Π over nodes on path to y: σ(±wₙ · d)
```

Training and inference cost drop from `O(K)` to `O(log K)`. The trade-off is that infrequent labels sit at deeper nodes, and probability estimates become slightly worse for the tail — usually acceptable when the alternative is not training at all.

Activate it in fastText with `loss="hs"`.

## Multi-label with fastText: `loss="ova"`

Multi-label problems (chapter 05) need `K` independent sigmoids, not a softmax that forces the probabilities to sum to 1. Set `loss="ova"` and prepend multiple `__label__` tags per line:

```
__label__toxic __label__insult he is a total idiot
__label__toxic __label__threat  I will find you and hurt you
```

Predict with a probability threshold:

```python
labels, probs = model.predict("...", k=-1, threshold=0.5)
```

Choose the threshold with the techniques from chapter 07, not `0.5` by default.

## Quantisation: how the ~1 GB model becomes ~10 MB

`fastText` supports product quantisation of the embedding table (Jégou, Douze & Schmid, ["Product Quantization for Nearest Neighbor Search"](https://ieeexplore.ieee.org/document/5432202), *TPAMI 2011*). One call:

```python
model.quantize(input="train.txt", qnorm=True, retrain=True, cutoff=100000)
model.save_model("classifier.ftz")
```

`cutoff` prunes to the top-`k` most-important features and `retrain=True` fine-tunes on top of the pruned set to recover accuracy. Typical results (from Joulin et al., ["FastText.zip: Compressing text classification models"](https://arxiv.org/abs/1612.03651), *arXiv 2016*): 10–100× size reduction with < 1 % accuracy loss on the paper's benchmarks. Rerun the numbers on your own data — the gap depends on the label distribution and vocabulary size.

## Where fastText wins

- **Cold-start on a fresh, moderately large, moderately messy corpus.** A single-CPU training run in minutes gives you a baseline that is often within 1–3 F1 points of a fine-tuned encoder.
- **Morphologically rich or OOV-heavy languages.** Finnish, Turkish, Arabic, agglutinative and inflectional languages benefit disproportionately from character n-grams. Same applies to social-media text, code, and biomedical corpora with heavy OOV.
- **Very large label sets.** Product categories in the thousands, medical codes in the tens of thousands, tag prediction in the millions. Hierarchical softmax makes these tractable where flat softmax does not.
- **Edge / on-device inference.** After quantisation, models fit in single-digit MB, run on CPU in microseconds per document, and are trivially portable (`.ftz` is a self-contained binary).
- **Language identification.** Meta's public `lid.176.bin` (176 languages) is a fastText model — chapter 09 of mod-102 covered it as the workhorse LID router.

## Where fastText caps out

- **Word order matters.** fastText averages word vectors, then classifies. It has no notion of order beyond `wordNgrams`. For entailment, argument roles, negation-scope tasks, contextual encoders remain the answer.
- **Cross-domain generalisation.** The embeddings are trained on your corpus, from scratch. There is no massive pretraining transfer; you cannot benefit from what Wikipedia + web corpora contain if you did not train on them.
- **Compositional generalisation to rare inputs.** Character n-grams handle *morphological* OOV; they do not confer semantic understanding of an unseen concept.

## Comparing fastText to tf-idf + linear

They are extremely close cousins:

| Dimension | tf-idf + logreg / SVM | fastText |
| --- | --- | --- |
| Feature space | Sparse, high-dim (100k–10M) | Dense, low-dim (50–300) |
| OOV robustness | Poor (unknown token = zero vector contribution) | Good (character n-grams reconstruct) |
| Multi-label scaling | One-vs-rest, standard | Native (`loss="ova"`) |
| Huge label sets | Slow with plain softmax | Fast with `loss="hs"` |
| Training speed on CPU | Seconds to minutes | Minutes |
| Interpretability | High (feature coefficients) | Low (dense embedding space) |
| Model artefact | Vocabulary + weight matrix | Embedding table + weight matrix |
| Quantisation to < 10 MB | Feature hashing gets you there | Native product quantisation |

Rule of thumb: if your text is clean, English, and moderately verbose, tf-idf + linear is often within noise of fastText and is easier to explain. If you have OOV-heavy or morphology-heavy text, or very large label sets, fastText tends to win.

## Chapter summary

- fastText is a linear classifier over averaged word embeddings enriched with character n-grams; training is joint, fast, CPU-friendly, and yields models that quantise to under 10 MB.
- Character n-grams handle morphology and misspellings; hierarchical softmax scales to hundreds of thousands of labels; `loss="ova"` is the native multi-label setting.
- Reasonable defaults: `dim=100`, `wordNgrams=2`, `minn=3, maxn=6`, `epoch=25`, `lr=0.5`. Sweep `lr` and `epoch` first.
- Wins over tf-idf + linear on OOV/morphology-heavy text and very large label sets; loses on tasks needing word order or contextual semantics — which is where chapter 04's encoder fine-tuning takes over.
