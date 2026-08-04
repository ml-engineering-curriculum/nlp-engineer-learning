# CRF Heads and Structured Decoders

## Motivation

Chapter 04 built the workhorse recipe: a per-token linear head that predicts BIO tags independently and takes argmax at inference. That model is fast, simple, and — for reasons this chapter unpacks — usually good enough. But there is an older, more principled decoder for BIO tagging: the linear-chain conditional random field (Lafferty, McCallum & Pereira, ["Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data"](https://repository.upenn.edu/cis_papers/159/), *ICML 2001*). CRFs model the joint probability of the whole tag sequence, learn a matrix of tag-to-tag transition scores, and decode with Viterbi. They were the default output head on top of BiLSTM NER for the entire 2015–2019 era and still ship in `flair`, `spaCy` (via `spacy-transformers`), and many biomedical NLP libraries.

This chapter explains what a CRF actually does, when the extra parameters still pay for themselves in the transformer era, and the cheaper alternatives — constrained Viterbi and BIO-invalidity post-processing — that capture most of the value at a fraction of the training cost.

## What the CRF adds over independent softmax

An independent softmax head assigns each token a distribution over the `K` tags and takes argmax token-by-token. Two problems follow:

1. **Invalid sequences.** `O → I-PER` is illegal under BIO. Independent softmax has no prohibition against it and, in practice, emits such transitions on 0.5–3 % of tokens in a well-trained model. Every one is a decoded false positive at some later position.
2. **No modelling of tag co-occurrence.** The head cannot express *"I've just emitted `B-ORG`, so `I-ORG` is more likely next"* — it treats each position independently.

A linear-chain CRF fixes both by scoring a full sequence `y = (y_1, ..., y_n)` as:

```
score(y | x) = Σ_t emission[t, y_t] + Σ_t transition[y_{t-1}, y_t]
```

`emission` comes from the encoder's per-token logits. `transition` is a `K × K` learnable matrix (plus start/end vectors). Training maximises the log-likelihood of the gold sequence normalised by the sum over *all* sequences — computed efficiently with the forward algorithm in `O(n × K²)`. Decoding is Viterbi, also `O(n × K²)`.

The transition matrix is the CRF's whole benefit: it can learn `transition[O, I-PER] = -∞` from data (or be constrained to it a priori), preventing invalid transitions structurally rather than by post-hoc cleanup.

## The empirical picture in 2026

The classic result on BiLSTM-CRF NER — Lample et al., ["Neural Architectures for Named Entity Recognition"](https://arxiv.org/abs/1603.01360), *NAACL 2016* — showed a consistent 0.5–1.5 F1 lift from adding a CRF on top of a BiLSTM. That result did not fully transfer to transformer encoders:

- **Base transformer + softmax vs. base transformer + CRF, CoNLL-2003 English.** Multiple reproductions (see e.g. Wang et al., ["Automated Concatenation of Embeddings for Structured Prediction"](https://arxiv.org/abs/2010.05006), *ACL 2021*; the flair benchmarks; the `pytorch-crf` community results) find the CRF adds ~0.2–0.5 F1 for `bert-base-cased`, often within seed noise.
- **Small labelled corpora (< 3 000 sentences).** CRF can add 0.5–1.5 F1 because the transition matrix is a strong inductive bias when data is scarce.
- **Domain corpora with long entities and rare tag transitions.** CRF sometimes helps by 0.5–1.0 F1 on biomedical and legal corpora, where hard-to-predict transitions (e.g., `I-DIS → I-CHEM` in *cisplatin-induced neuropathy*) benefit from an explicit prior.
- **`-large` transformers.** The encoder already learns most of the structural information; the CRF gain shrinks to ≤ 0.2 F1 and is not worth the training-time cost.

Concretely: if you are training a base-size transformer on a mid-size, standard NER corpus, do not put a CRF on it. If you are training on a small domain corpus and have a spare afternoon, try it.

## The cheap alternative: constrained decoding

You can capture most of the "no invalid transitions" benefit at zero training cost by constraining Viterbi decoding on top of a softmax model. Set forbidden transitions to `-∞` and decode:

```python
import numpy as np

def constrained_viterbi(logits, label_names):
    n, k = logits.shape
    # Build transition mask: -inf where transition is forbidden under BIO
    trans = np.zeros((k, k))
    for i, curr in enumerate(label_names):
        for j, nxt in enumerate(label_names):
            if nxt.startswith("I-") and (
                curr == "O" or (curr[2:] != nxt[2:])
            ):
                trans[i, j] = -np.inf  # forbid O → I-X and I-X → I-Y for X != Y

    dp = logits.copy()
    bp = np.zeros_like(dp, dtype=np.int64)
    for t in range(1, n):
        scores = dp[t - 1, :, None] + trans + logits[t, None, :]
        bp[t] = scores.argmax(axis=0)
        dp[t] = scores.max(axis=0)
    path = [int(dp[-1].argmax())]
    for t in range(n - 1, 0, -1):
        path.append(int(bp[t, path[-1]]))
    return list(reversed(path))
```

This is the "poor man's CRF" — a Viterbi decode with a hardcoded, prior-known transition mask. On CoNLL-2003, it typically matches a learned CRF within 0.1 F1 and adds < 1 ms per sentence at inference.

An even cheaper option is to greedy-decode with softmax argmax, then walk the output and coerce invalid transitions:

```python
def repair_bio(tags):
    out = []
    prev_type = None
    for tag in tags:
        if tag.startswith("I-"):
            t = tag[2:]
            if prev_type != t:
                tag = "B-" + t
            prev_type = t
        elif tag.startswith("B-"):
            prev_type = tag[2:]
        else:
            prev_type = None
        out.append(tag)
    return out
```

This "greedy + repair" approach loses ~0.1 F1 vs. constrained Viterbi and is what most production NER pipelines actually run.

## When the CRF is worth it: the honest list

- Small domain corpora (< 3 000 labelled sentences) with a base-size encoder — 0.5–1.5 F1 gain is possible.
- Domains with long entities (biomedical mentions of 5+ tokens, contract-party names of 8+ tokens) where the transition prior helps.
- You are already using `flair`'s `SequenceTagger` — CRF is the default there, keep it.
- You need calibrated *sequence* probabilities, not per-token probabilities. CRF gives you a proper `p(y | x)`.

## When it is not worth it

- `-large` encoders on any dataset.
- CoNLL-scale corpora (~15 000 sentences with balanced entity distribution).
- Latency-sensitive serving where the `O(n × K²)` Viterbi becomes the bottleneck (rare, but real at `K = 40+` for large ontologies).
- Long-context NER where the CRF forward pass becomes non-trivial at document scale.

## Practical implementations

- **`pytorch-crf`** — <https://github.com/kmkurn/pytorch-crf>. Drop-in `CRF` layer that produces log-likelihood loss and Viterbi decode. ~200 lines; easy to inspect.
- **`torchcrf`** — <https://github.com/rikeda71/TorchCRF>. Similar API.
- **`allennlp.modules.conditional_random_field`** — `CRF` with span-level constraint helpers.
- **`flair`'s `SequenceTagger`** — CRF is on by default; toggle with `use_crf=True/False`.
- **`spacy-transformers`** — no CRF; spaCy uses a transition-based decoder that is a different animal (chapter 06 style).

Sample integration on top of `AutoModel`:

```python
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel, AutoTokenizer

class TransformerCRF(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        h = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        emissions = self.classifier(self.dropout(h))
        mask = attention_mask.bool()
        if labels is not None:
            # Replace -100 with 0 for CRF (mask handles it), keep proper loss
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0
            log_likelihood = self.crf(emissions, crf_labels, mask=mask, reduction="mean")
            return -log_likelihood, emissions
        return self.crf.decode(emissions, mask=mask), emissions
```

Two footguns:

- **`-100` labels break the CRF loss.** CRF layers cannot ignore individual positions the way `CrossEntropyLoss` can. You must build a mask (usually the attention mask, sometimes a subword-first mask) and pre-clean the labels. Get this wrong and the loss silently trains against garbage.
- **Learning-rate mismatch.** The CRF's transition matrix trains much faster than the encoder. Either use a much larger LR on the CRF params (`lr=1e-3` on transitions vs. `2e-5` on encoder) or accept an extra 1–2 epochs of training.

## Other structured decoders worth naming

- **Semi-Markov CRF (semi-CRF)** — Sarawagi & Cohen, ["Semi-Markov Conditional Random Fields"](https://papers.nips.cc/paper_files/paper/2004/hash/eb06b9db06012a7a4179b8f3cb5384d3-Abstract.html), *NeurIPS 2004*. Models spans directly rather than per-token tags. Bridges into chapter 06 (span-based NER).
- **Transition-based parsers.** spaCy's default NER is a *transition-based* decoder — the model predicts SHIFT/BEGIN/IN/LAST/UNIT/OUT actions and maintains a partial-parse state. Fast, but harder to reason about at the token level.
- **Constrained beam search on decoder LMs.** For decoder-based extraction (chapter 10), constrained decoding with grammars (JSON Schema, regex, PDA-based) is the modern equivalent — a "CRF for LLMs."

## Chapter summary

- CRFs enforce valid BIO tag sequences by learning a transition matrix; they added 0.5–1.5 F1 in the BiLSTM era.
- On top of modern transformers, CRF gains are typically 0.0–0.5 F1 on standard corpora and only reliably help small domain datasets or base-size encoders.
- Constrained Viterbi decoding on top of softmax logits captures most of the benefit at zero training cost; greedy + BIO repair captures nearly all of it at inference.
- If you use a CRF, watch out for `-100` labels (need masking) and learning-rate mismatch (transitions want a larger LR).
- Semi-Markov CRFs, transition-based parsers, and constrained-decoding LMs are the same family with different assumptions — chapter 06 (span-based) and chapter 10 (decoder-LM) build on the structured-decoding idea.
