# Tagging Schemes and Entity-Level Evaluation with seqeval

## Motivation

Named-entity recognition looks like it wants to be a per-token classification problem: give every token a label, train cross-entropy, done. It is not that simple. A single entity spans multiple tokens, and the labels have to carry enough structure that a decoder can recover *spans* — not just *tokens* — from the model's output. The choice of tagging scheme is where that structure lives, and the choice of evaluation metric is where you find out whether your model actually gets those spans right.

This chapter builds two habits that carry through the rest of the module:

1. Every NER dataset you touch has a tagging scheme. Know which one, know the trade-offs, and know how to convert between them.
2. Every NER model you report on is scored *entity-level*, not token-level. `seqeval` and the CoNLL evaluation script are the tools; the difference between micro-F1 and macro-F1 matters as much as it did in classification.

## The tagging schemes, in order of complexity

### IO — the naïve scheme

Every token gets `I-<type>` if it is inside an entity of that type, and `O` otherwise. Simple, but ambiguous: two adjacent entities of the same type collapse into one. In the sentence *"John Alice met at HQ."* under IO, `John` and `Alice` become one PER span. Never use IO for training unless the corpus has no such adjacent same-type entities.

### BIO (a.k.a. IOB2) — the modern default

Every entity's first token is `B-<type>`; subsequent tokens are `I-<type>`; outside tokens are `O`. Two adjacent entities of the same type stay distinct because the second one starts with a fresh `B-`. This is the CoNLL-2003 default (Tjong Kim Sang & De Meulder, ["Introduction to the CoNLL-2003 Shared Task"](https://aclanthology.org/W03-0419/), *CoNLL 2003*) and the format Hugging Face's `datasets` library exposes for `conll2003`, `ontonotesv5`, `wnut_17`, and most other public NER corpora.

There is an older "IOB1" variant where `B-` is used *only* to disambiguate adjacent same-type entities. Almost every modern paper reports IOB2 (BIO), so unless you are explicitly loading a legacy corpus, treat "BIO" as "IOB2."

Example (BIO):

```
John        B-PER
Smith       I-PER
works       O
at          O
Bank        B-ORG
of          I-ORG
America     I-ORG
.           O
```

### BIOES (a.k.a. IOBES) — encode span boundaries explicitly

Adds `E-<type>` for the last token of a multi-token entity and `S-<type>` for a single-token entity. So a two-token PER is `B-PER E-PER`, and a one-token PER is `S-PER`. The tag distribution gives the model an explicit "this is the end" signal, which small CRF-headed models used to benefit from — the effect on modern transformer NER is usually within noise (Reimers & Gurevych, ["Reporting Score Distributions Makes a Difference"](https://arxiv.org/abs/1707.09861), *EMNLP 2017*, note the sensitivity of these comparisons).

### BILOU — the AllenNLP / Ratinov & Roth variant

Same idea as BIOES: `B` (begin), `I` (inside), `L` (last), `O` (outside), `U` (unit / single-token). Introduced in Ratinov & Roth, ["Design Challenges and Misconceptions in Named Entity Recognition"](https://aclanthology.org/W09-1119/), *CoNLL 2009*, who reported a small but consistent lift over BIO for classical (non-neural) models. `L`/`U` map one-to-one to `E`/`S` in BIOES, so it is BIOES relabelled — treat them as equivalent for evaluation.

### Which to use?

- **Default to BIO.** It is what every public dataset ships in, what `seqeval` expects, and what encoder-plus-linear-head NER models train fine on. Modern papers (2019+) rarely report a meaningful BIOES/BILOU gain on transformer-based NER.
- **Try BIOES if you use a CRF head** on top of an older encoder — the extra structure sometimes helps by 0.2–0.5 F1.
- **Never mix schemes across train / dev / test.** The evaluator has to see the same scheme it trained on.
- **When converting**, use `seqeval.scheme` or a small function that inserts `B-` at every entity boundary. Do not do it manually across a large corpus.

## Why token-level accuracy lies

Consider a 1000-token document with 20 tokens inside entities. A model that predicts `O` for every token scores 98 % token-level accuracy and zero entities correctly. This is not hypothetical — every naïve first-pass NER model does this, and every naïve first-pass report puts a comforting number in a slide.

Entity-level evaluation counts a prediction as correct only when *both* the span boundaries and the entity type match the gold annotation exactly. Precision, recall, and F1 are then computed over spans, not tokens.

The CoNLL-2003 evaluation script (`conlleval`) has been the community standard for this since 2003. `seqeval` (Nakayama, ["seqeval: A Python framework for sequence labeling evaluation"](https://github.com/chakki-works/seqeval)) is the modern Python reimplementation and the default in Hugging Face `evaluate` for NER. Both agree on the same definitions:

- **True positive** — a predicted span whose boundaries and type match a gold span.
- **False positive** — a predicted span with no matching gold span.
- **False negative** — a gold span with no matching prediction.
- **Precision, recall, F1** — computed per entity type, then aggregated micro / macro / weighted.

Partial matches count as *neither* TP nor as separate half-credit. A prediction of `B-PER I-PER` when the gold is `B-PER I-PER I-PER` is one false positive and one false negative — total zero credit. This is intentional: downstream consumers care about spans they can look up, not fractional overlap.

## Using seqeval

```python
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2

y_true = [
    ["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "O"],
    ["O", "B-LOC", "O"],
]
y_pred = [
    ["B-PER", "I-PER", "O", "B-ORG", "O",     "O"],
    ["O", "B-LOC", "O"],
]

print(f1_score(y_true, y_pred, mode="strict", scheme=IOB2))
print(classification_report(y_true, y_pred, mode="strict", scheme=IOB2))
```

Two flags matter:

- **`mode="strict"`** enforces valid BIO sequences — a span starting with `I-` without a preceding `B-` is treated as an invalid label and does *not* count as a match. The default (`mode="default"`) is lenient and is *not* what CoNLL reports.
- **`scheme=IOB2`** (or `IOBES`, `BILOU`) tells `seqeval` how to parse spans from the tag sequence. Getting this wrong silently miscounts spans.

Always set both. The Hugging Face `evaluate.load("seqeval")` wrapper defaults to lenient mode; pass `mode="strict"` when you call `compute` for CoNLL-comparable numbers.

## Micro, macro, weighted — the same three flavours as classification

- **Micro-F1** — pool TP/FP/FN across all entity types and compute one F1. Rewards a model that gets the frequent types (PER, ORG, LOC) right. This is what CoNLL calls the "overall F1."
- **Macro-F1** — compute F1 per type, then average unweighted. Rewards tail-type performance equally. What you report if you care about your MISC or FAC classes.
- **Weighted-F1** — per-type F1 weighted by support. Middle ground; less common in NER papers.

Publications almost universally report micro-F1 as the headline number for NER; a serious paper also reports per-type F1 in a table. Copy this convention.

## Nested and discontinuous entities — where BIO breaks

BIO/BIOES/BILOU assume entities are contiguous and non-overlapping. Real biomedical text (GENIA), legal text (contract parties nested inside jurisdictions), and financial text (Bank of *America* the entity vs. *America* the entity) violate both assumptions.

Two escape hatches:

1. **Layered BIO** — tag each entity type on a separate layer. Works for two-level nesting but explodes for genuine graphs.
2. **Span-based models** — abandon BIO entirely and score every candidate span directly. Chapter 06 covers this in depth; it is the current default for nested NER.

For discontinuous entities — e.g., *"heart and lung disease"* where DISEASE = {heart disease, lung disease} — you either merge to the outer span (loses information) or move to a span-graph model (Dai, ["Multi-task Learning for Discontinuous Named Entity Recognition"](https://aclanthology.org/2020.acl-main.577/), *ACL 2020*). This module treats discontinuous entities as a niche you should know exists but not build a chapter around.

## Common evaluation traps

- **Reporting token-level F1** as if it were entity-level. Always name the metric; always cite the scheme.
- **Running lenient `seqeval` and calling it CoNLL F1**. Lenient mode over-counts by accepting broken `I-`-starting spans; the delta can be 1–2 F1 points on messy tag sequences from BIO-trained softmax heads.
- **Comparing your BIO model against a paper's BIOES model** on the same F1 number. F1 is directly comparable across schemes when both are evaluated entity-level with the right scheme setting — but you have to convert the tags first, not assume `seqeval` handles it silently.
- **Not reporting per-type F1.** A model that regresses 15 F1 on MISC while gaining 2 F1 on PER + ORG can post the same micro-F1 as the previous version — the per-type breakdown is what surfaces this.
- **Silent tag remapping.** Some Hugging Face NER datasets include `B-`/`I-` for a type that never occurs at inference; leaving those labels in the `id2label` map inflates the effective label count and can confuse the softmax. Prune unused labels or check the confusion matrix.
- **Case-sensitivity of the scheme flag.** `seqeval` accepts `mode="strict"` but not `"Strict"`. Silent bugs live here.

## The evaluation loop you will paste into every project

```python
import numpy as np
from evaluate import load

label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
metric = load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    true_labels = [
        [label_names[l] for l in seq_labels if l != -100]
        for seq_labels in labels
    ]
    true_preds = [
        [label_names[p] for p, l in zip(seq_preds, seq_labels) if l != -100]
        for seq_preds, seq_labels in zip(preds, labels)
    ]
    results = metric.compute(
        predictions=true_preds,
        references=true_labels,
        mode="strict",
        scheme="IOB2",
    )
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
```

The `-100` filter is Hugging Face's convention for "ignore this position in loss and evaluation" — it is applied to sub-tokens that are not the first sub-token of their word. Chapter 03 is entirely about *how* that `-100` gets placed, and why it dominates the training recipe.

## Chapter summary

- BIO is the modern default tagging scheme; BIOES / BILOU add explicit end-of-entity signals but rarely move transformer NER by more than noise.
- Token-level accuracy is meaningless for NER; report entity-level precision, recall, and F1 with `seqeval` in `mode="strict"`, and always name the scheme.
- Report micro-F1 as the headline number and per-type F1 as the breakdown — the per-type table is where regressions hide.
- Nested and discontinuous entities break the BIO assumption; layered BIO handles the mild case, span-based models (chapter 06) handle the general one.
- The `compute_metrics` snippet at the end of this chapter is the evaluation loop you paste into every subsequent NER project in the module.
