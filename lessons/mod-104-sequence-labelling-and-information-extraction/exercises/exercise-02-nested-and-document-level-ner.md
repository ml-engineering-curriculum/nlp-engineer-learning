# exercise-02: Nested and Document-Level NER

**Estimated effort:** 3 hours

## Objective

Move past the flat-sentence CoNLL setting and confront the two ways real IE breaks the BIO assumption: **nested entities** (an entity inside another entity) and **document-level context** (entities that exceed the encoder's 512-token window or that need cross-sentence disambiguation). You will train one span-based model on a nested corpus, one long-context model on a document-scale corpus, and report the metrics both correctly *and* comparably to the flat baseline from exercise-01.

## Prerequisites

- Exercise-01 completed — you have a working `seqeval` evaluation loop and understand subword alignment.
- Chapters [02](../02-tagging-schemes-and-entity-level-evaluation-with-seqeval.md), [05](../05-crf-heads-and-structured-decoders.md), [06](../06-span-based-ner-for-flat-nested-and-discontinuous-entities.md), [07](../07-document-level-and-long-context-ner.md).
- Python 3.10+; `transformers`, `datasets`, `evaluate`, `seqeval`, `torch`. Optional: `SpanMarker`, `gliner`, `allenai/longformer-base-4096`.
- A GPU with ≥ 16 GB VRAM if you want to fine-tune Longformer on documents past 2 048 tokens; otherwise stick to `stride`-based windowed encoders.

## Datasets

Pick **one nested corpus** and **one document-level corpus**. Use these throughout the exercise.

### Nested-entity corpus (pick one)

- **GENIA** (Kim et al., ["GENIA corpus — a semantically annotated corpus for bio-textmining"](https://academic.oup.com/bioinformatics/article/19/suppl_1/i180/227927), *Bioinformatics 2003*). Biomedical; protein / DNA / RNA / cell types with heavy nesting. Available via `datasets.load_dataset("Rosenberg/genia")` and other community mirrors — check the licence before use.
- **ACE 2005** (Doddington et al., ["The Automatic Content Extraction (ACE) Program"](http://www.lrec-conf.org/proceedings/lrec2004/pdf/5.pdf), *LREC 2004*). LDC-licensed newswire with 7 entity types and moderate nesting. If your institution has LDC access, use the English portion.
- **NNE** (Ringland et al., ["NNE: A Dataset for Nested Named Entity Recognition in English Newswire"](https://aclanthology.org/P19-1510/), *ACL 2019*). Layered nested annotations on top of OntoNotes newswire; open-access.

### Document-level corpus (pick one)

- **CoNLL-2003** but **evaluated at the document level** — the raw dataset ships with `-DOCSTART-` markers; concatenate sentences within a document and evaluate with `seqeval` over the concatenated sequence. Cheapest option, tests the pipeline.
- **CDR** (Li et al., ["BioCreative V CDR task corpus"](https://doi.org/10.1093/database/baw068), *Database 2016*). Full PubMed abstracts, ~250 tokens each; chemical / disease entities.
- **SciERC** (Luan et al., ["Multi-Task Identification of Entities, Relations, and Coreference for Scientific Knowledge Graph Construction"](https://aclanthology.org/D18-1360/), *EMNLP 2018*). Scientific abstracts, ~130 tokens with nested spans and cross-sentence coreference — useful for exercise-03 later.
- **MedMentions** (Mohan & Li, ["MedMentions: A Large Biomedical Corpus Annotated with UMLS Concepts"](https://arxiv.org/abs/1902.09476), *AKBC 2019*). PubMed abstracts with fine-grained UMLS entity annotations; used again in exercise-04.

## Problem statement

### Part A — Baseline: flat BIO on the nested corpus

Before you build a span-based model, prove that flat BIO under-scores nested corpora.

1. Convert the nested annotations to a *flat* BIO stream by keeping only the outermost entity at each position (drop the inner mentions).
2. Fine-tune the same encoder recipe you used in exercise-01 (`microsoft/deberta-v3-base` or equivalent). Train for 3 epochs.
3. Evaluate with `seqeval` under `mode="strict"`.

Report micro-F1 **against the flat gold** and **against the full nested gold** (the latter will be substantially lower because you cannot predict inner mentions at all). The gap is the ceiling the span-based model has to beat.

### Part B — Span-based nested NER

Implement or use a span-classification model per chapter [06](../06-span-based-ner-for-flat-nested-and-discontinuous-entities.md):

- Either roll a minimal SpERT-style model from the code snippet in chapter 06, or use `SpanMarker` (<https://github.com/tomaarsen/SpanMarkerNER>) if it fits your dataset.
- Set `max_span_length` to the 99th percentile of your gold-entity token length. Log the value.
- Enumerate all candidate spans up to `max_span_length`; sample `k = 100` negatives per document.
- Encoder: same one you used in Part A.

Train, then evaluate with a **nested-aware span-F1 scorer** — a predicted span is a TP iff its start, end, and type match a gold span; inner and outer mentions are scored independently. Do **not** use `seqeval` here (it assumes non-overlapping spans); use a small custom scorer or the one that ships with `SpanMarker`.

Report:

- Overall micro-F1 across all mentions.
- Micro-F1 restricted to inner mentions (the ones flat BIO cannot see).
- Micro-F1 restricted to outer mentions (comparable to Part A's outer-only ceiling).

The interesting number is the inner-mention F1 — that is what the span model earns you.

### Part C — Sliding-window document-level NER

On the document-level corpus:

1. Fine-tune a base encoder (start with `microsoft/deberta-v3-base` or `roberta-base`) with `stride`-overlap tokenisation per chapter [07](../07-document-level-and-long-context-ner.md).
   ```python
   tokenizer(
       tokens, is_split_into_words=True,
       truncation=True, max_length=384, stride=128,
       return_overflowing_tokens=True,
   )
   ```
2. Train for 3 epochs.
3. At inference, decode each window independently and **merge windowed predictions with max-score deduplication**. Save the merged prediction list per document.
4. Evaluate `seqeval` `mode="strict"` over the full-document concatenated tag sequence — not per sentence, not per window.

Report:

- Overall micro-F1 at document level.
- **Cross-window disagreement rate** — fraction of overlap-region tokens where two windows predicted different tags. If > 5 %, note it and hypothesise why.

### Part D — Long-context encoder

Fine-tune `allenai/longformer-base-4096` on the same document-level corpus with `max_length = 1 024` (or 2 048 / 4 096 if you have the VRAM). Same training recipe otherwise; skip the sliding window entirely.

Report:

- Overall micro-F1 at document level.
- **Wall-clock** training and inference cost vs. Part C. Include both numbers — long-context encoders trade compute for context.

### Part E — Comparison and error analysis

Produce a comparison table:

| Model                                    | Micro-F1 (outer) | Micro-F1 (inner) | Wall-clock (train) | Wall-clock (infer, per doc) |
|------------------------------------------|------------------|------------------|--------------------|-----------------------------|
| Flat BIO baseline (Part A)               |                  | n/a              |                    |                             |
| Span-based nested (Part B)               |                  |                  |                    |                             |
| Sliding-window doc-level (Part C)        |                  | n/a              |                    |                             |
| Longformer doc-level (Part D)            |                  | n/a              |                    |                             |

Then pick **two failure cases per model** — one where nesting / long-range context clearly mattered, one where the model still got it wrong. Quote the sentence, show the gold and predicted spans, and explain in one sentence what the model missed.

### Part F — Write-up

A 400–600 word `README.md` covering:

- Which nested and document-level datasets you chose, and why.
- The comparison table from Part E.
- The most surprising number and one sentence of interpretation.
- One concrete production trade-off you would defend: e.g., "sliding window at `stride=128` beats Longformer by 0.4 F1 at 4× the throughput on this corpus — I would ship the sliding window."
- One thing you would try next (e.g., biaffine scorer, GLiNER zero-shot, coreference-augmented tagging from chapter 12).

## Starter guidance

- **Nested scoring is different from seqeval.** Write a tiny scorer: `TP = |predicted ∩ gold|` on the multiset of `(start, end, type)` tuples per document. Do not try to force nested predictions into a BIO tag stream.
- **Log truncation.** In Part C, log the fraction of documents that hit `max_length` in the *unstrided* setting. If it's > 20 %, sliding-window is genuinely necessary rather than a stylistic choice.
- **Sliding-window position drift.** The same token gets different position IDs in different windows. That is fine — the model handles it. But if you cache activations across windows to speed inference, you must recompute at overlap boundaries.
- **Longformer global-attention mask.** For token classification, set `global_attention_mask = 1` on `[CLS]` only; per-token global attention on long documents blows up memory. Chapter 07 goes deeper.
- **Reproducibility.** Fix seeds (`transformers.set_seed(42)`, `torch.use_deterministic_algorithms(True)` where possible). Nested-NER benchmarks are noisy — a single-seed number is not a claim.
- **Do not compare BIO F1 to nested F1 as if they were the same metric.** They are computed on different label sets. When comparing Part A vs. Part B, restrict Part B's scoring to outer mentions.

## Acceptance criteria

- [ ] Flat BIO baseline (Part A) trained and evaluated; F1 reported against both flat and full-nested gold, with the gap called out.
- [ ] Span-based nested model (Part B) trained; overall / inner / outer micro-F1 reported using a nested-aware scorer.
- [ ] Sliding-window document-level model (Part C) trained and merged with max-score deduplication; document-level `seqeval` micro-F1 reported.
- [ ] Long-context encoder model (Part D) trained; document-level micro-F1 and wall-clock reported.
- [ ] Comparison table in Part E populated, including at least one inner-mention F1 column for the span model.
- [ ] Two failure cases per model documented with a sentence-level analysis.
- [ ] 400–600 word write-up in `README.md` with the table, one surprising number, one production trade-off, and one next step.

## Stretch goals

- **Biaffine nested NER.** Reimplement Part B with the Yu, Bohnet & Poesio biaffine scorer (["Named Entity Recognition as Dependency Parsing"](https://arxiv.org/abs/2005.07150), *ACL 2020*). Report the F1 delta vs. SpERT-style span classification.
- **GLiNER zero-shot.** Run `gliner-community/gliner_small-v2.5` (Zaratiana et al., ["GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer"](https://arxiv.org/abs/2311.08526), *NAACL 2024*) zero-shot on the nested corpus with entity-type strings as labels. How close does it get to your fine-tuned span model?
- **Ensemble across windows.** Replace max-score merging in Part C with logit-averaging across overlapping windows. Report the F1 delta and the extra inference cost.
- **Coreference-augmented decoding.** Run a coreference model (chapter [12](../12-coreference-resolution-at-document-scale.md), e.g., `fastcoref`) as preprocessing on the document-level corpus and use its clusters to propagate entity-type labels across coreferent mentions. Report the F1 delta and where it came from (recall boost on pronouns? on abbreviations?).
- **Boundary-first span pruning.** Implement Shen et al.'s "Locate and Label" two-stage pruning (["Locate and Label: A Two-stage Identifier for Nested Named Entity Recognition"](https://arxiv.org/abs/2105.06804), *ACL 2021*) for Part B. Report the F1 and the drop in per-doc span-classification calls.
