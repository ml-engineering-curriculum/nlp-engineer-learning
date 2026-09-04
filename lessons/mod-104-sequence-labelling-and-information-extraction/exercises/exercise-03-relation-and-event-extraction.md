# exercise-03: Relation and Event Extraction

**Estimated effort:** 3 hours

## Objective

Build a working relation extractor with the typed-marker pipeline recipe, then extend it to event extraction (trigger + argument roles) on a small ontology. You will evaluate both under gold-entity and end-to-end conditions, and honestly report where NER errors bottleneck downstream RE / EE F1. By the end you should be able to defend one pipeline vs. joint vs. generation choice with numbers, not intuition.

## Prerequisites

- Exercise-01 (you have a working NER model) and preferably exercise-02.
- Chapters [08](../08-relation-extraction-pipeline-joint-and-marker-based.md), [09](../09-event-extraction-and-slot-filling.md).
- Python 3.10+; `transformers`, `datasets`, `evaluate`, `torch`.
- A GPU with ≥ 8 GB VRAM is comfortable.

## Datasets

Pick **one relation corpus**. If you have time, also do the event-extraction part on a small ontology — otherwise stick to slot filling (Part D-alt).

### Relation extraction

- **TACRED** (Zhang, Zhong, Chen, Angeli & Manning, ["Position-aware Attention and Supervised Data Improve Slot Filling"](https://aclanthology.org/D17-1004/), *EMNLP 2017*). Newswire; 41 relation types. LDC-licensed. Prefer **TACREV** (Alt et al., ["TACRED Revisited"](https://aclanthology.org/2020.acl-main.142/), *ACL 2020*) or **Re-TACRED** (Stoica et al., ["Re-TACRED: Addressing Shortcomings of the TACRED Dataset"](https://arxiv.org/abs/2104.08398), *AAAI 2021*) — same task, cleaner labels.
- **SemEval-2010 Task 8** (Hendrickx et al., ["SemEval-2010 Task 8: Multi-Way Classification of Semantic Relations Between Pairs of Nominals"](https://aclanthology.org/S10-1006/), *SemEval 2010*). 9 general relations; small and open-access.
- **DocRED** (Yao et al., ["DocRED: A Large-Scale Document-Level Relation Extraction Dataset"](https://arxiv.org/abs/1906.06127), *ACL 2019*). Document-level, 96 relation types. Harder; pick this for a stretch.
- **ChemProt / BioCreative VI** (Krallinger et al., ["Overview of the BioCreative VI chemical-protein interaction Track"](https://biocreative.bioinformatics.udel.edu/media/store/files/2017/chemprot_overview_v03.pdf), *BioCreative VI 2017*). Biomedical relations between chemicals and proteins.

### Event extraction (optional Part D)

- **ACE 2005** (Doddington et al., ["The Automatic Content Extraction (ACE) Program"](http://www.lrec-conf.org/proceedings/lrec2004/pdf/5.pdf), *LREC 2004*). 33 event types with argument roles. LDC-licensed.
- **MAVEN** (Wang et al., ["MAVEN: A Massive General Domain Event Detection Dataset"](https://arxiv.org/abs/2004.13590), *EMNLP 2020*). 168 event types; open-access. Trigger detection only in the base release — MAVEN-ARG (Wang et al., 2024) added arguments.
- **BioNLP-ST 2011 Genia** (Kim et al., ["Overview of Genia Event Task in BioNLP Shared Task 2011"](https://aclanthology.org/W11-1802/), *BioNLP 2011*). Biomedical event extraction.

### Slot filling (alternate Part D)

- **MASSIVE** (FitzGerald et al., ["MASSIVE: A 1M-Example Multilingual NLU Dataset"](https://arxiv.org/abs/2204.08582), *ACL 2023*). 60 intents, 55 slots. Open-access; multilingual.
- **SNIPS** or **ATIS** — smaller intent + slot datasets. Fine for a laptop.

## Problem statement

### Part A — Typed-marker relation extraction

Implement the Zhong & Chen typed-marker pipeline (["A Frustratingly Easy Approach for Entity and Relation Extraction"](https://arxiv.org/abs/2010.12812), *NAACL 2021*):

1. Add marker tokens `[E1:TYPE]`, `[/E1:TYPE]`, `[E2:TYPE]`, `[/E2:TYPE]` for each of your entity types to the tokenizer:
   ```python
   markers = [f"[E{i}:{t}]" for i in (1, 2) for t in entity_types]
   markers += [f"[/E{i}:{t}]" for i in (1, 2) for t in entity_types]
   tokenizer.add_tokens(markers)
   model.resize_token_embeddings(len(tokenizer))
   ```
2. For each training example, insert markers around the two candidate entities. Sample `no_relation` negatives at ~5× the positive count.
3. Encode with `microsoft/deberta-v3-base` (or `roberta-base`). Pool the hidden state at the position of `[E1:TYPE]` for the head, at `[E2:TYPE]` for the tail. Concatenate and classify over `{no_relation} ∪ RelationTypes`.
4. Train for 3–5 epochs. Standard hyperparameters: LR `3e-5`, batch 16, `warmup_ratio=0.06`, `weight_decay=0.01`.

Evaluate on the test split under **gold entities** — the standard RE-only condition. Report:

- Overall micro-F1 across relation types.
- Per-relation macro-F1 as a Markdown table with `Type / Support / Precision / Recall / F1`.
- `no_relation` precision — if your model predicts non-`no_relation` on 50 % of true `no_relation` pairs, your negatives were under-sampled.

### Part B — End-to-end evaluation

Run your NER model from exercise-01 on the test split, then feed its predicted entities into the RE model instead of gold entities. Enumerate all candidate pairs within a sentence; type-filter to plausible pairs when the schema allows (`works_at` only fires on `(PER, ORG)`).

Report:

- End-to-end micro-F1 on relations (a triple is correct iff both entity spans, both entity types, and the relation type all match a gold triple).
- The **gold-vs-end-to-end gap**. This gap is your NER's contribution to RE loss. A 15-point gap is common; a 30-point gap means your NER is the bottleneck.

### Part C — Marker ablation

Retrain the same relation classifier with **two ablated variants**:

1. **Bare markers** (`[E1]`, `[/E1]`, `[E2]`, `[/E2]`) — no type in the marker.
2. **No markers** — concatenate `[CLS] head_entity_text [SEP] tail_entity_text [SEP] sentence_text`.

Report the three F1s side-by-side under gold entities. Explain in one sentence what the typed markers buy you.

### Part D — Event extraction OR slot filling (pick one)

#### Option D-EE: Event extraction

On an event-extraction corpus:

1. **Trigger detection.** Train a token classifier over `{no_trigger} ∪ EventTypes` per chapter [09](../09-event-extraction-and-slot-filling.md). Use the same encoder + `seqeval` scoring as your NER exercise, but the labels are event types.
2. **Argument-role classification.** For each detected trigger + each candidate entity (from your NER model or gold), predict the argument role using a typed-marker recipe (same as Part A but with a trigger marker instead of a second entity marker).
3. Apply a type-compatibility filter: if `agent` roles only accept `PER` / `ORG`, drop implausible pairs.

Report **trigger F1**, **argument F1** (strict — trigger type, argument span, and role all correct), and **end-to-end** vs. **gold-trigger** argument F1.

#### Option D-SF: Slot filling

On MASSIVE / SNIPS / ATIS:

1. Train a joint intent + BIO-slot model per Chen, Zhuo & Wang, ["BERT for Joint Intent Classification and Slot Filling"](https://arxiv.org/abs/1902.10909), *arXiv 2019*. Shared encoder, `[CLS]` head for intent, per-token head for slot BIO.
2. Report **intent accuracy**, **slot F1** (`seqeval`), and **semantic frame accuracy** (intent + all slot values correct).

### Part E — Comparison and error analysis

Comparison table:

| Model / Condition                                   | Micro-F1 | Notes                       |
|-----------------------------------------------------|----------|-----------------------------|
| Typed-marker RE, gold entities (Part A)             |          |                             |
| Typed-marker RE, end-to-end (Part B)                |          | NER-loss contribution: ___  |
| Bare-marker RE, gold entities (Part C ablation 1)   |          |                             |
| No-marker RE, gold entities (Part C ablation 2)     |          |                             |
| Trigger F1 / Argument F1 / Frame accuracy (Part D)  |          |                             |

Then pick **three failure cases** across the ensemble. For each: quote the sentence, show the gold and predicted triple / event, and explain what the model got wrong (ambiguous coreference? type-filter overreach? overlapping relations?).

### Part F — Write-up

A 400–600 word `README.md` covering:

- Which corpus / schema you used.
- The comparison table from Part E.
- The typed-marker vs. no-marker delta and one sentence of interpretation.
- The gold-vs-end-to-end delta and one sentence about which stage of the pipeline you would improve first.
- One thing you would try next (joint model? generation-based extraction? LLM structured output?).

## Starter guidance

- **Marker vocab.** Adding new tokens with `tokenizer.add_tokens` then `model.resize_token_embeddings` is easy to forget. Your training will look fine but your marker positions will silently break — the model will read `[E1:PER]` as `[` + `E1` + `:` + `PER` + `]`. Assert that `tokenizer.tokenize("[E1:PER]") == ["[E1:PER]"]` before training.
- **Negative sampling.** For every positive pair `(e_h, e_t, r)`, sample 5× as many `(e_h, e_t, no_relation)` pairs. Without them, the classifier learns "never predict `no_relation`" and blows up in end-to-end scoring.
- **Direction matters.** `subsidiary_of(A, B)` ≠ `subsidiary_of(B, A)`. Enumerate both orderings for every pair and let the classifier pick the direction.
- **Type-compatibility filter.** Log its rejection rate. If it drops > 10 % of true positive pairs, your NER type errors are propagating; loosen the filter.
- **Gold-entity vs. end-to-end.** Publish both numbers or your comparison is meaningless. Papers that report only gold-entity F1 while claiming production readiness are papering over the NER bottleneck.
- **Event extraction is two models stitched together.** Train them separately, evaluate both. Do not try to train a single-shot joint model on your first pass unless the corpus is large.

## Acceptance criteria

- [ ] Typed-marker RE model (Part A) trained; overall + per-type F1 reported under gold entities.
- [ ] End-to-end RE F1 (Part B) reported; NER-vs-RE gap called out.
- [ ] Two marker ablations (Part C) reported; typed vs. bare vs. none.
- [ ] Either the event extraction stack (trigger + argument, D-EE) OR the joint intent+slot model (D-SF) trained and evaluated.
- [ ] Comparison table (Part E) populated.
- [ ] Three failure cases documented with sentence-level analysis.
- [ ] 400–600 word `README.md` write-up.

## Stretch goals

- **Joint SpERT-style model.** Reimplement Part A with a joint entity + relation model (Eberts & Ulges, ["Span-based Joint Entity and Relation Extraction with Transformer Pre-training"](https://arxiv.org/abs/1909.07755), *ECAI 2020*). Compare against your pipeline. Where does joint win — small data, overlapping relations, both?
- **Levitated markers.** Implement Ye et al.'s packed-marker encoding (["Packed Levitated Marker for Entity and Relation Extraction"](https://arxiv.org/abs/2109.06067), *ACL 2022*) to score multiple candidate pairs in one forward pass. Measure inference-cost savings.
- **REBEL / generation-based RE.** Run `Babelscape/rebel-large` (Cabot & Navigli, ["REBEL: Relation Extraction By End-to-end Language generation"](https://aclanthology.org/2021.findings-emnlp.204/), *EMNLP Findings 2021*) zero-shot on your test split. How does it compare to your typed-marker pipeline?
- **Document-level RE.** Run ATLOP (Zhou et al., ["Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling"](https://arxiv.org/abs/2010.11304), *AAAI 2021*) on DocRED and report micro-F1 with adaptive vs. global thresholds.
- **Distant-supervision bootstrap.** Take an unlabelled corpus (Wikipedia articles, PubMed abstracts) and Wikidata / UMLS triples; auto-label a training set via distant supervision. Retrain and report F1 vs. your gold-only model.
- **LLM-based extraction.** Prompt a strong LLM with structured output (chapter [10](../10-schema-driven-structured-extraction-with-decoder-llms.md)) for the same task; compare F1 and per-doc cost.
