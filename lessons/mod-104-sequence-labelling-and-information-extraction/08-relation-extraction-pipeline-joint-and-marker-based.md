# Relation Extraction: Pipeline, Joint, and Marker-Based

## Motivation

Once you have entities (chapters 04–07), the next question is: *which pairs of entities are related, and how?* Relation extraction (RE) is the workhorse task for populating knowledge graphs, running compliance queries, and driving structured search. It shows up in every serious IE pipeline:

- Given a protein and a disease in a biomedical paper, is this an *inhibits*, *treats*, or *unrelated* relation?
- Given a company and an executive in an SEC filing, is this a *ceo_of*, *board_member_of*, or *former_ceo_of* relation?
- Given a party and a jurisdiction in a contract, does *governed_by* hold?

RE turns pairs of entities into typed triples `(head_entity, relation_type, tail_entity)`. This chapter builds the three model families — pipeline, joint, and marker-based — and the design decisions that separate a competitive RE model from a mediocre one.

## The task, precisely

Given a text `x` and a set of entities `E = {(span, type)}` (either gold or predicted by a NER model), predict a set of relation triples `R = {(e_h, r, e_t) : e_h, e_t ∈ E, r ∈ RelationTypes ∪ {no_relation}}`.

Two common evaluation conditions:

- **Gold entities, predicted relations.** RE-only evaluation, isolating the relation model's contribution. Standard on ACE 2005, TACRED, SciERC "REL" tracks.
- **End-to-end.** Entities and relations both predicted. Errors compound; a relation only counts if both entity spans and their types are correct. Standard on CoNLL04, ADE, SciERC end-to-end.

Report both when possible; the gap between them tells you how much your NER errors bottleneck your RE.

## Family 1: pipeline

The oldest and simplest recipe:

1. Run NER (chapters 04–06) to get entity spans.
2. For each candidate entity pair `(e_h, e_t)` in a sentence or document, run a relation classifier over `{no_relation} ∪ RelationTypes`.
3. Emit non-`no_relation` triples.

The relation classifier is a text-classification model whose input is the raw text with the two candidate entities somehow marked. The marking strategy is the whole story.

## The entity marking strategies

You need to tell the encoder *which two entities are the candidates for this classification*. Three approaches, in order of modern preference:

### Bare (no marking)

Concatenate `[CLS] entity_1 [SEP] entity_2 [SEP] text` or `[CLS] text [SEP] entity_1 [SEP] entity_2 [SEP]`. The classifier reads the entity strings but has no idea where they are in the text. Fine on very short sentences with obvious pairs; loses badly on documents with multiple mentions of the same entity string.

### Positional markers (Zhang et al., 2018)

Insert special tokens around each candidate entity:

```
Text:  "[E1] Barack Obama [/E1] visited [E2] Berlin [/E2] on Friday."
```

Then pool over `[E1]` (or its adjacent hidden state) for the head representation and `[E2]` for the tail; concatenate; classify. Simple, effective, and the baseline for the RE task on the TACRED dataset (Zhang, Zhong, Chen, Angeli & Manning, ["Position-aware Attention and Supervised Data Improve Slot Filling"](https://aclanthology.org/D17-1004/), *EMNLP 2017*).

### Typed marker tokens (Zhong & Chen, 2021) — the current default

Extend positional markers with the entity types:

```
Text:  "[E1:PER] Barack Obama [/E1:PER] visited [E2:LOC] Berlin [/E2:LOC] on Friday."
```

Zhong & Chen, ["A Frustratingly Easy Approach for Entity and Relation Extraction"](https://arxiv.org/abs/2010.12812), *NAACL 2021*, showed that this "typed markers + pipeline" recipe beats jointly trained models on ACE 2005, SciERC, and CoNLL04 by 1–3 F1. The marker tokens are added to the tokenizer vocabulary and randomly initialised; the encoder learns to use them as position + type indicators.

For extraction, pool the hidden state at the position of `[E1:TYPE]` (the opening marker) as the head representation and at `[E2:TYPE]` for the tail:

```python
head_repr = hidden[batch_idx, start_of_e1_marker, :]
tail_repr = hidden[batch_idx, start_of_e2_marker, :]
pair_repr = torch.cat([head_repr, tail_repr], dim=-1)
logits = classifier(pair_repr)  # over |RelationTypes| + 1 (no_relation)
```

Two variants that trade fidelity for speed:

- **Levitated markers** (Ye et al., ["Packed Levitated Marker for Entity and Relation Extraction"](https://arxiv.org/abs/2109.06067), *ACL 2022*): pack multiple candidate pairs into a single encoded pass by adding "floating" marker tokens that do not disrupt the text — cuts inference cost 5–10× for documents with many candidate pairs.
- **Bare `[E1]/[E2]` without types**: less information but simpler when your NER is untyped. Rarely competitive with typed markers on modern benchmarks.

## Family 2: joint entity and relation extraction

The pipeline approach has an obvious failure mode: if NER misses an entity, RE cannot recover it. Joint models train entity and relation heads together, sharing the encoder and allowing gradients from RE to reshape the entity representation.

References worth knowing:

- **SpERT** — Eberts & Ulges, ["Span-based Joint Entity and Relation Extraction with Transformer Pre-training"](https://arxiv.org/abs/1909.07755), *ECAI 2020*. Span classifier (chapter 06) for entities; pairwise scorer for relations. Share encoder; sum losses.
- **UniRE** — Wang et al., ["UniRE: A Unified Label Space for Entity Relation Extraction"](https://arxiv.org/abs/2107.04292), *ACL 2021*. Puts entities and relations in the same table-filling label space.
- **TPLinker** — Wang et al., ["TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking"](https://arxiv.org/abs/2010.13415), *COLING 2020*. Token-pair matrix formulation; handles overlapping relations naturally.

Joint models were considered SOTA from 2019 to 2021. Zhong & Chen's typed-marker pipeline result changed that story — the *pipeline* approach with strong typed markers is now competitive or better on most benchmarks. Joint approaches still win on **small-data regimes** (where sharing the encoder helps) and on **schemas with heavy overlap** (an entity that participates in many relations).

Rule of thumb: default to typed-marker pipeline; try joint when data is scarce or relations overlap heavily.

## Family 3: extraction as sequence generation

The decoder-LLM alternative. Prompt an LM (T5, BART, GPT, Claude) to emit relation triples as text or JSON:

```
Prompt:
Extract relation triples from the following text.
Text: Barack Obama visited Berlin on Friday.
Output as JSON: [{"head": ..., "relation": ..., "tail": ...}]
```

References:

- **REBEL** — Cabot & Navigli, ["REBEL: Relation Extraction By End-to-end Language generation"](https://aclanthology.org/2021.findings-emnlp.204/), *EMNLP Findings 2021*. Fine-tuned BART that emits linearised triples; strong end-to-end results on multiple RE benchmarks with a single model.
- **TANL** — Paolini et al., ["Structured Prediction as Translation between Augmented Natural Languages"](https://arxiv.org/abs/2101.05779), *ICLR 2021*. Frames RE (and NER, and EE) as a translation task.

Chapter 10 covers the modern decoder-LLM approach and its constrained-decoding tooling. For now, note that generation-based RE:

- **Handles nested and overlapping relations naturally** — no `no_relation` combinatorics.
- **Suffers on rare relation types** — the LM has to hallucinate the type name character-by-character.
- **Is slower** — decoder autoregression vs. encoder single-pass.

## Distant supervision and data scarcity

RE datasets are small because annotating triples is expensive. The standard workaround for two decades has been distant supervision: align a knowledge base (Freebase, Wikidata, UMLS) to text, and every sentence mentioning both entities of a KB triple is a positive example for that relation.

- **Mintz, Bills, Snow & Jurafsky, ["Distant supervision for relation extraction without labeled data"](https://aclanthology.org/P09-1113/), *ACL 2009*.** The original paper.
- **Multi-instance learning** — Riedel, Yao & McCallum, ["Modeling Relations and Their Mentions without Labeled Text"](https://link.springer.com/chapter/10.1007/978-3-642-15939-8_10), *ECML-PKDD 2010*. Treats a bag of sentences mentioning the same pair as one training example — mitigates label noise.
- **NYT-Freebase** — the standard distant-supervision benchmark; noisy labels, ~50 relations.

Chapter 13 (active learning + weak supervision) generalises this idea. For domain RE, a distant-supervision pass followed by human review of the top-scoring predictions is often the fastest way to bootstrap a labelled corpus.

## Document-level RE

Sentence-level RE covers ~60 % of real relations. The other ~40 % are cross-sentence: subject in paragraph one, action in paragraph three. Document-level RE datasets and models:

- **DocRED** — Yao et al., ["DocRED: A Large-Scale Document-Level Relation Extraction Dataset"](https://arxiv.org/abs/1906.06127), *ACL 2019*. ~5 000 Wikipedia articles, 96 relation types, many cross-sentence.
- **ATLOP** — Zhou et al., ["Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling"](https://arxiv.org/abs/2010.11304), *AAAI 2021*. Adaptive per-instance thresholding; strong DocRED baseline.
- **SSAN** — Xu, Wang & Sun, ["Entity Structure Within and Throughout: Modeling Mention Dependencies for Document-Level Relation Extraction"](https://arxiv.org/abs/2102.10249), *AAAI 2021*.

Document-level RE typically pairs a long-context encoder (chapter 07) with an adaptive-threshold classifier — the number of true relations per document varies, so a single threshold across all documents is a bad fit.

## Metrics

- **Precision, recall, F1 over triples** — the standard. A triple `(e_h, r, e_t)` counts as correct if head, tail, and relation type all match a gold triple.
- **Micro-F1** is the headline; **macro-F1 across relation types** exposes rare-type performance.
- **Rel+ vs. Rel (Bekoulis et al., 2018)** — some benchmarks distinguish "entity spans correct AND relation correct" (strict, Rel+) from "relation correct given gold entities" (Rel). Report the one the paper you're comparing against uses.

## The full pipeline recipe, condensed

```python
# Step 1: NER (chapter 04)
ner_predictions = ner_pipeline(text)

# Step 2: Enumerate candidate pairs (all pairs of entities in the same sentence or document)
pairs = [(e1, e2) for e1 in entities for e2 in entities if e1 != e2]

# Step 3: Build typed-marker inputs
def make_input(text, e1, e2):
    # Insert [E1:TYPE] ... [/E1:TYPE] and [E2:TYPE] ... [/E2:TYPE] around the entity spans
    ...

# Step 4: Classify
inputs = [make_input(text, e1, e2) for e1, e2 in pairs]
logits = re_model(**tokenizer(inputs, ...))
preds = logits.argmax(-1)
triples = [
    (e1, id2rel[p.item()], e2)
    for (e1, e2), p in zip(pairs, preds)
    if id2rel[p.item()] != "no_relation"
]
```

`SpanMarker` and the `PL-Marker` reference code (<https://github.com/thunlp/PL-Marker>) are the current strong open-source implementations for typed-marker RE.

## Failure modes worth naming

1. **Pair explosion.** A 20-entity sentence has 380 ordered pairs. A 200-entity document has 39 800. Filter aggressively — same-sentence only, distance limit, type-compatibility filter (e.g., `works_at` only fires on `(PER, ORG)` pairs).
2. **No `no_relation` in training.** If your training data has only positive triples, the classifier learns "always predict some relation." Explicitly sample `no_relation` pairs (typically 5–10× positive count) during training.
3. **Marker tokens not added to vocab.** `[E1:PER]` becomes `[` + `E1` + `:` + `PER` + `]` under the tokenizer, defeating the purpose. Add markers with `tokenizer.add_tokens` and `model.resize_token_embeddings`.
4. **Threshold-per-relation drift.** Different relations need different thresholds — `subsidiary_of` may need `0.7`, `worked_at` needs `0.5`. Use adaptive thresholding (ATLOP) or per-type threshold tuning.
5. **Symmetric vs. directed relations.** `spouse_of` is symmetric; `subsidiary_of` is not. Some datasets encode both `A_subsidiary_of_B` and `B_parent_of_A`; some don't. Check the schema before training.
6. **Entity-type mismatch with NER.** RE trained on gold entities but deployed on NER-predicted entities: entity boundaries and types drift, and RE F1 drops. Train RE on NER-predicted entities (with gold relations aligned) for realistic deployment numbers.

## Chapter summary

- Relation extraction turns entity pairs into typed triples; typed-marker pipeline is the current default, joint models still win on data-scarce or overlap-heavy schemas.
- Marker strategy is where the whole recipe lives: typed positional markers (`[E1:TYPE]`) at the entity boundaries, pool the marker hidden state, classify.
- Joint models (SpERT, TPLinker, UniRE) share the encoder across NER and RE; useful when NER errors bottleneck RE.
- Sequence-generation RE (REBEL, TANL) handles overlapping relations naturally at the cost of decoder speed; chapter 10 covers the decoder-LLM version.
- Distant supervision from a KB is how you bootstrap labelled data; multi-instance learning is the standard noise-mitigation.
- Document-level RE (DocRED, ATLOP) pairs long-context encoders with adaptive thresholding; a single global threshold is a bad fit.
- Evaluate over triples (P/R/F1 micro), report per-relation macro-F1 to catch tail regressions, and always report both gold-entity and end-to-end conditions when the paper you compare against does.
