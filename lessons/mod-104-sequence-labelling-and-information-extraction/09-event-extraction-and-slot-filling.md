# Event Extraction and Slot Filling

## Motivation

Relation extraction (chapter 08) captures binary facts about pairs of entities. Real domains need higher-arity structure. When a news article reports *"On Tuesday, Airbus delivered five A320s to Delta in Toulouse"*, you want to extract a single **event** — not a bag of pairwise relations — with typed argument slots:

```
event_type: Delivery
trigger:    "delivered"
arguments:
  - role: agent        entity: Airbus
  - role: recipient    entity: Delta
  - role: theme        entity: five A320s
  - role: place        entity: Toulouse
  - role: time         entity: Tuesday
```

The dialogue analogue — **slot filling** — has a fixed schema per intent and populates it from a user utterance:

```
intent: book_flight
slots:
  origin_city:      "SFO"
  destination_city: "LAX"
  departure_date:   "2026-08-15"
  passenger_count:  2
```

Both tasks are "structured extraction with role-typed arguments," and both are dominated in 2026 by two model families: **classic pipeline models** (trigger detection → argument classification) and **decoder-LLM structured extraction** (chapter 10 covers the latter in depth).

## Event extraction: the vocabulary

The canonical formalisation comes from the ACE 2005 (Automatic Content Extraction) programme:

- **Event mention** — a specific occurrence of an event in text, anchored by a trigger.
- **Trigger** — the word or phrase that most clearly expresses the event (usually a verb: "delivered", "resigned", "acquired").
- **Event type** — the ontology class (ACE has 33; FrameNet has 1 200+; domain schemas have ~10–50).
- **Argument** — an entity or value participating in the event.
- **Argument role** — the typed slot the argument fills (`agent`, `theme`, `time`, `place`).

References for the ontology and dataset landscape:

- **ACE 2005** — Doddington et al., ["The Automatic Content Extraction (ACE) Program – Tasks, Data, and Evaluation"](http://www.lrec-conf.org/proceedings/lrec2004/pdf/5.pdf), *LREC 2004*. The canonical 33-type ontology.
- **FrameNet** — Baker, Fillmore & Lowe, ["The Berkeley FrameNet Project"](https://aclanthology.org/C98-1013/), *COLING 1998*. Lexical-semantic frames; broader coverage than ACE.
- **PropBank** — Palmer, Gildea & Kingsbury, ["The Proposition Bank: An Annotated Corpus of Semantic Roles"](https://aclanthology.org/J05-1004/), *Computational Linguistics 2005*. Verb-specific argument annotations.
- **MAVEN** — Wang et al., ["MAVEN: A Massive General Domain Event Detection Dataset"](https://arxiv.org/abs/2004.13590), *EMNLP 2020*. 168 event types, ~4× the trigger annotations of ACE.
- **BioNLP shared tasks** (2009–2013) — Kim, Ohta, Pyysalo, Kano & Tsujii, ["Overview of BioNLP'09 Shared Task on Event Extraction"](https://aclanthology.org/W09-1401/), *BioNLP Workshop 2009*. Biomedical event extraction (protein binding, gene expression, regulation).

## The classic pipeline

Two stages, both trained as sequence-labelling / classification problems:

### Stage 1: Trigger detection

A per-token classifier over `{no_trigger} ∪ EventTypes`. Same recipe as NER (chapter 04) — encoder + linear head + cross-entropy — but the classes are event types, and most tokens are `no_trigger`.

Multi-token triggers are handled with BIO (chapter 02) or span-classification (chapter 06). ACE triggers are almost always single-token; MAVEN and BioNLP corpora have some multi-token cases.

### Stage 2: Argument role classification

For each detected trigger and each candidate entity in the sentence (or document), classify the argument role:

- Input: text with the trigger and candidate argument both marked (same typed-marker recipe as chapter 08).
- Output: `{no_role} ∪ ArgumentRoles`.

Some pipelines add a **candidate entity type filter** — `agent` roles only accept `PER` or `ORG` entities, `place` roles only `LOC`, and so on. This filter cuts the candidate space 3–5× and improves precision without hurting recall.

Reference pipelines to know:

- **DYGIE++** — Wadden, Wennberg, Luan & Hajishirzi, ["Entity, Relation, and Event Extraction with Contextualized Span Representations"](https://arxiv.org/abs/1909.03546), *EMNLP 2019*. Joint NER + RE + EE with shared span representations. Strong baseline on ACE 2005.
- **OneIE** — Lin et al., ["A Joint Neural Model for Information Extraction with Global Features"](https://aclanthology.org/2020.acl-main.713/), *ACL 2020*. Joint decoding with global constraints (an event of type X requires an argument of type Y).
- **DEGREE** — Hsu et al., ["DEGREE: A Data-Efficient Generation-Based Event Extraction Model"](https://arxiv.org/abs/2108.12724), *NAACL 2022*. Encoder-decoder that generates event structures as text; effective in low-resource settings.

## Slot filling: the dialogue-flavoured cousin

Slot filling for conversational systems has a fixed set of intents, each with a fixed slot schema. Two mainstream recipes:

### Joint intent classification + slot BIO tagging

Chen, Zhuo & Wang, ["BERT for Joint Intent Classification and Slot Filling"](https://arxiv.org/abs/1902.10909), *arXiv 2019*. One encoder, two heads: `[CLS]` embedding for intent classification, per-token BIO tagging for slot values. Trained jointly.

```
User utterance:  "book a flight from SFO to LAX tomorrow"
Intent:          book_flight
Slot tags:       O O O O B-origin O B-dest B-departure_date
```

This is the reference recipe for MultiWOZ, SNIPS, ATIS, and MASSIVE benchmarks. Strong, simple, and shippable.

### Zero-shot / open-schema slot filling

When the slot set changes weekly (or is defined by end-user), fine-tuning per schema is impractical. Options:

- **Question-answering framing** — Namazifar, Papangelis, Hakkani-Tur & Tur, ["Language Model is All You Need: Natural Language Understanding as Question Answering"](https://arxiv.org/abs/2011.03023), *ICASSP 2021*. Each slot becomes a QA question ("What is the origin city?"); slot values are extracted spans.
- **Decoder-LLM extraction** — prompt an LLM with the schema and let it emit JSON. Chapter 10 is entirely about this.

## The MASSIVE benchmark and modern multilingual NLU

FitzGerald et al., ["MASSIVE: A 1M-Example Multilingual NLU Dataset"](https://arxiv.org/abs/2204.08582), *ACL 2023*, is the current default for evaluating multilingual intent + slot filling: 51 languages, 60 intents, 55 slots, ~1M utterances. Reference results:

- Fine-tuned XLM-R base: ~85 % intent accuracy, ~75 % slot micro-F1.
- Zero-shot from strong LLMs: within 3–5 points of fine-tuned on head languages, wider gap on tail.

Reproducibility notes: MASSIVE evaluation is per-language and requires macro-averaging across languages. Reporting a single number without saying which language mix hides the tail.

## Evaluation for event extraction

Two levels, both standard on ACE 2005:

- **Trigger F1** — a predicted trigger counts as correct if its token span and event type both match a gold trigger. Standard `seqeval`-style scoring.
- **Argument F1** — a predicted argument counts as correct if the argument entity span, the trigger it attaches to, and the role type all match. The strict version (used in most modern papers) also requires the trigger event type to be correct.

Report both, and report both under gold-entity and end-to-end conditions.

## Evaluation for slot filling

- **Intent accuracy** — top-1 over the intent classes.
- **Slot F1** — micro over slot spans (via `seqeval` on BIO slot tags).
- **Semantic frame accuracy (a.k.a. exact match)** — a joint metric: intent correct AND every slot value correct. Harshest and most product-realistic.

Report all three. Slot F1 alone hides the case where a model gets 90 % of slots right on 100 % of utterances but botches one slot per utterance — a shippable classifier by slot-F1, a broken product by frame accuracy.

## Common pitfalls

1. **Trigger scarcity.** Even MAVEN has ~40 average tokens per trigger. Trigger-detection F1 on rare types (< 50 examples) is typically 20–30 F1 lower than head types. Weight or oversample.
2. **Multi-span arguments.** ACE arguments are usually single-mention spans; real IE often needs multi-mention arguments (all authors of a paper, all subsidiaries in a merger). Extend the argument head to emit sets, not singletons.
3. **Coreference blindness.** *"Airbus delivered five A320s to Delta. **The airline** was pleased."* — "the airline" is Delta. A per-sentence argument classifier misses this; run coreference (chapter 12) first for document-level EE.
4. **Ontology drift.** ACE, FrameNet, PropBank, and every domain ontology name similar events differently. Freeze your ontology per project; do not switch mid-training.
5. **Argument role type-compatibility filter too aggressive.** If your entity typing is imperfect, a strict "only PER can be `agent`" filter drops valid arguments. Log the filter's rejection rate.
6. **Domain slots that are not spans.** Slot filling in commerce often needs value normalisation ("tomorrow" → `2026-08-05`; "two people" → `2`). BIO extraction gives you the surface span; a normaliser (chapter 09 of mod-102, plus dateutil or duckling) turns it into a value.

## When to reach for decoder-LLM extraction

The classic pipeline shines when:

- The schema is stable and well-defined.
- You have ≥ 500 examples per event type / slot type.
- Deterministic latency matters.

Decoder-LLM extraction (chapter 10) wins when:

- Schema changes weekly.
- You have < 50 examples per type (few-shot regime).
- The task has arbitrary-arity arguments or complex nested structure.

Increasingly, production event-extraction systems mix both: a fast pipeline handles ~90 % of the traffic on stable event types; an LLM handles novel types, ambiguous cases, and quality-audit sampling.

## Chapter summary

- Event extraction adds a trigger and typed argument roles on top of NER + RE; slot filling is the dialogue-flavoured cousin with a fixed schema per intent.
- The classic pipeline is trigger detection → argument classification; DYGIE++, OneIE, and DEGREE are the standard modern references.
- Joint intent classification + BIO slot tagging is the workhorse for closed-schema slot filling; MASSIVE is the current multilingual benchmark.
- Evaluate at both trigger and argument level for EE; report intent accuracy, slot F1, and frame accuracy for slot filling.
- Decoder-LLM extraction (chapter 10) is the pragmatic answer when the schema changes fast or examples are scarce.
