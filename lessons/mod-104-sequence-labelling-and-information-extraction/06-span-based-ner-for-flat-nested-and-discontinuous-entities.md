# Span-Based NER for Flat, Nested, and Discontinuous Entities

## Motivation

BIO tagging works when entities are contiguous and non-overlapping. That covers CoNLL-2003 and most newswire corpora, but it breaks the moment you look at real biomedical, legal, or financial text:

- **GENIA** (Kim et al., ["GENIA corpus — a semantically annotated corpus for bio-textmining"](https://academic.oup.com/bioinformatics/article/19/suppl_1/i180/227927), *Bioinformatics 2003*) has protein mentions nested inside protein-complex mentions.
- **ACE 2004/2005** ships entity mentions where a person is nested inside an organisation ("**the Bush** administration").
- **Discontinuous entities** like *"heart and lung disease"* denote *{heart disease, lung disease}* — two entities that share tokens but are non-contiguous.

Span-based NER abandons per-token tagging and instead scores every candidate *span* directly: "is `text[i:j]` an entity of type `T`?" This side-steps the BIO ambiguity, handles nesting natively, and — with a discontinuous-span extension — handles the general graph case.

## The span-classification recipe

The core idea is small enough to state in one loop:

1. Encode the input with a transformer.
2. Enumerate candidate spans `(i, j)` with `j - i ≤ L` (a max-span-length hyperparameter).
3. For each span, build a representation from the encoder's hidden states — typically concatenate the boundary tokens' hidden states plus a length feature: `[h_i; h_j; length_embedding(j-i)]`.
4. Feed each span representation through an MLP classifier over the label set `{O, PER, ORG, LOC, ...}`.
5. Loss: cross-entropy over the flat span-label space, with `O` for spans that do not match any gold entity.

Training turns into a many-way classification per span, with gold labels for spans that match a gold entity and `O` for the rest. Decoding is greedy: keep every span classified as an entity type, resolve conflicts (nesting is allowed; identical `(i, j, type)` duplicates are impossible; identical `(i, j)` with different types are broken with softmax argmax).

## SpERT — the canonical modern reference

Eberts & Ulges, ["Span-based Joint Entity and Relation Extraction with Transformer Pre-training"](https://arxiv.org/abs/1909.07755), *ECAI 2020*, introduced SpERT and made the span-classification recipe a modern baseline for joint entity and relation extraction. The span classifier is trained jointly with the relation classifier (chapter 08 covers the RE side).

Key SpERT choices worth internalising:

- **Span representation** = `[h_start; h_end; max_pool(h_start:h_end); width_embedding]`. The max-pool captures inner-span content; the width embedding disambiguates 2-word from 8-word spans of the same boundary shape.
- **Negative sampling.** Sampling every span up to length `L` per document is quadratic; SpERT samples `k` negative spans per document instead of enumerating all. Standard `k` = 100 for base-size models.
- **Softmax over labels ∪ {O}.** A span classified as `O` is not an entity. Softmax rather than sigmoid because a span has exactly one type in flat schemas.

## Handling nested entities

Under BIO, nested entities require layered tagging or model surgery. Under span classification, nesting is free: two spans `(i, j)` and `(i', j')` with `i ≤ i' < j' ≤ j` can each be labelled and both survive decoding. No modification needed.

Reference points for nested NER performance:

- **Yu, Bohnet & Poesio, ["Named Entity Recognition as Dependency Parsing"](https://arxiv.org/abs/2005.07150), *ACL 2020*.** Biaffine span scorer; became the strong baseline for nested NER on ACE 2004/2005 and GENIA.
- **Wang et al., ["Pyramid: A Layered Model for Nested Named Entity Recognition"](https://aclanthology.org/2020.acl-main.525/), *ACL 2020*.** Layered scoring that respects the tree structure of nested mentions.
- **Shen et al., ["Locate and Label: A Two-stage Identifier for Nested Named Entity Recognition"](https://arxiv.org/abs/2105.06804), *ACL 2021*.** Locate candidate spans first with a boundary detector, then classify. Faster than exhaustive enumeration.

For most projects the biaffine or SpERT-style scorer at `L = 30` is the right default.

## Handling discontinuous entities

Discontinuous entities are the real graph case. Two references worth knowing:

- **Dai, Karimi, Hachey & Paris, ["An Effective Transition-based Model for Discontinuous NER"](https://aclanthology.org/2020.acl-main.520/), *ACL 2020*.** Transition-based parser that emits contiguous and discontinuous spans in one pass.
- **Fei, Ren & Ji, ["Rethinking Boundaries: End-to-End Recognition of Discontinuous Mentions with Pointer Networks"](https://arxiv.org/abs/2003.11080), *AAAI 2021*.** Pointer-network approach.

Most industrial projects treat discontinuous entities as a "either merge to outer span or ignore" problem. The 1–3 % of the CADEC / ShARe corpora that are truly discontinuous rarely justify the model complexity. Know the pattern, know the two papers, move on.

## The training loop, concretely

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class SpanNER(nn.Module):
    def __init__(self, model_name, num_labels, max_span_length=30, width_dim=25):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.width_embed = nn.Embedding(max_span_length + 1, width_dim)
        self.classifier = nn.Sequential(
            nn.Linear(3 * h + width_dim, h),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(h, num_labels),
        )
        self.max_span_length = max_span_length

    def forward(self, input_ids, attention_mask, spans, span_labels=None):
        # spans: LongTensor of shape (B, S, 2) with (start, end) indices
        # span_labels: LongTensor of shape (B, S) with entity type ids (0 = O)
        hidden = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        B, S, _ = spans.shape
        h_start = torch.gather(hidden, 1, spans[..., 0:1].expand(-1, -1, hidden.size(-1)))
        h_end = torch.gather(hidden, 1, spans[..., 1:2].expand(-1, -1, hidden.size(-1)))
        widths = spans[..., 1] - spans[..., 0]
        widths = widths.clamp(0, self.max_span_length)
        w_emb = self.width_embed(widths)
        # max-pool over the span
        pooled = []
        for b in range(B):
            for s in range(S):
                i, j = spans[b, s, 0].item(), spans[b, s, 1].item()
                pooled.append(hidden[b, i:j + 1].max(0).values)
        pooled = torch.stack(pooled).view(B, S, -1)
        span_repr = torch.cat([h_start.squeeze(-2), h_end.squeeze(-2), pooled, w_emb], dim=-1)
        logits = self.classifier(span_repr)
        loss = None
        if span_labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                span_labels.view(-1),
                ignore_index=-100,
            )
        return loss, logits
```

The dataloader is more involved than for token classification: you generate candidate spans on-the-fly, keeping all gold-positive spans and sampling `k` negative spans per document. Reference implementations: SpanMarker (<https://github.com/tomaarsen/SpanMarkerNER>), SpERT (<https://github.com/lavis-nlp/spert>), and the `nested_ner` recipes in `flair`.

## Comparing to BIO in numbers

Rough rules of thumb from published comparisons (Yu et al., 2020; Eberts & Ulges, 2020; and the SpanMarker benchmark table):

- On **flat, closed-schema NER** (CoNLL-2003, OntoNotes), span-based models match BIO within 0.5 F1 with a similar encoder. Slightly slower training, similar inference latency.
- On **nested NER** (ACE, GENIA), span-based models beat layered BIO by 3–8 F1.
- On **zero-shot NER** (unseen types), span-based models with a *label-aware* head (see below) beat BIO by more, because they don't have to relearn `B-<newtype>` and `I-<newtype>` tags.
- On **discontinuous NER** (CADEC, ShARe), transition-based discontinuous decoders beat span-classification with post-hoc merging by 2–5 F1.

Practical guidance: default to BIO for flat, closed schemas; move to span-based (biaffine or SpERT) for nested or growing-schema tasks.

## Zero- and few-shot span NER

Span-classification with a *label-aware* encoding — where the label name is embedded and scored against each span — enables zero-shot NER over labels the model never saw at training time. Two production-relevant references:

- **UniversalNER** — Zhou et al., ["UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition"](https://arxiv.org/abs/2308.03279), *ICLR 2024*. LLM-distilled model that generalises to arbitrary entity types by prompt.
- **GLiNER** — Zaratiana et al., ["GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer"](https://arxiv.org/abs/2311.08526), *NAACL 2024*. Zero-shot span NER that takes a *list of entity type names as input* and returns matching spans. Runs on a laptop, generalises across domains, and is the current default for "I need a NER model for a new schema without labelled data."

`gliner-community/gliner_small-v2.5` inference:

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("gliner-community/gliner_small-v2.5")
text = "Amazon acquired MGM for $8.45 billion."
labels = ["organization", "person", "monetary value", "date"]
model.predict_entities(text, labels)
# [{'start': 0, 'end': 6, 'text': 'Amazon', 'label': 'organization', 'score': 0.98}, ...]
```

For a new NER project with no labelled data, GLiNER (or a similar label-aware span model) is often a better first move than either BIO fine-tuning *or* LLM prompting.

## Inference latency and span-count budget

Span classification cost scales with `n × L` per document (span count), not `n` (token count). At `L = 30` on a 512-token document, that is ~15 000 span classifications per doc. Base-size models handle this at 5–10× the latency of a token-classification NER; large-size models can be 20×.

Two levers:

1. **Boundary detection first** — Shen et al.'s "Locate and Label" trains a boundary-token classifier that emits candidate spans, then a classifier scores only those. Cuts the per-doc span count by 5–10×.
2. **Lower `L`.** If your entity length distribution has 99th percentile at 12, `L = 12` is enough. Log the length distribution before setting `L`.

For latency-critical serving with a flat schema, BIO is still faster. Span-based is worth its cost when the schema demands it.

## The library ecosystem

- **`spacy` + `spacy-transformers`** — flat NER with a transition-based decoder. Fast, production-hardened; nested NER is a plugin.
- **`flair`** — BIO with optional CRF, per-type F1 out of the box.
- **`SpanMarker`** — <https://github.com/tomaarsen/SpanMarkerNER>. Wraps `AutoModelForTokenClassification` with span-marker encoding (Zhong & Chen, 2021); great baseline for standard NER.
- **`spert`** — <https://github.com/lavis-nlp/spert>. Reference SpERT implementation with joint RE (chapter 08).
- **`gliner`** — <https://github.com/urchade/GLiNER>. Zero-shot label-aware span NER.
- **`allennlp`** — biaffine and layered nested NER; still maintained enough for reproducing 2020-era baselines.

## Chapter summary

- Span-based NER scores every candidate `(i, j)` span as an entity of some type, sidestepping BIO's contiguity assumption and handling nesting natively.
- SpERT-style span representation — boundary hidden states + max-pool + width embedding — is the default; biaffine scoring is the strongest alternative.
- Nested NER is where span models earn their cost; on flat schemas they match BIO within noise.
- Zero-shot span NER with label-aware scoring (GLiNER, UniversalNER) is the current default for new schemas with no labelled data.
- Span-based inference is more expensive than BIO; boundary detection first + a well-chosen `L` recover most of the gap.
