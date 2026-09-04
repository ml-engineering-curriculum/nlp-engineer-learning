# Composing Document-Level NLP Pipelines: spaCy and Hugging Face

## Motivation

A production NLP service almost never runs a single model on a raw string. Something has to segment the document, route each segment to the right classifier, run NER on the parts that matter, resolve entities against a knowledge base, and emit a structured record for the downstream system. Every step is a place where the wrong composition — wrong order, wrong I/O contract, wrong batch boundary — silently degrades quality or throughput.

Two ecosystems dominate this composition problem: **spaCy** (with its `Language` pipeline object and the spaCy Projects DAG) and **Hugging Face Transformers** (with `pipelines` and, for orchestration, custom code on top of `AutoModel...`). They make different trade-offs. spaCy is a Cython runtime built around the `Doc` object, with strict typed components and cheap per-token annotation storage; it is fast on CPU and excellent when your pipeline is many small statistical components. Hugging Face `pipelines` are thin ergonomic wrappers around Transformer models with per-task pre/post-processing; they are the right primitive when each step is one large Transformer and you need the flexibility of raw PyTorch under the hood.

This chapter is about picking, ordering, and wiring these primitives so the pipeline you ship is one thing you can profile, cache, batch, and monitor — not a Python script with implicit contracts between models.

## The composition problem, stated concretely

A document-level pipeline is a directed graph of components with three properties:

1. **A shared document representation** that each component reads from and writes to. In spaCy that is the `Doc` and its `Span`/`Token` views with `.set_extension(...)` attributes. In Hugging Face there is no shared object — you must define one (`dataclass` / TypedDict / Pydantic model) and pass it explicitly.
2. **An ordering** that respects data dependencies. Segmentation before per-sentence classification. NER before entity linking. Coreference before relation extraction if relations reference cluster IDs.
3. **A batching boundary policy** — where in the graph you accumulate items before a model call, and where you must stay one-at-a-time (usually because a downstream step depends on a routing decision that varies per item).

Everything below is how the two ecosystems answer those three questions.

## spaCy: the `Language` pipeline

A spaCy `Language` (loaded via `spacy.load("en_core_web_trf")` or built from `spacy.blank("en")`) is a pipeline of *components* that run sequentially over a `Doc`. Components are registered via `@Language.component` (function) or `@Language.factory` (stateful class), and each declares what it reads from and writes to.

```python
import spacy
from spacy.language import Language

nlp = spacy.load("en_core_web_lg")

@Language.component("segment_pii")
def segment_pii(doc):
    # writes doc.spans["pii"], reads doc.ents
    doc.spans["pii"] = [e for e in doc.ents if e.label_ in {"PERSON", "GPE"}]
    return doc

nlp.add_pipe("segment_pii", after="ner")

for doc in nlp.pipe(texts, batch_size=64, n_process=1):
    ...
```

Three properties matter for production:

- **`nlp.pipe(texts, batch_size=..., n_process=...)`** is the batched, multi-process entry point. `batch_size` controls per-component batching; `n_process` uses `multiprocessing` to parallelise. For Transformer-backed pipelines (`en_core_web_trf`), keep `n_process=1` and set `batch_size` to what fits in GPU memory — CPU multiprocessing does not help.
- **Component ordering is explicit and inspectable.** `nlp.pipe_names` returns the ordered list; `nlp.disable_pipes("parser")` turns off components you do not need (huge latency win). Pipelines that never call `disable_pipes` on unused components are the single most common cause of "spaCy is slow." Loading `en_core_web_sm` and calling only `nlp("...").ents` still runs the tagger and parser unless you disable them.
- **The `Doc` is the shared state.** Custom fields go on `Doc.set_extension`, `Span.set_extension`, `Token.set_extension`. A downstream component reads `doc._.my_field`. The contract is per-attribute and typed at extension time.

### spaCy Projects for end-to-end pipelines

[spaCy Projects](https://spacy.io/usage/projects) is spaCy's DAG runner for reproducible training-and-packaging pipelines. A `project.yml` declares assets (data), commands (steps), and workflows (ordered command sequences):

```yaml
title: "Customer-intent NER pipeline"
vars:
  version: "0.2.0"
  gpu_id: 0
assets:
  - dest: "assets/train.jsonl"
    checksum: "8e6d..."
    url: "s3://our-corpus/intent/train.jsonl"
commands:
  - name: "convert"
    script: ["python scripts/to_spacy.py assets/train.jsonl corpus/train.spacy"]
    deps: ["assets/train.jsonl"]
    outputs: ["corpus/train.spacy"]
  - name: "train"
    script: ["python -m spacy train configs/config.cfg --output training/ --gpu-id ${vars.gpu_id}"]
    deps: ["corpus/train.spacy", "configs/config.cfg"]
    outputs: ["training/model-best"]
  - name: "package"
    script: ["python -m spacy package training/model-best packages --version ${vars.version}"]
    deps: ["training/model-best"]
    outputs: ["packages/en_intent_ner-${vars.version}/dist"]
workflows:
  all: ["convert", "train", "package"]
```

`spacy project run all` runs the workflow; each command is skipped if `deps` and `outputs` hashes are unchanged, giving you Makefile-style caching. `spacy project push`/`pull` synchronises artifacts against a remote (S3, GCS, Azure, or a local path). Chapter 5 comes back to this from the packaging-and-shipping angle.

## Hugging Face: `pipelines` and hand-wired orchestration

`transformers.pipeline(task, model=..., tokenizer=..., device=...)` is the ergonomic entry point. It exists for common tasks (`ner`, `text-classification`, `token-classification`, `summarization`, `translation`, `zero-shot-classification`, `feature-extraction`, `question-answering`, ...) and each is a thin `Pipeline` subclass with three phases: preprocess (tokeniser), forward (model), postprocess (task-specific decoding).

```python
from transformers import pipeline

clf = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=0,        # -1 for CPU
    batch_size=32,
    truncation=True,
)

for out in clf(texts):
    ...
```

Two production properties are non-obvious:

- **Batching is opt-in and not always a win.** Pass `batch_size` to enable batched forward passes; without it, `pipeline` runs one item at a time. Batching helps GPU throughput but *hurts* per-item latency for varied-length inputs (the batch's slowest item drags everyone). See chapter 3 for the tuning rules.
- **Preprocess/postprocess run on the main thread.** For heavy postprocess (span decoding for `token-classification`, top-k for `text-classification`), that CPU time is on the critical path. `pipeline(..., num_workers=N)` uses a `DataLoader` with N background workers for preprocess when you iterate over a `Dataset` — worth setting for long batches.

### Composing several HF pipelines: no built-in orchestrator

`transformers` does not ship a DAG runner. For multi-step pipelines you have three options:

1. **Straight-line Python** — call each pipeline in order, pass intermediate results explicitly. Fine for prototypes and few-stage pipelines. Loses batching across stages: stage A returns to CPU before stage B batches again.
2. **A custom `Pipeline` subclass** — for tight two-stage pipelines (say NER → linker), subclass `Pipeline`, override `preprocess`/`_forward`/`postprocess` to keep tensors on-device between forwards. Advanced; do it only when a profile shows the CPU round-trip is the bottleneck.
3. **A separate orchestrator** — spaCy (yes, wrapping HF models as `spacy-transformers` components), Haystack (`Pipeline` DAG with typed edges), BentoML runners, or a homegrown async orchestrator with a typed shared record. This is the shape most production HF stacks end up in past three components.

## Shared-state contract: what actually passes between stages

The compositional-quality problem in NLP is usually a **schema drift** between stages, not a model-quality problem. Segmentation returns character offsets in one system, token offsets in another; the NER stage expects a `List[Span]` but the linker calls `span.text.lower()` and drops the offsets you needed for downstream highlighting. The fix is to define the record once and enforce it.

A production-shaped record for a document pipeline:

```python
from typing import Literal
from pydantic import BaseModel

class Mention(BaseModel):
    text: str
    start_char: int
    end_char: int
    label: Literal["PER", "ORG", "LOC", "MISC"]
    confidence: float
    kb_id: str | None = None       # filled by the linker
    kb_score: float | None = None

class ProcessedDoc(BaseModel):
    doc_id: str
    lang: str                       # from language ID stage
    sentences: list[tuple[int, int]]   # char offsets
    topic_labels: list[tuple[str, float]]
    mentions: list[Mention]
    schema_version: str = "1.2.0"
```

Two properties make this record production-shaped, not toy:

- **Offsets are character-based and monotonic.** Every downstream consumer — highlighter, redaction, replay harness — can re-index against the original text.
- **The record is versioned.** `schema_version` lets you evolve the pipeline without breaking downstream consumers; drift monitors (chapter 6) can slice metrics by schema version to detect regressions from a schema change.

Whether you build this in spaCy (`Doc._.processed = ProcessedDoc(...)`) or as the return type of your HF orchestrator, the important thing is that *one* dataclass exists and every stage reads and writes exactly the fields it declares. Ad-hoc dicts drift within one sprint.

## Batching boundaries in composed pipelines

The composition question that most affects latency and throughput is **where in the graph you batch**. Rules of thumb:

- **Batch at model boundaries, unbatch immediately after.** A stage runs `[doc, doc, doc] -> model -> [out, out, out]`, then returns to per-doc control flow for the next stage. Passing a batch further down forces every stage to be batch-aware and makes routing decisions clumsy.
- **Do not batch across routing decisions.** If stage B is one of {`ner-generic`, `ner-legal`, `ner-medical`} based on the topic label from stage A, batching after stage A means splitting the batch three ways. That is fine when you have enough traffic to keep each downstream model warm; it is a latency disaster when it means one item waits for a batch-of-one on the rare route.
- **Match the batch size to the SLA, not to the GPU.** A GPU that hits peak throughput at batch 64 is not helpful if 64 items take you past your p95 SLA. Chapter 2 covers the SLA math; chapter 3 covers batching-vs-latency tuning.
- **Prefer streaming APIs over materialised lists.** `nlp.pipe(iter(texts))` and iterating over a `Dataset` in HF let the runtime overlap I/O with compute. Materialising a `list(texts)` before feeding the pipeline adds a barrier.

## When to reach for each ecosystem

The choice is rarely absolute — the two mix well (`spacy-transformers` gives you Transformer models as spaCy components; `datasets` feeds HF pipelines with the same batching guarantees as spaCy). A working default:

- **Choose spaCy as the outer framework** when the pipeline is many small components (segment / tag / NER / rules / linker / redactor), CPU is your target, and you want the DocBin serialisation + Projects DAG for free.
- **Choose Hugging Face as the outer framework** when each stage is a large Transformer, GPUs are cheap, and you want fine-grained control over batch composition and mixed precision. Wrap the composition in your own async orchestrator with a versioned record.
- **Mix them** when the shape is spaCy-ish orchestration but one stage is a Transformer that only exists in HF (a custom RoBERTa for your domain). `spacy-transformers` gives you `nlp.add_pipe("transformer", ...)`; use it.

## Chapter summary

- A production NLP pipeline is a DAG of components sharing a typed record and running under an explicit batching policy — not a script that calls models in order.
- spaCy's `Language` pipeline is the right primitive for many-small-components on CPU: use `nlp.pipe(batch_size=...)`, `disable_pipes` unused components, keep `n_process=1` for Transformer-backed pipelines. spaCy Projects packages the DAG for reproducibility (chapter 5).
- Hugging Face `pipelines` are the right primitive when each stage is one large Transformer: pass `batch_size` deliberately, know that preprocess/postprocess run on CPU, and reach for `num_workers` or a custom `Pipeline` subclass only when a profile demands it.
- Neither framework will define your inter-stage record for you. Define a `Pydantic` / dataclass `ProcessedDoc` with character offsets and a `schema_version` field; every stage reads and writes only its declared fields.
- Batch at model boundaries, unbatch immediately after; do not batch across routing decisions unless traffic keeps every route warm; match batch size to the SLA rather than to the GPU's peak-throughput sweet spot.
- The rest of the module builds on this composition: chapters 2–4 tune the latency of the graph you compose here; chapter 5 packages it; chapters 6–7 monitor and postmortem it in production.
