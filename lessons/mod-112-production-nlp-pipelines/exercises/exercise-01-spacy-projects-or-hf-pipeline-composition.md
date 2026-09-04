# exercise-01: spaCy Projects or HF Pipeline Composition

**Estimated effort:** 3 hours

## Objective

Compose a **document-level NLP pipeline** — segmentation → per-sentence classification → NER → simple entity linking → structured output — as a **reproducible artifact** using either the spaCy Projects DAG or the Hugging Face `pipelines` + custom orchestrator pattern. Deliver both a runnable pipeline and evidence that it is composed properly: a typed shared record with a `schema_version`, deterministic outputs on a canonical input set, a project DAG (spaCy) or an orchestrator module (HF) with explicit inter-stage contracts, and a `manifest.json` (chapter 5 spec) at the top of the artifact directory.

## Prerequisites

- Chapter [01](../01-pipeline-composition-hf-and-spacy.md) and (for the manifest) chapter [05](../05-packaging-and-reproducible-shipping.md).
- Python 3.10+; if picking the spaCy track: `spacy>=3.7`, `spacy-transformers`, one `en_core_web_*` model (`sm` is enough), `pydantic>=2`. If picking the HF track: `transformers>=4.40`, `tokenizers`, `pydantic>=2`, `sentence-transformers` (for the linker candidate step).
- Access to any 200-document text corpus you can redistribute: sample from `datasets` (`ag_news`, `reuters21578`, `bbc_news`, or your own), or a public press-release feed. Keep documents ≤ 4 000 chars each.

## Problem statement

### Part A — Pick your track and design the record

Choose one of:

- **Track SP** (spaCy Projects): outer framework is spaCy; per-sentence classifier is a custom `@Language.factory` component or a small Transformer via `spacy-transformers`; NER is a spaCy `ner` component or a `spacy-transformers` head; linker is a rule-based lookup against a small in-memory KB you build.
- **Track HF** (Hugging Face + orchestrator): outer framework is your own Python orchestrator; segmentation via `nltk.sent_tokenize` or `spacy` sentence segmenter used as a library; per-sentence classifier via `transformers.pipeline("text-classification", ...)`; NER via `transformers.pipeline("token-classification", ...)`; linker as above.

Then design the **shared record** — a `pydantic.BaseModel` in a file `pipeline/record.py`. Required fields:

- `doc_id: str`
- `lang: str`
- `schema_version: str = "1.0.0"`
- `sentences: list[SentenceRecord]` where `SentenceRecord` carries `(start_char, end_char, text, topic_label, topic_conf)`.
- `mentions: list[Mention]` where `Mention` carries `(start_char, end_char, text, label, confidence, kb_id, kb_score)`. Character offsets are into the **original document**, not the sentence — this is the discipline chapter 1 called out.
- `processing_ms: dict[str, float]` — per-stage wall-clock time, keyed by stage name.

Write `pipeline/record.py`. It is the contract every stage reads from and writes to.

### Part B — Build the pipeline

**Track SP.** Structure the repo as a spaCy Projects layout:

```
project.yml
configs/config.cfg
scripts/
    segment_and_classify.py    # custom @Language.factory
    linker.py                  # custom @Language.factory
    kb.json                    # 20-50 hand-authored entities: {"kb_id": ..., "name": ..., "aliases": [...]}
    build_manifest.py
data/
    docs.jsonl                 # your 200-doc corpus
```

`project.yml` should declare commands `run` (execute the pipeline over `data/docs.jsonl`, write `outputs/results.jsonl` and `outputs/manifest.json`), `verify` (re-run and compare byte-for-byte against a stored reference), and workflow `all: [run, verify]`. Use `spacy project run all` end-to-end. Register the custom components so `nlp.add_pipe(...)` orders them: `sentencizer → classify_sentences → ner → link_entities → assemble_record`.

**Track HF.** Structure as a Python package:

```
pipeline/
    __init__.py
    record.py
    orchestrator.py            # composes the stages, produces DocumentRecord
    segment.py                 # wraps spaCy sentencizer
    classify.py                # wraps transformers.pipeline
    ner.py                     # wraps transformers.pipeline
    linker.py                  # rule-based
    kb.json
scripts/
    run.py                     # CLI: read docs.jsonl, emit results.jsonl + manifest.json
    verify.py                  # re-run and compare
Makefile
```

The orchestrator must accept an iterator of docs, batch appropriately at model boundaries (chapter 1 batching rule), and yield `DocumentRecord` instances. Batch sizes belong in a config file, not hard-coded.

Whichever track: **every stage records its wall-clock time** into `processing_ms`. The eventual reproducibility check depends on this being deterministic across runs (same offsets, same labels, same schema version — timings may vary).

### Part C — Determinism and the schema-version guarantee

Run the pipeline twice on the same input and prove:

- Every non-timing field in every `DocumentRecord` is byte-identical between the two runs. (Timings may differ.) Include a `scripts/verify.py` that diffs the outputs modulo `processing_ms`.
- Changing any stage's model or config (e.g. swap the NER model to a different checkpoint) bumps `schema_version` if the output *shape* changes (new label set, changed offsets) — otherwise the version stays but the manifest's model SHAs change.
- The pipeline runs with `PYTHONHASHSEED=0` and any framework seeds (spaCy `spacy.util.fix_random_seed(0)`, or `transformers.set_seed(0)`) set at entry.

### Part D — The manifest

Produce `outputs/manifest.json` with the fields specified in chapter 5:

- `name`, `version`, `created_at` (UTC ISO 8601).
- `git.repo`, `git.sha`, `git.branch`.
- For each model in the pipeline: framework, framework_version, model_id, revision/SHA.
- `runtime`: precision (`fp32` here), device (`cpu` or `cuda:0`).
- `inputs`: dataset path or ID, hash of the input file.
- `outputs`: hash of the results file (excluding timings).
- `eval`: a stub — pick any 20 documents, hand-label whether the NER + linking is broadly correct, report a rough precision. Not the module's evaluation exercise; just prove the field is populated.
- `schema_version` (matches the record).
- `contact` (your name / email).

Ship `scripts/build_manifest.py` that regenerates the manifest from a completed run.

### Part E — Failure-mode gallery

Write `failures.md` — for each of these deliberate mistakes, run the pipeline in a modified state and record what breaks. Show the actual error / bad output. Do not "fix" the pipeline to avoid them; the point is to exhibit them and articulate the guardrail.

1. Drop `schema_version` from the record; a downstream consumer that pins `schema_version == "1.0.0"` breaks with a specific error.
2. Change the NER stage to return token offsets instead of character offsets; a highlighter consumer produces off-by-one highlights on multi-byte input.
3. In the Track SP variant, forget to `disable_pipes("parser")` when loading `en_core_web_sm`; measure the per-stage timing difference on 200 docs.
4. In the Track HF variant, pass `batch_size=1` versus a reasonable batch size to the NER pipeline; measure end-to-end throughput on 200 docs.
5. Bump the base model without bumping the artifact version; show that downstream consumers cannot detect the change.

Two paragraphs per mistake in `failures.md`: what you changed, what the pipeline did, what the guardrail should be.

## Starter guidance

- **Character offsets are the discipline.** Chapter 1 emphasises this and part E exercise 2 forces it. Every stage that finds a span records `(start_char, end_char)` into the *original document text*. For HF `token-classification`, use `aggregation_strategy="simple"` and translate the returned `start`/`end` (which are character offsets in the input string) into document-space offsets when your segmentation split into sentences first.
- **Segmentation is the trickiest join.** After segmenting, keep a `(sentence_start_char, sentence_end_char)` for each sentence and add `sentence_start_char` to any downstream NER offset that comes back per-sentence.
- **Keep the KB tiny.** 20-50 hand-authored entities in `kb.json`. This exercise is about composition, not entity linking; a `str.lower()` alias match is fine.
- **The manifest is the artifact.** Reviewers should be able to read only `manifest.json` and answer "what code, on what data, produced what results." If they cannot, the manifest is incomplete.
- **Use `spacy project push` (Track SP) or a `Makefile push` target (Track HF)** to a local `remotes/` directory to prove the artifact is portable; this is the seed of chapter 5's registry story.
- **Do not add features not requested.** No web UI, no real entity linker, no evaluation harness. Composition + reproducibility is the exercise; scope creep is a distraction.

## Acceptance criteria

- [ ] `pipeline/record.py` (or spaCy equivalent) defines `DocumentRecord`, `SentenceRecord`, `Mention` with the required fields and `schema_version = "1.0.0"`.
- [ ] Pipeline runs end-to-end on the 200-doc input, producing `outputs/results.jsonl` — one `DocumentRecord` per line.
- [ ] Character offsets in `Mention` refer to positions in the original document text and are verifiable (`doc.text[mention.start_char:mention.end_char] == mention.text`) for every mention.
- [ ] `scripts/verify.py` confirms byte-identical outputs (modulo `processing_ms`) across two runs with seeded randomness.
- [ ] `outputs/manifest.json` populates every field in chapter 5's spec, including per-model SHAs, input/output hashes, and a manual 20-doc eval stub.
- [ ] `failures.md` documents the five mistakes with the actual observed break and the articulated guardrail.
- [ ] Track SP: `project.yml` with declared `commands`, `deps`, `outputs`, and `workflows`; `spacy project run all` is the entry point.
- [ ] Track HF: `Makefile` (or `pyproject.toml` scripts) with `make run` / `make verify` targets; orchestrator batches at model boundaries per the record.

## Stretch goals

- **Cross-track parity.** Build the same pipeline in *both* tracks and produce a diff of the outputs on the same input set. Which stages diverge, and why? A useful `parity.md` documents where the two ecosystems produce genuinely different tokens / spans.
- **Add a routing stage.** After the sentence classifier, route sentences with `topic_label == "legal"` to a domain-specific NER and everything else to the default NER. Instrument the batching consequences (chapter 1: batching across a routing decision) and report throughput impact.
- **Add span-level provenance.** Every `Mention` carries a `source: Literal["ner", "rule"]` field indicating whether it came from the model or a rule component; extend the pipeline with a small rule pack (e.g. regex-matched product codes) and show the merged output.
- **Bundle a rate-limited HTTP frontend.** Wrap the orchestrator in a FastAPI endpoint with per-request timing and a `/version` endpoint returning the manifest. Prove the manifest is the same the CLI produces.
- **Continuous-integration proof.** Add a GitHub Actions or GitLab CI job that runs `make verify` (or `spacy project run verify`) and rejects any PR that changes the output modulo timings without a schema-version bump.

## Deliverables

Track SP layout:

```
project.yml
configs/config.cfg
scripts/                  segment_and_classify.py  linker.py  kb.json  build_manifest.py  verify.py
data/                     docs.jsonl
outputs/                  results.jsonl  manifest.json
failures.md
README.md                 # how to run, choices made
```

Track HF layout:

```
pipeline/                 __init__.py  record.py  orchestrator.py  segment.py  classify.py  ner.py  linker.py  kb.json
scripts/                  run.py  verify.py  build_manifest.py
Makefile
data/                     docs.jsonl
outputs/                  results.jsonl  manifest.json
failures.md
README.md
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
