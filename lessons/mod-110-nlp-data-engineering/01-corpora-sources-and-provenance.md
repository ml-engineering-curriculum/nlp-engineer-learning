# Corpora, Sources, and Provenance

## Motivation

"Where did this text come from?" is not a librarian's question. It is the question that decides whether your model memorises an email address, whether your lawyer can defend the training set in discovery, whether a reviewer can reproduce your paper, and whether your next data refresh will quietly swap a public-domain book for a copyrighted one. Treat it as an engineering question from the first byte you touch.

Three concrete things go wrong when provenance is not first-class:

- **Memorisation and leakage.** A model trained on text whose source you cannot name will sometimes regurgitate it verbatim. If you cannot answer "where did this come from?" for a leaked string, you cannot answer "is anyone else going to find it there?" either. See Carlini et al., "Extracting Training Data from Large Language Models", *USENIX Security 2021*, [arXiv:2012.07805](https://arxiv.org/abs/2012.07805), and the follow-up Carlini et al., "Quantifying Memorization Across Neural Language Models", *ICLR 2023*, [arXiv:2202.07646](https://arxiv.org/abs/2202.07646).
- **Legal exposure.** The current wave of training-data lawsuits (New York Times v. OpenAI, Authors Guild v. OpenAI, Getty v. Stability, and the Books3-related suits) all turn on *which specific documents are in the training set*. If your pipeline cannot produce a per-document audit trail, you have no defence.
- **Reproducibility.** "We trained on a Common Crawl snapshot from last spring" is not a reproducible description. "We trained on `CC-MAIN-2024-10`, filtered with CCNet commit `abc1234`, deduplicated at MinHash threshold 0.8" is.

This chapter sets up the data layer the rest of the module operates on: where open-web text actually comes from, what the derived corpora you will encounter are built from, and how to attach a provenance record to every row so the later chapters (dedup, routing, quality, PII, licensing) have something to work with.

## The canonical open-web stack: Common Crawl

Almost every open LLM training corpus you will meet starts life as [Common Crawl](https://commoncrawl.org/). It is a non-profit that crawls a large sample of the public web roughly monthly and publishes the result on S3 (`s3://commoncrawl/`) and HTTP. Each monthly snapshot is identified by a tag of the form `CC-MAIN-YYYY-WW` (e.g. `CC-MAIN-2024-10` is the tenth week of 2024). Pin that tag in your pipeline the way you pin a package version.

Each snapshot ships in three parallel formats, and knowing which to read is half of the Common Crawl learning curve:

| Format | What it contains                                                             | Typical use                                     |
|--------|-------------------------------------------------------------------------------|-------------------------------------------------|
| WARC   | Full HTTP request + response records, raw HTML bytes                          | You want to re-extract text or inspect headers  |
| WAT    | JSON metadata derived from WARC (URL, response headers, extracted links)      | Link graphs, URL filtering, crawl analytics     |
| WET    | Plain-text extraction of each page (Common Crawl's own strip-to-text)         | Quick-start text corpora, cheap language stats  |

WARC is the authoritative artefact; WAT and WET are convenience derivatives. Serious corpus builds re-extract from WARC rather than trusting WET, because WET is produced by a one-pass boilerplate stripper that leaves a lot of nav-bar sludge in and throws some real content out. See chapter 04 for why that matters.

The URL index is at <https://index.commoncrawl.org/>; the file layout per snapshot is a flat set of `warc/*.warc.gz` paths listed in `warc.paths.gz` at the top of each snapshot's prefix. A single snapshot is on the order of hundreds of TB compressed, which is why nobody downloads it whole — you stream a shard list from S3, filter by URL / language / content-type, and keep what you want.

```bash
# List the WARC files in a snapshot.
SNAPSHOT=CC-MAIN-2024-10
curl -s https://data.commoncrawl.org/crawl-data/$SNAPSHOT/warc.paths.gz \
  | gunzip | head -5
# crawl-data/CC-MAIN-2024-10/segments/.../warc/CC-MAIN-20240226000000-...-00000.warc.gz
# ...

# Stream one WARC file and pull out HTTP responses.
aws s3 cp --no-sign-request \
  s3://commoncrawl/crawl-data/$SNAPSHOT/segments/.../CC-MAIN-...-00000.warc.gz - \
  | zcat | python -m warcio.cli index -
```

### Extracting text from WARC

Common Crawl deliberately does not ship a canonical "clean text" product. The de facto community extractor is **CCNet** (Wenzek et al., "CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data", *LREC 2020*, [arXiv:1911.00359](https://arxiv.org/abs/1911.00359)), which combines:

1. HTML-to-text with a fork of `trafilatura` / `jusText`-style boilerplate removal.
2. Paragraph-level language identification with fastText (see mod-102 chapter 9).
3. Per-language perplexity filtering using a KenLM n-gram model trained on Wikipedia.

The CCNet pipeline is itself source for several of the derived corpora below. If you are building your own corpus from WARC, start by reading CCNet's code — not because you will run it unmodified, but because the ordering of stages (dedup before LID before quality filter) is load-bearing and the paper justifies each choice.

Other extractors in common use: `trafilatura` (Barbaresi, "Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction", *ACL 2021 demo*, [paper](https://aclanthology.org/2021.acl-demo.15/)) and `resiliparse` (part of ChatNoir). FineWeb's extractor is `trafilatura` with specific settings — the FineWeb team publishes them.

## Derived corpora you will encounter

Most practitioners do not build from raw WARC; they consume a published corpus that already did. Know which Common Crawl snapshot and which extractor each one wraps, because that determines what is and is not in it.

| Corpus       | What it is                                                                 | Primary source                                                 |
|--------------|----------------------------------------------------------------------------|----------------------------------------------------------------|
| CC-100       | Per-language shards built with CCNet, ~2.5 TB text, 100+ languages         | Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale" (XLM-R), *ACL 2020*, [arXiv:1911.02116](https://arxiv.org/abs/1911.02116) |
| OSCAR / OSCAR-2301 | Per-language shards of Common Crawl via the goclassy pipeline        | Abadji et al., "Towards a Cleaner Document-Oriented Multilingual Crawled Corpus", *LREC 2022*, [arXiv:2201.06642](https://arxiv.org/abs/2201.06642) |
| mC4          | Multilingual C4; the training corpus for mT5                               | Xue et al., "mT5: A massively multilingual pre-trained text-to-text transformer", *NAACL 2021*, [arXiv:2010.11934](https://arxiv.org/abs/2010.11934) |
| C4           | Colossal Clean Crawled Corpus; the training corpus for T5                  | Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", *JMLR 2020*, [arXiv:1910.10683](https://arxiv.org/abs/1910.10683) |
| The Pile     | 22-source mix: Common Crawl + books + arXiv + GitHub + StackExchange + ... | Gao et al., "The Pile: An 800GB Dataset of Diverse Text for Language Modeling", *2020*, [arXiv:2101.00027](https://arxiv.org/abs/2101.00027) |
| RedPajama    | Open re-implementation of the LLaMA 1 training recipe                      | Together AI, [redpajama-data](https://github.com/togethercomputer/RedPajama-Data) |
| DOLMA        | AI2's open 3T-token corpus with per-document provenance                    | Soldaini et al., "Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research", *ACL 2024*, [arXiv:2402.00159](https://arxiv.org/abs/2402.00159) |
| FineWeb      | 15T-token Common Crawl-only corpus with released filter configs            | Penedo et al., "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale", *2024*, [arXiv:2406.17557](https://arxiv.org/abs/2406.17557) |
| FineWeb-Edu  | FineWeb filtered by an "educational quality" classifier                    | Same paper                                                     |

Picking between them is partly taste, but a few durable observations:

- **C4 and mC4 are old and aggressively filtered.** Known artefacts: a "bad-words" blocklist that over-filters queer and minority-language content (Dodge et al., "Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus", *EMNLP 2021*, [arXiv:2104.08758](https://arxiv.org/abs/2104.08758)). Still useful; know what you are getting.
- **The Pile is a mix, not a crawl.** Great for small-model research because the per-source composition is explicit; also the subject of the Books3 removal story (below).
- **DOLMA is the one to read if you care about provenance.** It ships per-document source IDs, a documented filter chain, and a data-sheet-grade release.
- **FineWeb / FineWeb-Edu are the current open-source default for "give me a lot of English web tokens that are already pretty clean".** Educational-quality variant is a strong signal for small models.
- **OSCAR-2301 and CC-100 are the usual defaults for non-English pretraining.** OSCAR ships document-level language annotation; CC-100 ships paragraph-level.

For a running system, pin not just the corpus but the *version*: `OSCAR-2301`, not "OSCAR"; `FineWeb v1.1.0`, not "FineWeb"; `DOLMA v1.6`, not "DOLMA".

## Non-web sources

Web crawl is the volume; curated sources are the quality and the diversity. Common ones:

- **Wikipedia dumps** — <https://dumps.wikimedia.org/>. Per-language, monthly, licence CC-BY-SA 3.0 (text) + GFDL. Parse with `mwparserfromhell` or Hugging Face's `wikipedia` loader. Wikipedia is a disproportionately strong signal per token because it is edited and typed consistently; mind the licence propagation if you redistribute (chapter 11).
- **Project Gutenberg** — <https://www.gutenberg.org/>. ~70k public-domain books, mostly English, heavy on pre-1928 US works. The licence situation is subtle: the *texts* are public domain in the US, but Project Gutenberg's trademark and header boilerplate are not, and some books are PD in the US but still copyrighted in the EU. Strip the PG header and footer before training (there is a canonical `PG***HEADER / FOOTER` marker).
- **OpenSubtitles** — <https://www.opensubtitles.org/>. Movie and TV subtitles across many languages; the common OPUS-packaged version is the training data for a lot of multilingual MT. Noisy, dialogue-shaped, often unlicensed at the source; useful for conversational register, legally awkward to redistribute.
- **Books3** — a ~37GB corpus of ~196k books assembled by The Eye and shipped as part of The Pile. The Rights Alliance filed a DMCA takedown in 2023 and Books3 was removed from The Eye and from the HuggingFace mirror of The Pile. The lawsuits that reference it are ongoing. The engineering lesson is simple: if your corpus manifest contains a source you cannot name a licence for, you have a Books3-shaped liability. (See chapter 11 for a licensing taxonomy.)
- **Stack Exchange** — <https://archive.org/details/stackexchange>. Quarterly data dumps, CC-BY-SA (with attribution requirements). High-quality Q&A text; mind that answers and comments share the same licence.
- **GitHub** — crawled or via BigQuery `githubarchive`. Each repository has its own licence; the aggregate "GitHub" is *not* uniformly licensed. The Stack (Kocetkov et al., "The Stack: 3 TB of permissively licensed source code", *2022*, [arXiv:2211.15533](https://arxiv.org/abs/2211.15533)) does the licence-filtering work and is the usual starting point for code pretraining.
- **Scientific open access** — PubMed Central (`PMC-OAS`) for biomedical; arXiv bulk access via S3 (`s3://arxiv/` with requester-pays) for physics / CS / maths; S2ORC (Lo et al., "S2ORC: The Semantic Scholar Open Research Corpus", *ACL 2020*, [arXiv:1911.02782](https://arxiv.org/abs/1911.02782)) for cross-publisher scientific text. All of these have per-article licence metadata you must preserve.

Every non-web source gives you better per-token quality than Common Crawl at the cost of smaller volume and more licence bookkeeping. Mix accordingly.

## Provenance as an engineering concern

The claim: **every document in your corpus carries a provenance record, stored next to the text, that is enough to reproduce the document from its source.** If you cannot reproduce the row from its provenance record, you have lost provenance.

Minimum fields:

| Field              | Why it is there                                                                 |
|--------------------|---------------------------------------------------------------------------------|
| `doc_id`           | Stable primary key for the document in your corpus                              |
| `source`           | Enum: `common_crawl`, `wikipedia`, `arxiv`, `stackexchange`, `github`, ...       |
| `source_ref`       | Source-specific pointer: Common Crawl snapshot ID + WARC path + record offset; Wikipedia dump date + page ID + revision ID; arXiv ID + version; GitHub owner/repo + commit SHA + path |
| `source_url`       | Original fetch URL where meaningful                                             |
| `fetched_at`       | ISO-8601 timestamp of fetch (or the dump timestamp)                             |
| `raw_sha256`       | SHA-256 of the raw bytes *before* extraction                                    |
| `text_sha256`      | SHA-256 of the extracted text that is stored on this row                        |
| `extractor`        | Name + semver of the extractor: `trafilatura==1.9.0`, `ccnet@abc1234`, ...      |
| `language`         | BCP-47 tag assigned by LID (see chapter 03), plus the confidence                |
| `license`          | SPDX identifier where known; `unknown` is a legitimate value you then act on    |
| `license_evidence` | How you determined the licence: HTTP header, HTML meta tag, repo `LICENSE` file |

Concrete JSON example for one row:

```json
{
  "doc_id": "cc-main-2024-10::en::0003f1a7c0e94b2d",
  "source": "common_crawl",
  "source_ref": {
    "snapshot": "CC-MAIN-2024-10",
    "warc_path": "crawl-data/CC-MAIN-2024-10/segments/1700/warc/CC-MAIN-20240226-...-00042.warc.gz",
    "record_offset": 418203471,
    "record_length": 24196
  },
  "source_url": "https://example.org/articles/why-postgres-is-still-great",
  "fetched_at": "2024-02-27T03:12:48Z",
  "raw_sha256": "9a3c...e71f",
  "text_sha256": "b2d4...08af",
  "extractor": "trafilatura==1.9.0+finweb-settings-v1",
  "language": {"tag": "en", "conf": 0.998},
  "license": "unknown",
  "license_evidence": {"html_meta_rights": null, "robots_txt": "allow"}
}
```

Store this record in the *same* parquet file as the text, as a sibling column, not in a side-car table that will drift. Partition by `(source, snapshot, language)` so downstream filtering is a predicate push-down rather than a scan.

Two rules this record exists to enforce:

1. **If you cannot regenerate `text_sha256` from `source_ref` + `extractor`, you have lost provenance.** This is testable: run the extractor against the source pointer and compare hashes on a sample every ingestion.
2. **If `license` is `unknown`, downstream training must treat the row according to your `unknown`-license policy** — reject, quarantine, or include-with-flag. The policy is a product decision; the record makes it enforceable.

A corollary: the extractor is part of provenance. If you upgrade `trafilatura` from 1.9.0 to 2.0.0, you have changed the text; a new ingestion gets a new extractor tag, and the diff is auditable.

## Licensing: a preview

Full treatment is in chapter 11; this is enough to make the provenance record above make sense.

| Licence class                           | Example                              | Train on it?                               |
|-----------------------------------------|--------------------------------------|--------------------------------------------|
| Public domain                           | Pre-1928 US works, US gov documents  | Generally yes; mind jurisdiction           |
| Permissive (CC-BY, CC0, MIT, Apache-2)  | Wikipedia (BY-SA), MIT-licensed code | Usually yes, with attribution obligations  |
| Share-alike (CC-BY-SA, GPL)             | Wikipedia, Stack Exchange, GPL code  | Yes, but outputs may inherit obligations   |
| Non-commercial (CC-NC)                  | Many research datasets               | Depends on your deployment                 |
| No derivative (CC-ND)                   | Rare in text corpora                 | Usually no for training                    |
| Platform ToS (no explicit licence)      | Reddit, Twitter, generic web pages   | Grey; current litigation                   |
| Unknown / mixed                         | Generic Common Crawl page            | Grey; current litigation                   |
| Known-restricted                        | Books3, newspaper paywalls           | No                                         |

Three gotchas worth internalising now, because they bite at the provenance layer:

- **Project Gutenberg "US-only" works.** Several Gutenberg texts are PD in the US but still under copyright in the EU and elsewhere. The Gutenberg metadata flags these; your ingestion must preserve the flag.
- **GitHub repository licence vs. snippet licence.** A repo's `LICENSE` file does not automatically licence every file in it (vendored code, generated files, embedded strings). The Stack's licence filter checks per-file where possible.
- **ToS vs. copyright.** A page may be freely accessible but its platform's ToS forbids crawling or redistribution. ToS is contract law, not copyright, and is a separate axis you must track.

## The corpus pipeline, end to end

Everything in this module composes into one pipeline. Here it is, with the chapter that owns each stage:

```
                 Common Crawl        Wikipedia       GitHub          arXiv / PMC
                 (WARC/WAT/WET)      (dumps)         (repo clone)    (bulk access)
                      │                 │               │                 │
                      ▼                 ▼               ▼                 ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  INGEST + PROVENANCE RECORD   (this chapter, ch 01)      │
                 │  - raw bytes stored, sha256                              │
                 │  - source_ref + fetched_at + extractor pinned            │
                 └──────────────────────┬───────────────────────────────────┘
                                        ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  TEXT EXTRACTION  (ch 04 — deboilerplating)              │
                 │  - trafilatura / resiliparse / CCNet                     │
                 └──────────────────────┬───────────────────────────────────┘
                                        ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  DEDUPLICATION    (ch 02 — exact + MinHash)              │
                 │  - URL / line / n-gram / document levels                 │
                 └──────────────────────┬───────────────────────────────────┘
                                        ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  LANGUAGE ROUTING (ch 03; see also mod-102 ch 09)         │
                 │  - BCP-47 tag + confidence per document                  │
                 └──────────────────────┬───────────────────────────────────┘
                                        ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  QUALITY FILTERING (ch 04)                               │
                 │  - heuristics + classifier (e.g. FineWeb-Edu)            │
                 └──────────────────────┬───────────────────────────────────┘
                                        ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  PII SCRUBBING    (ch 05)                                │
                 │  - regex + NER + hashing / redaction                     │
                 └──────────────────────┬───────────────────────────────────┘
                                        ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  TOXICITY / SAFETY FILTER (ch 06)                        │
                 │  - classifier + rules; distinct from quality             │
                 └──────────────────────┬───────────────────────────────────┘
                                        ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  DATASET CARD / DATASHEET (ch 11)                        │
                 │  - per-source composition, licences, known limitations   │
                 └──────────────────────┬───────────────────────────────────┘
                                        ▼
                                 training shards
                       (parquet, partitioned by source+lang)
                                        │
                                        ▼
                                   mod-101 tokenizer
```

A few invariants of this pipeline:

- **Provenance is upstream of extraction.** You hash and record *before* you parse HTML, so a bad extractor is an auditable regression, not a lost source.
- **Dedup runs before quality filtering.** Running quality filters on duplicates wastes compute and biases the kept distribution toward whatever gets duplicated most (templated boilerplate, SEO farms). CCNet, DOLMA, and FineWeb all order it this way.
- **Language routing runs before per-language steps.** PII and toxicity filters are language-specific; running an English PII regex on Hindi devanagari text is a null op.
- **The dataset card is written from the pipeline's own records, not from memory.** The per-source composition, filter configs, and licence breakdown in the card are populated by querying the provenance table.

Annotation-side pipelines (chapters 07-10) consume this output and add labels; the provenance record follows the row through labelling, so every annotation inherits a source.

## When "we'll fix it later" is wrong

Three temptations worth naming so you can resist them:

- **"We'll add provenance when we need it."** You will need it in a legal review, in a reproducibility audit, or when a user reports a memorisation leak — and at every one of those moments you will not be able to backfill. Capture it at ingest or do not have it.
- **"We'll just use HuggingFace `load_dataset`."** Fine for prototyping; dangerous for production. HuggingFace datasets are versioned and most carry licences, but a `load_dataset("c4", "en")` call does not pin a snapshot or a filter configuration unless you ask it to. Record the resolved commit hash of the dataset repo in your provenance.
- **"We'll dedupe after training."** Dedup changes which rows the model sees, which changes gradients, which changes weights. Post-hoc dedup is a different experiment, not the same one.

## Chapter summary

- Common Crawl's WARC / WAT / WET formats, monthly `CC-MAIN-YYYY-WW` snapshots, and the CCNet extraction pipeline are the open-web substrate every other corpus is built from — pin snapshot and extractor by version.
- The derived corpora you will encounter (C4, mC4, CC-100, OSCAR-2301, RedPajama, The Pile, DOLMA, FineWeb/FineWeb-Edu) differ mostly in which snapshot they wrap and which filters they ran; read the paper, pin the version.
- Non-web sources (Wikipedia, Gutenberg, OpenSubtitles, Stack Exchange, GitHub, PMC, arXiv) buy quality and diversity at the cost of per-source licence bookkeeping; Books3 is the standing lesson in what happens when that bookkeeping is skipped.
- Treat provenance as a first-class engineering artefact: every document carries source + source_ref + raw_sha256 + extractor-version + licence, stored next to the text; if you cannot regenerate the row from its record, you have lost provenance.
- The module pipeline is ingest -> extract -> dedup -> route -> quality -> PII -> toxicity -> dataset card -> shards; provenance threads through every stage and is the input to the dataset card.
- Next chapter: deduplication, which runs immediately after extraction and before every other filter, because every later statistic is wrong if duplicates are in.
