# exercise-03: spaCy Pipeline with Rule-Based Components

**Estimated effort:** 3 hours

## Objective

Assemble a working, extensible spaCy pipeline that combines the neural components you get out of the box with *rule-based* components you author yourself: a custom normalisation component, a gazetteer matcher, a dependency-pattern extractor, and a coarse language-identification fallback. Then evaluate rule-based vs. neural NER on the same held-out set and reason about the trade-off from chapter 10.

## Prerequisites

- Chapters [07](../07-parsers-dependency-and-constituency.md) and [08](../08-composing-pipelines-spacy-nltk-stanza-corenlp.md).
- Python 3.10+; `spaCy ≥ 3.7`; `en_core_web_sm` (or `en_core_web_trf` if you have a GPU); `fasttext` for the LID fallback.

## Problem statement

### Part A — Custom normalisation component

Author a `@Language.component("normalise_dashes_and_quotes")` that:

- Rewrites `token.norm_` for em/en dashes and curly quotes to their ASCII equivalents.
- Sets a boolean `Token.set_extension("is_normalised_smart_punct", default=False)` flag on tokens it touched.

Insert it as the *first* component. Verify with a unit test that `nlp("She said "hi"—loudly.")` produces the expected `.norm_` and flags.

### Part B — Gazetteer entity ruler

Build a `PhraseMatcher`-backed `EntityRuler` for a domain of your choice — a list of ≥200 items in at least two entity classes. Suggestions:

- Drug names + drug classes (from RxNorm's public subset).
- Publicly-traded companies + tickers (from Wikipedia's list of NASDAQ/NYSE companies).
- OSS project names + programming languages.

Requirements:

- Match on `LOWER` (case-insensitive) but respect token boundaries.
- Overlap policy: prefer the longer span on conflict, document it.
- Add the ruler *before* the neural `ner` component and confirm that the ruler's spans survive downstream (`overwrite_ents=True` on the ruler; or manually merge).

### Part C — DependencyMatcher extractor

Design at least three `DependencyMatcher` patterns over the parsed tree. Examples (pick or invent):

- "Company acquired Company" — subject/object of `acquire|buy|purchase` where both are `PROPN` chains or `ORG` entities.
- "Drug X administered at Y mg" — verb `administer|give|prescribe`, dobj = drug entity, prepositional modifier with a quantity.
- "Person joined Company as Title" — `join` with subject `PROPN`, `dobj` `PROPN`, `prep_as` phrase.

Each pattern should produce a structured record (`dict[str, str]`), not just a match.

### Part D — Language-identification fallback

Wrap fastText's `lid.176.ftz` in a `@Language.factory("lang_router")` component that:

- Sets `doc._.lang` to a BCP-47 tag with confidence-and-delta thresholds (per chapter 09).
- Sets `doc._.route` to either `"en"` (the current pipeline) or `"reject"` (below thresholds).

Add it as the *last* pipeline step. Show that when you feed non-English input the router flags it and the downstream code skips extraction.

### Part E — Rule-based vs. neural NER

On a held-out set of 50-100 sentences from your domain, score:

- Precision, recall, F1 of the neural `ner` component on your entity classes.
- Precision, recall, F1 of the gazetteer ruler alone.
- Precision, recall, F1 of ruler-then-neural (ruler wins on conflict).

Report the three numbers per entity class and write a one-paragraph recommendation for the production pipeline.

## Starter guidance

- Use `spacy project` or a plain `main.py` — but pin the model version (`en_core_web_sm==3.7.1`) in a `requirements.txt`.
- For the gazetteer, pre-normalise entries with the same casefold/NFKC policy your `normalise_dashes_and_quotes` uses, otherwise you will hit the "`iphone` vs. `iPhone` vs. `IPHONE`" trap.
- Batch processing: use `nlp.pipe(texts, batch_size=64, n_process=1)` for the eval. Report wall-clock time.
- For dependency patterns, start from `spaCy`'s [dependency matcher documentation](https://spacy.io/api/dependencymatcher) and `explacy` if you want a printed tree while debugging.
- Save eval outputs (matches, misses) as tables so the write-up can quote specific examples.

## Acceptance criteria

- [ ] Six pipeline components in the described order: `normalise_dashes_and_quotes` → tokenizer → tagger → parser → gazetteer `EntityRuler` → neural `ner` → `lang_router` (or an equivalent order documented in your write-up).
- [ ] Unit tests for the normalisation component and the three dependency patterns pass.
- [ ] At least three worked extraction examples per DependencyMatcher pattern in the write-up.
- [ ] NER comparison table (precision/recall/F1 per class, three configurations, wall-clock time per configuration).
- [ ] Language-router demonstration: at least one example that routes to `"reject"` and at least one borderline case where the thresholds decide it.
- [ ] A short write-up (`README.md`) applying the chapter 10 framework: which layers are load-bearing for correctness, which for cost, and where the seam is between rule-based and neural.

## Stretch goals

- **Serialise the pipeline.** `nlp.to_disk("./my-pipeline")` and reload; verify byte-identical output on the eval set.
- **Custom scorer.** Register a `spacy.Scorer` that scores your DependencyMatcher outputs against a labelled test set (rather than just NER).
- **Cross-lingual routing.** Add a second language's model (Stanza or spaCy) and have the router hand off to it when LID returns that language.
- **Speed profile.** Run `python -m cProfile` on `nlp.pipe`; identify the slowest component; either optimise or justify why the current cost is acceptable.
