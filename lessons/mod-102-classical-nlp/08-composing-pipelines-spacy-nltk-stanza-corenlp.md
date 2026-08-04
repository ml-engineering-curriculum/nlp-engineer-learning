# Composing Pipelines with spaCy, NLTK, Stanza, and Stanford CoreNLP

## Motivation

There are four toolkits an NLP engineer will meet on essentially any long-lived Python NLP codebase. They overlap; they have different design philosophies; they are chosen for different reasons. Picking wrong is not fatal but produces months of friction — spaCy pipelines that reimplement Stanza's segmenter poorly; Stanza pipelines that swap in NLTK's Snowball stemmer for a component that already handles lemmatisation better; NLTK pipelines that do not scale.

This chapter compares the four, then walks through composing a real production pipeline out of the best pieces of each.

## The four toolkits, at a glance

| Toolkit             | Primary purpose                            | Strength                                       | Weakness                                       |
|---------------------|--------------------------------------------|------------------------------------------------|------------------------------------------------|
| **spaCy**           | Production NLP pipelines (Python + Cython) | Fast, ergonomic, good rule-matching, extensible with custom components | Fewer languages than Stanza; some tasks (e.g. constituency parsing) not first-class |
| **Stanza**          | Multilingual research/production           | Best-in-class accuracy on many languages via Universal Dependencies; ships a huge treebank inventory | Slower than spaCy; Python-only; heavier startup |
| **Stanford CoreNLP** | Java-based classical NLP toolkit          | Mature, reference implementations of many classical algorithms; excellent rule-based matchers (Tregex/Semgrex/TokensRegex); reliable OpenIE / RTE / coreference | JVM startup; Python bindings are wrappers (`stanza` client, `stanfordnlp`) |
| **NLTK**            | Teaching, prototyping, classical algorithms | Enormous breadth of classical algorithms + corpora; canonical educational reference | Slow; APIs inconsistent; not production-ergonomic |

The short version: **spaCy for production, Stanza for multilingual coverage or research-grade accuracy, CoreNLP where JVM is acceptable and Semgrex/OpenIE are needed, NLTK for teaching and one-off scripts.**

## spaCy: the anatomy of a pipeline

A spaCy `Language` object is a container for a *pipeline* — an ordered sequence of components that share a `Doc`. Each component reads and writes attributes on the `Doc`, `Token`, or `Span` objects.

```python
import spacy
nlp = spacy.load("en_core_web_sm")
print(nlp.pipe_names)
# ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']
```

The components:

- `tok2vec` — a shared neural token embedder.
- `tagger` — POS tagger (uses `tok2vec` embeddings).
- `parser` — transition-based dependency parser.
- `attribute_ruler` — rule-based POS overrides.
- `lemmatizer` — rule/lookup-based lemmatiser.
- `ner` — transition-based NER.

You can add, remove, or reorder components:

```python
nlp = spacy.load("en_core_web_sm", disable=["ner"])                    # skip NER
nlp.add_pipe("sentencizer", before="tagger")                            # rule-based sentence splitter
nlp.add_pipe("entity_ruler").from_disk("configs/entities.jsonl")        # gazetteer NER
nlp.add_pipe("dependency_matcher", config={"patterns": patterns})       # tree-pattern matcher
```

Custom components are just Python functions decorated with `@Language.component`:

```python
from spacy.language import Language

@Language.component("normalize_dashes")
def normalize_dashes(doc):
    for token in doc:
        if token.text in {"—", "–"}:
            token.norm_ = "-"
    return doc

nlp.add_pipe("normalize_dashes", first=True)
```

The reason this matters: **spaCy is the toolkit that treats "add a rule between the tokenizer and the tagger" as a first-class operation.** That is exactly the classical-NLP layer this module is about.

### The three rule-matching APIs to know

- **`Matcher`** — token-sequence patterns with per-token attribute constraints. Fast; used for gazetteer-like matching.
- **`PhraseMatcher`** — optimized `Matcher` variant for very large exact-phrase gazetteers.
- **`DependencyMatcher`** — subgraph patterns over dependency trees (chapter 07).

Reference: <https://spacy.io/usage/rule-based-matching>. Every serious spaCy pipeline uses at least one of these; the same effect achieved by post-hoc regex over `doc.text` will be slower, less accurate, and impossible to introspect.

## Stanza: research-grade multilingual NLP

Stanza (formerly StanfordNLP-Python) ships trained pipelines for 70+ languages on Universal Dependencies treebanks. Every pipeline goes: tokenise → MWT expand → POS/UFEATS → lemmatise → depparse → (optional) NER.

```python
import stanza
stanza.download("fi")
nlp = stanza.Pipeline("fi", processors="tokenize,pos,lemma,depparse")
doc = nlp("Kissa istui matolla.")
for sent in doc.sentences:
    for word in sent.words:
        print(word.text, word.lemma, word.upos, word.head, word.deprel)
```

Choose Stanza when:

- You need consistent, treebank-based analysis across many languages.
- You need a lemmatiser or morphology tagger for a language spaCy does not cover well.
- You need Universal Dependencies output for a downstream consumer (semantic parsers, cross-lingual retrieval).
- Accuracy on the UD LAS metric matters more than latency.

Reference: Peng Qi et al., ["Stanza: A Python Natural Language Processing Toolkit for Many Human Languages"](https://arxiv.org/abs/2003.07082), *ACL 2020*.

## Stanford CoreNLP: the Java reference implementation

CoreNLP (Manning et al., ["The Stanford CoreNLP Natural Language Processing Toolkit"](https://nlp.stanford.edu/pubs/manning-EtAl_ACL2014-CoreNLP.pdf), *ACL 2014*) is the mature Java-based classical toolkit. It contains reference implementations of:

- **Tokeniser** (PTBTokenizer).
- **POS tagger** — MaxEnt / averaged perceptron.
- **NER** — CRF-based, plus rule-based `RegexNER`.
- **Constituency parser** — Stanford PCFG parser.
- **Dependency parser** — neural transition-based (Chen & Manning, 2014) or converted from constituency.
- **Coreference** — deterministic sieve system (Lee et al., ["Deterministic Coreference Resolution Based on Entity-Centric, Precision-Ranked Rules"](https://www.aclweb.org/anthology/J13-4004.pdf), *CL 2013*) or statistical.
- **OpenIE** — Angeli et al., ["Leveraging Linguistic Structure for Open Domain Information Extraction"](https://nlp.stanford.edu/pubs/2015angeli-openie.pdf), *ACL 2015*.
- **Rule matchers** — TokensRegex (over token sequences), Semgrex (over dependency graphs), Tregex (over constituency trees).

Access from Python is via the Stanza *client* (which shells to a running CoreNLP JVM server) or `stanfordnlp`:

```python
from stanza.server import CoreNLPClient

with CoreNLPClient(annotators=["tokenize","ssplit","pos","lemma","depparse","openie"], timeout=60000) as client:
    ann = client.annotate("Barack Obama was born in Hawaii.")
    for sent in ann.sentence:
        for triple in sent.openieTriple:
            print(triple.subject, triple.relation, triple.object)
```

Choose CoreNLP when:

- You want deterministic sieve coreference or OpenIE with a well-published lineage.
- You need Tregex/Semgrex/TokensRegex (they are strictly more expressive than spaCy's rule matchers for some patterns).
- Your platform already runs a JVM.

Avoid it when JVM startup and interop cost matters more than any of the above.

## NLTK: still the best classroom, sometimes a production hazard

NLTK (Bird, Klein, Loper, ["Natural Language Processing with Python"](https://www.nltk.org/book/), O'Reilly, 2009 — free online) is the reference textbook of classical NLP in Python. It exposes canonical implementations of:

- Tokenisation (`word_tokenize`, `sent_tokenize`).
- Stemmers (Porter, Snowball, Lancaster).
- WordNet lemmatiser.
- POS taggers (perceptron, HMM, Brill).
- CRF wrapper (over CRFsuite).
- N-gram LMs (`nltk.lm`, no smoothing at KenLM's quality).
- Parsers (chart, CFG, PCFG, dependency).
- Corpora — Brown, Reuters, Gutenberg, Movie Reviews, PropBank, ConLL, and much more.

Where NLTK shines: teaching and reference. Where it does not: NLTK's default tokenisers and POS tagger are slow and less accurate than spaCy's or Stanza's; its API is inconsistent across modules; its corpora require an explicit `nltk.download(...)` step that is easy to get wrong in a Docker image.

Guideline: **reach for NLTK when you want the algorithm; reach for spaCy/Stanza when you want the pipeline.**

## A worked hybrid pipeline

A production English NLP pipeline for medical notes might look like this — a real hybrid of the four toolkits and the components from earlier chapters:

```python
import icu, spacy, kenlm
from spacy.matcher import PhraseMatcher, DependencyMatcher

# --- 1. Load resources ---
nlp = spacy.load("en_core_sci_md", disable=["ner"])        # scispaCy encoder+parser
nlp.add_pipe("sentencizer", before="tagger")               # fast rule-based SBD
nlp.add_pipe("negex", last=True)                           # rule-based negation (community)

drug_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
drug_matcher.add("DRUG", list(nlp.pipe(open("configs/drug_gazetteer.txt"))))

extract = DependencyMatcher(nlp.vocab)
extract.add("PRESCRIBED", [[
    {"RIGHT_ID": "vrb", "RIGHT_ATTRS": {"LEMMA": {"IN": ["prescribe", "administer", "give"]}}},
    {"LEFT_ID": "vrb", "REL_OP": ">", "RIGHT_ID": "med",
     "RIGHT_ATTRS": {"DEP": "dobj", "ENT_TYPE": "DRUG"}},
]])

lm = kenlm.Model("models/medical-4gram.bin")               # for anomaly scoring

# --- 2. Process one document ---
def process(text: str):
    text = icu.UnicodeString(text).normalize().casefold()  # front-door normalisation
    doc = nlp(text)
    for _, tokens in drug_matcher(doc):
        span = doc[tokens[0]:tokens[-1]+1]
        span._.set("ent_type", "DRUG")
    matches = extract(doc)
    ppl = 10 ** (-lm.score(doc.text) / len(doc))            # rough anomaly signal
    return {
        "sentences": [s.text for s in doc.sents],
        "drug_events": [(doc[toks[0]:toks[-1]+1].text) for _, toks in matches],
        "lm_perplexity": ppl,
    }
```

What each layer contributes:

- **ICU** — Unicode-correct normalisation and case-folding, the boundary that everything else depends on.
- **spaCy** — the pipeline backbone: tokenisation, tagging, parsing, custom components, matcher APIs. All the *composition* logic.
- **scispaCy** — a domain-adapted spaCy model (Beltagy et al., ["scispaCy"](https://arxiv.org/abs/1902.07669), 2019). Substituted for the general English model.
- **KenLM** — cheap per-document perplexity for anomaly triage.
- **DependencyMatcher patterns** — replace an LLM call for the well-defined "prescribed drug" extraction; keep the LLM (not shown) for the fuzzy "assessment and plan" summary.

## Pipeline hygiene

- **Pin every model version.** spaCy models are versioned (`en_core_web_sm-3.7.1`); Stanza and CoreNLP models likewise. Store the version in the persisted output.
- **Serialise the pipeline config.** `nlp.to_disk(...)` writes both weights and pipeline config; use it as your artefact.
- **Batch, do not iterate.** `nlp.pipe(texts, batch_size=64)` is 5-20× faster than `[nlp(t) for t in texts]`.
- **Turn off components you do not need.** `disable=["ner"]` drops both the model and its dependency on `tok2vec` (if configured). Use `nlp.select_pipes(...)` context manager for temporary disables.
- **Test the pipeline as a unit.** Given a canonical input document, assert on the exact tokens, spans, and rule matches. This is how you catch a silent stemmer or model upgrade.

## Chapter summary

- spaCy for production Python pipelines; Stanza for multilingual UD-accuracy; CoreNLP for JVM-friendly reference implementations and Semgrex/OpenIE; NLTK for teaching and algorithms.
- Custom components (`@Language.component`) let you slot in normalisers, rule matchers, gazetteers, and LM-based anomaly scoring next to the neural components.
- The three rule-matching APIs (`Matcher`, `PhraseMatcher`, `DependencyMatcher`) are the workhorses of classical rule-based extraction in a modern pipeline.
- Hybrid pipelines mix all four toolkits deliberately; pin model versions, batch, and unit-test the pipeline output.
- Next chapter: the layer that runs *before* any of these — language identification and routing.
