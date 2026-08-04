# Dependency and Constituency Parsers

## Motivation

Parsing is the layer where NLP stops labelling tokens and starts producing structure: which words modify which, which nominal phrase is the subject, which relative clause attaches where. Even in a pipeline where the "smart" work is done by an LLM, a dependency tree is often the substrate for rule-based extraction, coreference resolution, syntactic negation handling, and query understanding — because rules over a tree are enormously more powerful than rules over a string.

This chapter covers the two dominant families — constituency and dependency parsing — with enough algorithmic depth to read parser output, choose the right toolkit, and build rule-based extractors on top.

## Two grammars, two views

**Constituency (phrase-structure) grammars** describe sentences as nested phrases:

```
                     S
                ┌────┴────┐
               NP         VP
             ┌──┴──┐   ┌──┴──┐
            DT   NN   VBD    NP
             │    │    │    ┌─┴─┐
            The cat  sat   PP
                          ┌─┴─┐
                         IN   NP
                          │  ┌─┴─┐
                          on DT NN
                             │  │
                            the mat
```

Grammar formalism: **CFG** (Context-Free Grammar). The Penn Treebank uses phrase-structure trees; Charniak, Collins, and Berkeley parsers all consume this format.

**Dependency grammars** describe sentences as directed relations between individual words:

```
sat ──nsubj──> cat
sat ──obj/nmod──> mat (via prep on)
cat ──det──> The
mat ──det──> the
sat ──prep──> on
on  ──pobj──> mat
```

Every non-root word has exactly one head; every arc has a labelled grammatical relation (`nsubj`, `obj`, `det`, `amod`, etc.). Reference: **Universal Dependencies** (<https://universaldependencies.org/>), a cross-linguistic dependency scheme used by spaCy, Stanza, and Trankit, with treebanks for 150+ languages.

Both views are useful. Constituency reveals phrase-level structure (which spans are noun phrases); dependency reveals word-level structure (who acts on whom). Modern NLP has largely settled on dependency parsing as the default because arc-labelled trees are easier to consume in downstream code and cross-lingually consistent.

## Constituency parsing algorithms

### CKY (Cocke-Kasami-Younger)

Given a **Chomsky Normal Form** CFG (all rules of the form `A → BC` or `A → w`), CKY builds all possible parses bottom-up in a triangular table:

- Cell `(i, j)` stores the set of nonterminals that derive the span `wᵢ … wⱼ`.
- For each split point `k`, look up `(i, k)` and `(k+1, j)` and combine.
- Complexity: `O(n³ · |G|)`.

CKY parses any CFG, but a "plain" CFG has terrible ambiguity. The productions do not care that "sat on the mat" is more likely a VP than an NP.

### PCFGs — probabilistic CFGs

Assign each rule a probability. Parse probability is the product of rule probabilities. Choose the highest-probability tree with weighted CKY. Reference: Michael Collins, ["Three Generative, Lexicalised Models for Statistical Parsing"](https://aclanthology.org/P97-1003/), *ACL 1997*, and the "Collins parser" that followed; and Dan Klein, Christopher Manning, ["Accurate Unlexicalized Parsing"](https://nlp.stanford.edu/pubs/unlexicalized-parsing.pdf), *ACL 2003*, i.e., the Stanford PCFG parser.

Modern **neural constituency parsers** (Kitaev, Klein, ["Constituency Parsing with a Self-Attentive Encoder"](https://arxiv.org/abs/1805.01052), *ACL 2018*) replace hand-engineered PCFG features with contextual embeddings; the CKY decoder underneath is the same.

Ship a constituency parser when:

- You have to output Penn-Treebank-style trees for a downstream consumer (some semantic parsers, some rule-based summarisers).
- You care about spans as first-class objects (chunking, sentence compression).

Otherwise, prefer dependency parsers.

## Dependency parsing algorithms

Two families dominate:

### Transition-based (arc-eager / arc-standard)

Model parsing as a sequence of decisions taken by a shift-reduce automaton:

- **State:** a stack of processed words, a buffer of remaining words, and the arcs built so far.
- **Actions:** `SHIFT`, `LEFT-ARC(label)`, `RIGHT-ARC(label)`, `REDUCE`.
- **Classifier:** a per-step classifier (perceptron, MaxEnt, neural) predicts the next action from features of the state.
- **Complexity:** `O(n)` — one pass through the sentence, greedy or beam.

Nivre's arc-eager algorithm (Joakim Nivre, ["Algorithms for Deterministic Incremental Dependency Parsing"](https://aclanthology.org/J08-4003/), *CL 2008*) is the canonical reference. **MaltParser** (<https://www.maltparser.org/>) is the reference implementation; **spaCy's parser** is a fast neural transition-based parser (Chen & Manning, ["A Fast and Accurate Dependency Parser using Neural Networks"](https://cs.stanford.edu/~danqi/papers/emnlp2014.pdf), *EMNLP 2014*, then evolved).

Trade-off: fast, streaming-friendly, but greedy decoding can commit early errors that later evidence would fix. Beam search partially mitigates this.

### Graph-based (Eisner / MST)

Model parsing as finding the highest-scoring spanning tree over all possible arcs:

- **Score:** each candidate `(head, dependent, label)` arc gets a score from a classifier.
- **Decoder:** for *projective* trees (no crossing arcs), Eisner's algorithm — `O(n³)` dynamic programming. For *non-projective* trees (needed for German, Czech, Latin, Russian, and any language with free word order), Chu-Liu-Edmonds' Maximum Spanning Tree algorithm — `O(n²)`.

**MSTParser** (Ryan McDonald, ["Non-projective Dependency Parsing using Spanning Tree Algorithms"](https://aclanthology.org/H05-1066/), *HLT-EMNLP 2005*) is the reference; **Stanza's parser** and **Trankit** use neural biaffine scoring on top of a graph-based decoder (Dozat & Manning, ["Deep Biaffine Attention for Neural Dependency Parsing"](https://arxiv.org/abs/1611.01734), *ICLR 2017*), currently the strongest published dependency parsers.

Trade-off: higher accuracy, especially on long or non-projective sentences; slower and less amenable to streaming.

### CoNLL-U — the interchange format

Every serious dependency parser reads and writes CoNLL-U. One token per line, ten tab-separated fields, blank line between sentences:

```
1   The     the     DET    DT    Definite=Def|PronType=Art   2   det       _   _
2   cat     cat     NOUN   NN    Number=Sing                 3   nsubj     _   _
3   sat     sit     VERB   VBD   Mood=Ind|Tense=Past         0   root      _   _
4   on      on      ADP    IN    _                           6   case      _   _
5   the     the     DET    DT    Definite=Def|PronType=Art   6   det       _   _
6   mat     mat     NOUN   NN    Number=Sing                 3   obl       _   _
7   .       .       PUNCT  .     _                           3   punct     _   _
```

Fields: `ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC`. Reference: <https://universaldependencies.org/format.html>.

## Rule-based matching over parse trees

Once you have a parse tree, rules over it are dramatically more expressive than string regex. The three canonical rule languages:

### spaCy `DependencyMatcher`

Pattern-match subgraphs of the dependency tree. Example: verbs of "acquisition" with the acquirer as subject and the target as object:

```python
import spacy
from spacy.matcher import DependencyMatcher

nlp = spacy.load("en_core_web_sm")
matcher = DependencyMatcher(nlp.vocab)
pattern = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": {"IN": ["acquire", "buy", "purchase"]}}},
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subj",
     "RIGHT_ATTRS": {"DEP": "nsubj", "POS": {"IN": ["NOUN", "PROPN"]}}},
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "obj",
     "RIGHT_ATTRS": {"DEP": "dobj", "POS": {"IN": ["NOUN", "PROPN"]}}},
]
matcher.add("ACQUISITION", [pattern])
doc = nlp("Microsoft acquired Activision Blizzard in 2023.")
for match_id, tokens in matcher(doc):
    print([doc[i] for i in tokens])
# [acquired, Microsoft, Activision]
```

The `REL_OP` operators (`<`, `>`, `<<`, `>>`, `$-`, `$+`, and more) express head/dependent/ancestor/descendant/sibling relations. Reference: [spaCy `DependencyMatcher` docs](https://spacy.io/api/dependencymatcher).

### Stanford Semgrex / Tregex

- **Tregex** matches over Penn Treebank *constituency* trees.
- **Semgrex** matches over Stanford Dependencies / UD *dependency* graphs.
- Both are shell tools and library APIs shipped with Stanford CoreNLP; readable syntax; used in academic IE and biomedical extraction pipelines.
- Reference: Roger Levy, Galen Andrew, ["Tregex and Tsurgeon"](https://nlp.stanford.edu/software/tregex.html); Stanford Semgrex documentation.

### spaCy `Matcher` and `PhraseMatcher` (surface / lemma matching, not tree matching)

Simpler; matches token sequences with lemma / POS / shape constraints. Faster to write, less expressive. Use for gazetteer matching and short-span patterns; use `DependencyMatcher` for anything involving structure.

## Where classical parsers still win

- **Extraction from structured genres.** Financial filings, biomedical abstracts, legal citations. A DependencyMatcher pattern for "subject verb object" is a maintainable extractor that a domain expert can review — an LLM prompt for the same is a black box.
- **Coreference and negation feature extraction.** Rules over dependency arcs are still competitive baselines for negation ("negated by `neg`") and hedging detection in clinical NLP.
- **Query understanding.** Rewriting a search query into its content words + modifiers uses a parser routinely.
- **Consistency checks on LLM output.** Ask an LLM to answer, parse the answer, verify the parse matches a schema.

Neural parsers are now the default engine — Stanza's Universal Dependencies parsers score in the 90s LAS on most treebanks — but the parsing *interface* (CoNLL-U trees, DependencyMatcher patterns, span queries) is what actually ships in production code.

## Evaluating a parser

- **UAS (Unlabelled Attachment Score)** — fraction of tokens whose head is correct.
- **LAS (Labelled Attachment Score)** — fraction of tokens whose head *and* dependency label are correct.
- **F1 on labelled bracketing** — for constituency parsers (EVALB is the reference tool).
- **Parseval / bracketing accuracy** — older constituency metric; less relevant now.

Cross-language, LAS numbers are only meaningful within the same UD treebank and version. Do not compare a parser trained on `en_ewt` v2.13 to one trained on `en_gum` v2.11 and conclude one is "better".

## Chapter summary

- Two grammatical views: constituency (nested phrases, PCFG + CKY) and dependency (word-level arcs, transition-based or graph-based decoders).
- CoNLL-U is the interchange format; Universal Dependencies is the cross-lingual annotation scheme every modern parser targets.
- Neural parsers (biaffine graph-based, transition-based with contextual embeddings) beat classical ones on accuracy, but the decoders (Eisner, Chu-Liu-Edmonds, Nivre-style transition systems) are the same algorithms.
- The real product-level win of parsing today is *rule-based matching over trees*: spaCy `DependencyMatcher`, Stanford Semgrex, Tregex. These make maintainable extractors that survive an LLM refactor.
- UAS/LAS for dependency parsers, span F1 for constituency; always evaluate on the same treebank version.
- Next chapter: how to compose these pieces together with spaCy, NLTK, Stanza, and Stanford CoreNLP.
