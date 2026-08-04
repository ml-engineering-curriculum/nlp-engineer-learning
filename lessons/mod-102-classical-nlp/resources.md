# Resources for mod-102 · Classical NLP

Prefer primary sources: papers, standards, and official documentation. Blog posts and tutorials are included only where they are the canonical or most-referenced explanation.

## Textbooks and canonical references

- **Jurafsky & Martin, *Speech and Language Processing*, 3rd ed. (in progress)** — <https://web.stanford.edu/~jurafsky/slp3/>. Chapters 2 (regular expressions), 3 (n-gram LMs), 8 (POS + HMM), 17 (dependency parsing) are the classical-NLP core.
- **Manning & Schütze, *Foundations of Statistical Natural Language Processing*, MIT Press, 1999.** The reference text for HMMs, PCFGs, and pre-2000 statistical NLP.
- **Bird, Klein & Loper, *Natural Language Processing with Python* (NLTK Book)** — <https://www.nltk.org/book/>. Free online; the canonical Python-first classical NLP textbook.
- **Beesley & Karttunen, *Finite State Morphology*, CSLI Publications, 2003.** The reference text for FST morphology and the `xfst` / `lexc` toolchain.

## Regex and pattern matching

- **Cloudflare, ["Details of the Cloudflare outage on July 2, 2019"](https://blog.cloudflare.com/details-of-the-cloudflare-outage-on-july-2-2019/)** — post-mortem on a regex-induced production outage; canonical reading on ReDoS.
- **Russ Cox, ["Regular Expression Matching Can Be Simple And Fast"](https://swtch.com/~rsc/regexp/regexp1.html), 2007.** Motivating essay for RE2 and the linear-time NFA approach.
- **RE2** (C++ / Go / Python via `google-re2`) — <https://github.com/google/re2>.
- **Rust `regex`** — <https://docs.rs/regex/>. Linear-time NFA/DFA hybrid, similar guarantees to RE2.
- **Intel Hyperscan** — <https://github.com/intel/hyperscan>. Multi-pattern high-throughput DFA matcher.
- **`regex` (Python, third-party)** — <https://pypi.org/project/regex/>. Unicode-property escapes, `\X` grapheme cluster, UAX #29 word/sentence breaks; the stdlib `re` upgrade.

## Finite-state transducers and text normalisation

- **OpenFST** (Allauzen et al., 2007) — <https://www.openfst.org/>. Weighted FST library and shell tools.
- **Thrax / pynini** — <https://www.openfst.org/twiki/bin/view/GRM/Thrax> and <https://www.openfst.org/twiki/bin/view/GRM/Pynini>. Grammar languages on top of OpenFST.
- **`nemo-text-processing`** (NVIDIA) — <https://github.com/NVIDIA/NeMo-text-processing>. Production TN/ITN grammars in pynini.
- **Zhang et al., "NeMo Inverse Text Normalization: From Development to Production", *Interspeech 2021***, [arXiv:2104.05055](https://arxiv.org/abs/2104.05055).
- **Ebden & Sproat, "The Kestrel TTS text normalization system", *Natural Language Engineering*, 2015** — the design of Google's production TN system that Sparrowhawk / NeMo TN descend from.
- **Sparrowhawk** — <https://github.com/google/sparrowhawk>. Google's open-source runtime for Kestrel-style grammars; historical baseline.
- **`foma`** (Hulden, EACL 2009) — <https://fomafst.github.io/>. Open-source `xfst`/`lexc`-compatible FST engine.
- **HFST — Helsinki Finite-State Toolkit** — <https://hfst.github.io/>.

## Unicode standards (re-used from mod-101)

- **UAX #29 · Unicode Text Segmentation** — <https://www.unicode.org/reports/tr29/>.
- **UAX #15 · Unicode Normalization Forms** — <https://www.unicode.org/reports/tr15/>.
- **UAX #44 · Unicode Character Database (Script property, General_Category)** — <https://www.unicode.org/reports/tr44/>.
- **BCP-47 · Tags for Identifying Languages (RFC 5646)** — <https://www.rfc-editor.org/info/bcp47>.
- **International Components for Unicode (ICU)** — <https://icu.unicode.org/>. Reference implementation of segmentation, normalisation, transliteration, collation.

## Sentence segmentation and lemmatisation

- **Kiss & Strunk, ["Unsupervised Multilingual Sentence Boundary Detection"](https://aclanthology.org/J06-4003/), *Computational Linguistics*, 2006.** The Punkt algorithm.
- **`pysbd` (Pragmatic Segmenter port)** — <https://github.com/nipunsadvilkar/pySBD>. Rule-based multilingual sentence splitter.
- **Porter, ["An algorithm for suffix stripping"](https://tartarus.org/martin/PorterStemmer/def.txt), *Program*, 14(3), 1980.**
- **Snowball stemmer family** — <https://snowballstem.org/>.
- **Universal Dependencies** — <https://universaldependencies.org/>. Cross-lingual treebank and lemma inventory.

## n-gram language models

- **Jelinek & Mercer, "Interpolated estimation of Markov source parameters from sparse data", *Pattern Recognition in Practice*, 1980.** Interpolation smoothing.
- **Katz, "Estimation of Probabilities from Sparse Data for the Language Model Component of a Speech Recognizer", *IEEE ASSP*, 1987.** Backoff.
- **Kneser & Ney, "Improved backing-off for M-gram language modeling", *ICASSP 1995*.** The core Kneser-Ney idea.
- **Chen & Goodman, ["An empirical study of smoothing techniques for language modeling"](https://www.cs.cmu.edu/~roni/11761/PreviousYearsHandouts/chen-goodman-99.pdf), *Computer Speech & Language*, 1999.** Modified Kneser-Ney; the definitive survey.
- **Brants et al., ["Large Language Models in Machine Translation"](https://aclanthology.org/D07-1090.pdf), *EMNLP 2007*.** Stupid backoff at scale.
- **Heafield, ["KenLM: Faster and Smaller Language Model Queries"](https://kheafield.com/papers/avenue/kenlm.pdf), *WMT 2011*.** The current production toolkit.
- **KenLM** — <https://github.com/kpu/kenlm>.
- **SRILM** — <http://www.speech.sri.com/projects/srilm/>.

## Statistical sequence taggers

- **Rabiner, ["A tutorial on hidden Markov models and selected applications in speech recognition"](https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf), *Proceedings of the IEEE*, 1989.** HMM canonical reference.
- **McCallum, Freitag & Pereira, ["Maximum Entropy Markov Models for Information Extraction and Segmentation"](https://www.cs.umass.edu/~mccallum/papers/memm-icml2000.pdf), *ICML 2000*.**
- **Lafferty, McCallum & Pereira, ["Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data"](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers), *ICML 2001*.** The label-bias problem and CRFs.
- **Sutton & McCallum, ["An Introduction to Conditional Random Fields"](https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf), *Foundations and Trends in Machine Learning*, 2012.** Book-length CRF tutorial.
- **Collins, "Discriminative Training Methods for Hidden Markov Models: Theory and Experiments with Perceptron Algorithms", *EMNLP 2002*.** The structured perceptron.
- **Honnibal, ["A Good Part-of-Speech Tagger in about 200 Lines of Python"](https://explosion.ai/blog/part-of-speech-pos-tagger-in-python).** The averaged-perceptron tagger as spaCy ships it.
- **Lample et al., ["Neural Architectures for Named Entity Recognition"](https://arxiv.org/abs/1603.01360), *NAACL 2016*.** The BiLSTM-CRF architecture that put a classical CRF layer on top of a neural encoder.
- **CRFsuite / `python-crfsuite`** — <https://github.com/scrapinghub/python-crfsuite>.
- **`sklearn-crfsuite`** — <https://sklearn-crfsuite.readthedocs.io/>.
- **`seqeval`** — <https://github.com/chakki-works/seqeval>. Span-level F1 for BIO/BIOES tags.

## Parsers

- **Collins, ["Three Generative, Lexicalised Models for Statistical Parsing"](https://aclanthology.org/P97-1003/), *ACL 1997*.** Lexicalised PCFGs.
- **Klein & Manning, ["Accurate Unlexicalized Parsing"](https://nlp.stanford.edu/pubs/unlexicalized-parsing.pdf), *ACL 2003*.** Stanford PCFG parser.
- **Kitaev & Klein, ["Constituency Parsing with a Self-Attentive Encoder"](https://arxiv.org/abs/1805.01052), *ACL 2018*.** The state-of-the-art constituency parser.
- **Nivre, ["Algorithms for Deterministic Incremental Dependency Parsing"](https://aclanthology.org/J08-4003/), *CL 2008*.** Transition-based parsing formalisms.
- **McDonald et al., ["Non-projective Dependency Parsing using Spanning Tree Algorithms"](https://aclanthology.org/H05-1066/), *HLT-EMNLP 2005*.**
- **Chen & Manning, ["A Fast and Accurate Dependency Parser using Neural Networks"](https://cs.stanford.edu/~danqi/papers/emnlp2014.pdf), *EMNLP 2014*.** The neural transition-based parser used in Stanford CoreNLP and (in evolved form) spaCy.
- **Dozat & Manning, ["Deep Biaffine Attention for Neural Dependency Parsing"](https://arxiv.org/abs/1611.01734), *ICLR 2017*.** The biaffine parser used in Stanza.
- **CoNLL-U format** — <https://universaldependencies.org/format.html>.
- **Semgrex / Tregex** — <https://nlp.stanford.edu/software/tregex.html>. Stanford's tree-pattern matchers.
- **MaltParser** — <https://www.maltparser.org/>.

## Toolkits

- **spaCy** — <https://spacy.io/>. Production Python NLP pipeline.
- **spaCy `Matcher`, `PhraseMatcher`, `DependencyMatcher`** — <https://spacy.io/usage/rule-based-matching>.
- **Stanza** (Qi et al., ["Stanza: A Python Natural Language Processing Toolkit for Many Human Languages"](https://arxiv.org/abs/2003.07082), *ACL 2020*) — <https://stanfordnlp.github.io/stanza/>.
- **Stanford CoreNLP** (Manning et al., ["The Stanford CoreNLP Natural Language Processing Toolkit"](https://nlp.stanford.edu/pubs/manning-EtAl_ACL2014-CoreNLP.pdf), *ACL 2014*) — <https://stanfordnlp.github.io/CoreNLP/>.
- **NLTK** — <https://www.nltk.org/>. Bird, Klein, Loper reference toolkit.
- **Trankit** (Nguyen et al., ["Trankit: A Light-Weight Transformer-based Toolkit for Multilingual Natural Language Processing"](https://aclanthology.org/2021.eacl-demos.10/), *EACL 2021*) — <https://trankit.readthedocs.io/>. Transformer-backed alternative to Stanza.
- **scispaCy** (Beltagy et al., ["ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing"](https://arxiv.org/abs/1902.07669), 2019) — <https://allenai.github.io/scispacy/>. Biomedical domain-adapted spaCy models.

## Language identification

- **Bojanowski, Grave, Joulin & Mikolov, ["Enriching Word Vectors with Subword Information"](https://arxiv.org/abs/1607.04606), *TACL 2017*.** Word representations underlying fastText.
- **Joulin et al., ["Bag of Tricks for Efficient Text Classification"](https://arxiv.org/abs/1607.01759), *EACL 2017*.** Linear classifier at the heart of fastText LID.
- **fastText language identification** — <https://fasttext.cc/docs/en/language-identification.html>.
- **CLD3** (Google) — <https://github.com/google/cld3>. Python: `gcld3`, `pycld3`.
- **Lui & Baldwin, ["langid.py: An Off-the-shelf Language Identification Tool"](https://aclanthology.org/P12-3005.pdf), *ACL 2012*.** — <https://github.com/saffsd/langid.py>.
- **Cavnar & Trenkle, "N-Gram-Based Text Categorization", *SDAIR 1994*.** DIY character-n-gram LM approach.
- **Aguilar et al., ["LinCE: A Centralized Benchmark for Linguistic Code-switching Evaluation"](https://aclanthology.org/2020.lrec-1.223/), *LREC 2020*.**

## Hybrid pipelines and industrial context

- **Sproat et al., "Normalization of non-standard words", *Computer Speech & Language*, 2001.** The paper that framed text normalisation as a semiotic-class problem.
- **Ram et al., ["Conversational AI: The Science Behind the Alexa Prize"](https://arxiv.org/abs/1801.03604), 2018.** Case study in hybrid classical + neural production NLP.
- **`Duckling`** (Facebook) — <https://github.com/facebook/duckling>. Haskell-based probabilistic parser for dates, times, currencies, quantities; classical NLP shipping in Messenger and Wit.ai.
