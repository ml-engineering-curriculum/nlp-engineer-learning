# Coreference Resolution at Document Scale

## Motivation

*"Marie Curie discovered polonium in 1898. She won the Nobel Prize in Physics five years later. The Polish-born chemist later added a second Nobel — this time in Chemistry — becoming the only person to win in two sciences."*

Every downstream IE task on this paragraph — NER, RE, EL, EE, summarisation — needs to know that "Marie Curie", "She", "The Polish-born chemist", and (arguably) "the only person" all refer to the same real-world entity. Coreference resolution (coref) is the task of grouping these mentions into clusters.

Coref is where document-level IE succeeds or fails. A relation-extraction system that catches *"Curie discovered polonium"* but misses *"She won the Nobel"* halves its recall. A knowledge-graph builder that treats each mention as a separate entity node produces a graph that is right by mention but useless by concept. Coref is the connective tissue.

## The task, precisely

Given a document with a set of mention spans, cluster the mentions so that mentions in the same cluster refer to the same real-world entity. Mentions include:

- **Proper nouns:** "Marie Curie", "Nobel Prize"
- **Nominal mentions:** "the Polish-born chemist", "the discovery"
- **Pronouns:** "she", "her", "it"
- **Cataphoric mentions:** rare in most domains; a pronoun that resolves later ("Before *he* answered, John hesitated")

Two evaluation settings, matching NER's two:

- **Gold-mention coref** — mentions are given; only cluster them. Standard on OntoNotes CoNLL-2012.
- **End-to-end coref** — model detects mention spans and clusters them jointly. What production runs.

## The canonical modern architecture

Lee, He, Lewis & Zettlemoyer, ["End-to-end Neural Coreference Resolution"](https://arxiv.org/abs/1707.07045), *EMNLP 2017*, set the modern paradigm and it has held for a decade:

1. **Encode the document** with a transformer (span-level embedding via boundary + attention-pool).
2. **Score every candidate span** as a mention (mention score).
3. **For each pair of mentions**, score compatibility (antecedent score).
4. **Cluster greedily** — for each mention, either link to its highest-scoring antecedent or start a new cluster (a "dummy" antecedent).

Refined and simplified in Joshi et al., ["BERT for Coreference Resolution: Baselines and Analysis"](https://arxiv.org/abs/1908.09091), *EMNLP 2019*, and Joshi et al., ["SpanBERT: Improving Pre-training by Representing and Predicting Spans"](https://arxiv.org/abs/1907.10529), *TACL 2020*, whose SpanBERT pretraining objective (predicting spans rather than tokens) is a natural fit for coref.

Modern strong baselines to know:

- **`coref-hoi`** (Xu & Choi, ["Revealing the Myth of Higher-Order Inference in Coreference Resolution"](https://arxiv.org/abs/2009.12013), *EMNLP 2020*) — refines the pairwise model with higher-order inference.
- **`s2e-coref`** (Kirstain, Ram & Levy, ["Coreference Resolution without Span Representations"](https://arxiv.org/abs/2101.00434), *ACL-IJCNLP 2021*) — reduces memory footprint by not materialising span representations.
- **wl-coref** (Dobrovolskii, ["Word-Level Coreference Resolution"](https://arxiv.org/abs/2109.04127), *EMNLP 2021*) — links words rather than spans, then expands to spans. Faster and simpler.
- **maverick** (Martinelli et al., ["Maverick: Efficient and Accurate Coreference Resolution Defying Recent Trends"](https://arxiv.org/abs/2407.21489), *ACL 2024*) — recent efficient baseline; competitive with the higher-order models at lower cost.

## Metrics: MUC, B³, CEAF, and the CoNLL average

Coref has four competing metrics because there is no single "obvious" way to score cluster agreement:

- **MUC** (Vilain et al., 1995) — counts the minimum number of edits needed to align predicted and gold clusters. Sensitive to over-linking; less so to over-splitting.
- **B³** (Bagga & Baldwin, 1998) — per-mention precision and recall based on cluster overlap. More balanced.
- **CEAF** (Luo, 2005) — Constrained Entity Alignment F-Measure. Aligns clusters via Hungarian match then scores.
- **BLANC** (Recasens & Hovy, 2011) — coreference / non-coreference link–based; less common.

The standard headline number is **CoNLL F1**, the arithmetic mean of MUC-F1, B³-F1, and CEAFφ4-F1. Report the average and each individual metric. `coval` (<https://github.com/ns-moosavi/coval>) is the reference scorer; every modern paper reports its output.

## Handling long documents

The Lee et al. architecture is `O(n²)` in candidate spans and `O(n × candidates)` in antecedent scoring. Straight-out-of-the-box coref does not fit on a 5 000-token document without either:

- **Span pruning** — score all candidates cheaply, keep the top `M` per document (typically `M ≈ 0.4 × n`).
- **Windowed encoding + document-level clustering** — encode the document in overlapping windows (chapter 07), score span pairs within a bounded distance, then run a cluster-merging pass over the full document.
- **Long-context encoders** — SpanBERT + Longformer combinations exist and beat sentence-level baselines on GAP and OntoNotes long-doc subsets.

For very long documents (contracts, discharge summaries), incremental coref (Xia, Sedoc & Van Durme, ["Incremental Neural Coreference Resolution in Constant Memory"](https://arxiv.org/abs/2005.00128), *EMNLP 2020*) is the memory-friendly option.

## The engineering trade-off nobody talks about

Coref is expensive to run and expensive to evaluate. Two practical shortcuts that show up in production:

### Rule-based fallback

For very long documents where neural coref is infeasible, string-match + simple heuristics get 70–80 % of the benefit:

- Cluster mentions with matching normalised surface forms.
- Attach pronouns to the nearest gender-matching, plausibly-typed antecedent.
- Attach definite noun phrases ("the CEO") to the nearest type-matching entity mention.

The Stanford Multi-Pass Sieve system (Lee et al., ["Deterministic Coreference Resolution Based on Entity-centric, Precision-ranked Rules"](https://aclanthology.org/J13-4004/), *Computational Linguistics 2013*) is the canonical reference for the rule-based approach and, up to about 2018, was the fastest way to add coref to a production pipeline.

### Skip coref, use EL to deduplicate

If your goal is a document-level entity set (not full clusters), an alternative is to link every mention to a KB (chapter 11) and treat entities with the same KB ID as coreferent by construction. This side-steps neural coref for named-entity mentions. It does not help with pronouns.

## Cross-document coreference

Coreference across documents ("*Apple*" in article A vs. "*Apple Inc.*" in article B) is the natural next step for building knowledge graphs from a corpus.

- **Cattan et al., ["Streamlining Cross-Document Coreference Resolution: Evaluation and Modeling"](https://arxiv.org/abs/2009.11032), *arXiv 2020*.** ECB+ benchmark; modern cross-doc coref architectures.
- **CDLM** (Caciularu et al., ["Cross-Document Language Modeling"](https://arxiv.org/abs/2101.00406), *EMNLP Findings 2021*) — pretrained encoder for cross-document tasks including coref.

For most production pipelines cross-document coref is a two-step process: intra-document coref, then cluster-level entity linking to a shared KB. Model-based cross-doc coref pays off when the KB is absent or unreliable.

## Bias and fairness in coref

Coref is one of the NLP tasks most studied for social bias — pronoun-antecedent choices reflect gender and racial assumptions in training data. Two references worth citing:

- **Rudinger et al., ["Gender Bias in Coreference Resolution"](https://aclanthology.org/N18-2002/), *NAACL 2018*.** Introduces the WinoBias / Winogender-style stress tests.
- **Zhao et al., ["Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods"](https://arxiv.org/abs/1804.06876), *NAACL 2018*.**

Test on WinoBias / Winogender if you ship coref in a product where gendered pronoun resolution matters (chatbots, HR tools, biography-processing). Report the accuracy gap between stereotype-consistent and stereotype-inconsistent test cases as a fairness metric alongside CoNLL F1.

## The library ecosystem

- **Stanford CoreNLP** — Java-based multi-pass sieve; still a solid rule-based baseline.
- **spaCy `coreferee`** — <https://github.com/msg-systems/coreferee>. Neural-plus-rules coref for English, German, Polish.
- **`crosslingual-coreference`** — spaCy pipeline component wrapping neural models.
- **`fastcoref`** (Otmazgin et al., ["F-COREF: Fast, Accurate and Easy to Use Coreference Resolution"](https://arxiv.org/abs/2209.04280), *AACL 2022*) — production-oriented coref with speed as a first-class concern.
- **`maverick-coref`** — <https://github.com/SapienzaNLP/maverick-coref>. The current efficient baseline.
- **AllenNLP `coref`** — Lee et al. reference implementation, still occasionally maintained.

## Failure modes

1. **Singleton spans polluting output.** Neural coref emits many "clusters of size one" — mentions the model was uncertain about. Downstream systems often want only clusters of size ≥ 2 as "coref clusters" and treat singletons as regular entities. Configure filtering explicitly.
2. **Gendered-pronoun resolution mistakes.** Common on domain text (legal filings, medical charts) where profession-gender priors mislead. Run WinoBias-style stress tests before shipping.
3. **First-mention span mismatch.** Coref clusters usually anchor on the first (longest, canonical) mention. If your NER emits a shorter version ("Curie" instead of "Marie Curie"), downstream code keys on the wrong string. Store the full cluster; use the longest mention as the canonical form.
4. **Cross-sentence but same-paragraph collapse.** Some models require paragraph or discourse features to link mentions across sentences. Loosen the antecedent-distance limit if recall on long documents is low.
5. **Coreference across speaker turns** in dialogue is a special case — "he" in speaker B's turn may refer to speaker A. Dialogue-adapted coref (e.g., PersuAsion, PersonaChat coref systems) is a separate literature.
6. **Memory blowup on long documents.** Span-representation coref explodes to 8+ GB VRAM on a 5 000-token document; switch to word-level (wl-coref) or incremental (Xia et al.) coref.

## Chapter summary

- Coreference resolution groups mentions of the same real-world entity into clusters; every document-level IE task depends on it.
- The Lee et al. mention-pair architecture with SpanBERT-style encoders is the workhorse; wl-coref and Maverick are the modern efficient variants.
- Evaluate with the CoNLL average of MUC-F1, B³-F1, and CEAFφ4-F1; `coval` is the reference scorer.
- Long documents require span pruning, windowed encoding, or incremental coref; rule-based fallbacks capture ~70–80 % of the benefit at a fraction of the cost.
- Cross-document coref matters for knowledge-graph construction; EL against a shared KB is the common shortcut.
- Bias testing (WinoBias / Winogender) is not optional if coref decides gendered-pronoun antecedents in a product.
