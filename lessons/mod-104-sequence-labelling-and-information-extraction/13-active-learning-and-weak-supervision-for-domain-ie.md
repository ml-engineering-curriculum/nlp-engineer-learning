# Active Learning and Weak Supervision for Domain IE

## Motivation

Standard benchmarks — CoNLL-2003, OntoNotes, ACE — ship with tens of thousands of gold-annotated entities. Domain projects do not. A first-week clinical NER project might have 200 labelled discharge summaries. A legal-clause tagger for a new contract type might start with zero. A financial-event extractor for a niche filing type might have exactly what one analyst has hand-labelled between meetings.

Two techniques bend the labelling cost curve:

- **Active learning** — instead of labelling documents uniformly at random, ask the model which documents to label next. Labels the highest-information examples first.
- **Weak supervision** — instead of asking humans for labels at all, write rules, use distant supervision, or prompt an LLM to label the corpus, then denoise the noisy labels programmatically.

Both are load-bearing in real domain IE. This chapter walks the concrete recipes and shows where each fits.

## Active learning: the loop

The canonical iterative loop:

1. Train a NER model on the currently labelled data.
2. Run it on the unlabelled pool.
3. Score every unlabelled example by a **query strategy** (below).
4. Send the top-`k` to human annotators.
5. Add the newly labelled examples to the training set; repeat.

Under a fixed labelling budget, active learning typically reaches a given F1 target with 30–60 % fewer labels than random sampling (Settles, ["Active Learning Literature Survey"](https://minds.wisconsin.edu/handle/1793/60660), *University of Wisconsin–Madison, 2010*, is the comprehensive reference; the general pattern reproduces in NER-specific studies like Shen et al., ["Deep Active Learning for Named Entity Recognition"](https://arxiv.org/abs/1707.05928), *ICLR 2018*).

## Query strategies for NER

Not every strategy transfers cleanly from classification to sequence labelling. The four you should know:

### Least confidence

Score each unlabelled document by `1 - p(most_likely_sequence)`. Requires either Viterbi + CRF (chapter 05) or an approximation from independent softmax. Simple and reliable; the default first-shot strategy.

### Token entropy (max over positions)

For each token, entropy of the tag distribution; document score = max entropy across tokens. Rewards documents with individually uncertain tokens.

### Sequence entropy

Entropy over the full tag-sequence distribution — the sum of per-token entropies, approximating the joint under the independence assumption. Rewards documents with globally spread-out uncertainty.

### Query-by-committee

Train `M` models (different seeds, subsampled data, or perturbed hyperparameters); score examples by inter-model disagreement (per-token vote entropy, tag KL). More compute but often the strongest strategy, especially early in the loop when a single model is unstable.

### Diversity-aware sampling

Confidence-only strategies concentrate queries in a narrow slice of the input distribution ("all the ambiguous drug names"). Diversity strategies balance uncertainty with representative coverage — e.g., k-means clustering of embeddings, then select the most uncertain example per cluster. See Zhdanov, ["Diverse mini-batch Active Learning"](https://arxiv.org/abs/1901.05954), *arXiv 2019*.

Rule: start with least confidence; upgrade to query-by-committee if the labelling budget justifies the training cost.

## Practical active-learning tools

- **`prodigy`** — <https://prodi.gy/>. Commercial; the reference labelling UI for NER + active learning. Integrates with spaCy.
- **`Label Studio`** — <https://labelstud.io/>. Open-source; supports active-learning workflows via ML backends.
- **`small-text`** — <https://github.com/webis-de/small-text>. Python active-learning framework with pluggable query strategies.
- **`modAL`** — <https://github.com/modAL-python/modAL>. Older but widely used AL framework; less NER-focused.

## Cost-aware active learning

Not all annotations cost the same. A 5-token entity in a short sentence costs 30 seconds of annotator time; a full 500-token clinical note might take 20 minutes. Cost-aware AL divides the query score by the annotation cost estimate — Settles, Craven & Friedland, ["Active Learning with Real Annotation Costs"](https://web.eecs.umich.edu/~aparanjp/eecs486/papers/Real_Annotation_Costs.pdf), *NIPS Workshop 2008*. In practice this shifts queries toward shorter, information-dense examples.

## Weak supervision: rules + LLMs + denoising

Weak supervision replaces "human labels" with "programmatically generated noisy labels." Three sources:

### Rule-based labellers

Regular expressions, gazetteers, PoS-pattern rules. Cheap, high-precision, low-recall.

```python
# Toy example: label PROTEIN mentions in biomedical text
def label_protein(text):
    matches = []
    for m in re.finditer(r"\b([A-Z][A-Z0-9]{1,6})\b", text):
        if m.group(1) in UNIPROT_SYMBOLS:
            matches.append((m.start(), m.end(), "PROTEIN"))
    return matches
```

### Distant supervision

Align a knowledge base to text: every sentence containing two entities known to be related gets a positive relation label. Every mention of a KB-listed entity gets a NER label of that entity's type. Introduced by Mintz et al., ["Distant supervision for relation extraction without labeled data"](https://aclanthology.org/P09-1113/), *ACL 2009*, for RE; the same idea works for NER against gazetteers or KBs.

### LLM-based labellers

Prompt an LLM to label a corpus. Higher recall than rules, decent precision if the prompt is careful and validated. Chapter 10's structured-extraction recipe *is* an LLM-based labeller when applied to unlabelled data. Common practice in 2024–2026 is:

- Run the LLM on the unlabelled pool.
- Sample 200 labelled examples for human review to estimate LLM precision per label type.
- If precision is high enough (typically ≥ 0.85), use the LLM labels as training data — either directly or after denoising.

### Denoising with Snorkel-style label models

Multiple noisy labellers on the same corpus produce a matrix of votes per example. A **label model** (Ratner et al., ["Snorkel: Rapid Training Data Creation with Weak Supervision"](https://arxiv.org/abs/1711.10160), *VLDB 2018*) learns each labeller's accuracy and correlations, then aggregates their votes into probabilistic labels. A discriminative model is then trained on those soft labels.

For sequence labelling, extensions exist:

- **`skweak`** (Lison et al., ["skweak: Weak Supervision Made Easy for NLP"](https://arxiv.org/abs/2104.09683), *ACL Demo 2021*) — HMM-based label aggregation for NER; the canonical modern tool.
- **`wrench`** — <https://github.com/JieyuZ2/wrench>. Weak-supervision benchmark and toolkit.
- **`Snorkel Flow`** — the commercial successor to open-source Snorkel; NER + RE workflows built-in.

## Combining active learning and weak supervision

The most effective real-world pipelines combine both:

1. **Cold start.** Write 10–30 simple rules that cover the easy cases with high precision. Run over the unlabelled pool.
2. **LLM labelling** on the rule-uncovered cases. Merge labels with `skweak` / Snorkel.
3. **Distil a small tagger** on the aggregated soft labels.
4. **Active-learning loop** from the distilled tagger — target human review at the highest-disagreement examples between the rules, the LLM, and the tagger.
5. **Iterate.** Every AL cycle produces more gold examples that improve both the tagger and the labelling-quality estimator.

This is the pattern that makes domain NER economically viable. It is also the pattern behind Amazon's product-attribute extraction, most healthcare-NLP startups, and every research NER paper on a domain not covered by OntoNotes.

## Data quality: the silent metric

Active learning and weak supervision put you in a position where your labels are not gold. You need to measure label quality alongside model performance:

- **Inter-annotator agreement (IAA)** on the human-labelled subset. Report Cohen's κ or F1 (annotator vs. annotator). Below κ = 0.7, your labels are the noise ceiling for your model.
- **Rule / LLM precision** on the human-reviewed subset. Report per-type; a labeller that is 95 % precise on PER but 60 % on FAC needs per-type filtering.
- **Coverage** — what fraction of the true entities in your labelled subset does each labeller catch? Rules typically 40–70 %; LLMs 70–90 %.
- **Redundancy** — how often do labellers agree on the same span? High agreement is not automatically good if all labellers share the same blind spots.

## Domain considerations

### Clinical

- **De-identification is a prerequisite.** i2b2 / N2C2 shared tasks (Uzuner et al., ["Extracting Medication Information from Clinical Text"](https://pubmed.ncbi.nlm.nih.gov/20819857/), *JAMIA 2010*) established the standard schemas.
- **UMLS as the linking KB** and as a weak-supervision gazetteer.
- **Long documents** — sliding-window or Longformer.
- **Clinical LMs** — `Bio_ClinicalBERT`, `Med-BERT`, `GatorTron`.

### Legal

- **Ontology-heavy.** LexNLP, Kira, and the CUAD (Contract Understanding Atticus Dataset, Hendrycks et al., ["CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review"](https://arxiv.org/abs/2103.06268), *NeurIPS Datasets 2021*) all bring detailed schemas.
- **Long documents.** Contracts commonly exceed 50 000 tokens; chunking + coreference are load-bearing.
- **Legal LMs** — LEGAL-BERT, InLegalBERT.

### Finance

- **Ticker / CUSIP / LEI gazetteers** are strong weak-supervision sources.
- **Event ontology from the FIBO** (Financial Industry Business Ontology).
- **FinBERT** (Yang, Uy & Huang, ["FinBERT: A Pretrained Language Model for Financial Communications"](https://arxiv.org/abs/2006.08097), *arXiv 2020*) for domain LM.
- **Real-time freshness** — new tickers, mergers, and executive appointments make the KB stale fast; retrain alias indexes weekly.

## The economics, honestly

A rough rule from published labelling studies and industry practice:

- **Human gold labels:** $1–$5 per entity for standard domains; $10–$50 for specialist domains (medical, legal).
- **LLM labels via API:** $0.0001–$0.001 per entity, at ~80–95 % precision.
- **Rule-based labels:** near-zero marginal cost after rule development; typically 60–80 % precision, 30–70 % recall.

The domain-IE budget question is not "will we use active learning or weak supervision" but "what mix of {rules, LLM, human review} minimises the cost of reaching our F1 target?" There is no universal answer; run the calculation with your actual labelling costs.

## Failure modes

1. **AL query strategy dominated by outliers.** Highest-uncertainty examples are often malformed input (encoding errors, OCR garbage). Filter by an input-quality score before ranking.
2. **Rule labellers with silent false positives** — a regex that also matches non-entities. Measure per-rule precision on a held-out set before adding it to the pool.
3. **LLM labeller drift.** The same prompt today and in three months returns slightly different labels because the API updated the model. Pin model versions; log labeller version in every emitted label.
4. **Denoising overweighting a correlated labeller.** Two rules that both fire on `[A-Z]{2,}` are not independent evidence. Snorkel-style label models handle correlations if you tell them; the naïve "average the votes" does not.
5. **Distribution shift between labelled pool and production.** AL selects from an unlabelled pool that may not match production traffic. Sample production traffic into the AL pool.
6. **Human annotators reading model predictions.** Presenting model predictions to annotators biases labels toward agreement with the model, inflating measured accuracy and reducing the diversity of new labels. Present blank text for a fraction of examples.

## Chapter summary

- Domain IE is data-scarce; active learning and weak supervision are the two levers that bend the labelling cost curve.
- Active-learning query strategies for NER — least confidence, sequence entropy, query-by-committee, diversity-aware — reach the same F1 target with 30–60 % fewer labels than random sampling.
- Weak supervision replaces gold labels with rules + distant supervision + LLM labels; Snorkel-style / skweak denoising aggregates noisy labellers into training-ready soft labels.
- The workhorse pattern combines both: rules cold-start, LLM covers the rules' gaps, active learning targets human review at maximum-disagreement examples, distil into a small tagger, iterate.
- Data quality metrics (IAA, per-type labeller precision, coverage, redundancy) are as important as model metrics — the ceiling of your model is the noise floor of your labels.
- Clinical, legal, and financial IE each bring domain-specific ontologies, KBs, and pretrained LMs; use them all before writing anything from scratch.
