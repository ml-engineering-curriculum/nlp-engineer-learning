# Domain Adaptation: Clinical, Legal, Scientific

## Motivation

A general-purpose embedding model — `BAAI/bge-large-en-v1.5`, `intfloat/e5-large-v2`, OpenAI `text-embedding-3-large` — is trained on the internet-shaped mixture of Wikipedia, Reddit, StackExchange, Common Crawl, MS MARCO. On any of those, it is excellent. On a corpus of ICD-10-coded discharge summaries, EU legislation, or biomedical abstracts, its retrieval quality degrades — sometimes by 10–20 nDCG@10 points — because the *domain vocabulary is under-represented in pretraining* and the *notion of "relevance" in the target domain is subtly different*.

This chapter is the recipe for closing that gap. It covers domain-adapted backbones (BioBERT, SciBERT, Legal-BERT, PubMedBERT, ClinicalBERT), domain-adapted embedding fine-tuning (GPL, TSDAE on the domain corpus, supervised fine-tune on domain-labelled pairs), and the domain-specific MTEB-style evaluation you need to trust the result. Chapter 10 does the equivalent for languages.

## Where the gap comes from

Two failure modes, both real:

1. **Tokenisation shatter.** Domain terms like `nasopharyngeal`, `pembrolizumab`, `stipulated`, `misappropriation`, `SARS-CoV-2`, `HbA1c` are tokenised into 4–10 subwords by a general SentencePiece / BPE. The encoder's per-token cost is higher and the shared cross-token signal is weaker. `nasopharyngeal` under `bert-base-uncased` becomes `[naso, ##pha, ##ryn, ##geal]`; under PubMedBERT it is one token.
2. **Distributional shift in "similarity."** In a general corpus, two documents about "cancer" and "diabetes" are meaningfully different. In an oncology-focused clinical corpus, the interesting distinctions are between "melanoma stage IIIA" and "melanoma stage IIIB." A general encoder considers those close-to-identical; a clinical encoder needs to separate them.

Adaptation attacks both problems.

## The domain-adapted backbones

Public checkpoints, all encoder-only BERT-family, all in-domain pretrained. Load them as HF `AutoModel` and wrap per chapter 03:

- **PubMedBERT** (Gu et al., ["Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing"](https://arxiv.org/abs/2007.15779), 2020) — `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`. Pretrained from scratch on PubMed abstracts + full text (not warm-started from general BERT). The strongest biomedical backbone for text.
- **BioBERT** (Lee et al., ["BioBERT: a pre-trained biomedical language representation model for biomedical text mining"](https://arxiv.org/abs/1901.08746), *Bioinformatics 2020*) — `dmis-lab/biobert-base-cased-v1.1`. General BERT warm-started, further pretrained on PubMed + PMC.
- **ClinicalBERT** (Alsentzer et al., ["Publicly Available Clinical BERT Embeddings"](https://arxiv.org/abs/1904.03323), 2019) — `emilyalsentzer/Bio_ClinicalBERT`. BioBERT further pretrained on MIMIC-III clinical notes. Reach for it if your corpus is clinical narrative (discharge summaries, notes), not published biomedical text.
- **SciBERT** (Beltagy, Lo & Cohan, ["SciBERT: A Pretrained Language Model for Scientific Text"](https://arxiv.org/abs/1903.10676), *EMNLP 2019*) — `allenai/scibert_scivocab_uncased`. Pretrained on 1.14 M full papers from Semantic Scholar. Its custom vocab covers a wider scientific range than BioBERT (chemistry, physics, biology, CS).
- **Legal-BERT** (Chalkidis et al., ["LEGAL-BERT: The Muppets straight out of Law School"](https://arxiv.org/abs/2010.02559), *EMNLP 2020 Findings*) — `nlpaueb/legal-bert-base-uncased`. Pretrained on EU legislation, ECHR case law, US contracts, and other legal text. Multiple variants (`legal-bert-small-uncased`, `bert-base-uncased-contracts`) for specific sub-domains.
- **FinBERT** (Yang, Uy & Huang, ["FinBERT: A Pretrained Language Model for Financial Communications"](https://arxiv.org/abs/2006.08097), 2020) — `yiyanghkust/finbert-tone` for sentiment, `ProsusAI/finbert` for text classification. Financial-corpus-pretrained.

For **retrieval-shaped** embeddings, the sentence-transformer bundles built on top of these backbones:

- **`pritamdeka/S-PubMedBert-MS-MARCO`** — PubMedBERT with an MS-MARCO retrieval fine-tune. Solid biomedical retrieval baseline.
- **`allenai/specter2` and `allenai/specter2_base`** — SciBERT with a scientific-citation-based training regime (Cohan et al., ["SPECTER: Document-level Representation Learning using Citation-informed Transformers"](https://arxiv.org/abs/2004.07180), *ACL 2020*, and Singh et al., ["SciRepEval"](https://arxiv.org/abs/2211.13308), 2023). The reference for "embed a paper by its title + abstract for citation similarity."
- **`malteos/scincl`** (Ostendorff et al., 2022) — SciNCL, an alternative citation-informed scientific embedding.
- **`nlpie/legalbert-base-uncased-4096-block`** and various legal sentence-transformer bundles on Hugging Face — the domain-adapted embedding shelf for legal is thinner than biomedical; often you have to train your own.

Rule of thumb: **start from the closest domain-adapted backbone available, then adapt further.** Do not fine-tune a general-domain encoder on domain pairs from scratch if a domain-pretrained backbone is available; you leave a lot on the table by ignoring the vocabulary and MLM signal that domain pretraining bought.

## Adaptation strategies, ranked by data cost

Four strategies, each with its own supervision requirement:

| Strategy | Labelled pairs needed | Compute cost | Expected lift |
|----------|----------------------|--------------|---------------|
| Swap backbone (use a domain-adapted checkpoint as-is) | 0 | Zero (just re-encode) | 2–5 points on domain retrieval |
| Domain **MLM pretraining** (further pretraining on domain text) | 0 | Weeks-scale on domain corpus | 3–7 points |
| **TSDAE / SimCSE** on domain text (unsupervised) | 0 | Hours-scale | 3–8 points |
| **GPL** (generative pseudo-labelling) | 0 real labels; synthetic queries | Days-scale | 5–15 points on hard cases |
| **Supervised fine-tune** on domain pairs | 1k+ (better: 10k+) | Hours-scale | 5–20 points |
| **Full curriculum** (pretrain + TSDAE + GPL + supervised) | Some | Days-scale | Highest achievable |

The best strategy depends on what you have. The right first move for a new domain with no labels is *almost always*:

1. Try the closest domain-adapted encoder off-the-shelf as a baseline.
2. Run GPL to adapt it to your specific corpus.
3. Evaluate on a small human-labelled domain retrieval set (~100 queries).

If GPL gives you the quality you need, stop there. If not, add domain MLM pretraining and/or supervised fine-tuning on any labelled pairs you can collect.

## GPL as the reference no-label adaptation recipe

**GPL** (Wang, Reimers, Gurevych, ["GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval"](https://arxiv.org/abs/2112.07577), *NAACL 2022*) is the pipeline for adapting an embedding model to a new corpus *without any labelled queries at all*. It's chapter 05's cross-encoder-denoising recipe pushed to its logical end.

The four steps:

1. **Query generation.** For each passage in the target corpus, generate 3–10 synthetic queries with a T5-based query generator (`doc2query/msmarco-t5-base-v1` is the reference generator, trained on MS MARCO). Each synthetic query pairs with the passage it came from.
2. **Hard-negative mining.** For each (synthetic-query, passage) pair, mine hard negatives from the target corpus with a bi-encoder (a general one is fine).
3. **Cross-encoder pseudo-labelling.** Score every `(q, d)` pair from steps 1–2 with a strong general-domain cross-encoder. Store the soft scores.
4. **Train the target bi-encoder** with `MarginMSELoss`: match the cross-encoder's score margins between positive and negative pairs.

The reference implementation lives in the `sentence-transformers` examples: [`training/domain_adaptation/train_gpl.py`](https://github.com/UKPLab/sentence-transformers/tree/master/examples/training/domain_adaptation).

Key knobs:

- **Queries per passage.** More is better up to ~10. Diminishing returns after.
- **Number of hard negatives per query.** 7–15.
- **Cross-encoder choice.** Bigger is better, but you pay it once. `BAAI/bge-reranker-large` is a strong default. For biomedical, `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` reranker or similar domain-adapted cross-encoder is worth trying.

The GPL paper reports 4–15 nDCG@10 improvement across BEIR benchmarks vs. general-domain baselines, and this is the shape you should expect in production.

## Supervised fine-tuning when you *do* have labels

If your team has (or can construct) even ~1000 labelled query-passage pairs from your domain, supervised fine-tuning on top of a domain-adapted backbone is the highest-leverage move:

- Warm-start from the domain-adapted encoder (or the general-domain encoder if none exists).
- Mine hard negatives *within your corpus* with the warm-start encoder (chapter 05).
- Fine-tune with `MultipleNegativesRankingLoss`, small batch (32–128), few epochs (1–2), low LR (1e-5).

Small labelled sets have a hazard: overfitting. Symptoms include perfect train recall and worse dev recall than the baseline. Mitigations:

- Weight-average the fine-tuned model with the base model at end of training (`0.5 * base + 0.5 * fine-tuned`), a simple model-soup lift.
- Freeze the bottom N layers of the encoder; only fine-tune the top layers.
- Use LoRA (Hu et al., 2021) on the encoder — chapter 10 in mod-107 for the general LoRA recipe; the domain-adaptation use case is identical.

## Domain-aware evaluation

Do not evaluate a legal embedding on MS MARCO. Do not evaluate a clinical embedding on Wikipedia STS-B. Symptoms of the mismatch are that domain-adapted models often *lose* on generic benchmarks — a legal-fine-tuned encoder scores worse on STS-B than the base model, because it has learned to distinguish legal-jargon subtleties that STS-B does not care about.

The right evaluation:

- **Public domain benchmarks.** `MTEB(law)`, `MTEB(medical)`, `CoIR` (code), BIOSSES (biomedical STS), TREC-COVID, NFCorpus, SciFact (scientific retrieval), CUREv1 (clinical trial abstracts), LEDGAR (contracts). Chapter 08.
- **In-house evaluation set.** ~100 real production queries with human-annotated relevance judgements. Small enough to build in a week, precise enough to detect real-world drift.
- **A/B test in production.** For any change that will be deployed, measure the business-metric impact — click-through, task completion, human-rated result quality — because MTEB and even in-house sets do not fully capture "did users find what they needed."

## Two worked-through examples

**Clinical discharge-summary retrieval.**

- Corpus: 500 k discharge summaries, patient-privacy-cleared. No labelled queries.
- Recipe: (1) start from `pritamdeka/S-PubMedBert-MS-MARCO`, (2) GPL: generate 5 queries per summary with a `doc2query`-clinical variant, denoise with `pritamdeka/BioBERT-mnli` cross-encoder, (3) train the S-PubMedBert bi-encoder with `MarginMSELoss` for 3 epochs.
- Evaluation: 100 human-labelled clinician queries, nDCG@10 vs. baseline.
- Expected shape: 4–10 nDCG@10 lift.

**EU legal search over legislation + case law.**

- Corpus: 2 M pieces of EU legislation and ECHR case law. ~5 000 labelled query-doc pairs from legal-research analyst logs.
- Recipe: (1) start from `nlpaueb/legal-bert-base-uncased` wrapped as a `SentenceTransformer` with mean pooling, (2) supervised fine-tune with `MultipleNegativesRankingLoss` on the 5k pairs + 7 mined hard negatives each, (3) 2 epochs, batch 64.
- Evaluation: MTEB(law) subset + held-out analyst queries.
- Expected shape: 5–15 nDCG@10 lift over general-domain encoder.

## Common failure modes

- **The domain backbone loses on the general benchmarks after fine-tuning.** Expected — it is now optimised for the domain distribution. Evaluate on domain, not on general.
- **GPL synthetic queries are all one-word.** The query generator did not generalise. Try a different `doc2query` variant, add domain-adapted generation, or fine-tune the generator on any in-domain query-passage pairs you have.
- **Supervised fine-tune destroys retrieval on unseen document types.** Overfit to the labelled set's document distribution. Fix: broaden the training set, freeze bottom layers, or weight-average with baseline.
- **Tokenisation shatter is *worse* after switching backbones.** You picked a domain BERT whose custom vocab does not actually cover your target sub-domain. Some Legal-BERT variants are EU-focused; US-contract-only vocab may not match. Check `tokenizer.tokenize()` on 20 sample domain terms before committing.
- **All GPL synthetic queries are near-paraphrases of the passage.** The generator is being lazy; the negatives are too easy. Add a diversity constraint on the generator (`num_return_sequences` with `top_k` sampling) and re-run.

## Chapter summary

- General-purpose embeddings degrade on domain corpora because of (a) tokenisation shatter of domain terms and (b) shifted similarity structure.
- Domain-adapted BERT-family backbones (PubMedBERT, BioBERT, ClinicalBERT, SciBERT, Legal-BERT, FinBERT) are the starting point for their respective verticals — always warm-start from the closest domain-adapted encoder available.
- Adaptation strategies scale with data: swap backbone (no data) → GPL (no labels) → supervised fine-tune (labelled pairs). Combine as needed.
- GPL — synthetic queries + cross-encoder denoising + MarginMSE distillation — is the no-labels reference recipe and is the default first move on a new corpus.
- Supervised fine-tuning with ~1000+ labelled pairs is the highest-leverage move when labels are available. Mine hard negatives within the target corpus.
- Evaluate on domain benchmarks (MTEB(law), MTEB(medical), CoIR, BIOSSES) plus a small in-house human-annotated set. General benchmarks (STS-B) are not diagnostic for domain models.
- Guard against overfitting on small labelled sets with weight averaging, layer freezing, or LoRA.
- Always inspect the tokeniser on 20 domain-representative terms before committing to a backbone — subword shatter is the silent killer.
