# exercise-04: Domain and Multilingual Embeddings

**Estimated effort:** 2 hours

## Objective

Adapt an embedding model to (a) a specialised vertical (clinical, legal, or scientific) *and* (b) a non-English language family, then evaluate the adapted models against general-purpose baselines on a domain-appropriate benchmark. Chapter 09 provides the domain adaptation playbook; chapter 10 covers the language adaptation counterpart. This exercise builds one of each end-to-end.

## Prerequisites

- Chapters [09](../09-domain-adaptation-clinical-legal-scientific.md) and [10](../10-multilingual-and-cross-lingual-embeddings.md).
- Exercise-02 (you will reuse the hard-negative mining + cross-encoder denoising machinery).
- Python 3.10+; `sentence-transformers>=3.0`, `datasets`, `torch`, `mteb>=1.10`, `transformers`, `faiss-cpu`/`faiss-gpu`.
- A GPU is strongly recommended.

## Scope

You will build two adapted encoders in this exercise:

- **Track A: domain adaptation.** Pick one vertical — biomedical *or* legal *or* scientific. Build a domain-adapted bi-encoder.
- **Track B: language adaptation.** Pick one non-English language family — recommended: French, German, Chinese (`zh`), or Japanese (`ja`). Build a language-adapted bi-encoder.

Both tracks reuse the same recipe skeleton: warm-start from the closest specialised backbone, adapt with GPL (no labels) or supervised fine-tune (labels), evaluate on the matching domain/language MTEB slice.

## Model shelves

### Track A — Domain backbones

Pick one vertical. Corresponding starting encoders:

- **Biomedical / clinical:** warm-start from `pritamdeka/S-PubMedBert-MS-MARCO` (bi-encoder bundle over PubMedBERT). Domain cross-encoder for GPL denoise: `cross-encoder/ms-marco-MiniLM-L-6-v2` (general) or a PubMed-adapted reranker if available. Corpus: 20 000-passage subset of PubMed abstracts (`load_dataset("pubmed_qa", "pqa_labeled")` gives contexts, or use `MedRAG/pubmed` on Hugging Face).
- **Legal:** warm-start from a bi-encoder built on `nlpaueb/legal-bert-base-uncased` (you may need to add mean pooling per chapter 03 if no `sentence-transformers` bundle exists). Corpus: 20 000 documents from `pile-of-law` or `lex_glue/ecthr_a`.
- **Scientific:** warm-start from `allenai/specter2_base`. Corpus: 20 000 papers from `allenai/scirepeval` or `allenai/scitldr` (title + abstract).

### Track B — Language backbones

Pick one language. Corresponding starting encoders:

- **Multilingual default:** `intfloat/multilingual-e5-base` (requires `"query: "` / `"passage: "` prefixes).
- **Alternative multilingual with long context:** `BAAI/bge-m3`.
- **Cross-lingual specialist:** `sentence-transformers/LaBSE` (translation-optimised).
- **Language-specific specialist** (add for comparison): `BAAI/bge-large-zh-v1.5` for Chinese, `intfloat/multilingual-e5-base` prefixed for French/German, `pkshatech/simcse-ja-bert-base-clcmlp` for Japanese.

Target-language corpus: pick a subset of the language's Wikipedia dumps (`wikimedia/wikipedia` on the HF Hub, `20231101.{fr,de,zh,ja}` etc.), or `xnli` train split, or `mMARCO` in your language.

## Problem statement

### Part A — Domain / language corpus preparation

For your chosen vertical (Track A) or language (Track B):

1. Load a 20 000-passage corpus. Report token-length distribution under the *chosen backbone's* tokenizer. Chapter 09 warns about subword shatter for domain terms; chapter 10 warns about script/normalisation for non-English languages.
2. Pick 20 domain-representative terms (Track A) or 20 language-representative surface forms including any diacritics/scripts (Track B) and print their tokenisations under the general baseline (`bge-large-en-v1.5`) vs. the domain/language-specialised backbone. Report shatter counts.
3. Report Unicode normalisation form (NFC / NFKC) and pin it consistently across indexing and querying. Chapter 10.

Save as `corpus_stats.md`.

### Part B — Baselines: general-purpose encoders

Evaluate the following *unadapted* models on the appropriate domain/language MTEB slice:

- Track A: `BAAI/bge-large-en-v1.5` and `intfloat/e5-large-v2` on `MTEB(medical)` / `MTEB(law)` / `CoIR` (whichever matches your vertical).
- Track B: `sentence-transformers/all-mpnet-base-v2` (English-only, meant to fail) and `intfloat/multilingual-e5-base` on `MTEB(fra)` / `MTEB(deu)` / `MTEB(cmn, v1)` / `MTEB(jpn)`.

Restrict to 3–5 tasks per MTEB slice for runtime. Report nDCG@10 for retrieval tasks and Spearman ρ for STS if the slice includes any. Chapter 09 predicts general encoders will lose 5–15 points on domain retrieval; chapter 10 predicts English-only encoders will lose double-digits on non-English retrieval and Tatoeba F1.

Save as `baselines.md`.

### Part C — Specialised backbones as baselines

Now evaluate the *specialised-but-unadapted* backbones from the shelves above on the same slices. Track A: `pritamdeka/S-PubMedBert-MS-MARCO` (or your chosen vertical's specialist). Track B: `LaBSE`, `bge-m3`, and (if available) a monolingual-specialist.

Report the same metric panel. This is chapter 09's step 1 ("swap backbone") and chapter 10's shelf comparison.

Add these rows to `baselines.md` and note where they beat the general encoders (should be everywhere in the target domain/language).

### Part D — Track A: GPL adaptation

Take the strongest specialised backbone from Part C and adapt it to your 20 k corpus with GPL (chapter 09):

1. **Query generation.** Generate 5 synthetic queries per passage with `doc2query/msmarco-t5-base-v1` (or a domain-fine-tuned doc2query variant if you find one). Sanity-check the output — chapter 09 warns about lazy generators.
2. **Hard-negative mining.** Use the specialised backbone from Part C to mine 7 negatives per synthetic query (`mine_hard_negatives`, `range_min=10`, `range_max=100`).
3. **Cross-encoder pseudo-labelling.** Score every `(q, positive)` and `(q, negative)` pair with your denoising cross-encoder. Persist the soft scores.
4. **Train with `MarginMSELoss`.** Match the cross-encoder's margin between positive and negative per query. Batch 64, LR `2e-5`, 3 epochs.

Save as `train_gpl.py` and the checkpoint.

### Part E — Track B: cross-lingual distillation OR supervised fine-tune

For your chosen language, pick *one* of the two adaptation moves from chapter 10:

- **Option 1: cross-lingual distillation** (Reimers & Gurevych, 2020). Teacher: an English `sentence-transformer` you trust (`all-mpnet-base-v2` or your exercise-01 model). Student: the multilingual backbone from Part C. Distillation data: 100 k parallel `(en, target-lang)` sentence pairs from OPUS-100 or Tatoeba. Loss: MSE between teacher(en) and student(target). Chapter 10.
- **Option 2: GPL in the target language.** Same recipe as Track A Part D but with a target-language `doc2query` (`doc2query/msmarco-{lang}-mt5-base-v1` from the Hub, or fine-tune one on `mMARCO`). Denoise with `BAAI/bge-reranker-v2-m3` (multilingual). Train the multilingual bi-encoder on the pseudo-labels.

Save as `train_lang.py` and the checkpoint. State which option you picked and why in `report.md`.

### Part F — Evaluate the adapted models

Re-run the domain/language MTEB slices from Part B for the two adapted models. Report nDCG@10, Recall@100, and any STS or bitext-mining scores in the slice.

Additional cross-lingual eval for Track B: run MTEB's Tatoeba bitext-mining subset for your language pair (chapter 10 calls this the cross-lingual diagnostic). Report F1.

Present as one combined table in `results.md`:

|                            | Retrieval nDCG@10 | Retrieval Recall@100 | STS ρ (if in slice) | Tatoeba F1 (Track B) |
|----------------------------|-------------------|----------------------|---------------------|----------------------|
| General baseline (Part B)  |                   |                      |                     |                      |
| Specialised baseline (Part C) |                |                      |                     |                      |
| **Adapted (Parts D/E)**    |                   |                      |                     |                      |

Chapter 09 predicts 4–15 nDCG@10 lift for Track A; chapter 10 predicts strong Tatoeba F1 (>90) for LaBSE-track or improved retrieval for E5/BGE-track.

### Part G — Failure inspection

Chapter 09 and chapter 10 both list failure modes. Pick two to reproduce or check for:

- **Track A candidates:** the adapted model loses on the general benchmarks (STS-B); the doc2query outputs are all one-word queries; tokenisation shatter is *worse* than baseline.
- **Track B candidates:** cross-lingual works for European languages but fails for Chinese/Arabic; Tatoeba F1 is high but domain-specific cross-lingual quality is bad; missing task prefix cuts retrieval by 5–8 points.

Reproduce two, quantify the damage, and write a paragraph in `report.md`.

### Part H — Write-up

500–700 word `report.md` covering:

- Corpus and tokenisation stats (Part A) — domain-term shatter or script normalisation.
- Baseline vs. specialised-backbone vs. adapted table (Parts B, C, F). Where the biggest lifts came from.
- The adaptation recipe you picked (Part D for A, one of two options for E) and *why*.
- Two reproduced failure modes (Part G) with quantified impact.
- One "what next" — e.g. push to a labelled supervised fine-tune, add a second language, ship a domain reranker (chapter 07 + 09).

## Starter guidance

- **Warm-start from the closest specialist.** Chapter 09's rule of thumb: do not train a general encoder on domain pairs from scratch when a domain-pretrained backbone exists. Chapter 10 says the same for languages.
- **`mine_hard_negatives` accepts a `cross_encoder=` argument.** Use it. GPL is easier to write correctly this way than as three separate scripts.
- **`MarginMSELoss` needs a cross-encoder score column in your dataset.** Precompute and cache — do not compute inside the training loop.
- **Task prefixes matter for multilingual-E5 and BGE-M3.** Same warning as exercise-03. Chapter 10 has the incantations.
- **Log Unicode normalisation form** at both index and query time. Chapter 10's silent-killer warning.
- **Do not evaluate a legal model on STS-B and worry.** Chapter 09: expected to lose on out-of-domain benchmarks. Evaluate on the domain slice.
- **doc2query outputs can be junk.** Read 20 generated queries per passage before running the whole training loop. If they are all one-word or all near-duplicates of the passage, switch generators or add sampling diversity (`top_k=50`, `num_return_sequences=5`).
- **Track B is faster to iterate than Track A** because multilingual encoders are more common than domain ones. If you are time-boxed, do Track B first for a full-run confidence check, then Track A.

## Acceptance criteria

- [ ] `corpus_stats.md` documents corpus size, token-length distribution, tokeniser shatter on 20 domain/language-representative terms, and pinned Unicode normalisation form.
- [ ] `baselines.md` reports general and specialised baselines on the appropriate MTEB slice (3–5 tasks).
- [ ] `train_gpl.py` implements GPL end-to-end for Track A: query generation → hard-negative mining → cross-encoder pseudo-labelling → `MarginMSELoss` training.
- [ ] `train_lang.py` implements cross-lingual distillation *or* language-specific GPL for Track B. The choice and rationale are documented.
- [ ] `results.md` shows the three-way comparison (general vs. specialised vs. adapted) for both tracks, with Tatoeba F1 for Track B.
- [ ] Two failure modes from chapters 09/10 reproduced and quantified in `report.md`.
- [ ] `report.md` (500–700 words) covers stats, adaptation recipes, results, failures, and one next step.

## Stretch goals

- **Supervised fine-tune on top of GPL.** Chapter 09: if you can collect even 1 000 labelled pairs in your domain, warm-start from the GPL-adapted model and fine-tune with `MultipleNegativesRankingLoss`. Report the additional lift.
- **Domain cross-encoder training.** Train a `cross-encoder/ms-marco-MiniLM-L-6-v2` on your GPL data and use it in a two-stage pipeline (exercise-02's Part F). Report end-to-end nDCG@10 and MRR@10.
- **Second language for Track B.** Repeat Part E for a *typologically different* language (if you did French, add Chinese; if you did Chinese, add Arabic). Chapter 10: cross-lingual performance is often lopsided by language family.
- **BGE-M3's three modes.** For Track B, use `bge-m3` in all three of its modes (dense / sparse / multi-vector / hybrid) on the same eval set. Chapter 10.
- **In-house domain eval.** Construct a 50-query hand-labelled retrieval set from your corpus, evaluate all three (general / specialised / adapted) on it, and compare the ranking with the public MTEB slice's ranking. Chapter 09 emphasises this final step.
- **Overfitting mitigation ablation.** For the supervised-fine-tune stretch, ablate three mitigations from chapter 09: weight-averaging with baseline, bottom-N layer freeze, LoRA. Report which recovers the most out-of-distribution quality.

## Deliverables

Ship as a directory with:

```
corpus_stats.md
baselines.py + baselines.md
train_gpl.py        # Track A
train_lang.py       # Track B
eval_domain.py + eval_lang.py
results.md
failure_audit.md
report.md
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
