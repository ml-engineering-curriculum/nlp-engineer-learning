# Multilingual and Cross-Lingual Embeddings

## Motivation

If your product ships in more than one language — or even ingests documents in one language and answers queries in another — a monolingual English encoder becomes an active drag. Chapter 07 of mod-107 covered multilingual *encoders* (XLM-R, mT5) for classification and tagging. This chapter is their embedding-shaped siblings: LaBSE, multilingual-E5, BGE-M3, LASER, `paraphrase-multilingual-*`, and the recipe for training your own.

Cross-lingual retrieval — a query in Swahili retrieving documents in English — is the sharpest test of a multilingual embedding, because success requires the encoder to place semantically equivalent sentences from *different languages* in nearby regions of the embedding space. This is a genuinely hard alignment problem, and models that ship this capability got there through specific training regimes we should be able to name.

## What "multilingual" and "cross-lingual" actually mean

Terminology worth pinning:

- **Multilingual embedding.** A single encoder that produces useful vectors for text in many languages. Each language works well *on its own* — Spanish-to-Spanish retrieval, French-to-French clustering.
- **Cross-lingual embedding.** A multilingual encoder additionally satisfying the property that semantically equivalent sentences in *different* languages land near each other in the same space — Spanish query retrieving French documents.
- **Language-agnostic embedding.** The strongest form: the embedding is (approximately) invariant to the source language. `sim(en_translation_of_x, ja_translation_of_x)` is close to 1.

Every "multilingual" model on Hugging Face falls somewhere on this spectrum. LaBSE is language-agnostic. `paraphrase-multilingual-MiniLM-L12-v2` is multilingual with weak cross-lingual alignment. Multilingual-E5 is multilingual with strong cross-lingual alignment. Read the model card; the training data reveals which category the model actually belongs to.

## The public multilingual shelf

Recognise these:

- **LaBSE** (Feng, Yang, Cer, Arivazhagan & Wang, ["Language-agnostic BERT Sentence Embedding"](https://arxiv.org/abs/2007.01852), *ACL 2022*) — `sentence-transformers/LaBSE`. Trained on 6 B parallel-translation pairs from CommonCrawl mining. 109 languages. Explicitly optimised for cross-lingual similarity. The reference "language-agnostic" bi-encoder.
- **Multilingual-E5** (Wang et al., ["Multilingual E5 Text Embeddings: A Technical Report"](https://arxiv.org/abs/2402.05672), 2024) — `intfloat/multilingual-e5-small`, `-base`, `-large`. Same E5 recipe extended to ~100 languages. Requires `"query: "` / `"passage: "` prefixes. Strong on cross-lingual retrieval.
- **BGE-M3** (Chen, Xiao, Zhang & Muennighoff, ["BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings"](https://arxiv.org/abs/2402.03216), 2024) — `BAAI/bge-m3`. Multilingual + supports three retrieval modes (dense, sparse, ColBERT-style) from one model. Long-context (8k tokens). Currently a strong all-round multilingual default.
- **`paraphrase-multilingual-MiniLM-L12-v2`** and **`paraphrase-multilingual-mpnet-base-v2`** — the original `sentence-transformers` multilingual bundles. Trained via cross-lingual distillation (Reimers & Gurevych, ["Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation"](https://arxiv.org/abs/2004.09813), *EMNLP 2020*). Still competitive on smaller latency budgets.
- **LASER / LASER-2 / LASER-3** (Artetxe & Schwenk, ["Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond"](https://arxiv.org/abs/1812.10464), *TACL 2019*; and NLLB Team, 2022) — Meta's massively multilingual sentence encoder, trained on translation pairs. LASER-3 covers 200+ languages via NLLB alignment. Best when your language mix skews low-resource.
- **`jina-embeddings-v3`** — recent multilingual encoder with an emphasis on long-context and task-specific LoRA adapters. Worth benchmarking.
- **Hosted:** OpenAI `text-embedding-3-*` (multilingual by design), Cohere `embed-multilingual-v3`, Voyage `voyage-multilingual-2`, Google `text-multilingual-embedding-002`. Trade the ability to inspect and adapt for a maintained multilingual API.

Rule: for a *new* multilingual product with no strong preference, benchmark `multilingual-e5-large` and `bge-m3` on the target-language MTEB slice and pick the better one. For a low-resource-heavy mix, add LaBSE and LASER-3 to the comparison.

## How multilingual embeddings are trained

Three training paradigms cover almost everything you see:

- **Translation pair contrastive training (LaBSE-style).** Mine `(sentence, translation)` pairs from CommonCrawl at billion-pair scale. Train with MNR — pull translation pairs together, push everything else apart. This is what makes an embedding *language-agnostic*: the model is explicitly told that a Spanish sentence and its English translation are the same point.
- **Cross-lingual distillation (Reimers & Gurevych 2020).** Take a strong English `sentence-transformer` teacher; take a multilingual student (`xlm-roberta-base`); train the student to reproduce the teacher's embeddings on parallel data. Cheap, effective, and the recipe behind the `paraphrase-multilingual-*` bundles.
- **Massively multilingual pretraining + supervised fine-tune on multilingual data** (multilingual-E5, BGE-M3). The E5-style three-stage recipe (chapter 06) applied with multilingual data at every stage. The most expensive, currently the strongest.

Each recipe has a signature failure mode:

- **Translation-only training** produces vectors that are language-agnostic but sometimes miss fine-grained same-language semantics — LaBSE occasionally underperforms specialised English encoders on English-only tasks.
- **Cross-lingual distillation** inherits the teacher's blind spots and adds multilingual generalisation on top. Only as good as the teacher.
- **Full three-stage** works well across the board but needs an enormous multilingual pair corpus at pretraining time.

## Bitext mining: the diagnostic cross-lingual task

MTEB's cross-lingual diagnostic is **bitext mining** on Tatoeba and BUCC — given two large monolingual collections in different languages, find the pairs that are translations of each other. The metric is F1 over the top-1 candidate.

Why it is the right diagnostic: it directly measures "can this encoder tell that a Spanish sentence and its English translation are the same thing" over a large pool, at scale, with no supervision at query time. A model that scores >95 % Tatoeba F1 has strong cross-lingual alignment; one that scores <80 % has weak alignment (fine for monolingual tasks, bad for cross-lingual retrieval).

LaBSE typically scores 95%+ on most Tatoeba pairs. `paraphrase-multilingual-mpnet-base-v2` scores in the low 80s–90s. multilingual-E5-large scores in the 90s. BGE-M3 is in the 90s. If cross-lingual retrieval is your use case, this is the number to look at.

## Cross-lingual retrieval evaluation

Public benchmarks worth naming:

- **mMARCO** (Bonifacio et al., 2022) — machine-translated MS MARCO in 14 languages. The reference cross-lingual retrieval benchmark.
- **MIRACL** (Zhang et al., ["MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages"](https://arxiv.org/abs/2210.09984), 2022) — human-annotated retrieval judgements across 18 languages. Higher quality than mMARCO but smaller.
- **XOR-TyDi** (Asai et al., 2021) — cross-lingual open-domain QA over Wikipedia in 7 typologically diverse languages.
- **MLDR** (Chen et al., 2024) — long-document multilingual retrieval; the benchmark BGE-M3 was designed against.

Run these on your candidate encoders before committing. The pattern that will emerge: some models are strong on European-language cross-lingual and weak on Chinese / Arabic / Indic; others (LaBSE, LASER-3, BGE-M3) are more balanced but sacrifice European-language peak quality.

## Script and tokenisation warnings

Same warnings as mod-107 chapter 09 — they apply here too:

- **Chinese, Japanese, Thai:** no whitespace. SentencePiece handles this; whitespace-first tokenisers destroy the input.
- **Arabic:** RTL rendering, ligatures, diacritic optionality. Normalise consistently (NFC or NFKC) at both index and query time.
- **Indic scripts:** virama-conjunct sequences require careful normalisation; different scripts for the same language (Punjabi in Gurmukhi vs. Shahmukhi) are treated as different languages by many multilingual tokenisers.
- **Turkish, Finnish, Hungarian:** high morphological complexity leads to token-shatter. Domain terms often need custom preprocessing.

The trap: your multilingual encoder passes MTEB and Tatoeba but silently fails on your Arabic corpus because the normalisation is inconsistent between index-time (NFC) and query-time (NFKC). Log Unicode-normalisation form at both points.

## Language coverage vs. quality trade-off

Every multilingual model makes an explicit trade: broader coverage costs per-language quality. The rough shape:

| Model                            | Languages | Per-language quality (relative) | Cross-lingual quality | Where it fits |
|----------------------------------|-----------|--------------------------------|-----------------------|---------------|
| `all-mpnet-base-v2` (English)    | 1         | Best on English                | N/A                   | English-only  |
| `multilingual-e5-large`          | ~100      | Strong on 20+                  | Strong                | Everyday default |
| `BAAI/bge-m3`                     | ~100      | Strong on 20+                  | Strong                | Everyday default (long-context) |
| `sentence-transformers/LaBSE`    | 109       | Middle-of-pack per-language    | Best (near-perfect Tatoeba) | Cross-lingual heavy |
| LASER-3 (via `sentence-transformers` fork) | 200+ | Weaker per-language           | Strong (translation-optimised) | Low-resource languages |
| Cohere `embed-multilingual-v3`   | 100+      | Strong                         | Strong                | Hosted preference |

A concrete decision procedure:

1. If your language mix is 90 % English, use an English encoder and only enable multilingual on the ~10 % non-English traffic.
2. If your mix spans European + a handful of Asian languages, `multilingual-e5-large` or `bge-m3` is the right default.
3. If your mix is heavy on low-resource languages, add LaBSE or LASER-3 to the shortlist.
4. If you need production-quality on a specific language pair (e.g. Japanese-Chinese cross-lingual), evaluate specifically for that pair — averages hide it.

## Adapting to a specific language family

Two adaptation moves that work:

- **Fine-tune on target-language pairs.** Same recipe as chapter 09 for domains: warm-start from a multilingual base, fine-tune with `MultipleNegativesRankingLoss` on labelled or GPL-generated pairs in the target language.
- **Cross-lingual distillation from a strong same-language teacher.** If you have a strong Chinese encoder (e.g. `BAAI/bge-large-zh-v1.5`) and want to add it to your multilingual stack, distil the multilingual student to match the specialist teacher on Chinese examples. Same recipe as Reimers & Gurevych (2020).

The GPL recipe (chapter 09) works for languages too — just use a target-language query generator (fine-tuned mT5 or similar) and a multilingual cross-encoder for denoising.

## Common failure modes

- **Cross-lingual retrieval works for European languages, fails for Chinese.** You probably picked a model that was trained mostly on European parallel data. Switch to BGE-M3 or LaBSE for balanced coverage.
- **`multilingual-e5-large` retrieval quality is bad on your corpus.** You forgot the `"query: "` / `"passage: "` prefixes.
- **The model produces different vectors for the same text on different runs.** Non-deterministic subword tokenisation from a version mismatch between the model card's expected tokeniser and the loaded one. Pin `transformers` and `sentence-transformers` versions.
- **Retrieval on a specific language degrades after adding new languages to the mix.** Multilingual interference — the model's shared parameters redistribute capacity. Usually fine, sometimes not. If a specific language matters, benchmark it explicitly after any change.
- **Tatoeba F1 is 95 % but production cross-lingual quality is bad.** Tatoeba is short, clean, single sentences. Your production text is long, messy, and multi-sentence. Build a domain-specific cross-lingual eval set (100 queries + 5000 docs is enough).

## Chapter summary

- Multilingual embedding = one encoder producing useful vectors for many languages. Cross-lingual = additionally aligning translation-equivalent sentences.
- Recognisable public shelf: LaBSE (language-agnostic), multilingual-E5 (leaderboard default), BGE-M3 (multi-function, long-context), `paraphrase-multilingual-*` (older cheap default), LASER-3 (low-resource coverage), hosted APIs.
- Three training paradigms: translation-pair contrastive (LaBSE), cross-lingual distillation from an English teacher (`paraphrase-multilingual-*`), full multilingual three-stage (mE5, BGE-M3).
- Bitext mining (Tatoeba F1) is the reference cross-lingual diagnostic. mMARCO and MIRACL are the reference retrieval benchmarks.
- Script and normalisation issues (Chinese/Arabic/Indic/RTL/CJK) show up here as they did in mod-107. Log Unicode-normalisation form at both index and query time.
- Language coverage vs. per-language quality is a trade-off. Match the model to the language mix you care about.
- Adapt to a specific language with GPL (no labels) or supervised fine-tuning on labelled pairs. Same recipe as domain adaptation, applied on the language axis.
- Do not trust global averages. If a specific language pair matters, benchmark it directly.
