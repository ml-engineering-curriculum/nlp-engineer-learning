# The Machine Translation and Multilingual NLP Landscape

## Motivation

Machine translation is the oldest still-running problem in NLP — Weaver's 1949 memorandum, ALPAC's 1966 report, IBM Model 1 through 5, Moses and phrase-based SMT, the seq2seq revolution of 2014, transformer NMT in 2017, and the massively multilingual models — M2M-100, mBART-50, NLLB-200 — that let a single checkpoint translate between hundreds of languages. Every era's approach still ships somewhere in production; the difference now is that a competent NLP engineer is expected to reach for a transformer NMT model as the default and to know when the "small, boring, in-domain" Marian model is still the right call.

Multilingual NLP is the same problem viewed from the other side. Instead of translating *between* languages, we ask a single encoder to *represent* text from many languages in a shared space, then attach classifiers, taggers, or extractors on top. That shared space is what makes cross-lingual zero-shot transfer possible: fine-tune on English NER, evaluate on Swahili, and — if the model is XLM-R or mT5 and the fine-tuning is careful — do better than a from-scratch Swahili classifier ever could.

This module owns both. We start from the model families (Marian, mBART, NLLB, M2M-100) and their tokenisers, move through fine-tuning recipes for high- and low-resource pairs, add the terminology and domain-adaptation levers you need in production, then swap perspective and cover XLM-R / mT5 for cross-lingual transfer, script and locale handling, and the evaluation stack — BLEU, chrF, COMET, BLEURT, and the human protocols (DA, SQM, MQM) that keep the automatic metrics honest.

## The two problems, formally

- **Machine translation (MT).** Given a source sentence $x$ in language $L_s$, produce a target sentence $y$ in language $L_t$ such that $y$ preserves the meaning of $x$ and is fluent in $L_t$. Modern systems model $p_\theta(y \mid x, L_s, L_t)$ with an encoder-decoder transformer.
- **Multilingual NLP.** Given a text $x$ in language $L$ and a task $T$ (classification, tagging, extraction, retrieval), produce a task-specific output. Modern systems use a shared multilingual encoder $\phi_\theta$ that maps text from any supported language into a common representation and attach $T$-specific heads.

They share almost every engineering concern — tokenisation, script normalisation, subword vocabulary, script coverage, evaluation across locales — but pin different training objectives. MT is a generation task with hard, sentence-aligned parallel supervision. Multilingual NLP is usually a discriminative task with per-language monolingual supervision and (often) *no* cross-lingual labels at all.

## A taxonomy for MT systems

Pin the following three axes before you commit to a model family. Getting them wrong is the leading source of "we picked the wrong translator" incidents.

| Axis                     | Options                                                                                         | What it decides                                                                                                                     |
|--------------------------|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| **Language coverage**    | Bilingual (one pair) · Multilingual many-to-one · Multilingual one-to-many · Multilingual many-to-many | Whether you can share parameters, whether zero-shot pivoting is available, and whether the model can serve unseen language codes.   |
| **Resource tier**        | High-resource (>10 M parallel) · Mid (~1 M) · Low (~100 k) · Extremely low (<10 k)                | Whether you fine-tune from scratch, from a bilingual base, or from a massively multilingual checkpoint — and whether back-translation is required. |
| **Domain**               | General web · News · Legal / medical / financial · Software / support · Speech transcripts       | Whether you need domain adaptation, terminology injection, and domain-specific evaluation sets on top of the general benchmark.     |

A concrete deployment usually pins each axis. "Bilingual, high-resource, general web" is the WMT English-German setup — an open Marian or NLLB fine-tune is the baseline. "Multilingual many-to-many, low-resource, legal" is a customer-support translator for a global SaaS product — start from NLLB-200 and expect to bring terminology and domain-adapted checkpoints per vertical.

## The resource tiers you will see in the wild

The single biggest predictor of "which recipe works" is the amount of parallel data available for the target pair. The tiers below are approximate but predictive:

- **High-resource (> 10 M parallel sentences).** English ↔ German / French / Spanish / Chinese / Japanese. Fine-tune Marian or NLLB with straightforward cross-entropy, expect to beat public baselines with in-domain data alone.
- **Mid-resource (~ 1 M).** English ↔ Polish / Turkish / Vietnamese / Indonesian. Fine-tune from a multilingual base (mBART, NLLB); back-translation buys 2–5 BLEU.
- **Low-resource (~ 10–100 k).** Many African, Southeast Asian, and Indic pairs; Basque; Welsh. Start from NLLB-200 or M2M-100; back-translation is essential; expect noisy references and to rely on chrF/COMET more than BLEU.
- **Extremely low-resource (< 10 k parallel, monolingual-only tail).** ~1 500 of the world's ~7 000 languages have any structured data at all. Approaches: massively multilingual pretraining (NLLB-200 covers 200 languages, of which ~150 are genuinely low-resource), self-supervised pretraining on monolingual data (XLM-R, mBART), unsupervised NMT, transfer from related high-resource languages. Evaluation itself is a research problem — FLORES-200 was built to give this tier a common yardstick.

The **FLORES-200** benchmark (NLLB Team et al., ["No Language Left Behind"](https://arxiv.org/abs/2207.04672), 2022) is the reference test set for the low- and extremely-low tiers: professionally translated, English-pivoted parallel sentences across 200 languages, with `devtest` splits public. You should recognise it by name.

## Multilingual model families you should know

Four lineages dominate modern multilingual NLP. This module lives inside them; the rest are historical or specialised.

- **Marian NMT** ([`Helsinki-NLP/opus-mt-*`](https://huggingface.co/Helsinki-NLP)). Small, fast, bilingual (usually) encoder-decoder models trained on OPUS. ~74 M parameters typical. Best when you need latency-sensitive translation for a specific pair and have or can pull in-domain data. Also the reference "small MT" baseline throughout the chapter.
- **mBART / mBART-50** (Liu et al., 2020; Tang et al., 2020). Multilingual denoising pretraining plus optional many-to-many fine-tuning on 50 languages. ~610 M parameters. Good general-purpose starting point when you want an encoder-decoder for either translation *or* multilingual generation (summarisation, paraphrase).
- **M2M-100 / NLLB-200** (Fan et al., 2021; NLLB Team, 2022). Massively multilingual many-to-many MT. M2M-100 covers 100 languages; NLLB-200 covers 200. NLLB is the modern default for any language pair where you are not sure a high-quality bilingual model exists. Available at ~600 M (`nllb-200-distilled-600M`), 1.3 B, 3.3 B, and (research-only) 54.5 B parameters.
- **XLM-R / mT5** (Conneau et al., 2020; Xue et al., 2021). Multilingual encoders (XLM-R, XLM-V, mDeBERTa) and encoder-decoders (mT5) pretrained on 100+ languages of Common Crawl. Not translation models; instead the backbone for multilingual classification, NER, extractive QA, and (mT5) multilingual generation. Cross-lingual zero-shot transfer with these is the topic of chapters 07–08.

Adjacent tools you will meet:

- **SentencePiece** (Kudo & Richardson, 2018). The subword tokeniser almost all multilingual models use — unified BPE / Unigram with language-agnostic byte-fallback. Chapter 09 is where it matters most.
- **fairseq / Marian / OpenNMT** as training frameworks — you will occasionally see them in research papers but Hugging Face `transformers` covers everything this module needs.

## What "controlled" means in an MT context

Just like summarisation (mod-106), translation is a generation task where the raw model output is almost never exactly what production ships. The levers you will pull:

1. **Terminology constraints.** Force the translation of specific source terms — brand names, product SKUs, legal jargon — to specific target strings. Handled by placeholder tokens, forced-decoding constraints, and disjunctive-constraint beam search. Chapter 06.
2. **Format constraints.** Preserve markup (HTML/XML tags, Markdown), placeholders (`{user_name}`, `%s`), and structural inline elements. Handled by tag-aware tokenisation, placeholder round-tripping, and constrained decoding.
3. **Register / style control.** Force formal vs. informal pronouns (German du/Sie, French tu/vous, Japanese です/だ), gender-marked morphology, or dialect selection. Handled by control codes in the input template or by fine-tuning per register.
4. **Length control.** Enforce that the translation fits inside a UI budget (subtitles, button labels, push notifications). Handled by `length_penalty`, hard length constraints, or paraphrase / compression models on top.
5. **Faithfulness / adequacy.** Prevent hallucination and dropped content. Handled by decoding-time re-ranking (COMET or NLI), MBR decoding, quality-estimation gates, and abstention. Chapters 10–11.

## Where multilingual NLP sits

Everything the other modules teach for English still applies per-language, but three problems dominate the multilingual side:

- **Tokenisation dominates cost.** A multilingual SentencePiece with 250 k tokens (mT5) is 5× the size of an English BERT vocabulary and roughly 1.5–3× the tokens per sentence for many non-Latin scripts. Budget accordingly.
- **Script and normalisation eat weeks.** Arabic ligatures, Han-Kanji-Hanzi overlap, Indic virama and matra sequences, Thai wordless script, Vietnamese diacritics: covered in chapter 09.
- **Evaluation across languages is a research problem.** A model at 30 BLEU on English → German is not "twice as good" as one at 15 BLEU on English → Swahili — BLEU is not comparable across languages, chrF is closer but still biased, and human evaluation is scarce for low-resource pairs.

## What each chapter owns

- **Chapters 02–06: NMT.** Model families, fine-tuning for high- and low-resource pairs, domain adaptation, terminology and glossary injection.
- **Chapters 07–09: multilingual representations and script.** XLM-R and mT5, cross-lingual zero-shot transfer, script and BCP-47 locale handling.
- **Chapters 10–12: evaluation.** Automatic metrics, human protocols, and multilingual benchmarks — FLORES-200, WMT, XTREME, XNLI.

## The two questions to keep in your head

Two questions run through the whole module. Answer both up front and most design decisions fall out.

1. **How much parallel data do I have, and in what domain?** This decides whether you fine-tune from scratch (high), from a multilingual base (mid), or lean on NLLB-200 and back-translation (low). It also decides which evaluation set is meaningful.
2. **Am I *translating* or *transferring*?** Translating (MT) needs parallel data and an encoder-decoder. Transferring (multilingual NLP) needs monolingual supervision in a pivot language plus a multilingual encoder. They share tokenisation and evaluation concerns but almost nothing else in the training loop.

## Chapter summary

- Machine translation and multilingual NLP share a stack (tokenisation, scripts, evaluation) but differ in objective: MT generates target-language text; multilingual NLP builds task-specific heads on a shared encoder.
- Pin three axes up front: language coverage (bilingual / multilingual), resource tier (high / mid / low / extremely low), and domain (general / vertical). Each axis narrows your model choice sharply.
- Four model lineages own this module: Marian (small bilingual NMT), mBART / mBART-50 (multilingual denoising encoder-decoder), M2M-100 / NLLB-200 (massively multilingual MT), XLM-R / mT5 (multilingual representations for downstream tasks).
- FLORES-200 is the reference low-resource MT benchmark; you should recognise it by name.
- Two anchor questions: how much parallel data (decides recipe and metric), and translate-or-transfer (decides architecture).
