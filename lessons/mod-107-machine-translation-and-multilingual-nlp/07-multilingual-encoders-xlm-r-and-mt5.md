# Multilingual Encoders: XLM-R and mT5

## Motivation

Chapters 02–06 covered translation — the *generation* side of multilingual NLP. This chapter switches perspective. Instead of translating between languages, we build a single encoder that *represents* text in many languages inside a shared vector space, then attach classification, tagging, or extraction heads. If the shared space is good enough, fine-tuning on English labels and evaluating on Swahili (or Hindi, or Basque) *works* — the multilingual encoder has already learned that "president" in English lives near "presidente", "président", "رئيس", and "राष्ट्रपति" in its embedding space. That is the phenomenon that made **cross-lingual zero-shot transfer** possible and turned XLM-R and mT5 into the default multilingual encoders of the era.

This chapter is about the models themselves — architecture, tokenisation, cost, and per-task defaults. Chapter 08 covers the transfer patterns you build on top.

## The two lineages

Two families dominate the current stack:

- **Encoder-only masked-LM (XLM-R, XLM-V, mDeBERTa).** Descendants of BERT, trained with masked language modelling on multilingual Common Crawl. Best for classification, tagging, extractive QA, and retrieval. Cannot generate text.
- **Encoder-decoder text-to-text (mT5, ByT5, mBART).** Descendants of T5 (span-corruption on 101 languages of mC4) or BART (denoising, mBART). Encoder-only tasks work; generation tasks (summarisation, structured extraction, multilingual QA generation) also work.

They serve different jobs. Attach a classifier head on XLM-R for multilingual sentiment; attach nothing on mT5 and prompt it to produce a JSON extraction. Both cover 100+ languages; both fine-tune with the same recipes as their monolingual predecessors.

## XLM-R: the encoder-only default

**XLM-RoBERTa** (Conneau et al., ["Unsupervised Cross-lingual Representation Learning at Scale"](https://arxiv.org/abs/1911.02116), *ACL 2020*) is a RoBERTa-style encoder pretrained with MLM on 2.5 TB of Common Crawl covering 100 languages (CC-100). It is the multilingual encoder you should reach for by default.

Sizes on the Hub:

| Checkpoint                        | Parameters | Hidden dim | Layers | Vocab   | Best for                                                    |
|-----------------------------------|-----------:|-----------:|-------:|--------:|-------------------------------------------------------------|
| `FacebookAI/xlm-roberta-base`     | 278 M      | 768        | 12     | 250 002 | Default. Latency-sensitive per-request classification.       |
| `FacebookAI/xlm-roberta-large`    | 559 M      | 1024       | 24     | 250 002 | Higher-quality classification, extraction, retrieval.        |
| `FacebookAI/xlm-roberta-xl`       | 3.5 B      | 2048       | 36     | 250 002 | Research; occasionally batch inference; rarely served live.  |
| `FacebookAI/xlm-roberta-xxl`      | 10.7 B     | 4096       | 48     | 250 002 | Research; heavy inference; usually distilled before serving. |

Two design decisions worth remembering:

- **No language embeddings.** Unlike its predecessor XLM (which had a per-language embedding lookup), XLM-R relies purely on subword identity to distinguish languages. Cross-lingual transfer is emergent.
- **Language-aware sampling during pretraining.** Sentences from lower-resource languages were upweighted with a temperature-sampling scheme (same $\tau=0.3$ variant later used in NLLB), so the model does not collapse to English.

Adjacent XLM-R descendants worth knowing:

- **XLM-V** (Liang et al., ["XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models"](https://arxiv.org/abs/2301.10472), 2023). Same architecture; 1 M-token vocabulary designed to reduce over-splitting for low-resource scripts. Often the better choice for Indic, Ethiopic, or Amharic classification.
- **mDeBERTa-v3** (`microsoft/mdeberta-v3-base`, He et al., ["DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training"](https://arxiv.org/abs/2111.09543), *ICLR 2023*). Disentangled-attention encoder; consistently ~1–2 points above XLM-R-base on XTREME / XNLI. Reach for it when you want XLM-R-base-scale quality with base-scale cost.
- **LaBSE** (Feng et al., ["Language-agnostic BERT Sentence Embedding"](https://arxiv.org/abs/2007.01852), *ACL 2022*). Sentence-level dual-encoder trained on parallel data — the multilingual sentence-encoder analogue of Sentence-BERT. Not a general-purpose encoder; use for cross-lingual retrieval and semantic similarity.

## mT5: the encoder-decoder default

**mT5** (Xue et al., ["mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer"](https://arxiv.org/abs/2010.11934), *NAACL 2021*) is T5's multilingual descendant — same span-corruption objective, pretrained on **mC4**, a 101-language subset of Common Crawl. Sizes:

| Checkpoint          | Parameters | Best for                                                          |
|---------------------|-----------:|-------------------------------------------------------------------|
| `google/mt5-small`  | 300 M      | Latency-sensitive multilingual generation.                        |
| `google/mt5-base`   | 580 M      | The default.                                                       |
| `google/mt5-large`  | 1.2 B      | Better generation quality; heavier.                                |
| `google/mt5-xl`     | 3.7 B      | Research; sometimes served for QA generation.                      |
| `google/mt5-xxl`    | 13 B       | Research; usually distilled or LoRA-tuned before serving.          |

**ByT5** (`google/byt5-*`, Xue et al., ["ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models"](https://arxiv.org/abs/2105.13626), *TACL 2022*) is the token-free sibling — operates on raw UTF-8 bytes, no SentencePiece. Slower per step but more robust to noisy or under-covered scripts; the go-to for character-level tasks (transliteration, script conversion, robust text-normalisation in low-resource languages).

Practical defaults:

- **Multilingual classification, NER, extractive QA.** XLM-R-large or mDeBERTa-v3-base.
- **Multilingual generative QA, summarisation, structured extraction.** mT5-base or mT5-large.
- **Multilingual retrieval or sentence similarity.** LaBSE, or a multilingual variant of Sentence-Transformers.
- **Character-level / heavy script robustness.** ByT5.

## Tokeniser footprint and cost

The vocabulary is where multilingual encoders pay their coverage tax:

| Model           | Vocab size | Approx. tokens / word (Latin) | Approx. tokens / word (Devanagari) | Notes                                     |
|-----------------|-----------:|------------------------------:|-----------------------------------:|-------------------------------------------|
| BERT-base (en)  |     30 522 | 1.3                            | shatter (byte fallback)              | English-only baseline for context.        |
| XLM-R           |    250 002 | 1.3                            | 2.2                                  | Balanced across 100 languages.            |
| XLM-V           |  1 000 000 | 1.2                            | 1.6                                  | Larger vocab, less shattering.            |
| mT5             |    250 112 | 1.4                            | 2.0                                  | Similar footprint to XLM-R.               |
| ByT5            |        384 | N/A (per byte)                 | N/A (per byte)                       | ~4–8× more compute per sentence.          |

Two implications:

- **Batching gets uneven across languages.** A 128-token budget covers a longer English sentence than a longer Hindi or Amharic sentence. Sort your batches by tokenised length within a language to keep padding overhead down.
- **Vocabulary embeddings dominate small-model parameter counts.** For `xlm-roberta-base`, the 250 k × 768-dim embedding is ~192 M of the 278 M total parameters. Fine-tuning the embedding is the largest gradient in the model — freeze the embedding when working with tight VRAM.

## Fine-tuning XLM-R for classification

The recipe is a straight port of monolingual BERT fine-tuning (see mod-103 for the general recipe):

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
)

MODEL = "FacebookAI/xlm-roberta-large"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=NUM_LABELS)

def preprocess(batch):
    return tok(batch["text"], truncation=True, max_length=256)

args = TrainingArguments(
    output_dir="xlmr-classifier",
    learning_rate=2e-5,                # Standard XLM-R LR
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_ratio=0.06,
    bf16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    save_total_limit=2,
)
```

Two XLM-R-specific things:

- **`learning_rate=2e-5`** is the standard. XLM-R-large in particular is unstable at higher LRs — do not skip the warmup.
- **Fine-tuning XLM-R-large with fp32 head is often more stable than mixed-precision head.** If you see NaN losses, try `fp16_full_eval=False` or force the classifier head to fp32.

For NER / sequence tagging, replace `AutoModelForSequenceClassification` with `AutoModelForTokenClassification`; for extractive QA, `AutoModelForQuestionAnswering`. The rest of the recipe is unchanged.

## Fine-tuning mT5 for text-to-text tasks

mT5 is fine-tuned exactly like T5 — the task is expressed as a text prefix and the output as a text answer:

```python
def preprocess(batch):
    inputs  = [f"classify sentiment: {t}" for t in batch["text"]]
    targets = [f"{lab}" for lab in batch["label_str"]]
    x = tok(inputs, max_length=256, truncation=True)
    with tok.as_target_tokenizer():
        y = tok(targets, max_length=8, truncation=True)
    x["labels"] = y["input_ids"]
    return x
```

Two idioms that differ from mT5's monolingual T5 sibling:

- **No task prefix is *required*.** mT5's pretraining did not include the T5 task-prefix soup. You can still use a prefix (it usually helps) but do not treat it as mandatory.
- **Learning rate is lower than T5.** `1e-4` with AdamW is a safer default than T5's `1e-3` Adafactor recipe — mT5 has been observed to diverge with the T5 LR.

For summarisation, extraction, and QA-generation, mT5 is the multilingual answer to BART / T5.

## The cross-lingual transfer surface

This is the phenomenon that motivates the whole chapter. Fine-tune XLM-R on **English** NLI data (SNLI or MNLI); evaluate on **Chinese**, **Swahili**, **Urdu**, or **Basque** (XNLI covers 15 languages). Numbers are down from monolingual English NLI but stay competitive with — sometimes above — from-scratch monolingual training in each language. Conneau et al. (2020) show XLM-R-large at ~80.9 % English NLI accuracy transferring to ~74–79 % on high-resource NLI languages and ~65–70 % on the low-resource tail. That is a shockingly good result to get without a single non-English training example.

Chapter 08 covers the transfer patterns (translate-train, translate-test, true zero-shot) and their trade-offs.

## Choosing between the two lineages

| Task shape                                                             | Reach for                                              |
|-----------------------------------------------------------------------|--------------------------------------------------------|
| Multilingual sentence / document classification                         | XLM-R-large or mDeBERTa-v3-base (classifier head).      |
| Multilingual NER, POS, chunk tagging                                    | XLM-R-large (token classification head).                |
| Multilingual extractive QA (span selection)                             | XLM-R-large (QA head).                                  |
| Multilingual sentence similarity, retrieval                             | LaBSE or paraphrase-multilingual-mpnet-v2.              |
| Multilingual generative QA (free-form answer)                           | mT5-base or mT5-large.                                  |
| Multilingual summarisation, paraphrase, structured extraction           | mT5-large or mBART-50.                                  |
| Character-level / underserved script                                    | ByT5.                                                    |
| Latency-sensitive multilingual classification (< 30 ms per request)     | Distilled XLM-R (`Geotrend/distilbert-base-25lang-cased` or similar). |

## Serving cost realities

The multilingual encoders are 2–5× the parameter count of their monolingual cousins largely because of the vocabulary. Serving implications:

- **Latency.** XLM-R-large is ~2× the wall-clock of BERT-large in most inference stacks.
- **Distillation is standard.** Distilled multilingual encoders (`nreimers/mMiniLMv2-*`, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) exist and are the production defaults for classification and retrieval. Do the fine-tune on the full XLM-R-large; distil the fine-tuned model down for serving.
- **ONNX / TensorRT export works but tokenisation dominates for short sequences.** Batch tokenisation ahead of the forward pass; measure end-to-end, not model-only, latency.

## Common failure modes and their fixes

- **XLM-R-large training diverges (loss NaN or plateau).** LR too high or fp16 numerical issue. Try `lr=1e-5`, `warmup_ratio=0.1`, and `bf16=True` if available; or force the classifier head to fp32.
- **Poor per-language performance for a language in the coverage list.** The pretraining was heavy on the high-resource tail. Check per-language F1 during eval; consider XLM-V or a language-specific fine-tune if a single language is critical.
- **Tokenisation shatters low-resource script.** XLM-R base tokeniser tokenises Amharic word into 5+ pieces. Switch to XLM-V or ByT5.
- **mT5 generates in the wrong language.** Prepend the target language name or code in the prompt (`"translate to Swahili: ..."`). Unlike NLLB, mT5 has no `forced_bos_token_id` mechanism — the model relies on the prompt.
- **Cross-lingual transfer looks worse than expected.** The label distribution in your target language may differ from the source. Check per-language label priors, and consider translate-train (chapter 08) as a fallback.

## Chapter summary

- Two multilingual encoder lineages dominate: XLM-R / mDeBERTa (encoder-only, best for classification/tagging/QA/retrieval) and mT5 / mBART / ByT5 (encoder-decoder, best for generation).
- XLM-R-large + XNLI-style fine-tune is the workhorse for multilingual classification; mDeBERTa-v3-base often matches it at lower cost.
- mT5 is fine-tuned like T5 but at a lower learning rate (`1e-4` AdamW is the safe default).
- Vocabulary is the biggest cost driver — 250 k tokens is the norm. Batching by tokenised length and freezing the embedding are useful VRAM levers.
- ByT5 (token-free) is the specialist for underserved scripts and character-level tasks.
- Serve distilled fine-tunes; keep the full-scale checkpoints for training and offline evaluation.
- Chapter 08 builds cross-lingual transfer patterns on top of these encoders.
