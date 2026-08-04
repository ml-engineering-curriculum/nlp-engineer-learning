# Multilingual Classification with XLM-R and Language-Aware Sampling

## Motivation

The moment a product exists in more than one language, single-language classifiers fail silently. English-only sentiment models tag French traffic with the majority class. Per-language classifiers double the training and serving burden and cap out on low-resource languages where you have no labelled data. The multilingual encoder — trained on many languages jointly and fine-tuned to a task on whatever labels you have — is the modern default.

This chapter covers the model most teams reach for (XLM-R), the training decisions that actually move the number (language-aware sampling, effective-batch composition), and the evaluation discipline that keeps you from being fooled by a monolingual model averaging across languages you can't read.

## XLM-R at a glance

Conneau, Khandelwal, Goyal, Chaudhary, Wenzek, Guzmán, Grave, Ott, Zettlemoyer & Stoyanov, ["Unsupervised Cross-lingual Representation Learning at Scale"](https://arxiv.org/abs/1911.02116), *ACL 2020*, introduced XLM-RoBERTa. Key facts, from the paper and the model cards:

- **Architecture**: RoBERTa (encoder-only, no next-sentence prediction).
- **Sizes**: `xlm-roberta-base` (~270 M parameters), `xlm-roberta-large` (~550 M parameters). Also `xlm-roberta-xl` and `xxl` for GPU-rich settings.
- **Tokenizer**: SentencePiece Unigram model with a 250 000-token vocabulary shared across all languages (module 101 covers SentencePiece).
- **Training data**: CC-100, a 2.5 TB filtered Common Crawl corpus spanning 100 languages (Wenzek et al., ["CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data"](https://arxiv.org/abs/1911.00359), *LREC 2020*).
- **Objective**: masked language modelling.

Two follow-on models often preferred over vanilla XLM-R in 2026 for classification:

- **mDeBERTa-v3** (`microsoft/mdeberta-v3-base`) — He, Gao & Chen, *ICLR 2023*. Applies DeBERTaV3's disentangled attention and ELECTRA-style pretraining to a multilingual setup on CC-100. Typically outperforms XLM-R base on GLUE-like tasks; comparable on lower-resource languages.
- **XLM-V** — Liang et al., ["XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models"](https://arxiv.org/abs/2301.10472), *NAACL 2023*. Larger, better-allocated multilingual vocabulary; helps on languages XLM-R under-tokenises.

Read the model card for language coverage, licence, and known biases before deploying.

## The zero-shot cross-lingual pattern

The default use case for XLM-R is train on the highest-resource language you have (usually English), evaluate on all target languages, and ship. This works because pretraining aligned representations across languages during masked-LM training; classification fine-tuning largely inherits that alignment.

A minimal script:

```python
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorWithPadding,
)

MODEL = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# XNLI: NLI dataset translated across 15 languages
xnli = load_dataset("xnli", "en")   # train on English
xnli_test_fr = load_dataset("xnli", "fr", split="test")
xnli_test_sw = load_dataset("xnli", "sw", split="test")

def tokenise(batch):
    return tokenizer(batch["premise"], batch["hypothesis"],
                     truncation=True, max_length=128)

xnli = xnli.map(tokenise, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)

args = TrainingArguments(
    output_dir="out", learning_rate=1e-5,
    per_device_train_batch_size=32, num_train_epochs=3,
    warmup_ratio=0.1, weight_decay=0.01,
    fp16=True, eval_strategy="epoch", seed=42, report_to="none",
)
trainer = Trainer(
    model=model, args=args,
    train_dataset=xnli["train"], eval_dataset=xnli["validation"],
    tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
)
trainer.train()

# Zero-shot evaluate on French and Swahili
for name, ds in [("fr", xnli_test_fr), ("sw", xnli_test_sw)]:
    tokenised = ds.map(tokenise, batched=True)
    print(name, trainer.evaluate(tokenised))
```

Conneau et al. report substantial zero-shot transfer with this recipe on XNLI and MLQA; XTREME (Hu et al., ["XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization"](https://arxiv.org/abs/2003.11080), *ICML 2020*) is the canonical benchmark suite for this pattern.

## Language-aware sampling: the training-data lever

When you have labelled data in more than one language, you rarely have *balanced* multilingual data. English dominates by orders of magnitude. Training with a uniform sampler over the concatenation will steer the encoder toward English and hurt low-resource-language quality.

The XLM-R paper (Conneau et al. 2020, §3.1) uses **temperature-scaled sampling** over languages during pretraining:

```
q_l = (p_l)^α / Σ_l' (p_l')^α
```

- `p_l` is the empirical fraction of examples in language `l`.
- `α ∈ [0, 1]` is a temperature-like smoothing exponent.
- `α = 1.0` recovers proportional sampling (majority language dominates).
- `α = 0.0` recovers uniform sampling (all languages equal, minority-language noise dominates).
- The XLM-R pretraining uses `α = 0.3`.

Fine-tuning follows the same logic. You have per-language datasets `D_l` with sizes `n_l`; at each step, sample a language according to `q_l` and then sample an example uniformly within that language. The multilingual translation literature calls this "language sampling" or "batch composition."

A practical PyTorch sampler:

```python
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np

def language_aware_weights(languages, alpha=0.3):
    counts = np.array([np.sum(np.array(languages) == l) for l in sorted(set(languages))])
    probs = counts / counts.sum()
    tempered = probs ** alpha
    tempered = tempered / tempered.sum()
    lang_to_weight = dict(zip(sorted(set(languages)), tempered / counts))
    return np.array([lang_to_weight[l] for l in languages])

weights = language_aware_weights(train_langs, alpha=0.3)
sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
DataLoader(train_dataset, sampler=sampler, batch_size=32)
```

Two things to think about:

1. **Combine with class weighting.** Language-aware sampling addresses language imbalance; you still need chapter 06's class weighting for label imbalance *within each language*.
2. **Batch composition matters.** With `α = 0.3` and many languages, most batches will still be dominated by the highest-resource language. If gradient noise across languages helps, force batches to be per-language homogeneous by grouping; if it hurts, mix. Test both.

## The "curse of multilinguality"

Conneau et al. (2020) and Chung et al., ["Improving Multilingual Models with Language-Clustered Vocabularies"](https://arxiv.org/abs/2010.12777), *EMNLP 2020*, document a real phenomenon: as you add more languages to a multilingual model at fixed capacity, per-language performance degrades. High-resource languages get worse to make room for low-resource ones. Larger models mitigate but do not eliminate the effect.

Practically:

- On a task where you only care about English + one or two other high-resource languages, a monolingual encoder per language often beats XLM-R base.
- XLM-R large closes most of the gap; XLM-V and XLM-R XL/XXL close more, at compute cost.
- The wins are largest on truly low-resource languages, where monolingual models don't exist or perform poorly.

Pick XLM-R when: you have < 500 labelled examples in a target language, your product ships in more than ~3 languages, or your traffic mix shifts language distribution over time. Pick per-language monolingual encoders when: 2–3 major languages, plenty of data in each, and latency-sensitive serving where a smaller monolingual model beats XLM-R base.

## Evaluation: the discipline you cannot skip

The trap: a single "overall accuracy" number across languages hides monolithic failure on languages you can't read. The rules:

- **Report per-language metrics separately.** Always. Never a single aggregate without the breakdown.
- **Report `mean` and `min` across languages.** The min is what your worst users experience.
- **Match training and eval language distributions.** If you fine-tuned on English and evaluate on 15 languages, note it — that is zero-shot transfer, not multilingual training.
- **Human-in-the-loop spot checks.** Pick a native-speaker reviewer for at least one low-resource language and eyeball a hundred predictions. Automated metrics on translated evaluation sets can obscure fluency-driven false positives.

Benchmarks and datasets to know:

- **XNLI** (Conneau et al., ["XNLI: Evaluating Cross-lingual Sentence Representations"](https://arxiv.org/abs/1809.05053), *EMNLP 2018*) — natural language inference in 15 languages.
- **PAWS-X** (Yang et al., ["PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification"](https://arxiv.org/abs/1908.11828), *EMNLP-IJCNLP 2019*) — paraphrase identification in 7 languages.
- **MARC** (Keung et al., ["The Multilingual Amazon Reviews Corpus"](https://arxiv.org/abs/2010.02573), *EMNLP 2020*) — product review classification in 6 languages.
- **MASSIVE** (FitzGerald et al., ["MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages"](https://arxiv.org/abs/2204.08582), *ACL 2023*) — intent classification and slot filling in 51 languages.
- **XTREME** (Hu et al. 2020, above) — multi-task multilingual evaluation.
- **XTREME-R** (Ruder et al., ["XTREME-R: Towards More Challenging and Nuanced Multilingual Evaluation"](https://arxiv.org/abs/2104.07412), *EMNLP 2021*) — the update.
- **FLORES-200** (NLLB Team, ["No Language Left Behind: Scaling Human-Centered Machine Translation"](https://arxiv.org/abs/2207.04672), *arXiv 2022*) — 200-language translation benchmark, useful when your classification pipeline includes translation.

## Tokeniser behaviour on unseen languages

XLM-R's SentencePiece vocabulary was trained on CC-100's 100 languages. Languages *not* in that set fall back to Unicode-byte-level fragments — the tokeniser will not fail, but it will emit very long sequences of tiny subword tokens, and classification quality on such inputs is usually poor.

Check before deploying:

```python
tokens = tokenizer.tokenize(sample_text)
print(len(tokens), tokens[:20])
```

If a 100-character sentence tokenises to 200+ tokens with mostly `▁` byte-fragments, your language is not represented and you should either:

- Continue pretraining on your language's Common Crawl subset (see mod-108 on adaptation), or
- Route that language to a language-specific model, or
- Pick a model with broader vocabulary coverage (XLM-V, mDeBERTa-v3).

## Language identification as a preprocessing step

If your input can be *any* language and you need to route or annotate it before classifying, run a fastText LID pass first (see mod-102 chapter 09 for the deep dive). Cases where LID is worth adding:

- You are training a per-language classifier and need to filter input to the right one.
- You are using XLM-R but want to apply language-specific calibration or thresholds (chapter 07).
- You want to log language distribution over time.

Even with a multilingual encoder, LID as an upfront tag is cheap and useful.

## Combining XLM-R with the other levers from this module

- **Imbalance handling** (chapter 06) applies per-language *and* across languages. Do both.
- **Calibration** (chapter 07) needs to be done per-language when the base rate varies by language, which it usually does. A single global temperature is a first pass; per-language temperatures often help.
- **Thresholds** (chapter 07) should be tuned per-language when precision-recall trade-offs differ per market.

## Where multilingual encoders are not the answer

- **Extremely low-resource languages**, where the encoder simply lacks representation. Either continue pretrain on that language (mod-108), or accept the ceiling.
- **Dialects or code-switching heavy inputs.** XLM-R was pretrained on written CC-100; social-media colloquial mixed-language text is closer to unseen territory. The LinCE benchmark (Aguilar et al., 2020, cited in mod-102) is the classic testbed.
- **Products with only 1–2 languages and abundant per-language data.** Monolingual encoders (`bert-base-german-cased`, `flaubert`, `bert-base-japanese-v3`) are cheaper to serve and often better at the top.

## Chapter summary

- XLM-R (and mDeBERTa-v3, XLM-V) is the default encoder when your inputs span multiple languages, especially when you have data in only some of them.
- Zero-shot cross-lingual transfer works because pretraining aligned representations across languages; train on your highest-resource language and evaluate on the rest.
- Language-aware sampling with `α ≈ 0.3` (Conneau et al.'s pretraining-time recipe) balances high- and low-resource languages during fine-tuning; combine with class weighting for label imbalance.
- The "curse of multilinguality" caps per-language performance at fixed capacity; larger models help. Monolingual encoders can still beat XLM-R when languages are few and data is abundant.
- Evaluate per-language, always. Report `mean` and `min` across languages, watch out for silent monolithic failures on languages you can't read, and use XNLI, PAWS-X, MASSIVE, XTREME or MARC as the standard benchmarks.
- Combine multilingual training with per-language calibration and threshold tuning; a single global threshold across all markets is usually the wrong default.
