# Abstractive Summarisation with BART, T5, PEGASUS, and mT5

## Motivation

Once you have accepted the abstractive-vs-extractive trade-offs from chapter 01 and calibrated against the baselines in chapter 02, the modern recipe for English single-document summarisation is: pick a pretrained encoder-decoder, fine-tune with cross-entropy on `(source, summary)` pairs, decode with beam search, evaluate with ROUGE plus a paraphrase-tolerant metric plus a faithfulness probe. This chapter walks that recipe end-to-end and explains why the four canonical checkpoints — **BART**, **T5**, **PEGASUS**, **mT5** — exist as separate options.

## The task, formally

Given a source $x$ (a document, dialogue, or transcript) and a target summary $y$, we train a sequence-to-sequence model $p_\theta(y \mid x)$ to maximise

$$
\sum_t \log p_\theta(y_t \mid y_{<t}, x)
$$

with teacher-forced next-token cross-entropy. No summarisation-specific loss is required; the pretraining objectives (denoising for BART, span-corruption for T5, gap-sentence prediction for PEGASUS) all reduce to seq2seq at fine-tune time.

## Which encoder-decoder to reach for

The four checkpoints below are the ones you should know by heart. All are drop-in-replaceable under `AutoModelForSeq2SeqLM`; the differences are pretraining objective, tokeniser, context window, and language coverage.

| Model                                 | Params           | Pretraining objective                                      | Context (train) | Best for                                                                 |
|---------------------------------------|------------------|------------------------------------------------------------|-----------------|--------------------------------------------------------------------------|
| `facebook/bart-base` / `-large`       | 140 M / 400 M    | Denoising (Lewis et al., 2020) — corrupt, then reconstruct | 1024            | English news, dialogue, general abstractive.                             |
| `facebook/bart-large-cnn`             | 400 M            | BART + fine-tuned on CNN/DailyMail                          | 1024            | Zero-shot English news summarisation baseline. Ships strong on day one.  |
| `t5-base` / `-large` / `-3b` / `-11b` | 220 M–11 B       | Span-corruption (Raffel et al., 2020)                       | 512             | Text-to-text tasks including summarisation with a `summarize:` prefix.   |
| `google/flan-t5-base` / `-large`      | 250 M / 780 M    | T5 + instruction-tuned on many tasks (Chung et al., 2022)   | 512             | Better zero-shot and few-shot summarisation than raw T5.                 |
| `google/pegasus-large`                | 568 M            | Gap-sentence generation (Zhang et al., 2020)                | 1024            | English summarisation — best per-parameter on many benchmarks.           |
| `google/pegasus-xsum` / `-cnn_dailymail` | 568 M         | PEGASUS + task-specific fine-tune                          | 1024            | Zero-shot XSum-style extreme summarisation / CNN-DM.                     |
| `google/mt5-base` / `-large` / `-xl`  | 580 M–3.7 B      | T5 span-corruption on mC4 (101 languages)                   | 1024            | Multilingual summarisation over 40+ languages.                           |

Practical defaults:

- **English single-doc summarisation, unknown domain.** `facebook/bart-large` or `google/pegasus-large`.
- **English, need zero-shot immediately.** `facebook/bart-large-cnn` for CNN/DailyMail-style; `google/pegasus-xsum` for XSum-style.
- **Instruction-tuned or few-shot without fine-tune.** `google/flan-t5-large`.
- **Non-English or multilingual.** `google/mt5-large` (or XL-Sum-tuned checkpoints).
- **Long documents (> 1024 tokens).** LongT5, PEGASUS-X, or LED — see chapter 04.

## Why the pretraining objective matters

The four objectives were designed with generation in mind, but they emphasise different skills:

- **BART's denoising.** Corrupts the input with a mixture of noise functions (span masking, token deletion, sentence permutation, document rotation) and reconstructs the original. Produces a generator that is robust to messy or ill-formed inputs and strong on paraphrase.
- **T5's span-corruption.** Masks contiguous spans in the input with sentinel tokens and asks the decoder to emit the spans in order. Produces a general-purpose text-to-text model whose "task" is just a text prefix.
- **PEGASUS's gap-sentence generation.** Deletes whole *important* sentences (selected by ROUGE self-overlap) and asks the decoder to reconstruct them. This objective is deliberately close to abstractive summarisation and is why PEGASUS punches above its parameter count on summarisation benchmarks.
- **mT5's span-corruption on mC4.** Same objective as T5 but over Common Crawl in 101 languages. Multilingual capability at the cost of English-only quality.

The practical upshot: on many English summarisation benchmarks, PEGASUS-large edges out BART-large at similar parameter counts; on multilingual, mT5 (or XL-Sum-tuned checkpoints) is your only option; on instruction-style or few-shot prompting, FLAN-T5 wins.

## Data preparation

Every dataset lands as `(source, summary)` pairs. Preprocessing needs three things right, and one thing every tutorial gets wrong:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

MAX_INPUT  = 1024
MAX_TARGET = 128

def preprocess(examples):
    model_inputs = tokenizer(
        examples["article"],
        max_length=MAX_INPUT,
        truncation=True,
        padding=False,       # let the collator pad dynamically
    )
    labels = tokenizer(
        text_target=examples["highlights"],
        max_length=MAX_TARGET,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

Three things right:

- **Use `text_target=...`** (or the older `as_target_tokenizer()` context manager). Some encoder-decoder tokenisers switch prefixes/languages between input and target — mT5 for cross-lingual work is the canonical case.
- **Pad dynamically at collation time**, not at preprocess time. Wildly variable summary lengths — CNN/DailyMail vs. XSum vs. Multi-News — punish static padding.
- **Truncate the source, never the target.** Truncating the summary silently teaches the model to emit incomplete summaries. If your summaries do not fit at `max_target = 128`, raise the cap; do not truncate.

And the thing every tutorial gets wrong:

- **Use `DataCollatorForSeq2Seq`, not `DefaultDataCollator`.** It pads inputs and labels independently *and* replaces label pad tokens with `-100` so cross-entropy ignores them. Using `DefaultDataCollator` produces a silently-wrong loss that trains a model to emit padding — you will notice at ROUGE time, weeks later.

For T5-family models, prepend a task prefix to the input:

```python
inputs = ["summarize: " + doc for doc in examples["article"]]
```

BART and PEGASUS do not need a prefix. mT5 for cross-lingual summarisation needs *language codes* — check the specific checkpoint's card.

## Training recipe (single-doc English)

Sensible defaults for BART-large / PEGASUS-large on CNN/DailyMail-scale data:

```python
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments,
)

model     = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

args = Seq2SeqTrainingArguments(
    output_dir="summariser",
    learning_rate=3e-5,                   # BART / PEGASUS with AdamW
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,        # effective batch 32
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_ratio=0.06,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=4,
    bf16=True,                             # or fp16 on older GPUs
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    tokenizer=tokenizer,
    compute_metrics=make_rouge_metrics(tokenizer),
)
trainer.train()
```

The differences that matter across the four models:

- **BART / PEGASUS** — AdamW with LR `3e-5`, three epochs, label smoothing `0.1` often helps.
- **T5 / FLAN-T5** — either Adafactor with LR `1e-3` (paper regime) or AdamW with LR `1e-4`. T5 does not need label smoothing; some regimes hurt.
- **mT5** — same as T5 but be careful with the tokeniser. mT5's SentencePiece has a very large vocabulary; batching is memory-heavy.

The `warmup_ratio=0.06` and `weight_decay=0.01` numbers come from the BART paper's fine-tuning recipe and remain a solid default. Do not skip warmup — encoder-decoder fine-tuning is unstable in the first few hundred steps.

## Decoding for summarisation

Chapter 06 covers decoding strategies in full. For summarisation specifically:

- **Beam search** with `num_beams=4` or `5` is the standard. Higher `num_beams` rarely helps and *sometimes hurts* summarisation quality (Stahlberg & Byrne, 2019 — the "beam-search curse").
- **`no_repeat_ngram_size=3`** suppresses the classic decoder failure mode where the model gets stuck in a repeated phrase.
- **`length_penalty=1.0`** is neutral. `>1.0` favours longer summaries; `<1.0` favours shorter. Tune on dev.
- **`min_length` and `max_length`** enforce hard bounds. BART-CNN's default is `min_length=56`, `max_length=142` — the CNN/DailyMail summary length distribution.
- **Sampling (nucleus, typical)** produces more diverse outputs but reduces ROUGE and often faithfulness. Reserve for creative rewriting or paraphrase tasks — chapters 06 and 07.

## Evaluation stack

Chapters 10–13 cover this in depth. The minimum panel for a fine-tuned summariser:

- **ROUGE-1 / ROUGE-2 / ROUGE-L** (F-measure) via `evaluate.load("rouge")`.
- **BERTScore** with `microsoft/deberta-xlarge-mnli` as the scoring model — paraphrase-tolerant.
- **A faithfulness probe** — SummaC or FactCC — on your dev predictions to catch hallucination that ROUGE cannot see.
- **Length statistics** — mean and 90th-percentile output length, against the reference distribution. Length drift is one of the earliest signs of training-data leakage or the model gaming beam search.

Never report a single metric. ROUGE-L, BERTScore, and a faithfulness rate together give the picture; alone they lie.

## Multilingual: mT5 and the XL-Sum family

For non-English or multi-language summarisation, replace BART/PEGASUS with mT5 or one of the multilingual PEGASUS-style checkpoints. The recipe is the same with three additional considerations:

- **Language codes.** mT5 has no language prefix, but its tokeniser handles 101 languages. XL-Sum-tuned checkpoints often expect a language identifier — check the card.
- **Tokenisation cost.** mT5's SentencePiece vocabulary is ~250 k. Batching is memory-heavier per example than English BART.
- **Data imbalance.** Multilingual training corpora skew heavily English. XL-Sum (Hasan et al., 2021) is the community standard multilingual summarisation dataset with balanced per-language splits.

For cross-lingual summarisation (source in language A, summary in language B), use an mT5 or NLLB-based recipe with the language codes made explicit in the input template.

## PEGASUS's gap-sentence trick — why it matters

The single most important thing to know about PEGASUS is that its pretraining objective was *designed* for summarisation:

1. Score every sentence in a document by its ROUGE overlap with the *rest* of the document.
2. Mask the top-$k$ scoring sentences and ask the model to reconstruct them.

This is essentially self-supervised extractive summarisation on a massive scale, and it explains why PEGASUS-large routinely beats BART-large at fewer parameters on summarisation-specific benchmarks. For any summarisation task where an off-the-shelf pretrained encoder-decoder is your starting point, PEGASUS is a legitimate first pick — usually within 1 ROUGE of BART-large-cnn zero-shot on CNN/DailyMail, and stronger on XSum with `pegasus-xsum`.

## Common failure modes and their fixes

- **Repetitive summaries.** Set `no_repeat_ngram_size=3`. If it persists, your training data is repetitive (bad label quality).
- **Truncated summaries.** Raise `generation_max_length`. Check that your training targets are not being silently truncated.
- **Extractive-copy outputs.** The model degenerates into copying sentences verbatim. Symptom: high ROUGE, low BERTScore delta over LEAD-N. Fix: check for near-duplicates between source and target in the training data.
- **Hallucinated entities.** Common on XSum-style data. Chapter 11–12 have the diagnostics and mitigations. Do *not* try to fix this by increasing beam width — beam search's mode-seeking behaviour makes hallucination *more* consistent, not less.
- **Silent-fail loss.** You forgot `DataCollatorForSeq2Seq`. Fix the collator; retrain.

## Chapter summary

- Fine-tuning an encoder-decoder for summarisation reduces to `(source, summary)` seq2seq with cross-entropy. All four canonical checkpoints — BART, T5/FLAN-T5, PEGASUS, mT5 — plug into the same recipe.
- **BART** is the general-purpose default; **PEGASUS** is the summarisation specialist; **T5/FLAN-T5** are text-to-text generalists; **mT5** covers multilingual.
- Use `text_target=`, dynamic padding, and `DataCollatorForSeq2Seq` — the last one is the bug every seq2seq tutorial gets wrong.
- Beam search with `num_beams=4–5`, `no_repeat_ngram_size=3`, and dev-tuned length penalties is the standard inference recipe.
- Never report a single evaluation metric — pair ROUGE with BERTScore and a faithfulness probe. Chapters 10–12 formalise that stack.
