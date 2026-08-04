# Fine-tuning Encoders: BERT, RoBERTa, DeBERTa, XLM-R

## Motivation

When the linear baselines cap out — you need paraphrase invariance, negation handling, entailment, or transfer to a related domain — the next rung is fine-tuning a pretrained encoder. Modern practice on Hugging Face Transformers makes this a ~50-line script, but that hides a stack of decisions that separate a competitive model from a mediocre one: which encoder to start from, how to pool, what learning rate, how many epochs, how to handle sequence length, and how to make the run reproducible.

This chapter is the reference for those choices. Chapters 05–07 layer task shape, imbalance, and calibration on top.

## Which encoder to start from

Four encoder families dominate 2026 text-classification work:

- **BERT** — Devlin, Chang, Lee & Toutanova, ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805), *NAACL 2019*. The original masked-LM encoder. Pretrained on BookCorpus + English Wikipedia. Still a fine baseline; on GLUE and many downstream tasks it now underperforms newer options at the same parameter count.
- **RoBERTa** — Liu et al., ["RoBERTa: A Robustly Optimized BERT Pretraining Approach"](https://arxiv.org/abs/1907.11692), *arXiv 2019*. Same architecture as BERT, better recipe: 10× more data, longer training, no next-sentence-prediction, dynamic masking, byte-level BPE tokenizer. Typically 1–3 points better than BERT on the same task at the same size.
- **DeBERTa** (v1 / v2 / v3) — He, Liu, Gao & Chen, ["DeBERTa: Decoding-enhanced BERT with Disentangled Attention"](https://arxiv.org/abs/2006.03654), *ICLR 2021*, and He, Gao & Chen, ["DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing"](https://arxiv.org/abs/2111.09543), *ICLR 2023*. Disentangled content/position attention, ELECTRA-style pretraining in v3, and consistently near the top of the GLUE / SuperGLUE leaderboards among encoders of comparable size. Default for English classification when compute allows.
- **XLM-R** — Conneau et al., ["Unsupervised Cross-lingual Representation Learning at Scale"](https://arxiv.org/abs/1911.02116), *ACL 2020*. 100-language multilingual encoder trained on CC-100. The default when your inputs span more than one language. Chapter 08 covers it in depth.

Size-tier picks that are almost always the right first swing:

| Task setting | English-only pick | Multilingual pick |
| --- | --- | --- |
| CPU inference, minimal budget | `distilbert-base-uncased` / `distilroberta-base` | `distilbert-base-multilingual-cased` |
| Standard GPU serving | `roberta-base` / `microsoft/deberta-v3-base` | `xlm-roberta-base` |
| Accuracy first, budget available | `microsoft/deberta-v3-large` | `xlm-roberta-large` / `microsoft/mdeberta-v3-base` |

Check the `Model Card` on Hugging Face for licence, tokenizer notes, and known biases before committing to any specific checkpoint.

## Anatomy of a fine-tune run

The moving parts:

```
raw text
  │  tokenizer  (BPE / WordPiece / SentencePiece)
  ▼
input_ids  ─┐
attention_mask ─┤─► Encoder (12–24 transformer layers)
token_type_ids ─┘        │
                         ▼
                   hidden_states (batch, seq, hidden)
                         │  pooling
                         ▼
                   sentence_repr (batch, hidden)
                         │  classification head
                         ▼
                     logits (batch, num_labels)
                         │  loss
                         ▼
                     cross_entropy / BCE
```

Hugging Face's `AutoModelForSequenceClassification` builds this stack for you and picks the right head shape from `num_labels`. What you own is the tokeniser configuration, the pooling choice, the training loop's hyperparameters, and the reproducibility discipline.

## The minimum viable fine-tune

```python
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorWithPadding,
)
import evaluate

MODEL = "microsoft/deberta-v3-base"

ds = load_dataset("ag_news")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenise(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)

ds = ds.map(tokenise, batched=True)
collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=4,
    id2label={0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
    label2id={"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3},
)

metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric_f1.compute(predictions=preds, references=labels, average="macro")

args = TrainingArguments(
    output_dir="out",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_ratio=0.06,
    lr_scheduler_type="linear",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    seed=42,
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=ds["train"], eval_dataset=ds["test"],
    tokenizer=tokenizer, data_collator=collator,
    compute_metrics=compute_metrics,
)
trainer.train()
```

Every hyperparameter above is defensible for a "middle of the road" fine-tune. What follows is why each is set that way, and what to change when.

## The hyperparameters that actually matter

- **Learning rate.** For encoders, `1e-5` to `5e-5` is the whole range. `2e-5` is the safe default (BERT paper, RoBERTa paper, and DeBERTa-v3 recipes all use values in this band). Larger models (v3-large) tolerate `1e-5`; smaller ones (`distil-*`) tolerate up to `5e-5`. Cosine schedules edge out linear on longer training; use linear with warmup for anything under 5 epochs.
- **Warmup ratio.** 6–10 % of steps. Skipping warmup causes gradient explosion in the first few hundred steps of a large-model fine-tune. `warmup_ratio=0.06` matches the RoBERTa recipe.
- **Batch size.** As large as you can fit — but effective batch size, not per-device. Combine `per_device_train_batch_size` with `gradient_accumulation_steps` to hit an effective batch of 32–64 for base models, 16–32 for large. Very small batches (< 8) hurt significantly on encoder fine-tuning.
- **Epochs.** Three is a good default for balanced tasks with tens of thousands of examples. Small datasets sometimes need up to 10; huge datasets converge in one. Watch validation loss; encoder fine-tunes overfit early and often.
- **Weight decay.** `0.01` universally works. Do not go to `0` — you will overfit on small tasks.
- **Sequence length.** `max_length=128` or `256` for short-text (tweets, headlines); `512` for review or article classification. Longer sequences quadruple compute; use `Longformer` / `BigBird` if your inputs genuinely exceed 512 tokens (Beltagy, Peters & Cohan, ["Longformer: The Long-Document Transformer"](https://arxiv.org/abs/2004.05150), *arXiv 2020*).
- **Dropout.** Leave the encoder's pretraining dropout in place; do not touch it. Add ~0.1 dropout on the classifier head — the default in `AutoModelForSequenceClassification`.
- **Mixed precision.** `fp16=True` on Ampere or newer, `bf16=True` on Ampere and later. Doubles throughput; the loss-scaling gotchas that plagued the 2019 era are mostly solved by `Trainer` and `accelerate`.

If you sweep hyperparameters, sweep learning rate first, then epochs, then batch size. Everything else usually stays at the defaults.

## Pooling: `[CLS]` vs. mean vs. attention

`AutoModelForSequenceClassification` pools by default the way the pretraining task expected:

- **BERT / RoBERTa / XLM-R** pool the first token (`[CLS]` or `<s>`) hidden state through a linear + tanh layer.
- **DeBERTa** uses a slightly different pooler; the class abstracts it away.
- **`sentence-transformers`-style mean pooling** averages token hidden states with the attention mask. This is what you use when computing embeddings for downstream retrieval (see mod-108), not for direct classification.

Rule: use the default pooler for classification. Mean pooling is only worth trying when documents are long and the useful signal is distributed across many tokens; even then, an attention-weighted pooler often helps more than a fixed mean (Sun, Qiu, Xu & Huang, ["How to Fine-Tune BERT for Text Classification?"](https://arxiv.org/abs/1905.05583), *CCL 2019*).

## Sequence-length budget: the classification-specific concerns

Truncation policy is a decision, not an afterthought:

- **`truncation=True, padding=False`** with a `DataCollatorWithPadding` — the standard. Documents longer than `max_length` are truncated *from the right* by default; this loses the tail of long documents.
- **Truncate-from-the-middle** — for tasks where the beginning and end both carry signal (e.g., reviews with a headline and a conclusion). Custom truncation, not a `tokenizer` flag.
- **Sliding-window classification.** For long documents, split into overlapping windows, classify each, aggregate by mean or max. Standard for long medical or legal documents; adds latency but recovers accuracy.
- **`max_length` past 512.** Base BERT/RoBERTa/DeBERTa embeddings have positional encodings only up to 512 tokens. Use a long-context encoder (Longformer, BigBird, LongT5-encoder) if you truly need it.

Log the truncation rate on your training set (`sum(len(ids) > max_length) / N`). If it exceeds a few percent, you are silently dropping signal and any accuracy gain from a bigger model may be a bigger `max_length` in disguise.

## Reproducibility discipline

Fine-tunes are stochastic. Seed everything and pin everything:

- Set `seed=` on `TrainingArguments` and `torch.manual_seed`; `Trainer` does the rest.
- Pin the exact model revision: `AutoModelForSequenceClassification.from_pretrained(MODEL, revision="...")`.
- Pin the exact tokenizer: instantiate from the same revision.
- Pin the exact `transformers` version in your requirements file — subtle preprocessing tweaks between minor versions can change tokenisation edge cases.
- Report **at least 3 seeds** in any paper-quality comparison — encoder fine-tunes on small datasets have standard deviations that dwarf small architectural improvements (Mosbach, Andriushchenko & Klakow, ["On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines"](https://arxiv.org/abs/2006.04884), *ICLR 2021*).
- Log tokenizer, model revision, PyTorch version, CUDA version, and precision alongside your metrics.

## Head design

`AutoModelForSequenceClassification` adds a `nn.Linear(hidden_size, num_labels)` on top of the pooled representation. Nine times in ten this is what you want. Cases where custom heads help:

- **Multi-label**: same head, but you set `problem_type="multi_label_classification"` on `from_pretrained`, which switches the loss to `BCEWithLogitsLoss` and expects float labels in `[0, 1]` (chapter 05 covers this properly).
- **Ordinal classification** (rating prediction): stack `K-1` binary heads for the ordinal cumulative-link formulation (see Frank & Hall, ["A Simple Approach to Ordinal Classification"](https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf), *ECML 2001*).
- **Multi-task learning** across related classification tasks — subclass `AutoModel` and add per-task heads on the shared encoder.

An MLP head (`Linear → GeLU → Linear`) rarely helps once you have a decent encoder — the encoder is already doing the heavy nonlinear work.

## What can go wrong, in the order it usually does

1. **Learning rate too high.** Loss diverges within a few hundred steps or plateaus at chance. Fix: drop LR by 2–5×, re-add warmup.
2. **Tokenizer/model mismatch.** Wrong encoder revision + tokenizer revision. Symptom: near-chance accuracy that improves slowly. Fix: instantiate both from the same repo/revision.
3. **Class imbalance.** Model trivially predicts the majority class. Fix: chapter 06.
4. **Label leakage from the tokenizer.** Your label appears literally in the input (e.g., subject line contains "SPAM"). Fix: strip during preprocessing, or accept it and stop pretending it's a hard task.
5. **Truncation dropping signal.** Log truncation rate; fix `max_length` or use a long-context encoder.
6. **Not enough epochs, or too many.** Watch `eval_loss` per epoch; `load_best_model_at_end=True` guards against the "too many" side.
7. **Seed variance masquerading as improvement.** Any single-seed comparison at ≤ 1 F1 point delta is noise. Repeat.

## When fine-tuning beats — and when it does not

Beats the linear baselines when:

- The task requires paraphrase invariance or entailment.
- The label distribution is moderately balanced and you have ≥ ~500 examples per class.
- Your input is natural language, not codes or logs.
- You have GPU serving budget and can tolerate 10–100 ms per document at batch size 1.

Does not beat when:

- You have < 100 examples per class (LLM few-shot or prompt-based methods will typically beat).
- Latency budget is under a few milliseconds on CPU (fastText or tf-idf + linear will beat).
- The signal is literal token match, not composition (linear baselines will match).
- Your label set changes weekly (retraining cost dominates; LLM zero-shot wins).

## Chapter summary

- Encoder fine-tuning is the third rung of the ladder. DeBERTa-v3 for English classification and XLM-R for multilingual are the current safe defaults; RoBERTa remains a strong fallback.
- The hyperparameter budget is small: LR in the `1e-5`–`5e-5` band, 3 epochs, warmup ~6 %, weight decay `0.01`, mixed precision, default pooling. Sweep LR first.
- Reproducibility discipline (seed, model + tokenizer revision, transformers version, multiple seeds) is not optional — encoder fine-tunes have real seed variance.
- Truncation policy and effective batch size are the two "silent" hyperparameters that most often explain unexpected results.
- Fine-tuning wins on paraphrase-heavy semantically compositional tasks with moderate labelled data; it does not automatically dominate baselines and requires the diagnostics from chapters 05–07 to actually ship.
