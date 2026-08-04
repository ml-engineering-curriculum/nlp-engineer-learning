# Token-Classification NER: the Workhorse Recipe

## Motivation

With BIO evaluation (chapter 02) and subword alignment (chapter 03) in place, you can now train the workhorse NER model: an encoder plus a per-token linear head, trained with cross-entropy against BIO labels, evaluated with `seqeval`. This chapter is the reference recipe — the ~80-line script that dominates production NER, the hyperparameters that actually matter, and the failure modes that repeat across every project.

Modern practice in 2026 still leans hard on this recipe. Span-based models (chapter 06) beat it on nested entities; decoder LLMs (chapter 10) beat it on flexibility; but for a flat, closed schema with a labelled corpus, `AutoModelForTokenClassification` on a decent encoder is the shortest path to a shippable model.

## Which encoder to start from

The same tier chart as classification (mod-103 chapter 04) applies, with a few NER-specific notes:

| Setting | English pick | Multilingual pick |
| --- | --- | --- |
| CPU inference, minimal budget | `distilroberta-base` / `distilbert-base-cased` | `distilbert-base-multilingual-cased` |
| Standard GPU serving | `roberta-base` / `microsoft/deberta-v3-base` | `xlm-roberta-base` |
| Accuracy first | `microsoft/deberta-v3-large` | `xlm-roberta-large` |
| Biomedical | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` (Gu et al., ["Domain-Specific Language Model Pretraining for Biomedical NLP"](https://arxiv.org/abs/2007.15779), *ACM Trans. Comput. Healthcare 2022*) or `dmis-lab/biobert-v1.1` |  |
| Clinical | `emilyalsentzer/Bio_ClinicalBERT`, `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` |  |
| Legal | `nlpaueb/legal-bert-base-uncased` (Chalkidis et al., ["LEGAL-BERT: The Muppets straight out of Law School"](https://arxiv.org/abs/2010.02559), *EMNLP Findings 2020*) |  |
| Scientific | `allenai/scibert_scivocab_uncased` (Beltagy, Lo & Cohan, ["SciBERT"](https://arxiv.org/abs/1903.10676), *EMNLP 2019*) |  |

Two NER-specific caveats:

- **Casing matters.** PER/ORG/LOC lean heavily on capitalisation. Prefer *cased* variants (`bert-base-cased`, `roberta-base`, `xlm-roberta-*`) for standard NER; uncased models are a small default handicap you should only accept when your input is all-lowercase (chat, social media).
- **Domain-adapted encoders help most in low-resource regimes.** With ≥ 10 000 labelled entities, a general `deberta-v3-base` often catches or beats a domain-BERT. With ≤ 2 000 labelled entities, the domain encoder usually wins by 2–5 F1.

## The minimum viable NER fine-tune

```python
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification, AutoTokenizer,
    DataCollatorForTokenClassification, Trainer, TrainingArguments,
)
from evaluate import load

MODEL = "microsoft/deberta-v3-base"
ds = load_dataset("conll2003")

label_names = ds["train"].features["ner_tags"].feature.names
label2id = {n: i for i, n in enumerate(label_names)}
id2label = {i: n for i, n in enumerate(label_names)}

tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"], is_split_into_words=True,
        truncation=True, max_length=256,
    )
    all_labels = []
    for i, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(word_labels[word_id])
            else:
                label_ids.append(-100)  # first-subword-only (variant A)
            previous_word_id = word_id
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized

ds = ds.map(tokenize_and_align_labels, batched=True, remove_columns=ds["train"].column_names)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL, num_labels=len(label_names),
    id2label=id2label, label2id=label2id,
)

collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
metric = load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    true_labels = [[id2label[l] for l in seq if l != -100] for seq in labels]
    true_preds = [
        [id2label[p] for p, l in zip(pseq, lseq) if l != -100]
        for pseq, lseq in zip(preds, labels)
    ]
    results = metric.compute(
        predictions=true_preds, references=true_labels,
        mode="strict", scheme="IOB2",
    )
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
    }

args = TrainingArguments(
    output_dir="out-ner",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
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

Trainer(
    model=model, args=args,
    train_dataset=ds["train"], eval_dataset=ds["validation"],
    tokenizer=tokenizer, data_collator=collator,
    compute_metrics=compute_metrics,
).train()
```

That is the whole recipe. It will get you within noise of published CoNLL-2003 numbers for `deberta-v3-base` (≈ 92 F1). The rest of this chapter is why each piece is set that way.

## Hyperparameters that actually matter

- **Learning rate.** `2e-5` to `5e-5`, `3e-5` a safe default for base-size encoders on NER. `1e-5` for `-large`. Larger LRs than classification because the head is fresh per-token and needs to move faster.
- **Batch size.** Effective batch of 16–32 (per-device times grad-accum). Larger batches on NER dilute the entity-vs-`O` signal because most tokens are `O`.
- **Epochs.** 3–5 for CoNLL-scale data; 10 for smaller corpora (a few thousand sentences); 1–2 for very large ones. Watch dev F1 per epoch and enable `load_best_model_at_end`.
- **Warmup ratio.** 6 % of steps. Same as classification.
- **Weight decay.** `0.01`. Do not zero it — small NER datasets overfit fast.
- **Sequence length.** 128 for sentence-level datasets, 256 for typical documents, 512 if you cannot re-chunk. Log the truncation rate; > 5 % is a signal to switch to sliding-window (chapter 07).
- **Data collator.** Use `DataCollatorForTokenClassification`, not the classification collator — it pads labels to the same length with `-100`.
- **Mixed precision.** `fp16=True` (Ampere+) or `bf16=True` (Ampere+, more stable for larger models).

If you sweep, sweep LR first, then epochs, then batch size. Everything else typically stays at the defaults.

## The label-imbalance question — different from classification

NER's imbalance is not "class B rare" but "most tokens are `O`." On CoNLL-2003 English, roughly 83 % of tokens are `O`. On many domain corpora it is 95 %+. Because the model is scored *entity-level*, this imbalance is less catastrophic than it looks — the loss is dominated by `O` but the metric only counts spans — but it does introduce two failure modes:

1. **Recall drops on rare types.** MISC / PROD / FAC classes with < 200 training entities lose 10–20 F1 relative to PER/ORG/LOC.
2. **Boundary errors on long entities.** The model prefers shorter spans because most spans it sees are short.

Fixes, in order of pain:

- **Class weights** on cross-entropy (`nn.CrossEntropyLoss(weight=...)`) inversely proportional to class frequency, but *not* down-weighting `O` too aggressively — a `0` weight on `O` teaches the model that everything is an entity. `weight_O ≈ 0.1–0.3`, `weight_entity ≈ 1.0` is a defensible starting point.
- **Focal loss** on top of cross-entropy (Lin et al., ["Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002), *ICCV 2017*). `γ = 2` typically. Small gains on unbalanced NER; sometimes hurts on balanced data. Test before committing.
- **Entity-oversampling** at the *document* level — over-represent documents containing rare entity types in your training batches. `torch.utils.data.WeightedRandomSampler` with per-document weights. Rarely needed for base models, occasionally helpful for `-large` on small corpora.
- **Move to span-based models** (chapter 06). Span-based decoding side-steps the `O` imbalance by scoring positive spans directly against a hard-negative pool.

## Do you still need a CRF head?

CRF (Lafferty, McCallum & Pereira, ["Conditional Random Fields"](https://repository.upenn.edu/cis_papers/159/), *ICML 2001*) sat on top of every serious neural NER model from 2015 to 2019. On top of BiLSTM encoders it added a consistent 0.5–1.5 F1 by enforcing valid tag transitions.

For transformer encoders in 2026 the answer is: **usually no.**

- On CoNLL-2003 with `bert-base-cased`, adding a CRF changes F1 by ≤ 0.3 (multiple reproductions in the `pytorch-crf` and `flair` communities).
- On domain corpora with small labelled sets (< 3 000 sentences), a CRF can still add 0.3–0.8 F1 by enforcing valid transitions the model has too few examples to learn.
- On document-level and long-context settings, the CRF's forward-backward cost becomes non-trivial (linear in sequence length times `K²` in tag count) and often does not pay for itself.

If you want to try one, `pytorch-crf` and `flair`'s `SequenceTagger` both expose ready CRF layers. Configure it with valid transition masking so the model cannot even learn `O → I-X`. Chapter 05 goes deeper on when structured decoders still help.

A cheaper alternative: **BIO-invalidity post-processing.** Run greedy softmax at inference, then walk the output and coerce invalid transitions (`O → I-X` becomes `O → B-X`). Sub-0.1 F1 difference from a full CRF at ~1 % of the training cost.

## Inference: the pipeline you probably want

```python
from transformers import pipeline

nlp = pipeline(
    "ner",
    model="out-ner",
    aggregation_strategy="first",   # match training's first-subword-only alignment
    device=0,
)

nlp("Barack Obama visited Berlin on Friday.")
# [{'entity_group': 'PER', 'score': 0.9987, 'word': 'Barack Obama', 'start': 0, 'end': 12}, ...]
```

`aggregation_strategy="first"` is the right default when you trained with variant A/C from chapter 03. `"simple"` and `"max"` are alternatives; `"none"` is debug-only.

For production, prefer the manual loop:

```python
inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True)
offset_mapping = inputs.pop("offset_mapping")
with torch.no_grad():
    logits = model(**inputs).logits
preds = logits.argmax(-1)[0]
# walk preds + offset_mapping to emit (start_char, end_char, label) spans
```

You control batching, truncation, and can plug in a temperature or calibrator on the logits — none of which the pipeline exposes cleanly.

## Reproducibility, again

- Report at least three seeds; NER on small corpora has 0.5–1.0 F1 seed variance (Reimers & Gurevych, ["Reporting Score Distributions Makes a Difference"](https://arxiv.org/abs/1707.09861), *EMNLP 2017*).
- Pin model + tokenizer revision.
- Log the `transformers` version, PyTorch version, CUDA version, and precision.
- Log which alignment variant, which scheme, and which `seqeval` mode you used. This information is what someone reproducing your result needs.

## What usually goes wrong, in order

1. **Tokenizer/model mismatch or wrong alignment variant.** Symptom: F1 stuck below 30 for many epochs. Fix: audit 20 aligned examples (chapter 03).
2. **Learning rate too high or too low.** Too high: loss NaNs or plateaus at chance. Too low: F1 climbs 1 point/epoch and never converges. Fix: sweep `1e-5, 3e-5, 5e-5`.
3. **Uncased model on a corpus where casing carries signal.** Symptom: PER/ORG recall lags LOC by 15+ F1. Fix: switch to cased.
4. **Sequence truncation dropping tail entities.** Symptom: recall for entity types clustered near document end is 5–15 F1 lower than others. Fix: log truncation rate; move to sliding-window (chapter 07).
5. **Lenient `seqeval` on a broken tag sequence.** Symptom: your F1 looks better than the paper's, but you cannot reproduce it downstream. Fix: `mode="strict"`, always.
6. **`O` class weight set to 0.** Symptom: model predicts entities everywhere; precision collapses. Fix: keep a nonzero `O` weight or use focal loss instead.
7. **Fine-tuning the pretrained tokenizer.** You almost never want this. `tokenizer.save_pretrained` at the end preserves the input pipeline; do not retrain the tokenizer.

## Chapter summary

- A ~80-line Hugging Face script — `AutoModelForTokenClassification` + `DataCollatorForTokenClassification` + `seqeval` + `Trainer` — is the modern default NER recipe. Everything else is a variation on this stack.
- LR `2e-5`–`5e-5`, effective batch 16–32, 3–5 epochs, cased base-size encoder is the safe first-shot configuration.
- CRF heads rarely pay for themselves on transformer encoders; BIO-invalidity post-processing is a cheaper alternative that captures most of the benefit.
- `O` dominance is NER's imbalance shape; class weighting and focal loss occasionally help, but span-based models (chapter 06) side-step the problem entirely.
- Reproducibility discipline (seeds, revisions, alignment variant, scheme, `seqeval` mode) is non-negotiable — NER seed variance is a real 0.5–1.0 F1.
