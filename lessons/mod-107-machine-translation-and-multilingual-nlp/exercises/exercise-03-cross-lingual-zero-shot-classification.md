# exercise-03: Cross-Lingual Zero-Shot Classification

**Estimated effort:** 3 hours

## Objective

Fine-tune a multilingual encoder (XLM-R or mDeBERTa-v3) on **English** classification labels, then evaluate on **five target languages** across three transfer patterns — **zero-shot**, **translate-train**, and **translate-test** — and report per-language accuracy with confidence intervals. Diagnose zero-shot's silent-failure mode (tokenisation shatter) and decide which pattern to ship. This is the canonical multilingual NLP recipe you should be able to reproduce whenever a customer asks "does this classifier work in <language X>?"

## Prerequisites

- Chapters [07](../07-multilingual-encoders-xlm-r-and-mt5.md), [08](../08-cross-lingual-transfer-and-zero-shot.md), [09](../09-scripts-transliteration-and-locale.md).
- Python 3.10+; `transformers`, `datasets`, `evaluate`, `torch`, `scikit-learn`, `sentencepiece`.
- Access to a GPU (`xlm-roberta-large` fine-tunes comfortably on a single L4/A100).

## Task and dataset

**XNLI** (`datasets.load_dataset("xnli", "<lang>", split="test")`) — the canonical cross-lingual NLI benchmark: three-way NLI (entailment / neutral / contradiction) with a shared source (English MultiNLI, professionally translated into 15 languages).

- **Training data:** English **MultiNLI** (`datasets.load_dataset("multi_nli", split="train")`). ~390 k examples. Downsample to 100 k for a runnable-in-an-afternoon exercise.
- **Evaluation:** XNLI **test** split for five languages spanning a resource spectrum:
    - `en` — English (in-distribution reference).
    - `de` — German (high-resource, Latin script).
    - `zh` — Chinese (high-resource, non-Latin, no spaces).
    - `sw` — Swahili (mid-resource).
    - `ur` — Urdu (lower-resource, Arabic-family script).

Alternative task (if NLI feels stale to you): **Amazon Reviews Multi (MARC)** or **IndicSentiment** with 5 target languages of your choosing. The recipe is identical.

## Problem statement

### Part A — Baseline: monolingual English

Fine-tune `FacebookAI/xlm-roberta-large` (or `microsoft/mdeberta-v3-base`) on English MultiNLI. Recipe (chapter 07):

- `AutoModelForSequenceClassification`, `num_labels=3`.
- LR `2e-5`, batch 16, `num_train_epochs=3`, `warmup_ratio=0.06`, `weight_decay=0.01`, `bf16=True`.
- `metric_for_best_model="accuracy"`, `load_best_model_at_end=True`.

Evaluate on **XNLI-en test**. This is your reference monolingual accuracy — everything else compares against it.

Save as `train_en.py` + `report_en.md`.

### Part B — Zero-shot transfer

Take the model from Part A **as-is** and evaluate on the XNLI test splits for `de`, `zh`, `sw`, `ur`. No re-training, no translation.

Report:

- Per-language accuracy and macro-F1.
- 95 % bootstrap CI (over test examples, 1 000 resamples).
- Delta vs. English (accuracy gap).

Save as `zero_shot.py` and `zero_shot_report.md`.

### Part C — Tokenisation-shatter diagnostic

Before trusting zero-shot, run the diagnostic from chapter 08:

- For each of the four non-English languages, tokenise 1 000 held-out sentences (any monolingual corpus for that language works — Wikipedia via `datasets.load_dataset("wikipedia", "20220301.<lang>")` or the XNLI test premise/hypothesis).
- Compute the **average subword pieces per whitespace word**. For Chinese (no spaces), report **tokens per Han character** instead.
- Report the results in a table alongside the per-language zero-shot accuracy from Part B.

Comment on whether the shatter rate predicts the accuracy drop.

Save as `shatter.py`.

### Part D — Translate-train

For each of `de`, `zh`, `sw`, `ur`:

1. Translate the (English) MultiNLI training premise + hypothesis into the target language with **`facebook/nllb-200-distilled-1.3B`**. Sample size: 100 k (translate the same subset used in Part A).
2. Optionally quality-filter with COMET-Kiwi (`Unbabel/wmt22-cometkiwi-da`); drop the bottom decile.
3. Fine-tune a fresh `xlm-roberta-large` on the translated data. Same recipe as Part A.
4. Evaluate on the XNLI test for the target language.

Report per-language accuracy vs. the zero-shot baseline. Include the translation time in the runtime budget.

Save as `translate_train.py`.

### Part E — Translate-test

Take the English-trained model from Part A. At inference:

1. Translate the target-language XNLI test set into English with the same NLLB model.
2. Predict with the English model.
3. Evaluate accuracy against the (language-invariant) gold labels.

Report per-language accuracy alongside zero-shot and translate-train.

Save as `translate_test.py`.

### Part F — Comparison and decision

Produce a comparison table:

| Language | Zero-shot | Translate-train | Translate-test | Monolingual English |

Include CIs and highlight the winning pattern per language. In `decision.md`, write:

- Which pattern wins on which language and why (tokenisation shatter? MT quality? monolingual proxy strength?).
- The trade-off table: per-language accuracy, cost per training run, cost per inference request.
- Recommendation: if you had to ship *one* pattern across all five languages, which and why.

### Part G — Per-language error analysis

Sample 10 misclassified examples from the *worst-performing* language in your best pattern. Classify each error:

- **Model didn't understand the target-language sentence** (tokenisation / representation failure).
- **MT (Part D/E) mistranslated the premise/hypothesis relationship** (translation-induced entailment flip).
- **Ambiguous / noisy label in XNLI** (translation loss in the test set).
- **Genuine hard NLI case.**

Save as `errors.md`.

### Part H — Write-up

500–700 word `report.md` covering:

- Setup (task, dataset, languages, models).
- Metric table across the three patterns and monolingual reference.
- Tokenisation-shatter diagnostic and interpretation.
- Per-language error analysis summary.
- One extension (multi-source joint fine-tuning? XLM-V for low-resource? MAD-X adapters?).

## Starter guidance

- **`xlm-roberta-large` needs `lr=2e-5` and `warmup_ratio=0.06`.** Higher learning rates diverge. If you see NaN losses, check `bf16` vs. `fp16` and consider forcing the classifier head to fp32.
- **Report per-language, never averaged.** The average hides tail failures — the whole point of the exercise.
- **Translate-train ≠ translate-test.** They are different training regimes and different inference regimes; the model files are different.
- **Structured labels survive translation, but examples still change.** NLI premise-hypothesis relationship *usually* survives NLLB translation but can flip on subtle logical connectives; this is a known translate-train failure mode.
- **NLLB code lookup.** `eng_Latn`, `deu_Latn`, `zho_Hans`, `swh_Latn`, `urd_Arab`. Wrong code = silent wrong-language output.
- **Bootstrap CI.** Resample examples with replacement; standard trick with `np.random.choice` — 1 000 resamples is plenty.
- **COMET-Kiwi filter is optional but often adds 1–2 points.** Ablate it if time allows.
- **Translate-test at inference latency counts.** NLLB inference adds hundreds of ms per request; this is why translate-test is often ruled out for latency-sensitive production even when it wins on accuracy.

## Acceptance criteria

- [ ] `train_en.py` fine-tunes `xlm-roberta-large` (or `mdeberta-v3-base`) on English MultiNLI; reports XNLI-en test accuracy with CI.
- [ ] `zero_shot.py` reports per-language accuracy and macro-F1 with CI on `de`, `zh`, `sw`, `ur`.
- [ ] `shatter.py` reports average subword pieces per word (or per Han character) for each non-English language.
- [ ] `translate_train.py` translates 100 k MultiNLI to each of `de`, `zh`, `sw`, `ur` with NLLB, fine-tunes fresh models, reports per-language accuracy.
- [ ] `translate_test.py` translates each target test set to English, predicts with the Part A model, reports per-language accuracy.
- [ ] Comparison table in `decision.md` with a shipped-pattern recommendation.
- [ ] `errors.md` with 10 classified errors from the worst-performing language of the best pattern.
- [ ] `report.md` (500–700 words).

## Stretch goals

- **Multi-source joint fine-tuning.** Mix labelled data from `en`, `de`, `zh`, `sw`, `ur` (translate-train for the non-en ones) into a single training set with temperature sampling ($\tau=5$). Report against the single-source arms.
- **XLM-V or ByT5 for low-resource.** Repeat the Urdu arm with `facebook/xlm-v-base` or `google/byt5-base`. Does the larger vocabulary / byte-level tokeniser help?
- **mDeBERTa-v3-base cost comparison.** Run the entire pipeline with `mdeberta-v3-base` instead of `xlm-roberta-large`. Report the accuracy / cost trade-off.
- **Adapter-based transfer (MAD-X).** Implement per-language adapters with the `adapters` library and compare against full fine-tuning.
- **Distillation.** Distil the best-performing model into a smaller student for serving; report per-language accuracy retention.

## Deliverables

```
train_en.py
zero_shot.py
shatter.py
translate_train.py
translate_test.py
decision.md
errors.md
report.md
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
