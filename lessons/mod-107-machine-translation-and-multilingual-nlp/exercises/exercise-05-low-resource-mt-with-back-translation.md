# exercise-05: Low-Resource MT with Back-Translation

**Estimated effort:** 2 hours

## Objective

Fine-tune a massively-multilingual NMT model (`facebook/nllb-200-distilled-600M`) on a **low-resource language pair** using the standard low-resource playbook — start from a small real-parallel seed, augment with **sampled back-translation** from monolingual target-language text, then compare the augmented model against (a) the pretrained NLLB baseline and (b) a fine-tune on only the real-parallel data. Report on **FLORES-200 devtest** with chrF + COMET, and diagnose at least one augmentation failure mode.

## Prerequisites

- Chapters [02](../02-encoder-decoder-nmt-architectures.md), [04](../04-low-resource-nmt-and-back-translation.md), [10](../10-mt-automatic-evaluation-bleu-chrf-comet-bleurt.md), [12](../12-multilingual-benchmarks-flores-and-friends.md).
- Python 3.10+; `transformers`, `datasets`, `evaluate`, `unbabel-comet`, `sacrebleu`, `torch`, `peft` (for the LoRA option), `sentencepiece`.
- One GPU (fine-tuning `nllb-200-distilled-600M` with LoRA fits on an L4 or single A100).

## Language pair and data

Pick **one low-resource pair** covered by NLLB with FLORES support. Recommended options:

- **English ↔ Swahili** (`eng_Latn` ↔ `swh_Latn`). Manageable parallel data + huge monolingual target corpus.
- **English ↔ Yoruba** (`eng_Latn` ↔ `yor_Latn`). Truly low-resource; MasakhaNER community has strong references.
- **English ↔ Kinyarwanda** (`eng_Latn` ↔ `kin_Latn`).
- **English ↔ Icelandic** (`eng_Latn` ↔ `isl_Latn`). Mid-resource; a good "less-severe" alternative.

**Real parallel seed** — pick one:

- `Helsinki-NLP/opus-100` for the pair (`en-sw`, `en-yo`, etc.); downsample to 20 k pairs to simulate a low-resource regime.
- `MAFAND-MT` (Adelani et al., 2022) for African languages.

**Monolingual target-language data** for back-translation:

- Wikipedia dump for the target (`datasets.load_dataset("wikimedia/wikipedia", "20231101.<lang>", split="train")`), sentence-split and downsampled to ~40 k sentences.
- Or the target-language side of OPUS-100 stripped of alignment info; or CC-100 / OSCAR filtered by language.

Deduplicate against FLORES-200 devtest source and reference sides before use.

**Evaluation:** FLORES-200 devtest for the pair (both directions if time allows; primary is `eng → target`).

## Problem statement

### Part A — Baseline: pretrained NLLB, no fine-tune

Evaluate `facebook/nllb-200-distilled-600M` **as-is** on FLORES-200 devtest for the chosen pair. Report:

- SacreBLEU (`--tokenize 13a`, or the right tokenizer for the target).
- chrF++.
- COMET (`Unbabel/wmt22-comet-da`).
- 95 % bootstrap CI on each.

Save as `baseline.py` + `baseline_report.md`.

### Part B — Fine-tune on real parallel data only

Fine-tune NLLB on the 20 k real parallel pairs. Because 20 k pairs on a 600 M model overfits catastrophically with a full fine-tune, use **LoRA** (chapter 04):

- `peft` with `LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])`.
- LR `3e-4` (LoRA tolerates higher LR), effective batch 16, `num_train_epochs=3`, `warmup_ratio=0.1`.
- `DataCollatorForSeq2Seq`, `label_smoothing_factor=0.1`.

Save training logs and adapter weights. Evaluate on FLORES-200 devtest with the same metric panel as Part A.

Save as `train_real_only.py` + `report_real_only.md`.

### Part C — Back-translation

Generate synthetic parallel data:

1. Use the pretrained NLLB (`facebook/nllb-200-distilled-600M`) in the **reverse direction** (target → English) to translate the 40 k monolingual target sentences.
2. Use **sampled** back-translation, not beam (Edunov et al., 2018): `do_sample=True`, `top_p=0.9`, `temperature=1.0`, `max_new_tokens=128`.
3. **Length-ratio filter.** Drop pairs where `len(hyp_src) / len(mono_tgt)` is outside `[0.5, 2.0]`.
4. **Optional COMET-Kiwi filter.** Score each `(hyp_src, mono_tgt)` pair with `Unbabel/wmt22-cometkiwi-da`; drop the bottom 20 %.
5. Report how many synthetic pairs remain after filtering.

Save the synthetic corpus as `back_translated.jsonl`; the generation script as `back_translate.py`.

### Part D — Fine-tune on real + back-translated data

Fine-tune a **fresh** NLLB with LoRA on the union of:

- 20 k real parallel pairs.
- Up to 20 k, then up to 40 k synthetic back-translated pairs (two runs — a 1× ratio and a 2× ratio).

Same LoRA recipe as Part B; use the same total number of gradient steps if possible so the comparison is fair.

Evaluate both runs on FLORES-200 devtest with the same metric panel.

Save as `train_bt.py` + `report_bt.md`.

### Part E — Comparison

Produce the table:

| System | Fine-tune data | SacreBLEU (CI) | chrF++ (CI) | COMET (CI) |
|--------|----------------|----------------|-------------|------------|
| A. Pretrained NLLB | — | | | |
| B. LoRA on real only | 20 k real | | | |
| C. LoRA on real + BT 1× | 20 k real + 20 k BT | | | |
| D. LoRA on real + BT 2× | 20 k real + 40 k BT | | | |

Include paired-bootstrap significance vs. baseline and vs. real-only for each metric. Bold the winner on each metric.

### Part F — Failure diagnosis

Sample **10 outputs** from the best-performing model where the COMET score is below the 10th percentile of the eval set. For each:

- Show the source, reference, and hypothesis.
- Classify the failure: **hallucination**, **dropped content**, **wrong-dialect / wrong-register**, **script / diacritic issue**, or **fluent but off-topic**.
- Note if it correlates with any characteristic of the back-translated corpus (e.g. the model over-uses phrases from the mono corpus).

Save as `failures.md`.

Also sample **10 back-translated pairs** at random and manually inspect. Rate each 1–5 on plausibility of the synthetic source. Report the mean and any observed pathologies (repetition, hallucination, dropped content).

Save as `bt_quality.md`.

### Part G — Write-up

500–700 word `report.md` covering:

- Language pair, dataset sizes, model, LoRA hyperparameters, training runtime.
- Baseline / real-only / BT 1× / BT 2× comparison with CIs and paired-bootstrap p-values.
- Effect of the COMET-Kiwi filter on the BT corpus (if you ablated it).
- Failure-mode summary from Part F.
- Back-translation quality note from Part F.
- One "what next" (curriculum learning? iterative BT? transfer from a related language?).

## Starter guidance

- **Do not full-fine-tune a 600 M model on 20 k pairs.** It will overfit; the LoRA adapter is not optional here.
- **NLLB reverse-direction plumbing.** `tokenizer.src_lang = "swh_Latn"` and `forced_bos_token_id = tok.convert_tokens_to_ids("eng_Latn")`. Wrap in the helper from chapter 02.
- **Sampled BT, not beam.** Beam back-translation collapses diversity and reduces the regularisation benefit (Edunov et al., 2018).
- **Deduplicate the monolingual pool against FLORES.** Otherwise your evaluation set leaks into training as (BT source, gold target) synthetic pairs and your numbers are inflated.
- **Length-ratio filter is cheap and catches a lot of noise.** Do it before you spend cycles on the COMET-Kiwi filter.
- **Do not tag synthetic pairs by default at 1× or 2× ratio.** Tagging (`<BT>`-prefix) starts to help above 5× (Caswell et al., 2019) — outside this exercise's scope.
- **Report chrF as primary, not BLEU.** For a low-resource / morphologically rich target, BLEU is noisy.
- **Watch for dialect drift.** For pairs with multiple scripts or standards (Serbian, Chinese, Arabic), enforce script/dialect in the NLLB language code. If you pick a pair like this, note it explicitly in the report.

## Acceptance criteria

- [ ] Baseline (pretrained NLLB) SacreBLEU + chrF++ + COMET on FLORES-200 devtest with 95 % bootstrap CIs.
- [ ] `train_real_only.py` fine-tunes with LoRA on the 20 k real parallel pairs; adapter weights saved.
- [ ] `back_translate.py` generates ≥ 20 k synthetic pairs with sampled back-translation and length-ratio filtering; reports before/after counts.
- [ ] `train_bt.py` fine-tunes two runs — real + BT 1× and real + BT 2× — with matched LoRA config.
- [ ] Comparison table with CIs and paired-bootstrap significance for baseline / real-only / BT 1× / BT 2×.
- [ ] `failures.md` with 10 low-COMET outputs classified.
- [ ] `bt_quality.md` with 10 hand-rated back-translated pairs.
- [ ] `report.md` (500–700 words).

## Stretch goals

- **COMET-Kiwi filter ablation.** Repeat the BT 2× run with and without the bottom-20 % COMET-Kiwi filter. Report the delta.
- **Iterative back-translation.** After Part D, use the trained model to back-translate a fresh monolingual batch; retrain. Does round 2 help?
- **Transfer from a related language.** Warm-start the LoRA from a Swahili fine-tune before fine-tuning on Kinyarwanda (or Icelandic → Faroese). Report the win.
- **Full fine-tune vs. LoRA head-to-head at low resource.** Just to feel the overfitting; expect worse chrF than LoRA.
- **Curriculum: pretrain on mixed, clean-tune on real.** Train first on real+BT then a final few epochs on real only. Chapter 04 predicts a small win.
- **Both directions.** Fine-tune the reverse direction (target → English) with the same recipe and report both.

## Deliverables

```
baseline.py
train_real_only.py
back_translate.py
back_translated.jsonl
train_bt.py
failures.md
bt_quality.md
report.md
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
