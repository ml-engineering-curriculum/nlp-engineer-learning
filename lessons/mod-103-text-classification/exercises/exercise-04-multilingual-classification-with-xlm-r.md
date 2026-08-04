# exercise-04: Multilingual Classification with XLM-R

**Estimated effort:** 2 hours

## Objective

Fine-tune XLM-R on a multilingual classification task, evaluate zero-shot cross-lingual transfer, add language-aware sampling to the training loop, and compare against per-language monolingual baselines. Produce a per-language metrics breakdown that survives scrutiny — no single-number aggregates hiding failure on languages you can't read.

## Prerequisites

- Chapter [08](../08-multilingual-classification-with-xlm-r.md); familiarity with chapters [04](../04-fine-tuning-encoders-bert-roberta-deberta-xlmr.md), [06](../06-class-imbalance-weighting-sampling-and-focal-loss.md), and [07](../07-calibration-and-threshold-tuning.md).
- Python 3.10+; `transformers`, `datasets`, `torch`. GPU strongly recommended.

## Dataset

Pick a multilingual classification dataset. Options:

- **XNLI** (Conneau et al., ["XNLI: Evaluating Cross-lingual Sentence Representations"](https://arxiv.org/abs/1809.05053), *EMNLP 2018*) — natural language inference in 15 languages. Available as `load_dataset("xnli", <lang>)`.
- **PAWS-X** (Yang et al., ["PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification"](https://arxiv.org/abs/1908.11828), *EMNLP-IJCNLP 2019*) — paraphrase identification in 7 languages. `load_dataset("paws-x", <lang>)`.
- **MARC** (Keung et al., ["The Multilingual Amazon Reviews Corpus"](https://arxiv.org/abs/2010.02573), *EMNLP 2020*) — 5-class review rating in 6 languages.
- **MASSIVE** (FitzGerald et al., ["MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages"](https://arxiv.org/abs/2204.08582), *ACL 2023*) — intent classification in 51 languages.

Pick at least 5 languages including a mix of high-resource (English, German, French, Spanish, Chinese) and lower-resource (Swahili, Urdu, Thai, Kazakh — whichever your dataset covers). If your dataset covers only 6–7 languages, use all of them.

## Problem statement

### Part A — Zero-shot cross-lingual transfer

1. Fine-tune `xlm-roberta-base` on **English only** using the chapter 08 recipe. Reasonable defaults: `lr=1e-5`, 3 epochs, `warmup_ratio=0.1`, `weight_decay=0.01`, `max_length=128`, `fp16=True`.
2. Evaluate on every target language's test set.
3. Produce a per-language metrics table: accuracy, macro-F1, minority-class F1. Include mean and min across languages.
4. Rank languages by transfer quality. Which lose the most vs. English? Is that correlated with pretraining data size (see the XLM-R paper's language table) or typological distance?

### Part B — Multilingual fine-tuning with uniform sampling

1. Now fine-tune on the **concatenation** of all target-language train sets with uniform batch composition (i.e., `WeightedRandomSampler` is not used — each example weighted equally).
2. Evaluate on every target language's test set.
3. Compare per-language F1 to Part A. Which languages improve? Which get worse?
4. Note whether English quality drops — this is the "curse of multilinguality" showing up locally.

### Part C — Language-aware sampling

1. Rerun Part B's training with temperature-smoothed language sampling at `α = 0.3` (Conneau et al. 2020's pretraining recipe). Use the `WeightedRandomSampler` snippet from chapter 08 as a starting point.
2. Also run `α = 0.7` (closer to proportional) as an ablation.
3. Compare all three multilingual configurations on the per-language table:
   - Uniform (Part B)
   - `α = 0.3`
   - `α = 0.7`
4. Report `mean`, `min`, and `max` per-language F1 across languages for each configuration. Which `α` gives the best `min` — the "worst-user experience" number?

### Part D — Monolingual baselines

For at least **two** target languages (one high-resource, one lower-resource), fine-tune a monolingual encoder as a baseline:

- Language-specific models: `bert-base-german-cased` (de), `camembert-base` (fr), `bert-base-japanese-v3` (ja), etc. Fall back to `xlm-roberta-base` fine-tuned on that language only if a monolingual model doesn't exist.
- Match hyperparameters to Part A.

Compare each monolingual baseline to the multilingual model's per-language number. On the high-resource language, does the monolingual model win? By how much? On the lower-resource language?

### Part E — Tokeniser diagnostic

Pick one lower-resource language. Sample 100 examples and:

1. Tokenise with the XLM-R tokenizer. Compute mean and median tokens per example.
2. Compute the same for English on 100 English examples.
3. Report the ratio. A ratio above ~1.5× means XLM-R under-tokenises that language, which caps quality regardless of fine-tuning.

If your chosen language shows severe under-tokenisation, discuss what you would do: continue pretraining, switch to XLM-V (Liang et al. 2023), route to a monolingual model, or accept the ceiling.

### Part F — Write-up

A 400–700 word write-up covering:

- The zero-shot vs. multilingual-training vs. `α`-swept comparison, with per-language numbers.
- Whether language-aware sampling helped, and on which languages.
- Whether monolingual baselines won on their language, and under what deployment scenario you would ship them instead.
- The tokeniser diagnostic result for your lowest-resource language and what you would do about it.
- Any language where a single global threshold from the multilingual model is clearly wrong (chapter 07 material — per-language calibration might belong in a follow-up).

## Starter guidance

- Zero-shot transfer with `xlm-roberta-base` on XNLI is well-studied; the paper reports specific numbers per language. Do not try to match them exactly — reproduce the *shape* of the result, not the absolute values.
- Language-aware sampling changes what your model sees per epoch. If you keep the number of gradient steps fixed, the model sees *fewer* examples per language than uniform sampling would allow. Report training loss curves per language to see where each language plateaus.
- Multilingual fine-tuning is sensitive to learning rate. If Part B/C loss stalls, drop LR to `5e-6`. XLM-R base can be finicky; XLM-R large is often more stable but slower.
- Do not evaluate on languages that the model has never seen at fine-tune time and then report the aggregate as if it were the same setup as Part B. Be explicit about which languages you fine-tuned on and which are held out.
- If a language shows near-chance F1, check first that (a) the tokenizer handled it (Part E), (b) labels weren't accidentally shuffled, and (c) the language string in your dataset matches the split you loaded.

## Acceptance criteria

- [ ] Fine-tuned `xlm-roberta-base` for at least three configurations: English-only (Part A), multilingual-uniform (Part B), multilingual with `α = 0.3` (Part C).
- [ ] Per-language metrics table for all three, plus `mean`, `min`, `max` across languages.
- [ ] Two monolingual baselines with per-language comparison against the multilingual model.
- [ ] Tokeniser diagnostic for at least one lower-resource language: mean/median tokens per example vs. English.
- [ ] 400–700 word write-up covering the shipping recommendation (multilingual vs. monolingual per language) and how language-aware sampling changed the story.

## Stretch goals

- **`α`-sweep.** Sweep `α ∈ {0.0, 0.3, 0.5, 0.7, 1.0}` and plot per-language F1 vs. `α`. Which `α` maximises the min? The mean? Are those the same?
- **Per-language calibration.** Apply chapter 07's temperature scaling per language on a held-out calibration set. Does per-language calibration substantially beat a single global temperature?
- **mDeBERTa-v3 vs. XLM-R base.** Repeat Parts A–C with `microsoft/mdeberta-v3-base`. Does it beat XLM-R on your languages? On which?
- **XLM-V.** If your lower-resource language showed severe tokenizer under-fitting in Part E, try `facebook/xlm-v-base` (Liang et al. 2023) and compare tokens-per-example plus downstream F1.
- **Continue pretraining.** Take XLM-R base and continue MLM pretraining on a monolingual corpus of your under-resourced language (100 MB–1 GB is enough to see movement). Then fine-tune and evaluate. Does adapted-XLM-R beat vanilla XLM-R on that language? By how much?
- **Cross-lingual data augmentation.** Machine-translate your English training set into your target languages and add the translations to the training pool. Compare against Part C. When does this help vs. hurt?
