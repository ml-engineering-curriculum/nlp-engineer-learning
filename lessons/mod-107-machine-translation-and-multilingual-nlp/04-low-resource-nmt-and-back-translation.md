# Low-Resource NMT and Back-Translation

## Motivation

The high-resource recipe from chapter 03 assumes you have millions of parallel sentence pairs. Most of the world's ~7 000 languages have nothing of the sort — and even a well-supported "mid-resource" pair like English ↔ Amharic or English ↔ Icelandic falls off the recipe's assumptions. This chapter walks through the low-resource playbook: pick a massively multilingual base, mine or augment more parallel data, exploit monolingual data with back-translation, transfer from a related language, and evaluate honestly with metrics that do not lie on scarce references.

Almost every production launch into an underserved market runs through this playbook. Getting it right is what separates a translator that "kind of works" from one that ships.

## Where you start: pick a base with the language already in it

For low-resource fine-tuning, the *base model* choice is far more consequential than the training loop.

- **NLLB-200** covers 200 languages, most of which have < 10 M parallel training pairs in NLLB's own data. It is the default first pick when your target pair is in its coverage list — the pretraining has already done most of the work for you.
- **M2M-100** covers 100 languages; use only if you specifically need one that NLLB does not.
- **mBART-50** covers 50 languages; older; use if NLLB has a memory/latency ceiling you cannot afford.
- **Marian OPUS-MT** has thousands of bilingual checkpoints, including for pairs that fell out of newer models' coverage (Faroese, Cornish, Māori). Worth checking the [OPUS-MT dashboard](https://opus.nlpl.eu/dashboard/) for your pair.
- **From scratch.** Almost never the right call for a low-resource pair. If NLLB / M2M do not cover your language, prefer transferring from a *related* covered language (e.g. Kinyarwanda → Kirundi, Icelandic → Faroese) over training from zero.

The rule of thumb: if the language is in the model card's coverage list and has a chrF > 20 on FLORES, treat it as a fine-tuning problem, not a from-scratch problem.

## Data augmentation: what actually works

Four techniques dominate. All four are worth trying; the wins compound.

### 1. Back-translation

The single most reliable low-resource technique — invented in the phrase-based era, formalised for NMT by Sennrich et al., ["Improving Neural Machine Translation Models with Monolingual Data"](https://arxiv.org/abs/1511.06709) (*ACL 2016*). The idea:

1. Train (or take pre-trained) a reverse-direction model $M_{\text{tgt}\to\text{src}}$.
2. Take **monolingual target-language** text $\{y_j\}$ (usually plentiful even for low-resource languages — Common Crawl, Wikipedia, national news).
3. Translate it back to the source with $M_{\text{tgt}\to\text{src}}$, producing synthetic pairs $\{(\hat{x}_j, y_j)\}$.
4. Concatenate synthetic pairs with real parallel pairs; fine-tune $M_{\text{src}\to\text{tgt}}$.

The synthetic source is noisy but the target is *human*, so cross-entropy on it teaches the target-side language model without polluting the target distribution. Two subtleties matter:

- **Ratio of synthetic to real.** Common range: 1× to 5×. Above 5× the model starts overfitting to the reverse-model's artefacts. Start at 1×, sweep.
- **Sampled vs. beam back-translation.** Edunov et al., ["Understanding Back-Translation at Scale"](https://arxiv.org/abs/1808.09381) (*EMNLP 2018*) showed that *sampled* back-translations (temperature sampling from the reverse model) beat beam-search back-translations at scale — the sampled synthetic sources are more diverse and better regularise the forward model. For low-resource, sampled with `top_p=0.9` is a good default.

**Iterative back-translation.** Alternate rounds — train forward, use it to back-translate more monolingual source data, train reverse, use it to back-translate more monolingual target data, retrain forward. Diminishing returns after 2–3 rounds; the first round buys most of the win.

### 2. Forward-translation (self-training)

The mirror of back-translation. Take monolingual **source** text, forward-translate to the target with the current model, filter with a quality-estimation metric (COMET-QE), and add to training. Usually complements back-translation, does not replace it. Useful when the source-side monolingual pool is much larger than the target-side.

### 3. Data mining from web crawls

The systems that top WMT and NLLB were trained on **mined parallel data** — sentence pairs discovered by aligning monolingual crawls with multilingual embeddings. The community-standard corpora:

- **CCMatrix** — Fan et al., ["CCMatrix: Mining Billions of High-Quality Parallel Sentences on the Web"](https://arxiv.org/abs/1911.04944) (*ACL 2021*). 4.5 B sentence pairs across ~90 language pairs, mined with LASER.
- **CCAligned** — El-Kishky et al., "A Massive Collection of Cross-Lingual Web-Document Pairs" (*EMNLP 2020*). Document-aligned parallel web pages.
- **NLLB "NLLB-Seed" and mined bitext** — released alongside NLLB-200; the mined subset covers many pairs otherwise unseen in public corpora.
- **OPUS** — the umbrella project (Tiedemann, 2012+) aggregating hundreds of open corpora: Paracrawl, TED talks, Ubuntu manuals, KDE localisations, Bible translations. Not clean by default; expect to filter.

For a new production launch: pull everything OPUS has for the pair, add NLLB's mined data if available, filter with Bicleaner AI, deduplicate against your test sets, and train on top.

### 4. Transfer from a related language

If your target is Kinyarwanda (low-resource) and Swahili (higher-resource) exists in your model with high quality, warm-start your fine-tune from a Swahili-tuned checkpoint (or fine-tune first on Swahili, then on Kinyarwanda). For scripts and closely-related morphology, transfer buys 5+ chrF on very low-resource pairs. Details: Kocmi & Bojar, ["Trivial Transfer Learning for Low-Resource Neural Machine Translation"](https://arxiv.org/abs/1809.00357) (*WMT 2018*); Neubig & Hu, ["Rapid Adaptation of Neural Machine Translation to New Languages"](https://arxiv.org/abs/1808.04189) (*EMNLP 2018*).

## Back-translation, worked

Concrete recipe for English → Swahili given (1) 20 k real parallel pairs and (2) 2 M monolingual Swahili sentences from a web crawl:

```python
# Step 1 — reverse model (Swahili → English). Use NLLB straight off the shelf.
sw_to_en = build_translator("facebook/nllb-200-distilled-600M",
                            src_lang="swh_Latn", tgt_lang="eng_Latn")

# Step 2 — sampled back-translation for diversity.
synthetic = []
for batch in chunk(monolingual_swahili, 32):
    src_hat = sw_to_en(batch, do_sample=True, top_p=0.9, temperature=1.0,
                        max_new_tokens=128)
    synthetic.extend(zip(src_hat, batch))

# Step 3 — combine with real parallel data. 1:1 ratio to start.
real   = load_parallel_en_sw()             # 20 k pairs
synth  = random.sample(synthetic, 20_000)   # match at 1× first
combined = real + synth
random.shuffle(combined)

# Step 4 — fine-tune forward. Same recipe as chapter 03; NLLB base.
```

Two guardrails:

- **Do not tag synthetic pairs by default.** Some recipes prepend a special `<BT>` token to the synthetic source so the model can distinguish. In practice, at 1–5× ratios and with a strong reverse model, tagging rarely helps and complicates inference. Consider only if you go above 5×.
- **Quality-filter the synthetic pairs.** Cheap filter: length ratio outside `[0.5, 2.0]` → drop. Better filter: COMET-QE score (chapter 10) on the synthetic pair; drop the bottom 20 %.

## Curriculum learning

For very low-resource pairs, ordering the training data matters. A common shape:

1. **Pretrain on all mined + back-translated data** with a smaller learning rate.
2. **Fine-tune on real parallel data only** with a larger learning rate for the final few epochs.

The "clean-tune" phase pushes the model back toward the real reference distribution. This is what most WMT low-resource submissions do — treat it as a default, not an ablation.

## Cross-lingual transfer from related languages

Two patterns that work in practice:

- **Warm-start from a related-language fine-tune.** If Amharic ↔ English is your target and Tigrinya ↔ English data exists, first fine-tune on Tigrinya, then continue on Amharic. The shared Ge'ez script and cognate vocabulary transfer.
- **Multilingual joint training with sampling temperature.** Fine-tune on {Amharic, Tigrinya, Oromo, Somali} → English *simultaneously* with sentence-count-weighted sampling raised by a temperature $\tau$:

  $$p(L) \propto \left(\frac{n_L}{\sum_{L'} n_{L'}}\right)^{1/\tau}$$

  Setting $\tau = 5$ upweights low-resource languages substantially without collapsing them. The recipe is from Arivazhagan et al., ["Massively Multilingual Neural Machine Translation in the Wild"](https://arxiv.org/abs/1907.05019) (2019) and is what NLLB uses internally.

## Parameter-efficient fine-tuning for low resource

Full-parameter fine-tuning on a 600 M NLLB checkpoint with 20 k parallel pairs is a recipe for overfitting. Two mitigations:

- **LoRA on the transformer blocks** (Hu et al., ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685), *ICLR 2022*). Freeze the base; add rank-8 or rank-16 adapters to `q_proj`, `v_proj` (optionally also `k_proj`, `out_proj`). Trains in ~15 minutes on a single A100 for 20 k pairs; often *outperforms* full fine-tune on very low resource because it does not disturb the base.
- **Adapter modules** (Bapna & Firat, ["Simple, Scalable Adaptation for Neural Machine Translation"](https://arxiv.org/abs/1909.08478), *EMNLP 2019*). Bottleneck feed-forward modules inserted after each transformer block; original per-language adapter recipe for MT. Serves the same purpose as LoRA but with a slightly larger overhead and better historical evaluation on MT specifically.

Chapter 05 covers both in more depth for the *domain* adaptation case — the techniques are the same.

## Evaluation on low-resource: mind your metric

Chapter 10 covers this in depth. Two immediate points:

- **BLEU is unreliable on low-resource.** Small reference sets, morphologically-rich targets, and non-word-segmented scripts all break BLEU. Report **chrF** and **COMET** as primary; BLEU as secondary if at all.
- **FLORES-200 is often your only shared yardstick.** Even if your internal evaluation set is domain-specific, report FLORES numbers alongside for reproducibility.
- **Human evaluation is scarce and expensive.** Direct Assessment (DA) on 200–500 sentences with 2–3 raters per sentence is the WMT-track default when budget allows. Chapter 11.

## Unsupervised NMT — where the line is

For genuinely zero parallel data, **unsupervised NMT** (Artetxe et al., ["Unsupervised Neural Machine Translation"](https://arxiv.org/abs/1710.11041), *ICLR 2018*; Lample et al., ["Unsupervised Machine Translation Using Monolingual Corpora Only"](https://arxiv.org/abs/1711.00043), *ICLR 2018*) uses shared multilingual embeddings, denoising autoencoding, and iterative back-translation to bootstrap a translator from monolingual data alone. In practice, in 2026, massively multilingual pretrained models (NLLB, mBART) subsume unsupervised NMT for most pairs — they already have some parallel signal internally. Keep unsupervised NMT in mind for truly isolated languages not covered by any pretrained model.

## Common failure modes and their fixes

- **Overfitting on 20 k pairs.** Full fine-tune of a 600 M model → catastrophic memorisation. Switch to LoRA / adapters, or freeze the base embedding and train the transformer blocks only.
- **Back-translation degrades on-target quality.** Ratio is too high (drop to 1×) or the reverse model is too weak (upgrade to a bigger NLLB, or filter with COMET-QE).
- **Synthetic sources dominate the target-side language model.** Some pairs have a monolingual pool that is orders of magnitude larger than the parallel corpus. Cap synthetic-to-real ratio at 5:1; do not just throw all monolingual data at it.
- **BLEU goes up, chrF goes down.** You have picked up an artefact — often over-punctuation or over-translation. Trust chrF and inspect predictions.
- **Model translates into the wrong dialect.** Common when the language has multiple written forms (Serbian Cyrillic vs. Latin; Traditional vs. Simplified Chinese; Egyptian vs. Modern Standard Arabic). Fix at data-tagging time — enforce script/dialect in the language code and in the training data. NLLB's `zho_Hans` vs. `zho_Hant` and `arb_Arab` vs. `ary_Arab` is exactly this.

## Chapter summary

- The low-resource playbook starts with the *base model*: NLLB-200 is the modern default; check its coverage list first.
- Back-translation is the highest-leverage technique for the mid-resource tier. Use *sampled* back-translation (`top_p=0.9`) at 1–5× the real-parallel volume; filter with length ratio and COMET-QE.
- Data mining (CCMatrix, CCAligned, OPUS) and transfer from related languages are the other two big-ticket wins.
- For very-low-resource (< 20 k pairs), full fine-tuning overfits — use LoRA or adapter-based fine-tuning to preserve the base.
- Curriculum learning (mixed → clean-tune) and multilingual joint training with temperature-sampling are what WMT-style submissions actually do.
- Evaluate with chrF and COMET, not BLEU alone. Report FLORES numbers for reproducibility even if your production distribution is different.
