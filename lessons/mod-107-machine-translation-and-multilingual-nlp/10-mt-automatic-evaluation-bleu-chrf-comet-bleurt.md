# Automatic MT Evaluation: BLEU, chrF, COMET, BLEURT

## Motivation

Every serious MT paper reports at least three metrics and every serious MT deployment monitors at least two. The reason: no single metric captures translation quality across languages, domains, and error types. BLEU became famous and became infamous — infamous because it is unstable across languages, tokenisation, and even punctuation. chrF is more robust but still surface-level. Learned metrics — COMET and BLEURT — correlate much better with human judgement but require inference on a neural model to score.

This chapter walks the metrics that ship: **BLEU** (SacreBLEU), **chrF**, **COMET**, **BLEURT**, plus the reference-free variants (COMET-QE, TransQuest) that let you score outputs without a reference. The goal is not to memorise the formulas; it is to know which metric to reach for when, and — crucially — how to report a number that another team can reproduce.

## The classical metrics: BLEU and chrF

### BLEU

**BLEU** (Papineni et al., ["BLEU: A Method for Automatic Evaluation of Machine Translation"](https://aclanthology.org/P02-1040/), *ACL 2002*) is a modified n-gram precision score with a brevity penalty:

$$\text{BLEU} = \text{BP} \cdot \exp\left( \sum_{n=1}^{N} w_n \log p_n \right)$$

where $p_n$ is the clipped precision of $n$-grams (usually $n = 1, 2, 3, 4$) between candidate and reference, and $\text{BP}$ penalises translations shorter than the reference. It is a corpus-level metric — micro-averaged over all sentence n-grams.

The problems are well-catalogued:

- **Sensitive to tokenisation.** Different tokenisers produce different BLEU scores on the same output. You can (and papers do) game BLEU by tweaking tokenisation. The industry response is **SacreBLEU** (Post, ["A Call for Clarity in Reporting BLEU Scores"](https://arxiv.org/abs/1804.08771), *WMT 2018*) — pins the tokeniser, reports a signature string, and lets any team reproduce the number.
- **Word-based precision.** Broken for languages without spaces (Chinese, Japanese, Thai) and unfair to morphologically rich languages (Turkish, Finnish, Arabic) where a single semantic unit is one word to English but a whole clause to the target.
- **Poor per-sentence discrimination.** BLEU was designed to be corpus-level; sentence-BLEU exists but correlates poorly with judgement.
- **Insensitive to paraphrase.** "The cat sat on the mat" and "The feline was on the rug" score near-zero BLEU against each other.

Always use **SacreBLEU** in code, never `nltk.translate.bleu_score.corpus_bleu`. Always report the SacreBLEU signature string alongside the number:

```python
import evaluate
sacrebleu = evaluate.load("sacrebleu")

res = sacrebleu.compute(
    predictions=hyps,
    references=[[r] for r in refs],   # references is a list-of-lists
    tokenize="13a",                    # default; "zh" for Chinese, "ja-mecab" for Japanese
)
print(res["score"])                    # 27.4
print(res.get("verbose"))              # includes brevity, precisions, ratio
```

For Chinese use `tokenize="zh"`; for Japanese `tokenize="ja-mecab"` (requires `mecab-python3`); for other CJK/Thai variants check the SacreBLEU docs.

### chrF and chrF++

**chrF** (Popović, ["chrF: character n-gram F-score for automatic MT evaluation"](https://aclanthology.org/W15-3049/), *WMT 2015*) computes an F-score over character n-grams (default $n = 6$) rather than word n-grams. **chrF++** adds a small word-n-gram term for stability.

Two big advantages over BLEU:

- **Tokenisation-invariant.** No tokeniser needed; character n-grams are language-agnostic.
- **Better for morphologically rich languages.** Character-level matching gives partial credit for morphological variants ("dog" ↔ "dogs" ↔ "dog's") without hard-matching failure.

chrF is the modern default for MT metric reporting across languages — WMT recommends chrF as the *primary* surface metric for shared tasks. Report chrF++ (or plain chrF, both are fine — always specify) alongside BLEU:

```python
chrf = evaluate.load("chrf")
res = chrf.compute(predictions=hyps, references=[[r] for r in refs], word_order=2)
# word_order=2 → chrF++; word_order=0 → chrF
```

### TER, WER, and CER

- **TER** (Translation Edit Rate; Snover et al., 2006). Number of edits (insertions, deletions, substitutions, shifts) to transform the candidate into the reference, normalised by reference length. Historically used in post-editing pipelines (translator productivity) — every edit is proportional to work saved.
- **WER / CER** (Word / Character Error Rate). Common in speech but occasionally used in MT for very short strings (UI labels).

TER remains useful in post-editing metrics but is not typically reported for research MT. Report if your product includes a human post-editing loop.

## Learned metrics: COMET and BLEURT

The state of the art in MT evaluation is *learned* — train a neural model on human quality judgements and use it to score. Two families dominate.

### COMET

**COMET** (Rei et al., ["COMET: A Neural Framework for MT Evaluation"](https://arxiv.org/abs/2009.09025), *EMNLP 2020*, with reference-based and reference-free variants at [`Unbabel/COMET`](https://github.com/Unbabel/COMET)) is the current de-facto learned metric. Architecture: XLM-R encodes (source, translation, reference), a small feedforward head predicts a scalar quality score trained on WMT direct-assessment judgements.

Checkpoints on Hugging Face / Unbabel:

- **`Unbabel/wmt22-comet-da`** — reference-based, trained on WMT 2017–2021 DA data. The default reference-based COMET.
- **`Unbabel/wmt22-cometkiwi-da`** — reference-free (source + hypothesis only). "COMET-Kiwi." Use when references are unavailable.
- **`Unbabel/XCOMET-XL`** — the newer XCOMET family (Guerreiro et al., 2023) that also predicts sentence-level error spans (MQM-style).

```python
from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(srcs, mts, refs)]
result = model.predict(data, batch_size=8, gpus=1)
# result.scores[0] ∈ [~0, ~1] approximate, but the raw range depends on the checkpoint.
```

Two properties worth internalising:

- **Segment-level correlation with human judgement is much higher than BLEU or chrF.** WMT metric shared tasks have shown COMET-family metrics as leaders since ~2020.
- **Report the checkpoint.** "COMET" without a checkpoint name is not reproducible. `wmt22-comet-da` and `Unbabel/wmt20-comet-da` produce different numbers on the same input.

### BLEURT

**BLEURT** (Sellam et al., ["BLEURT: Learning Robust Metrics for Text Generation"](https://arxiv.org/abs/2004.04696), *ACL 2020*) is a BERT-based learned metric fine-tuned on WMT quality judgements. Similar to COMET conceptually; different implementation lineage. The `BLEURT-20` checkpoint (Pu et al., 2021) is the current recommended version.

BLEURT is less commonly reported in 2020s MT papers than COMET but is worth knowing — some downstream teams standardised on it and its correlations with human judgement are competitive.

```python
# via evaluate
bleurt = evaluate.load("bleurt", "BLEURT-20")
res = bleurt.compute(predictions=hyps, references=refs)  # note: no source
```

Notably, BLEURT is reference-based only and does not take a source. For reference-free scoring, use COMET-Kiwi.

## Reference-free (quality estimation)

When references are unavailable — production monitoring, back-translation filtering, glossary compliance during decoding — reach for **quality-estimation (QE)** metrics that score `(source, hypothesis)` without a reference.

- **COMET-Kiwi (`Unbabel/wmt22-cometkiwi-da`).** The workhorse. Same architecture as COMET, trained on WMT QE data.
- **TransQuest / MonoTransQuest** (Ranasinghe et al., 2020). Predecessor family; still used for some low-resource pairs.
- **LLM-as-judge for MT.** GEMBA (Kocmi & Federmann, ["Large Language Models Are State-of-the-Art Evaluators of Translation Quality"](https://arxiv.org/abs/2302.14520), *EAMT 2023*) prompts a GPT-4-scale LLM to score translations directly. Works surprisingly well for high-resource pairs; expensive.

QE metrics have opened a new class of decoding-time and production-time uses:

- **MBR decoding.** Generate N candidates, score each with COMET-Kiwi, pick the highest-scoring — often beats plain beam search (Eikema & Aziz, ["Is MAP Decoding All You Need?"](https://arxiv.org/abs/2005.10283), *COLING 2020*).
- **Reference-free monitoring.** Track mean COMET-Kiwi on production traffic per language pair; flag drops.
- **Filter back-translated data.** Drop synthetic pairs with low COMET-Kiwi (chapter 04).

## The evaluation stack you should report

For a serious MT evaluation, report a *panel*, not a single number:

1. **SacreBLEU** with the tokeniser signature (default `13a` for most; `zh` / `ja-mecab` for CJK).
2. **chrF** or **chrF++**. Report as primary for morphologically rich or non-space-segmented targets.
3. **COMET** (`Unbabel/wmt22-comet-da`) with the checkpoint name.
4. **Optionally, BLEURT** if your downstream team standardised on it, or **COMET-Kiwi** if you also want reference-free.

Plus **95 % bootstrap confidence intervals** (1 000 resamples over sentence pairs) so you can tell when a delta is significant. `sacrebleu --paired-bs` runs paired bootstrap between two systems on the same test set; use it before claiming "system B is better."

## Statistical significance

Two systems with a 0.5 BLEU delta on a 1 000-sentence test set are usually not significantly different. Paired bootstrap resampling (Koehn, ["Statistical Significance Tests for Machine Translation Evaluation"](https://aclanthology.org/W04-3250/), *EMNLP 2004*) is the standard test:

```bash
sacrebleu ref.txt --input hyp_a.txt hyp_b.txt --paired-bs -m bleu chrf
```

Report BLEU delta with $p$-value and CI. Skipping this step is the most common reason "our new model is better" ends up meaning "our new model is not distinguishable from the old one."

## Multilingual evaluation caveats

- **BLEU is not comparable across languages.** A 30 BLEU on English → German is not "twice as good" as a 15 BLEU on English → Swahili. Different reference styles, tokenisation, and morphology all shift the scale.
- **chrF is closer to comparable but still biased.** Character-inventory size matters (100 chars in Latin, ~4 000 CJK).
- **COMET is closer to comparable** because it uses a shared multilingual encoder — but the training data was heavier on high-resource pairs, so COMET's calibration on low-resource languages is less reliable.
- **Per-language reports.** Never average metrics over a heterogeneous language set — the average hides tail failures. Report per-language and highlight the worst-performing.

## Common failure modes and their fixes

- **BLEU numbers are non-reproducible.** You are re-tokenising before scoring, or using a non-SacreBLEU implementation. Always use SacreBLEU and paste the signature string.
- **BLEU is high but COMET is low.** The model is producing something surface-similar to the reference but semantically wrong (paraphrase failure, hallucination). Trust COMET and inspect examples.
- **COMET score suddenly regressed after a checkpoint upgrade.** You changed the COMET checkpoint. `wmt20-comet-da` and `wmt22-comet-da` are not on the same scale.
- **A 0.5 BLEU improvement is claimed as significant.** Run paired bootstrap. It probably is not.
- **The Chinese BLEU is very high because you scored it word-tokenised the wrong way.** Use `--tokenize zh` in SacreBLEU. Or use chrF/COMET and ignore BLEU for Chinese.

## Chapter summary

- Report a *panel*, not a single number: SacreBLEU + chrF (or chrF++) + COMET, with paired-bootstrap significance and per-language breakdowns.
- Always use **SacreBLEU** and paste the signature string; `nltk.translate.bleu_score` on re-tokenised text is non-reproducible.
- **chrF** is the tokenisation-invariant surface metric — the default primary for morphologically rich and non-space-segmented targets.
- **COMET** (`Unbabel/wmt22-comet-da`) is the learned-metric default; always name the checkpoint. **COMET-Kiwi** (`wmt22-cometkiwi-da`) is the reference-free variant for QE and production monitoring.
- Learned metrics enable new production patterns: MBR decoding with QE scoring, back-translation filtering, reference-free monitoring.
- Multilingual: BLEU is not comparable across languages; chrF is closer; COMET is best but still biased toward high-resource. Always report per-language.
- Statistical significance via paired bootstrap (`sacrebleu --paired-bs`); a 0.5 BLEU delta is almost never significant.
- Human evaluation (chapter 11) is still the ground truth for consequential deployments; automatic metrics are the fast approximation.
