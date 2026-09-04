# Multilingual Benchmarks: FLORES-200 and Friends

## Motivation

You have a model, a metric panel (chapter 10), and a human protocol (chapter 11). The last piece is the *test set*. A number without a shared benchmark is a claim your peers cannot verify; a benchmark without a shared metric is a leaderboard nobody agrees on. This chapter walks the multilingual benchmarks that the field actually uses — for translation (**FLORES-200**, **WMT**, **NTREX-128**), for classification and NLI (**XNLI**, **XTREME**, **XTREME-R**), for QA (**TyDiQA**, **MLQA**, **XQuAD**), for NER (**WikiANN / PAN-X**, **MasakhaNER**), and for summarisation (**MLSum**, **XL-Sum**) — plus the practical rules for choosing one and reporting numbers a stranger can reproduce.

The single sentence to internalise: **report FLORES-200 numbers whenever your target pair is covered, even if your primary test set is domain-specific.** It is the closest thing multilingual MT has to a common yardstick.

## FLORES-200

**FLORES-200** (NLLB Team et al., ["No Language Left Behind: Scaling Human-Centered Machine Translation"](https://arxiv.org/abs/2207.04672), 2022) is the reference test set for the massively-multilingual era. It replaced FLORES-101 (Goyal et al., ["The FLORES-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation"](https://arxiv.org/abs/2106.03193), *TACL 2022*), which replaced the earlier FLORES benchmark for Sinhala and Nepali.

What is in it:

- **200 languages**, ~150 of them genuinely low-resource. Every language paired with English, and — for most pairs — with every other language.
- **~3 000 sentences** drawn from English Wikinews / Wikijunior / Wikivoyage, professionally translated into every target language by native-speaker translators.
- **Three splits.** `dev` (~ 1 000), `devtest` (~ 1 000), and a held-out `test` (~ 1 000, only released to shared-task participants). **You evaluate on `devtest`.** `dev` is for hyperparameter sweeps; `test` is not public.
- **Same source sentences translated to every target.** Because the source is fixed and translations are professional, cross-language *comparison* is possible in a way it was not with independently-sampled per-pair test sets.

Loading and evaluating:

```python
from datasets import load_dataset

ds = load_dataset("facebook/flores", "eng_Latn-deu_Latn", split="devtest")
sources    = ds["sentence_eng_Latn"]
references = ds["sentence_deu_Latn"]

# Translate `sources` with your model, then:
import evaluate
sb = evaluate.load("sacrebleu")
score = sb.compute(predictions=hyps, references=[[r] for r in references])
print(score["score"], score.get("signature"))
```

Language codes are the NLLB convention: ISO 639-3 + ISO 15924 with an underscore (`eng_Latn`, `swh_Latn`, `zho_Hans`, `arb_Arab`, `hin_Deva`, `pes_Arab`). The full 200-language list is in the NLLB paper's Appendix and in the [FLORES model card](https://huggingface.co/datasets/facebook/flores).

**Why FLORES matters for engineers, not just researchers:**

- **Reproducibility.** Everyone can pull the same devtest and re-score.
- **Cross-language sanity check.** If your production number on English → Swahili is 40 chrF but FLORES devtest is 25 chrF, either your production distribution is very different from Wikinews or something is wrong with your evaluation pipeline.
- **Regression floor.** If a new checkpoint drops FLORES chrF by > 1.0 for any pair, block the release and investigate.

## WMT and its yearly test sets

The **Workshop on Statistical Machine Translation** (now Conference on Machine Translation), running since 2006, is the annual venue that ships the shared tasks the field measures itself against. Each year produces:

- **News-translation task test sets** for ~10–15 language pairs. Fresh news articles translated by professionals; the primary MT leaderboard.
- **Biomedical, chat, terminology, sign-language, gender-inclusive** and other sub-tracks in various years.
- **Metrics shared task** (Freitag et al., annual; e.g. ["Results of the WMT23 Metrics Shared Task"](https://aclanthology.org/2023.wmt-1.51/)) — the evaluation of the evaluations. Where COMET and chrF earned their reputations.

Practical usage:

- **`wmt<year>` test sets** are on Hugging Face under names like `wmt/wmt19`, `wmt/wmt22`. For a specific language pair, filter the split by `--lp` argument or by `config_name="cs-en"` etc.
- **Compare to WMT-published system outputs.** WMT publishes all submitted systems' outputs alongside the test set. If you are trying to beat "the state of the art on WMT22 English–German," fetch the winning system's outputs and run paired-bootstrap SacreBLEU + COMET against yours — that is what a reviewer will ask you to do.
- **Do not train on WMT test sets.** They rotate yearly precisely so old test sets can become training data; check the year against your training-data cutoff.

## NTREX-128 and other translation-focused sets

- **NTREX-128** (Federmann et al., ["NTREX-128 — News Test References for MT Evaluation of 128 Languages"](https://aclanthology.org/2022.sumeval-1.4/), *SUMEVAL 2022*). WMT-style news translated to 128 languages. Overlap-heavy with FLORES but with different source sentences; useful as a *second* eval set to cross-check.
- **TICO-19** (Anastasopoulos et al., ["TICO-19: the Translation Initiative for COvid-19"](https://arxiv.org/abs/2007.01788), 2020). Public-health text translated to 35 languages; useful as an in-domain test for medical/health-adjacent MT.
- **OPUS-100 test sets** (Zhang et al., 2020). 100 language pairs derived from OPUS; use as an additional sanity check when you have already trained on OPUS training splits.

## Cross-lingual understanding benchmarks

For the multilingual-encoder side of the module (chapters 07–08), the benchmarks that matter:

### XNLI

**XNLI** (Conneau et al., ["XNLI: Evaluating Cross-lingual Sentence Representations"](https://arxiv.org/abs/1809.05053), *EMNLP 2018*) is the canonical cross-lingual NLI benchmark. Take MultiNLI, professionally translate the dev and test splits into **15 languages** (en, fr, es, de, el, bg, ru, tr, ar, vi, th, zh, hi, sw, ur), evaluate zero-shot from English or from any pivot. Every serious multilingual encoder reports XNLI zero-shot accuracy per language.

```python
xnli = load_dataset("xnli", "sw", split="test")  # Swahili test
# Fine-tune your encoder on English MNLI; predict on `xnli["premise"]`, `xnli["hypothesis"]`.
```

### XTREME and XTREME-R

**XTREME** (Hu et al., ["XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalisation"](https://arxiv.org/abs/2003.11080), *ICML 2020*) bundles **9 tasks across 40 languages** — classification, structured prediction (NER, POS), QA, retrieval — with a unified zero-shot leaderboard. Its successor **XTREME-R** (Ruder et al., ["XTREME-R: Towards More Challenging and Nuanced Multilingual Evaluation"](https://arxiv.org/abs/2104.07412), *EMNLP 2021*) trimmed easy tasks and added harder ones (50 languages, 10 tasks).

XTREME-R is the modern default: an XLM-R replacement should report per-language numbers on the full XTREME-R suite in the paper. Reproduce it via [`google-research-datasets/xtreme`](https://github.com/google-research/xtreme).

### Task-specific multilingual benchmarks

- **QA — TyDiQA** (Clark et al., ["TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages"](https://arxiv.org/abs/2003.05002), *TACL 2020*). 11 typologically diverse languages, information-seeking QA collected natively (not translated). Two tasks: `GoldP` (extractive), `SelectP` (passage selection).
- **QA — MLQA** (Lewis et al., ["MLQA: Evaluating Cross-lingual Extractive Question Answering"](https://arxiv.org/abs/1910.07475), *ACL 2020*). Extractive QA in 7 languages, parallel across languages so you can compare a system on the same question in different languages.
- **QA — XQuAD** (Artetxe et al., ["On the Cross-lingual Transferability of Monolingual Representations"](https://arxiv.org/abs/1910.11856), *ACL 2020*). SQuAD 1.1 dev translated into 11 languages by professionals.
- **NER — WikiANN / PAN-X** (Pan et al., ["Cross-lingual Name Tagging and Linking for 282 Languages"](https://aclanthology.org/P17-1178/), *ACL 2017*). Silver-labelled NER (PER/LOC/ORG) mined from Wikipedia in 282 languages. Noisy but broad.
- **NER — MasakhaNER 1.0 / 2.0** (Adelani et al., ["MasakhaNER: Named Entity Recognition for African Languages"](https://arxiv.org/abs/2103.11811), *TACL 2021*; ["MasakhaNER 2.0"](https://aclanthology.org/2022.emnlp-main.298/), *EMNLP 2022*). Human-annotated NER for 21 African languages. The community reference for African NER; note that MasakhaNER is *cleaner* than PAN-X for the same languages.
- **Sentiment — AfriSenti, IndicSentiment, MARC**. Regional sentiment benchmarks for African, Indic, and Amazon-review-cross-lingual sentiment respectively.
- **Summarisation — MLSum, XL-Sum, WikiLingua**. Multilingual summarisation over news (MLSum, XL-Sum) and how-to content (WikiLingua).
- **Reasoning / commonsense — XCOPA, XStoryCloze, XCSQA**. Multilingual analogues of English commonsense benchmarks.
- **Toxicity / safety — MULTIQ, JIGSAW multilingual, RTP-LX**. Multilingual toxicity and red-team datasets.

Pick the one that matches your task shape; report per-language.

## Region- and language-family-specific benchmarks

The community has invested heavily in benchmarks for chronically-underserved language families. If you are shipping into those markets, these are what your customers' teams will use to evaluate you.

- **AfricaNLP: MasakhaNER, MasakhaPOS, AfriSenti, [AfroLM](https://arxiv.org/abs/2211.03263), MAFAND-MT.**
- **Indic: [IndicNLP Suite](https://ai4bharat.iitm.ac.in/), IndicXNLI, IndicGLUE, FLORES Indic subset, Samanantar (parallel bitext).**
- **Southeast Asian: [SEACrowd](https://aclanthology.org/2024.emnlp-main.296/), NusaCrowd, IndoNLU / IndoNLG.**
- **East Asian: [C-Eval](https://arxiv.org/abs/2305.08322) (Chinese), [JGLUE](https://aclanthology.org/2022.lrec-1.317/) (Japanese), [KLUE](https://arxiv.org/abs/2105.09680) (Korean).**
- **Arabic and Semitic: ALUE, ARLUE, Alghafa.**

The FLORES-200 numbers give a global-comparability floor; the regional benchmarks are what tell you whether your model actually works for that region's users.

## Choosing which benchmark to report

Rules of thumb that will save you a review cycle:

1. **Translation.** Report FLORES-200 devtest for every pair you cover, plus (if applicable) the WMT test set for the most recent year, plus your production distribution.
2. **Cross-lingual classification.** Report XNLI (if NLI) or the relevant XTREME-R task per language.
3. **Cross-lingual QA.** Report TyDiQA GoldP for typological diversity and XQuAD/MLQA for cross-lingual parallel comparison.
4. **Multilingual NER.** Report MasakhaNER for the 21 African languages when relevant; WikiANN / PAN-X as the broad-coverage but noisy fallback.
5. **Multilingual summarisation.** Report XL-Sum (44 languages) as the primary; MLSum for European news.
6. **Custom / production.** Always report your production distribution numbers alongside the public benchmark. The public benchmark is for peer comparison; your production distribution is for your customer.

## Reporting numbers people can reproduce

The bar for a reproducible multilingual result:

- **Test-set version / split.** `FLORES-200 devtest`, not "FLORES." `WMT22 news test`, not "WMT."
- **Metric with signature.** `sacrebleu ... signature nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0` — paste it.
- **Model checkpoint name and hash.** `facebook/nllb-200-distilled-600M @ commit <sha>`. Hub commits change silently.
- **Language code convention.** `eng_Latn → deu_Latn` (NLLB style) vs. `en-DE` vs. `en_XX → de_DE` (mBART) are not the same code, and mixing them at the tokenizer boundary is the top-3 source of "why is my score suddenly zero."
- **Beam / decoding hyperparameters.** `num_beams=4, max_new_tokens=256, no_repeat_ngram_size=0`. Beam width shifts SacreBLEU by 1+ point routinely.
- **Per-language table, not average.** Averaged multilingual scores hide tail failures.

The [`sacrebleu` README](https://github.com/mjpost/sacrebleu) and the [WMT submission guidelines](https://www2.statmt.org/) are the canonical style guides.

## Benchmark contamination and data hygiene

The rate at which pretraining corpora leak into benchmarks is now a first-class problem. Steps:

- **Check overlap between your training data and the benchmark.** `sacrebleu`-style n-gram overlap between training sentences and test sentences reveals gross leakage. For mined bitext (CCMatrix, CCAligned), FLORES-200 sentences show up regularly — deduplicate before training.
- **Prefer newer test sets when you can.** Test sets released after your model's training cutoff are guaranteed unseen. FLORES-200 (2022) is safe for pre-2022 pretraining but by definition contaminates every model trained after mid-2022 that scraped Common Crawl.
- **Report training-cutoff / test-release-year alignment.** "NLLB-200 on FLORES-200" is not a leak (FLORES is a *held-out translation* of Wiki source with the target side not in NLLB's crawls); "GPT-4 on FLORES-200" almost certainly is. Different levels of trust.

## Benchmarks are not deployments

Two failure modes worth naming:

- **"We top FLORES; ship it."** FLORES sentences are Wikinews-style; your customer messages are colloquial, code-mixed, and full of product jargon. FLORES tells you the model *can* translate; only your production distribution tells you it *will*.
- **"Zero-shot XNLI is 78 %; NLI in Swahili is solved."** XNLI-Swahili is a translation of MNLI, not natively-authored Swahili NLI. Native-speaker NLI datasets often reveal a 5–15 point gap that XNLI hides.

Benchmark numbers are a floor and a communication tool; they are not a guarantee of production quality.

## Common failure modes and their fixes

- **"My FLORES numbers do not match the NLLB paper."** You are on a different checkpoint (`distilled-600M` vs. `1.3B` vs. `3.3B`), a different beam width, or you evaluated on `dev` not `devtest`. Or you tokenised the source with a stripped-punctuation preprocess. Match the paper's config precisely.
- **"My WMT number is 3 BLEU higher than the WMT winner."** Almost always a decoding-time bug — reference in the prediction file, empty predictions counted as perfect matches, or a language-code mixup. Sanity-check by scoring the reference against itself (should be 100 SacreBLEU).
- **"XTREME average is high but per-language numbers are terrible for the tail."** The average is dominated by high-resource languages. Report per-language; publish the worst-performing.
- **"My model on MasakhaNER is worse than on WikiANN for the same language."** Not a bug — MasakhaNER is cleaner and harder. Trust MasakhaNER for African NER; WikiANN's silver labels are noisy.
- **"FLORES score plummeted after a Hub checkpoint update."** The tokenizer or the special-token layout changed. Pin `revision=` explicitly in `from_pretrained` calls for production evaluation.

## Chapter summary

- **FLORES-200 devtest** is the reference translation benchmark: 200 languages, ~1 000 professionally-translated sentences per pair, shared source across all pairs. Report it whenever the pair is covered.
- **WMT news test sets** are the year-over-year MT leaderboard; compare to the year's winner outputs with paired-bootstrap SacreBLEU and COMET.
- Cross-lingual understanding: **XNLI** (15-lang NLI), **XTREME / XTREME-R** (multi-task suite), **TyDiQA / MLQA / XQuAD** (QA), **MasakhaNER** and **WikiANN** (NER), **XL-Sum / MLSum** (summarisation).
- Regional benchmarks — MasakhaNER, IndicNLP, SEACrowd, C-Eval / JGLUE / KLUE, ALUE — are what your regional teams will use to judge you.
- Report benchmark **version**, **split**, **metric signature**, **checkpoint hash**, **decoding hyperparameters**, and **per-language numbers**. Averages hide tail failures.
- Watch for benchmark contamination: check n-gram overlap between training data and test sets; prefer benchmarks released after your training cutoff.
- Benchmarks are a floor, not a guarantee. FLORES tells you the model *can* translate; production traffic tells you it *will*.
