# exercise-04: COMET / BLEU / chrF MT Evaluation

**Estimated effort:** 3 hours

## Objective

Score three MT systems on the **same** multilingual test set with a full metric panel — **SacreBLEU**, **chrF++**, **COMET**, **COMET-Kiwi** — plus a small **human direct-assessment (DA)** pilot; establish **paired-bootstrap significance** between the systems; and diagnose at least one **metric disagreement** case where BLEU and COMET rank differently. Deliver a metric report that a downstream team can (a) reproduce and (b) use to make a shipping decision.

## Prerequisites

- Chapters [10](../10-mt-automatic-evaluation-bleu-chrf-comet-bleurt.md), [11](../11-human-evaluation-for-mt.md), [12](../12-multilingual-benchmarks-flores-and-friends.md).
- Python 3.10+; `transformers`, `datasets`, `evaluate`, `sacrebleu`, `unbabel-comet`, `torch`, `numpy`.
- Access to a GPU for COMET scoring (COMET on CPU is slow but works).

## Setup

**Test set:** `FLORES-200 devtest` for **three language pairs** covering the resource / script spectrum:

- `eng_Latn → deu_Latn` (high-resource, Latin).
- `eng_Latn → zho_Hans` (high-resource, non-Latin, no spaces).
- `eng_Latn → swh_Latn` (mid-resource, Latin).

**Three MT systems to compare** (all publicly loadable):

- **System A — `Helsinki-NLP/opus-mt-en-<x>`** (Marian). Small bilingual per-pair; falls back to `nllb` if the pair isn't in OPUS-MT.
- **System B — `facebook/nllb-200-distilled-600M`** (NLLB distilled).
- **System C — `facebook/nllb-200-1.3B`** (or `-3.3B` if GPU allows).

## Problem statement

### Part A — Reproducible generation

Translate the FLORES-200 devtest for all three pairs with all three systems. Requirements:

- Wrap language-code plumbing in a helper (`build_translator`, chapter 02).
- Fixed decoding: `num_beams=4`, `max_new_tokens=256`, `no_repeat_ngram_size=0`.
- Log the model checkpoint hash (`from_pretrained(..., revision="<sha>")` if the Hub SHA is available).
- Save one `.txt` file per (system, pair) with one hypothesis per line, matching the source line order exactly.

Save as `translate.py`. Ship a `manifest.json` recording model names, checkpoint SHAs, and generation kwargs per file.

### Part B — Surface metrics: SacreBLEU + chrF

Score every hypothesis file:

- **SacreBLEU** via `sacrebleu` CLI (canonical), tokenizer:
    - `--tokenize 13a` for German, Swahili.
    - `--tokenize zh` for Chinese.
- **chrF++** via `sacrebleu --metrics chrf --chrf-word-order 2`.

Record the SacreBLEU **signature string** for every score. Signatures go in the report.

Save as `surface_metrics.py` (or a shell script wrapping the CLI).

### Part C — Learned metrics: COMET + COMET-Kiwi

For every (system, pair):

- **COMET (reference-based)** with `Unbabel/wmt22-comet-da`. Record the checkpoint version / hash.
- **COMET-Kiwi (reference-free)** with `Unbabel/wmt22-cometkiwi-da`. Same recording.

Use `comet.load_from_checkpoint(...)` and `.predict(data, batch_size=8, gpus=1)`.

Save as `learned_metrics.py`.

### Part D — Confidence and significance

For every metric and every (system, pair):

- **95 % bootstrap CI** over segments (1 000 resamples). Reuse or copy the `paired_bootstrap` helper from chapter 11.
- **Paired-bootstrap significance** between each pair of systems on SacreBLEU + chrF (use `sacrebleu ... --paired-bs -m bleu chrf`). Also implement paired bootstrap for COMET yourself (`sacrebleu`'s tool does not support it).

Produce a per-pair matrix that answers "is A better than B on metric M with $p < 0.05$?" for every {A,B,M} combination.

Save results as `significance.json`.

### Part E — Metric disagreement analysis

Find **three sentences per language pair** (nine total) where **SacreBLEU and COMET rank the two best systems differently**. For each:

- Show the source, reference, and both hypotheses.
- Report per-segment SacreBLEU and per-segment COMET.
- Classify the disagreement: **paraphrase-friendly** (COMET better than BLEU on a fluent paraphrase), **BLEU-noise** (systems produce similar output; BLEU is chance-noisy), **hallucination** (BLEU high, COMET low — check for extra content), or **other**.

Save as `disagreements.md`.

### Part F — Human DA pilot

Pick **the closest pair of systems** on your best metric — the two whose CIs overlap most. Sample **30 sentences** where the two systems produced different outputs.

Design and run a minimal DA protocol (chapter 11):

- Show both hypotheses in randomised order alongside the source (source-based DA).
- Rate each on a 0–100 slider for "adequacy" (does it convey the source meaning).
- 3 raters per segment if you can find them; 1–2 is acceptable for the exercise; be honest about who rated.

Report:

- Per-rater z-standardised system means.
- Paired-bootstrap significance on the DA differences.
- Whether the human ranking matches the automatic-metric ranking; if not, discuss which you would trust.

Save as `human_da.md` with the anonymised rating data in `da_ratings.csv`.

### Part G — Deliverable metric report

Assemble a single `report.md` (600–900 words) organised as:

1. **Setup.** Test set (FLORES-200 devtest), three systems, decoding config, checkpoint SHAs.
2. **Metric panel** — the big table:

    | Pair | System | SacreBLEU (CI, sig) | chrF++ (CI, sig) | COMET (CI, sig) | COMET-Kiwi (CI, sig) |

    Include SacreBLEU signature string in a footnote and COMET checkpoint version.
3. **Statistical significance summary.** Which pairwise comparisons are significant on which metrics.
4. **Disagreement cases.** The nine examples from Part E with brief interpretation.
5. **Human DA pilot.** 30-sentence protocol, results, agreement / disagreement with automatic metrics.
6. **Shipping recommendation** for each language pair — which system, and how confident.
7. **Reproducibility appendix.** All hashes, signatures, kwargs, and dependencies.

## Starter guidance

- **Never re-tokenise before SacreBLEU.** `nltk.translate.bleu_score.corpus_bleu` and hand-tokenised inputs will silently produce different numbers. Always use SacreBLEU on the *raw* text.
- **Paste the SacreBLEU signature.** `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0` — without it your number is not reproducible.
- **`--tokenize zh` for Chinese**, `--tokenize ja-mecab` for Japanese (requires `mecab-python3`). Wrong tokenizer = wrong number.
- **COMET is slow.** Use `batch_size=8` on a single GPU; `16` if VRAM allows. Cache encoded source embeddings if scoring multiple systems on the same source.
- **COMET-Kiwi scores on a different scale from COMET.** Do not compare COMET and COMET-Kiwi scores directly; both live on `~[0, 1]` but calibrations differ.
- **The COMET model card has changed between versions.** `wmt20-comet-da` and `wmt22-comet-da` are not on the same scale; always name the checkpoint.
- **1 000 bootstrap resamples is standard.** More does not add signal; fewer risks noisy CIs.
- **DA pilot ethics.** If your raters are colleagues, disclose so; don't publish rater identities.

## Acceptance criteria

- [ ] Three MT systems × three pairs = nine hypothesis files, one line per FLORES-200 devtest source.
- [ ] `manifest.json` with model names, checkpoint SHAs, and generation kwargs.
- [ ] SacreBLEU with signatures + chrF++ scores for every (system, pair).
- [ ] COMET + COMET-Kiwi with checkpoint versions for every (system, pair).
- [ ] 95 % bootstrap CIs on all metrics.
- [ ] Pairwise significance matrix for SacreBLEU + chrF + COMET.
- [ ] `disagreements.md` with three metric-disagreement examples per language pair.
- [ ] `human_da.md` with 30-sentence DA pilot, ratings, and analysis.
- [ ] `report.md` (600–900 words) with metric panel, significance summary, human-DA results, shipping recommendation, reproducibility appendix.

## Stretch goals

- **BLEURT arm.** Add BLEURT-20 to the metric panel. Compare its rankings to COMET's on the disagreement examples.
- **XCOMET spans.** Use `Unbabel/XCOMET-XL` to also predict per-sentence error spans; compare to your human-DA judgements.
- **LLM-as-judge (GEMBA).** Prompt an accessible strong LLM in GEMBA-DA style; compare to your COMET / human rankings on the high-resource pairs.
- **Domain slice.** Split FLORES-200 devtest by source topic (`domain` field) and report per-domain metrics; comment on where the smaller model catches up.
- **Length slice.** Split by source token length and report per-length-bucket metrics. Does one system dominate on long sentences?
- **Extended DA.** Raise the DA sample size to 100 with 3 raters and report inter-annotator agreement.

## Deliverables

```
translate.py
manifest.json
hyps/                       # nine .txt files
surface_metrics.py          # SacreBLEU + chrF
learned_metrics.py          # COMET + COMET-Kiwi
significance.json
disagreements.md
human_da.md
da_ratings.csv
report.md
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
