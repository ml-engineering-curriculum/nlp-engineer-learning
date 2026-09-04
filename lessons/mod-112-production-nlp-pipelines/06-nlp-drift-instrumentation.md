# Instrumenting NLP-Specific Drift Signals

## Motivation

Every ML system in production degrades. Some drift is upstream (users start typing in a new language, a source app changes formatting, a new product category enters the funnel); some is downstream (labellers change guidelines, a rule pack behind the classifier is updated); some is model-internal (a redeploy quietly ships a checkpoint whose calibration is different). The one property they all share: they are invisible without instrumentation.

Generic MLOps drift monitoring — Population Stability Index on tabular features, KS tests on numeric distributions — was designed for tabular models. It transfers to NLP badly. Text is high-dimensional, discrete, and semantically structured; the meaningful drift signals live in the vocabulary, the language mix, and the label distribution, not in a numeric feature vector. This chapter is the NLP-specific catalogue: four signal shapes worth instrumenting, what each detects, what each misses, and how to alert on them without drowning oncall in noise.

## Four drift shapes that matter for NLP

Before the tools, the taxonomy — every NLP drift signal is one of four shapes, and the mitigation is different for each.

- **Input-distribution drift.** The inputs your model sees no longer look like the inputs it was trained on. Detected on the *text side*, before the model. Symptoms: OOV-rate rising, average length changing, language mix shifting, formatting patterns changing (emoji density, URL density, capitalisation ratio).
- **Output-label drift.** The model's predicted-label distribution has changed. Detected on the *prediction side*. May be a legitimate response to input drift (users are asking more refund questions after a pricing change) or a symptom of a model regression (a redeploy shipped a checkpoint that never predicts label X).
- **Concept drift.** The relationship between input and correct label has changed — the input distribution may look the same, but the labels a competent annotator would assign have shifted. Only detectable with fresh ground truth (labelled samples from recent production). This is the drift shape MLOps generally underinstruments.
- **Prediction-quality drift.** The model's actual quality on production data has changed. Detectable only against a fresh labelled slice or a strong proxy signal (downstream metrics, user feedback, revert rates). Usually the *last* signal to fire — by the time quality has moved, business impact is happening.

The instrumentation order matches the detection order. Input drift is cheapest and earliest; concept and quality drift are expensive but definitive.

## Signal 1: input-distribution shift

The catch-all for "the text looks different." Track a fixed panel of input statistics per rolling window (5 min, 1 h, 1 d) and alert on deviation from a baseline (last 30 days, or the training-set distribution stored as a reference profile at packaging time).

The panel that catches most real drift:

- **Length distribution.** Char count and token count, per input. Report p50, p95, p99. Halved p50 or doubled p95 means an upstream client changed.
- **Language-ID distribution.** Every input passes through a language ID model at ingestion; log the identified language and confidence. Track the proportion per language, and the proportion of "low-confidence" identifications. Uses [fastText's `lid.176`](https://fasttext.cc/docs/en/language-identification.html), CLD3, or Google's newer [MediaPipe language detector](https://developers.google.com/mediapipe/solutions/text/language_detector). See mod-107 chapter 09 for the language-ID discussion in depth.
- **Character-class ratios.** Proportion of ASCII vs. non-ASCII, letter vs. digit vs. punctuation vs. whitespace, presence of emoji, URL, mention, hashtag markers. Cheap to compute; catches "app version upgrade started sending JSON instead of plain text."
- **Empty-input rate and near-empty-input rate.** The proportion of inputs shorter than a sensible floor (e.g. < 3 tokens). Rises silently when a client bug ships.
- **Embedding-space drift** — take a fast embedding (`sentence-transformers/all-MiniLM-L6-v2` or your production encoder's `[CLS]` pooled output) of a random 1 % sample per window; compare against the baseline embedding distribution with MMD, Wasserstein, or a domain classifier's AUC. Reference: Lipton, Wang, & Smola, ["Detecting and Correcting for Label Shift with Black Box Predictors"](https://arxiv.org/abs/1802.03916), *ICML 2018*, formalises label-shift detection; the same estimator shape works for input drift.

Instrumenting this in production is a one-time investment that pays off every time the alert fires early:

```python
from prometheus_client import Histogram, Counter

CHAR_LEN = Histogram("nlp_input_char_len", "input length in chars",
                     buckets=[0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
LANG_COUNT = Counter("nlp_input_language_total", "detected input language", ["lang"])
EMPTY = Counter("nlp_input_empty_total", "inputs below length floor")

def record_input_stats(text: str, lang_id: str):
    CHAR_LEN.observe(len(text))
    LANG_COUNT.labels(lang=lang_id).inc()
    if len(text.strip()) < 3:
        EMPTY.inc()
```

Prometheus histograms give you percentiles; a rate-of-change alert on `p95(nlp_input_char_len)[1h] / p95(nlp_input_char_len)[7d]` outside `[0.5, 2.0]` catches shape shifts without paging on normal daily variation.

## Signal 2: vocabulary OOV rate

Every tokeniser has an out-of-vocabulary (OOV) rate against a specific input distribution. For word-piece / BPE / SentencePiece tokenisers, "OOV" is a technicality — everything can be tokenised to `<unk>`-less subwords — but the *effective* OOV signal is real: tokens that get split into many single-character pieces, or that hit the `<unk>` token in a smaller vocabulary, or that appear only rarely in the training corpus and thus have unreliable representations.

Practical OOV metrics to track:

- **`<unk>` rate.** For tokenisers that emit `<unk>` (older WordPiece, some SentencePiece configurations): the fraction of tokens per input that are `<unk>`. A doubling here means the input distribution has moved outside your tokeniser's coverage.
- **Subword-splits-per-word.** For BPE / word-piece, track the average number of subword tokens per whitespace-separated word. A rising number means novel tokens (URLs, code snippets, non-Latin scripts, proper nouns) that split into many pieces. This is the effective OOV signal for modern tokenisers.
- **Rare-token rate.** Compare token IDs against a frequency profile computed on the training corpus at packaging time. Fraction of tokens whose training-corpus frequency was below some threshold (e.g. bottom 10th percentile, or fewer than 5 occurrences). Rising rare-token rate is the modern-tokeniser equivalent of "OOV drift."
- **Vocabulary novelty.** For each rolling window, the set of never-before-seen tokens (against a running training-plus-production vocabulary). Rate of novel tokens per input.

```python
from collections import Counter

def rare_token_rate(input_ids: list[int], rare_id_set: set[int]) -> float:
    if not input_ids:
        return 0.0
    return sum(1 for tid in input_ids if tid in rare_id_set) / len(input_ids)
```

Ship the `rare_id_set` (or its checksum) with the artifact so it is part of the reproducibility manifest (chapter 5).

## Signal 3: output-label drift

Track the distribution of predicted labels over a rolling window and compare it to a baseline. Two comparison shapes:

- **Population Stability Index (PSI)** per label — the KL-divergence-flavoured banking-industry standard. Given baseline proportions `b_i` and observed proportions `o_i`, `PSI = sum((o_i - b_i) * log(o_i / b_i))`. Rule of thumb: PSI < 0.1 no drift, 0.1–0.25 moderate drift (investigate), > 0.25 significant drift (alert). Cheap, well-understood, works out of the box in [Evidently](https://github.com/evidentlyai/evidently), [NannyML](https://github.com/NannyML/nannyml), and [Alibi Detect](https://github.com/SeldonIO/alibi-detect).
- **Chi-square goodness-of-fit** — a proper statistical test on the multinomial label distribution against the baseline. Gives a p-value; combine with PSI for a magnitude + significance pair.

For token-labelling models (NER), track:

- **Per-entity-type prediction rate.** Entities-per-thousand-tokens by type, per window. A rising ORG rate + falling PER rate often signals a domain shift or a labeller-guideline change reflected in the model.
- **Confidence distribution.** Mean and p10 of the model's confidence per prediction. Rising low-confidence rate is a leading indicator of quality drop before the labels themselves shift.

For classification models, track:

- **Per-class prediction rate**, per window.
- **Abstention rate** (if the model or a downstream threshold has an abstain option) — a rising abstention rate is a clean signal of model uncertainty.

For extractive QA / generation, add:

- **Answer-length distribution** (extractive) — halved p95 answer length across a redeploy usually means the model started predicting empty or truncated spans.
- **`no-answer` rate** (SQuAD 2.0-style) — direct proxy for confidence.

## Signal 4: language-mix drift

Language ID is a lightweight, high-leverage signal for multilingual services. Even for a monolingual English pipeline, a rising non-English proportion is important — either you need to route those inputs to a different model, or you need to reject them with a clear message, or your training data was mislabelled and the model is silently working on off-distribution inputs.

Track:

- **Per-language proportion**, per window. Rising per-language PSI against the baseline flags shifts.
- **Per-language quality proxies.** If you have any per-language ground truth (from a labelled evaluation set that gets re-sampled monthly), track per-language quality separately. Aggregate quality can be flat while non-English quality collapses.
- **Language-model perplexity as a coarse quality proxy** — feed a random per-window sample through a small multilingual LM (`facebook/xlm-v-base` or similar) and track per-input perplexity. A rising perplexity indicates the inputs are unusual by generic language modelling.
- **Script mix.** For services that see mixed-script inputs (Chinese, Arabic, Devanagari, Cyrillic mixed with Latin), track the character-script distribution — often a leading indicator of the language proportion changing.

## Reference profiles: what "baseline" means

Every drift signal is a comparison against something. Three baseline shapes:

- **Training-corpus profile.** Computed at packaging time and stored in the artifact manifest. Answers "does production look like what I trained on." Static — never drifts on its own — but stops being informative as time passes and the world changes.
- **Rolling reference window.** Last 30 or 90 days of production. Answers "does today look like recent normal." Automatically tracks slow real drift; misses slow-and-steady drift that the reference is drifting with.
- **Point-in-time snapshot.** A specific dated window (e.g. the two weeks immediately after the last successful deploy). Answers "has anything changed since the last thing we validated." The right choice for regression-detection alerts.

Use all three. Alert differently: training-corpus deviation is informational (data-quality dashboard); rolling-reference deviation is investigative (weekly review); point-in-time deviation is actionable (page oncall).

## Alerting: what to page on

Every drift signal is a candidate alert. Every candidate alert is a candidate page. Every unnecessary page destroys trust in the whole alerting stack. Standard SRE / SLO thinking applies: only page on symptoms that require immediate human action, and use lower-severity channels for the rest.

A working three-tier setup:

- **Tier 1 — page oncall.** Latency SLO burn (from chapter 2), abstention-rate spike beyond a hard threshold, empty-input-rate spike, deploy-correlated PSI jump > 0.5 on the label distribution. These correspond to a service that is measurably broken.
- **Tier 2 — Slack notification to the team, no page.** PSI in `[0.25, 0.5]` for output labels, sustained rise in subword-splits-per-word, per-language proportion crossing a rolling threshold. The team investigates during the next working hours.
- **Tier 3 — weekly review dashboard.** Slow-moving stats: rolling training-corpus PSI, novel-token rate against the running vocabulary, per-slice prediction rate trends. Discussed in the weekly model-review meeting.

Tune thresholds against a **backtest** — replay 6-12 months of historical logs and check that Tier 1 fires only on windows where real incidents happened. Alerts that would have fired on non-incident windows must be moved to Tier 2 or 3 or their thresholds relaxed. This is the same tuning-against-history discipline as SRE incident-review workbooks.

## Tools you can start with

The mainstream open-source options for building this out:

- **[Evidently AI](https://github.com/evidentlyai/evidently).** Python library and dashboards; ships with data-drift, target-drift, and text-drift presets, including PSI, KS, chi-square, and text-specific descriptors (length, OOV, sentiment). Good starting point for a project that needs monitoring today.
- **[NannyML](https://github.com/NannyML/nannyml).** Focused on performance estimation without ground truth (Confidence-Based Performance Estimation) plus multivariate drift detection. Strong statistical machinery; less NLP-specific.
- **[Alibi Detect](https://github.com/SeldonIO/alibi-detect).** Broader detector library (KS, MMD, LSDD, model-uncertainty drift, learned kernels); has explicit text-drift detectors built on embedding representations.
- **[Fiddler](https://www.fiddler.ai/), [Arize](https://arize.com/), [WhyLabs](https://whylabs.ai/).** Hosted / commercial. Faster to set up; higher ongoing cost; typical enterprise pick.
- **Roll your own on Prometheus + Grafana.** For teams already deep in Prometheus, the primitives (`Histogram`, `Counter`, `recording rules`) are enough for tiers 2 and 3 above; a small Python worker computes PSI hourly and exposes it as a gauge.

The tool matters less than the choice of what to instrument. Pick one, wire the four signal shapes above, iterate the thresholds against a backtest, and you have covered 90 % of the NLP drift surface. Chapter 7 walks through what happens when this stack fires — and what happens when it does not.

## Chapter summary

- Four NLP drift shapes worth instrumenting: input-distribution shift, output-label drift, concept drift, and prediction-quality drift. Input drift is cheapest to detect and fires earliest; quality drift is definitive but late.
- Input-distribution panel: length percentiles, language-ID proportions, character-class ratios, empty-input rate, embedding-space distance (MMD / Wasserstein on a sampled subset). Cheap, catches most real shifts.
- Vocabulary OOV signals for modern tokenisers: subword-splits-per-word, rare-token rate against a training-corpus frequency profile, novel-token rate against a running vocabulary. Ship the reference profile with the artifact.
- Output-label drift: PSI per label (< 0.1 no drift, 0.1–0.25 moderate, > 0.25 significant), chi-square goodness-of-fit for significance. For NER add per-entity-type prediction rate and confidence distribution; for QA add answer-length and no-answer rate.
- Language-mix drift: per-language proportion, per-language quality proxies (LM perplexity, or per-language ground truth if you have it), script-mix distribution. Non-English proportion rising in a monolingual service is a routing bug or a training-data mislabelling.
- Baselines: training-corpus profile (from the manifest), rolling reference window, point-in-time snapshot. Use all three; alert differently on each.
- Alerting is tiered — page on symptoms that require immediate action, Slack lower-severity signals, dashboard slow-moving stats. Tune thresholds against a historical backtest.
- Tools: Evidently, NannyML, Alibi Detect (open source), Fiddler / Arize / WhyLabs (hosted), or hand-rolled Prometheus for teams already there. Pick one and instrument the four shapes above.
