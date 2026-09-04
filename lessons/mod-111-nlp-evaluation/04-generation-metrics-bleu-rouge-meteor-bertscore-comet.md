# Generation Metrics: BLEU, chrF, ROUGE, METEOR, BERTScore, BLEURT, COMET

## Motivation

Free-form generation — translation, summarisation, question answering with generated (not extracted) answers, data-to-text — has no single metric that captures quality. The community's compromise is a *panel*: a cheap surface-form metric (BLEU / chrF / ROUGE) plus a learned neural metric (COMET / BLEURT / BERTScore) plus, for consequential deployments, human evaluation (chapter 09).

This chapter walks the generation-metric catalogue at the level a working NLP engineer needs: what each metric measures, when it lies, and how to report a number that another team can reproduce. mod-107 chapter 10 covers MT metrics in more depth; mod-106 chapters 10–12 cover summarisation-specific metrics (ROUGE, faithfulness, hallucination). This chapter is the unified reference for both plus the cross-cutting lessons.

## Surface-form metrics: what they measure and where they fail

### BLEU

**BLEU** (Papineni et al., ["BLEU: A Method for Automatic Evaluation of Machine Translation"](https://aclanthology.org/P02-1040/), *ACL 2002*) is modified n-gram precision with a brevity penalty:

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

where $p_n$ is the clipped precision of $n$-grams ($n = 1..4$ typically). Corpus-level micro-average over sentence n-grams.

The well-known limitations:

- **Sensitive to tokenisation.** Different tokenisers → different BLEU on the same text. The industry response is **SacreBLEU** (Post, ["A Call for Clarity in Reporting BLEU Scores"](https://arxiv.org/abs/1804.08771), *WMT 2018*), which pins the tokeniser, reports a signature, and lets anyone reproduce the number. **Always use SacreBLEU**; never `nltk.translate.bleu_score` on re-tokenised text.
- **Word-based.** Broken for Chinese / Japanese / Thai (no word boundaries) and unfair to morphologically rich languages. Use `--tokenize zh` / `ja-mecab` etc.
- **Insensitive to paraphrase.** "The cat sat on the mat" vs. "The feline was on the rug" scores near zero.
- **Poor per-sentence discrimination.** Corpus-level by design; sentence-BLEU exists but correlates poorly with human judgement.
- **Not comparable across languages.** A 30 BLEU on en→de is not "twice as good" as 15 BLEU on en→sw.

```python
import sacrebleu
bleu = sacrebleu.corpus_bleu(hyps, [refs])   # references is a list of lists
print(bleu.score, bleu.signature)            # 27.4  nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
```

**Always paste the signature string** alongside the number. Without it, the number is not reproducible.

### chrF and chrF++

**chrF** (Popović, ["chrF: character n-gram F-score for automatic MT evaluation"](https://aclanthology.org/W15-3049/), *WMT 2015*) computes an F-score over character n-grams (default $n=6$). **chrF++** adds a small word-n-gram term for stability.

Advantages over BLEU: **tokenisation-invariant** (character n-grams are language-agnostic) and **better on morphologically rich languages** (partial credit for morphological variants). WMT recommends chrF as the primary surface metric for shared tasks.

```python
chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)   # word_order=2 → chrF++
```

Report chrF or chrF++ (specify which) alongside BLEU for MT — chrF as primary for morphologically rich or non-space-segmented targets.

### ROUGE

**ROUGE** (Lin, ["ROUGE: A Package for Automatic Evaluation of Summaries"](https://aclanthology.org/W04-1013/), *WS 2004*) is a family of recall-flavoured n-gram overlap scores designed for summarisation:

- **ROUGE-N.** F1 over $n$-grams. ROUGE-1 (unigrams) and ROUGE-2 (bigrams) are standard.
- **ROUGE-L.** Longest common subsequence-based F-score; captures sentence-level structure without requiring contiguous matches.
- **ROUGE-Lsum.** ROUGE-L computed sentence-wise then averaged — the canonical variant for multi-sentence summaries.

Report ROUGE-1, ROUGE-2, and ROUGE-Lsum together. The `rouge-score` package (Google's reference implementation) is the current standard; the `evaluate` metric `"rouge"` wraps it. Avoid `rouge` (the older PyPI package by pltrdy) — it uses different stemming and produces different numbers.

```python
import evaluate
rouge = evaluate.load("rouge")
res = rouge.compute(predictions=hyps, references=refs, use_stemmer=True)
# {"rouge1": 0.42, "rouge2": 0.19, "rougeL": 0.31, "rougeLsum": 0.38}
```

ROUGE's failure modes:

- **Rewards extraction over abstraction.** A summariser that copies verbatim from the source will score high ROUGE. This is a known bias; pair with a learned metric.
- **Reference-quality sensitive.** ROUGE against one crowd-written reference underweighs valid alternative summaries. Multi-reference ROUGE (when available) reduces the noise.
- **No faithfulness signal.** ROUGE happily rewards fluent hallucinations that repeat words from the reference. See mod-106 chapters 11–12 for faithfulness metrics.

### METEOR

**METEOR** (Banerjee & Lavie, ["METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments"](https://aclanthology.org/W05-0909/), *ACL WS 2005*) is unigram-precision-and-recall with stemming, synonym matching (WordNet), and paraphrase alignment. Historically stronger than BLEU on segment-level correlation with human judgement; less popular than chrF or the learned metrics in current MT/summarisation reporting because WordNet coverage is uneven across languages.

Report METEOR when you have a strong reason (comparing to older results that used it, or English-language shared tasks that mandate it) — otherwise chrF is more portable.

## Learned neural metrics: BERTScore, BLEURT, COMET

Learned metrics correlate much better with human judgement than surface-form metrics. Every current WMT metrics shared task (Freitag et al., ["Results of the WMT22 Metrics Shared Task: Stop Using BLEU"](https://aclanthology.org/2022.wmt-1.2/), *WMT 2022*, and successors) shows learned metrics dominating.

### BERTScore

**BERTScore** (Zhang et al., ["BERTScore: Evaluating Text Generation with BERT"](https://arxiv.org/abs/1904.09675), *ICLR 2020*) embeds candidate and reference tokens with a Transformer (default: `roberta-large` for English, `bert-base-multilingual-cased` for multilingual), then computes greedy cosine-similarity alignment and averages to precision, recall, F1.

The `bert-score` package supports many backbones. Two configuration decisions determine the number:

- **`model_type`.** `roberta-large` for English, `microsoft/deberta-xlarge-mnli` for the current top English correlations, `bert-base-multilingual-cased` for multilingual. **Always report the backbone name**; scores are not comparable across backbones.
- **`rescale_with_baseline=True`.** Applies a per-backbone linear rescaling so scores span roughly `[0, 1]` instead of clustering in `[0.85, 0.95]`. The absolute numbers change; report whether you rescaled.

```python
from bert_score import score
P, R, F1 = score(hyps, refs, model_type="roberta-large", rescale_with_baseline=True, lang="en")
print(F1.mean().item())
```

BERTScore-F1 is a solid default learned metric for summarisation and paraphrase-heavy generation. For MT, COMET usually outperforms it because COMET conditions on the source and is trained on human MT judgements.

### BLEURT

**BLEURT** (Sellam, Das, Parikh, ["BLEURT: Learning Robust Metrics for Text Generation"](https://arxiv.org/abs/2004.04696), *ACL 2020*) is BERT fine-tuned on WMT direct-assessment judgements. Current recommended checkpoint: `BLEURT-20` (Pu et al., ["Learning Compact Metrics for MT"](https://arxiv.org/abs/2110.06341), *EMNLP 2021*).

Reference-based only (no source input). Reasonable correlations with human ratings; less commonly reported in 2020s MT papers than COMET but standardised for some downstream teams. Use via `evaluate.load("bleurt", "BLEURT-20")`.

### COMET

**COMET** (Rei et al., ["COMET: A Neural Framework for MT Evaluation"](https://arxiv.org/abs/2009.09025), *EMNLP 2020*) — the de facto learned MT metric. XLM-R encodes `(source, hypothesis, reference)`; a small feedforward head predicts a scalar quality score trained on WMT DA data.

Checkpoints:

- **`Unbabel/wmt22-comet-da`** — reference-based, the default.
- **`Unbabel/wmt22-cometkiwi-da`** — reference-free (source + hypothesis only). Use when references are unavailable — production monitoring, MBR decoding, back-translation filtering.
- **`Unbabel/XCOMET-XL`** and larger XCOMET checkpoints (Guerreiro et al., ["xCOMET: Transparent Machine Translation Evaluation through Fine-grained Error Detection"](https://arxiv.org/abs/2310.10482), *TACL 2024*) — also predict sentence-level error spans, MQM-style.

```python
from comet import download_model, load_from_checkpoint
model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)
data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(srcs, mts, refs)]
result = model.predict(data, batch_size=8, gpus=1)
```

**Always name the COMET checkpoint.** `wmt20-comet-da` and `wmt22-comet-da` produce different scores on the same input; without the checkpoint name, the number is not reproducible.

### GEMBA and LLM-as-judge

**GEMBA** (Kocmi & Federmann, ["Large Language Models Are State-of-the-Art Evaluators of Translation Quality"](https://arxiv.org/abs/2302.14520), *EAMT 2023*) prompts a frontier LLM to score translations directly, either on a 0–100 scale (`GEMBA-DA`) or with an error-tagged MQM-style rubric (`GEMBA-MQM`). Competitive with COMET on WMT high-resource pairs; sharp accuracy drop on low-resource languages the base LLM saw little of.

Costly and non-deterministic. Useful as a third metric alongside COMET for iterative development; not a substitute for human evaluation on consequential shipping decisions. Chapter 09 treats LLM-as-judge as a general pattern including its biases.

## Reference-free (quality estimation)

When references are unavailable — production monitoring, filtering synthetic data, scoring at decode-time — reach for reference-free metrics:

- **COMET-Kiwi (`Unbabel/wmt22-cometkiwi-da`).** The workhorse for MT.
- **BLEURT-QE / TransQuest.** Predecessor families, still used for some low-resource pairs.
- **QAFactEval / SummaC / FactCC** for summarisation faithfulness — see mod-106 chapter 11.

Reference-free QE metrics have enabled a class of decode-time patterns:

- **MBR decoding.** Generate N candidates, score each with QE, pick highest (Eikema & Aziz, ["Is MAP Decoding All You Need?"](https://arxiv.org/abs/2005.10283), *COLING 2020*).
- **Production monitoring.** Track mean COMET-Kiwi on production traffic per language pair; alert on drops.
- **Data filtering.** Drop synthetic pairs with low QE score before training (chapter 04 of mod-107 covers back-translation filtering).

## Reporting a generation-metric panel

For a serious generation evaluation, report a panel:

1. **One surface metric.** SacreBLEU (with signature) for MT; ROUGE-1/2/Lsum for summarisation; chrF as a language-portable surface baseline.
2. **One learned metric.** COMET (with checkpoint) for MT; BERTScore-F1 (with backbone + rescale flag) for summarisation and open-ended generation.
3. **One task-specific structural metric** where the task has one. Faithfulness scores for summarisation. Answer equivalence for open-ended QA. Length compliance for summarisation.
4. **Bootstrap 95 % CIs** on every number. `sacrebleu ... --paired-bs` for system-vs-system MT deltas; a home-rolled paired bootstrap for other metrics (chapter 07).
5. **Per-slice breakdowns.** Per language, per domain, per input-length bucket. Never aggregate averages across a heterogeneous set — the average hides the tail failures.

## Common failure modes

- **BLEU numbers not reproducible.** You are re-tokenising before scoring, or using `nltk` instead of SacreBLEU. Use SacreBLEU and paste the signature.
- **BLEU high, COMET low.** The model produced something surface-similar to the reference but semantically wrong (paraphrase failure or hallucination). Trust COMET; inspect examples.
- **COMET regressed after a "no-op" change.** You changed the COMET checkpoint. Name the checkpoint.
- **ROUGE inflated by extractive baseline.** Pair ROUGE with BERTScore or a faithfulness metric. Do not ship a summariser on ROUGE alone.
- **BERTScore not comparable across two papers.** Different backbones or one used `rescale_with_baseline` and the other did not. Report the exact config.
- **"Average over all languages" reported for a multilingual system.** Report per-language and macro-average with per-language callouts. A macro-average without callouts hides the tail.
- **0.3-point delta claimed as improvement.** Paired bootstrap. Almost always not significant.

## Chapter summary

- Report a *panel*: a surface metric (SacreBLEU / chrF / ROUGE-1/2/Lsum) + a learned metric (COMET / BLEURT / BERTScore) + a task-specific structural metric where applicable + bootstrap CIs + per-slice breakdown.
- Surface metrics are cheap and reproducible with pinned implementations (SacreBLEU signature; `rouge_score` for ROUGE) but blind to paraphrase.
- Learned metrics correlate much better with human judgement but require named checkpoints — `Unbabel/wmt22-comet-da`, `BLEURT-20`, `bert-score` `model_type` — to be reproducible. Report the checkpoint.
- Reference-free (COMET-Kiwi, GEMBA, QAFactEval) enables decode-time and production-time patterns: MBR decoding, filtering, monitoring — pattern coverage in the parent-task modules.
- LLM-as-judge (GEMBA-style) is competitive on high-resource pairs; not a substitute for human eval on consequential shipping decisions.
- The common failure modes are all reproducibility bugs: unpinned tokeniser, unnamed checkpoint, silent averaging across a heterogeneous language set, missing paired bootstrap on system-vs-system deltas.
