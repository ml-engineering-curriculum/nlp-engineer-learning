# Benchmark Suites: GLUE, SuperGLUE, XTREME, FLORES, MTEB, lm-evaluation-harness

## Motivation

Benchmark suites give you a canonical basket of tasks, a canonical scoring convention, and — critically — a canonical way to compare your model to every other model in the literature. Picking the wrong suite for the model you are evaluating produces numbers that read as impressive on the leaderboard but do not predict production behaviour. Picking the right suite but using it uncritically — pretending its coverage is exhaustive, or ignoring its known contamination and saturation — leads to overclaims.

This chapter walks the six benchmark suites you will encounter most as an NLP engineer: **GLUE** and **SuperGLUE** for English NLU, **XTREME** for cross-lingual NLU, **FLORES-200** for multilingual MT, **MTEB** for text embeddings, and **`lm-evaluation-harness`** for zero/few-shot LLM evaluation. For each: what it measures, when to reach for it, and where it falls short.

## GLUE — the English NLU sampler

**GLUE** (Wang et al., ["GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding"](https://arxiv.org/abs/1804.07461), *ICLR 2019*) packages nine English NLU tasks under one evaluation harness:

- **CoLA** (Warstadt et al., 2018) — linguistic acceptability, MCC.
- **SST-2** — sentiment (binary), accuracy.
- **MRPC** — paraphrase (binary), accuracy + F1.
- **QQP** — paraphrase from Quora, accuracy + F1.
- **STS-B** — sentence similarity (regression), Pearson + Spearman.
- **MNLI** — multi-genre NLI (3-way), accuracy on matched + mismatched dev.
- **QNLI** — SQuAD-derived NLI (binary), accuracy.
- **RTE** — textual entailment (binary), accuracy.
- **WNLI** — Winograd NLI (binary), accuracy. (Historically buggy — most models report but discount it.)

Reported as per-task numbers *and* the "GLUE score" (macro-average). Diagnostic set (an NLI probe over linguistic phenomena) reports per-category accuracy.

**When to reach for GLUE.**

- Comparing a new fine-tuned encoder (BERT / RoBERTa / DeBERTa / ELECTRA lineage) to the literature.
- Quick sanity-check that a training pipeline works — GLUE fine-tuning takes hours on a single GPU.

**Limitations.**

- **Saturated.** Top models score above the human baseline on the macro-average since ~2020. Discriminating current models on GLUE is like discriminating athletes on 100 m sprint time to the nearest millisecond — the tail is dominated by run-to-run noise. The GLUE homepage now recommends SuperGLUE for new work.
- **English only.**
- **Small test sets.** Many tasks have <5 000 dev examples. Statistical significance on small deltas is hard.
- **WNLI is broken.** A negative baseline (always predict the majority class) achieves higher accuracy than most models. Community convention: report it, discount it in the average.
- **Contamination-prone.** GLUE data is on the open web; most large pretrained LMs have seen it. Chapter 08 covers detection.

## SuperGLUE — harder NLU

**SuperGLUE** (Wang et al., ["SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems"](https://arxiv.org/abs/1905.00537), *NeurIPS 2019*) replaces GLUE with eight harder tasks: BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC. All harder to game with lexical heuristics; broader task-format variety (span selection, multi-sentence reasoning, word sense).

**When to reach for SuperGLUE.**

- Same use as GLUE but for a model you expect to be at or above the GLUE ceiling. Discriminates current SOTA models more effectively than GLUE.

**Limitations.**

- **Also approaching saturation.** Top-tier LLMs are at or above human baselines on most tasks.
- **English only.**
- **CB has 250 examples in the dev set** — deltas below a few percent are not distinguishable.
- **Contamination.** Same concerns as GLUE.

For English NLU evaluation of large post-2022 models, GLUE / SuperGLUE alone are not enough. Combine with MMLU-style knowledge probes and open-ended benchmarks (chapter's `lm-evaluation-harness` coverage below).

## XTREME and XTREME-R — cross-lingual NLU

**XTREME** (Hu et al., ["XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalisation"](https://arxiv.org/abs/2003.11080), *ICML 2020*) evaluates cross-lingual transfer across 40 languages and nine tasks: XNLI, PAWS-X (paraphrase), POS tagging, NER (WikiANN), XQuAD + MLQA + TyDi (QA), BUCC + Tatoeba (retrieval).

The evaluation protocol: fine-tune on English training data, evaluate on all 40 languages (zero-shot cross-lingual transfer). Report per-language and the macro-average.

**XTREME-R** (Ruder et al., ["XTREME-R: Towards More Challenging and Nuanced Multilingual Evaluation"](https://arxiv.org/abs/2104.07412), *EMNLP 2021*) refines XTREME — drops saturated tasks (PAWS-X, WikiANN NER), adds harder ones (MASAKHANER, LAReQA, XCOPA, MEWSLI-X), and adds a diagnostic suite for typological analysis.

**When to reach for XTREME / XTREME-R.**

- Evaluating a multilingual encoder (XLM-R, mDeBERTa, mT5) for cross-lingual transfer.
- Comparing a cross-lingual fine-tuning approach (e.g., translate-train vs. zero-shot) across a broad language sample.

**Limitations.**

- **Language coverage is skewed toward high-resource languages** even in the "40 languages" formulation. Report per-language and highlight low-resource performance; do not lean on the macro-average.
- **WikiANN NER is silver-standard** — auto-generated from Wikipedia link structure, not gold. WikiANN scores overstate real NER performance. XTREME-R drops it in favour of MASAKHANER (gold-standard African languages NER).
- **Zero-shot transfer** is one axis of cross-lingual evaluation — few-shot and translate-train also matter. Report all three when the resources exist.

## FLORES-200 — multilingual MT

**FLORES-200** (part of the NLLB release; NLLB Team, ["No Language Left Behind"](https://arxiv.org/abs/2207.04672), *2022*; the original FLORES-101 in Goyal et al., ["The FLORES-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation"](https://arxiv.org/abs/2106.03193), *TACL 2022*) is a professionally-translated MT evaluation set covering 200 languages, with `dev` (997 sentences), `devtest` (1 012 sentences), and blind `test` (2 009 sentences) splits. Every language pair is many-to-many (source is available in all languages).

**When to reach for FLORES.**

- Any multilingual MT evaluation. FLORES-200 is the current MT evaluation standard for cross-language comparability.
- Reporting on a language pair: use `devtest` (public reference-based) and, if you have leaderboard access, the blind `test`.

**Limitations.**

- **Wikipedia-derived source text.** Genre-narrow — encyclopedic prose. Does not stress conversational, colloquial, or code-mixed inputs.
- **One reference per pair.** Metric variance is higher than on multi-reference test sets. Use paired bootstrap and consider chrF or COMET (less sensitive to single-reference noise than BLEU).
- **Test set is blind.** Only `dev` and `devtest` are public references; the public `test` is unblinded via the leaderboard, but you must submit to compare on it. For iterative development report on `devtest`; save `test` for final submission.

Complementary MT test sets: **NTREX-128** (news, 128 languages), **TICO-19** (COVID-19 information, LMIC-focus), **WMT News Test** (annual, plus specialised sub-tracks — biomedical, chat, terminology, literary).

## MTEB — text embedding benchmarks

**MTEB** (Muennighoff et al., ["MTEB: Massive Text Embedding Benchmark"](https://arxiv.org/abs/2210.07316), *EACL 2023*) is the current standard for evaluating text embedding models. Covers eight task types across 58 datasets, most English but with a growing multilingual arm:

- **Bitext mining** — cross-lingual sentence pair retrieval.
- **Classification** — embedding + logistic regression on top.
- **Clustering** — V-measure of unsupervised clusters on embeddings.
- **Pair classification** — binary similarity (SICK-R, etc.).
- **Reranking** — MRR / MAP for query-doc rerankers.
- **Retrieval** — nDCG@10 on BEIR-style retrieval tasks.
- **STS** — Spearman correlation on standard STS benchmarks.
- **Summarisation** — score generated summaries against multiple references via embedding similarity.

Reported as per-task and per-category averages. The MTEB leaderboard on Hugging Face is the go-to reference.

**When to reach for MTEB.**

- Evaluating a sentence-embedding model — SBERT lineage, `e5`, `bge`, `gte`, `nomic-embed`, `Qwen3-embedding` families.
- Choosing an embedding model for a RAG / retrieval / clustering deployment.

**Limitations.**

- **English-heavy.** The multilingual MTEB subset (MMTEB, Muennighoff et al., 2024 extension) is smaller and less mature. For multilingual retrieval-heavy use cases, report on multilingual BEIR sub-datasets plus MMTEB where available.
- **Retrieval tasks dominate the aggregate.** A model that is strong on retrieval and weak on STS can look strong overall. Report per-task-category.
- **Contamination.** Many MTEB datasets predate current pretrained models. Retrieval tasks are especially prone — the query and passage text are on the open web.

## `lm-evaluation-harness` — zero/few-shot LLM eval

**`lm-evaluation-harness`** (EleutherAI, [`EleutherAI/lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness); Gao et al., ["A framework for few-shot language model evaluation"](https://zenodo.org/records/10256836), *2021 onward*) is the standard framework for evaluating causal LLMs in zero/few-shot settings. Ships hundreds of tasks under a common runner — MMLU, ARC, HellaSwag, TruthfulQA, GSM8K, HumanEval, BBH, WinoGrande, PIQA, BoolQ, plus perplexity benchmarks (WikiText, LAMBADA) and multilingual sub-suites.

Every serious open-model release reports headline numbers from `lm-evaluation-harness` for reproducibility.

**When to reach for `lm-evaluation-harness`.**

- Zero/few-shot evaluation of a causal LM you did not fine-tune per task.
- Comparing a new base or instruct model to public releases.

**Critical usage notes.**

- **Report the harness version.** The scoring conventions, few-shot prompt formats, and per-task normalisers have changed non-trivially between versions. `v0.4.x` numbers are not always comparable to `v0.3.x` numbers on the same task. Cite `--include_path` and the git SHA.
- **Choose the metric variant deliberately.** Multiple-choice tasks ship both `acc` (raw log-likelihood argmax over choices) and `acc_norm` (log-likelihood normalised by choice length in characters or tokens). `acc_norm` corrects for length bias; different papers report different variants — align to whichever your comparison targets.
- **Report the few-shot count.** `n=0`, `n=5`, `n=25` all appear in the literature. Same model, same task, different `n` → different numbers.
- **Prompt template matters.** `lm-evaluation-harness` has canonical prompt templates per task; deviating from them produces different numbers. Note when you use a custom template.

**Limitations.**

- **Multiple-choice framing bias.** Many `lm-eval-harness` tasks are multiple-choice — the model scores each answer's log-likelihood and picks the highest. This is not how the model would be used in production and correlates imperfectly with generation-quality metrics.
- **Coverage gaps.** Long-context capability, instruction following on open-ended tasks, and safety-relevant behaviour are under-covered. Complement with `AlpacaEval`, `MT-Bench`, `Arena-Hard`, or a task-specific eval.
- **Contamination.** MMLU, ARC, HellaSwag, TriviaQA — all on the open web, all plausibly in pretraining data of recent LLMs. Chapter 08 covers detection.

## Adjacent suites worth knowing

- **BEIR** (Thakur et al., ["BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models"](https://arxiv.org/abs/2104.08663), *NeurIPS 2021 D&B*) — 18 zero-shot IR datasets. Subsumed into MTEB but often reported separately for retrieval.
- **HELM** (Liang et al., ["Holistic Evaluation of Language Models"](https://arxiv.org/abs/2211.09110), *2022, ongoing*) — Stanford's cross-cutting benchmark that evaluates accuracy *plus* calibration, robustness, fairness, bias, toxicity, and efficiency across many tasks. Slower and heavier than `lm-eval-harness`; complementary.
- **Big-Bench** (Srivastava et al., ["Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models"](https://arxiv.org/abs/2206.04615), *2022*) — 200+ tasks covering "capability probes" the community submitted. **BBH** (Suzgun et al., ["Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them"](https://arxiv.org/abs/2210.09261)) is the 23-task subset most reported.
- **GLUE / SuperGLUE / XTREME are all included in `lm-eval-harness`** in adapted zero/few-shot form. The numbers are not directly comparable to fine-tuned-encoder results on the same tasks — different problem framing.

## The rule for picking a suite

- **Fine-tuning an encoder for English NLU** → SuperGLUE (or GLUE for legacy comparison).
- **Fine-tuning a multilingual encoder for NLU** → XTREME-R.
- **Training a multilingual MT model** → FLORES-200 devtest + task-specific test sets (WMT, NTREX, TICO-19).
- **Training an embedding model** → MTEB.
- **Evaluating a base or instruct causal LM zero/few-shot** → `lm-evaluation-harness` (pinned version and template).
- **Evaluating a chat-tuned LM** → Combination of `lm-eval-harness` core tasks, `MT-Bench` / `AlpacaEval` / `Arena-Hard` for open-ended, and human eval (chapter 09) for consequential decisions.
- **Any of the above** → Contamination check on the specific test slices you will report (chapter 08). Statistical significance (chapter 07). Per-slice reporting (per language, per subject, per difficulty).

## Common failure modes

- **Averaging over a heterogeneous suite as if the average were the interpretation.** A GLUE macro-average or an MTEB overall score hides which tasks the model is broken on. Report per-task.
- **Cross-version comparisons.** `lm-eval-harness v0.3.x` MMLU numbers vs. `v0.4.x` MMLU numbers on the same model can differ by 2–3 points due to prompt-template changes. Cite the version.
- **Cross-framing comparisons.** GLUE fine-tuned RoBERTa MNLI 90.2 vs. `lm-eval-harness` few-shot LLaMA MNLI 68.0 does *not* mean RoBERTa is better at NLI — the problem framing is different.
- **Contamination denial.** "The model was released after the benchmark, so it can't be contaminated." Only true if the benchmark's *training* set (or public web-hosted eval set) was released after the model's pretraining data cutoff. Verify — chapter 08.
- **Saturated leaderboard as ranking signal.** GLUE macro-averages within 0.3 of the human baseline are noise. Move to a harder benchmark or to task-specific human eval.

## Chapter summary

- GLUE / SuperGLUE: English NLU sampler for fine-tuned encoders. Now saturated at the top; use SuperGLUE for current models, treat WNLI carefully, watch contamination.
- XTREME / XTREME-R: cross-lingual NLU across 40 languages. Report per-language and highlight low-resource; XTREME-R drops silver-standard tasks like WikiANN.
- FLORES-200 devtest: current standard for multilingual MT. Wikipedia-genre — complement with NTREX, TICO-19, WMT sub-tracks for other genres.
- MTEB: standard for text embeddings across retrieval, STS, classification, clustering, reranking. Report per-task-category; English-heavy; growing multilingual arm.
- `lm-evaluation-harness`: standard for zero/few-shot causal LM evaluation. Pin the version, `n`-shot, prompt template, and `acc` vs. `acc_norm` variant.
- Adjacent: HELM (accuracy + calibration + fairness + robustness), Big-Bench / BBH (broad capability probes), BEIR (retrieval, subsumed into MTEB).
- Every suite is a *sampler* with known coverage gaps. Report per-task, not per-suite-average, and pair the suite with contamination checks (chapter 08) and statistical significance (chapter 07).
