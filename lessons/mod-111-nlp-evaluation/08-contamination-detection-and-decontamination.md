# Contamination Detection and Decontamination

## Motivation

An "evaluation" of an LLM against a benchmark it saw during pretraining is not an evaluation — it is a partial memorisation check. Every popular benchmark (SQuAD, MMLU, GSM8K, HellaSwag, ARC, HumanEval, GLUE, SuperGLUE, WMT, FLORES) has some non-trivial fraction of its test set indexable on the open web, and every large pretrained model plausibly ingested some of it. The reported number then reflects a mix of *capability* and *lookup*, and the two are not separable without a contamination audit.

This chapter walks the two questions you need to answer for any evaluation you ship:

1. **Detection.** For the specific test set and specific pretrained model you are about to score, how contaminated is the pretraining data with respect to your eval?
2. **Design.** How do you construct a held-out eval that is *contamination-resistant by construction* — either because the data postdates pretraining, or because the metric is robust to memorised gold answers?

The literature on contamination is fast-moving; this chapter covers the stable core (n-gram membership tests, canaries, publication-cutoff strategy, per-example accuracy vs. familiarity) and points to the current tooling.

## Where contamination comes from

Three routes into pretraining corpora:

- **Direct.** The test set is on the open web (Hugging Face dataset page, benchmark homepage, GitHub) at pretraining scrape time. Common Crawl scrapes benchmark pages regularly.
- **Indirect via papers and blog posts.** Even if the raw test file is not on the web, individual test items appear in papers, blog posts, and Stack Overflow answers that *are* scraped. LLMs may not have seen the whole SQuAD dev, but the model can plausibly recite the top-5 most-quoted SQuAD questions.
- **Indirect via derivative datasets.** Training corpora built for adjacent tasks often *include* test items — MMLU questions appear in flashcard sites, ARC questions appear on quiz sites, HumanEval prompts have been discussed in Stack Overflow. `HuggingFace Hub` and `The Pile` both contain such derivative content.

Contamination is not binary; it is a matter of degree. A useful mental model: for each test item, ask "did the model see this item's *input* during pretraining, and did it also see the *gold answer*?" Both-seen is the most dangerous — the model can recall the pair directly.

## Detection: n-gram membership tests

The cheap detection procedure is a sliding-window n-gram overlap between test items and pretraining data:

- Take each test item's input (question, source sentence, prompt, prefix).
- Extract character-level or token-level n-grams (typical: 13-gram tokens or 50-char character n-grams).
- Check whether any n-gram appears in the pretraining corpus.
- If a threshold fraction (say, 8 of 13 tokens) matches consecutively, mark the item as contaminated.

The GPT-3 paper (Brown et al., ["Language Models are Few-Shot Learners"](https://arxiv.org/abs/2005.14165), *NeurIPS 2020*) formalised this as their contamination check: 13-gram overlap between test items and Common Crawl scrape, with an ablation reporting metrics on clean and dirty subsets separately. The PaLM (Chowdhery et al., ["PaLM: Scaling Language Modeling with Pathways"](https://arxiv.org/abs/2204.02311)) and LLaMA (Touvron et al., ["LLaMA: Open and Efficient Foundation Language Models"](https://arxiv.org/abs/2302.13971)) technical reports use similar 13-gram checks.

If you have access to the pretraining corpus (open-source models — LLaMA 2/3, OLMo, Pythia, MPT — publish enough about their data mix that you can approximate), run the check directly. If you do not (proprietary models), you cannot definitively test membership — only *behaviour-based* proxies remain (below).

Open-source tools:

- **`decontamination` in Big-Bench** ([Big-Bench repo](https://github.com/google/BIG-bench), see `decontamination_check`) — 13-gram check against a supplied corpus.
- **`nl-augmenter`, `datasets_helper`, `deduplicate-text-datasets`** (Google's `dedupe`) — near-duplicate detection between corpora at scale using MinHash / LSH.
- **`bigscience-workshop/data-preparation`** — includes decontamination scripts used in BLOOM.

For a specific test set:

```python
from datasketch import MinHash, MinHashLSH

def shingles(text, k=13):
    tokens = text.lower().split()
    for i in range(len(tokens) - k + 1):
        yield " ".join(tokens[i : i + k])

def build_lsh(pretraining_corpus_iter, threshold=0.5, num_perm=128):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, doc in enumerate(pretraining_corpus_iter):
        m = MinHash(num_perm=num_perm)
        for s in shingles(doc):
            m.update(s.encode())
        lsh.insert(f"doc-{i}", m)
    return lsh

def contamination_check(test_items, lsh):
    hits = []
    for idx, item in enumerate(test_items):
        m = MinHash(num_perm=128)
        for s in shingles(item):
            m.update(s.encode())
        if lsh.query(m):
            hits.append(idx)
    return hits
```

The output is a list of test items whose input matches an n-gram window in the pretraining corpus. Report:

- **Contamination rate.** `|hits| / |test|`. Any non-trivial number (> a few percent) is a signal.
- **Clean vs. dirty subset metrics.** Compute the primary metric on the contaminated subset and on the clean subset separately. If the model scores substantially higher on the dirty subset, some of the reported number is memorisation.

## Detection when you cannot see the pretraining data

For closed-weight, closed-corpus models (GPT-4, Claude, Gemini, etc.), direct membership testing is impossible. Two behavioural proxies:

### Verbatim continuation test

Show the model the first half of a test item; check whether the completion matches the second half token-for-token. If the model can complete a low-entropy string it has never seen, that is the ceiling for generalisation; if it can complete a specific test-set string more accurately than a similar-format string it plausibly has not seen, contamination is likely.

Golchin & Surdeanu (["Time Travel in LLMs: Tracing Data Contamination in Large Language Models"](https://arxiv.org/abs/2308.08493), *ICLR 2024*) formalise this as the "guided instruction" test — ask the model to reproduce a specific split of a specific dataset and check verbatim overlap.

### Membership inference via log-probability

If the model assigns a much higher likelihood to test-set text than to distributionally similar non-test text, membership is likely. Formalised by Shi et al. (["Detecting Pretraining Data from Large Language Models"](https://arxiv.org/abs/2310.16789), *ICLR 2024*) as **Min-K% Prob** — the mean log-probability of the lowest-probability K% of tokens is a stronger signal than raw perplexity, because on unseen text the model's *worst* tokens are much worse than on seen text.

Both proxies produce probabilistic evidence, not proof. For closed models, report both, but present the conclusion as "consistent with contamination" rather than "the model was trained on this."

## Design: contamination-resistant held-out evaluation

Detection is remediation after the fact. The stronger move is to design an evaluation that cannot be contaminated by construction. Five patterns.

### 1. Time-boxed data

Use data published *after* the model's pretraining data cutoff. Every major closed model publishes a rough cutoff (GPT-4 Turbo: ~2023-04; Claude 3.5: ~2024-04; Gemini 1.5: varies). For open models, cutoffs are published exactly. If your test items were on the web only after the cutoff, the model has not seen them.

Concrete recipes:

- **News-of-the-month** — sample news articles from a cutoff-postdating window and construct QA / summarisation tasks over them.
- **Recent-arXiv classification** — fine-grained topic labels on papers submitted after the cutoff.
- **Recent-GitHub** — code understanding tasks on repositories created after cutoff.
- **Rolling benchmarks** — continuously updated evaluation streams. LiveBench (White et al., ["LiveBench: A Challenging, Contamination-Free LLM Benchmark"](https://arxiv.org/abs/2406.19314), *ICLR 2025*) is the canonical example.

Time-boxing does not protect against models that re-scrape and retrain — always name the model version and the data cutoff you are asserting against.

### 2. Canary strings

Insert a unique canary — a long random ASCII string like `IPS_CANARY_2024_2Kg9Wq...` — into any evaluation you distribute privately. If a later pretrained model regurgitates the canary when prompted with its prefix, the model saw your eval.

The BIG-bench organisers used canary strings for exactly this reason (Srivastava et al., ["Beyond the Imitation Game"](https://arxiv.org/abs/2206.04615)). If you publish an eval publicly, publish the canary alongside instructions to exclude the canary token from training — model providers can then verify.

Canaries are a *forensic* tool, not a prevention. Once a model has seen the canary, remedy is the next model.

### 3. Held-out-by-construction generation

For any task where you can synthesise the test items from a generator, hold out the generator (not just the outputs) and generate fresh test data per evaluation run. Two patterns:

- **Templated tasks.** GSM8K-style math problems can be re-instantiated from templates with fresh numbers; the model cannot memorise fresh instantiations. See GSM-Symbolic (Mirzadeh et al., ["GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models"](https://arxiv.org/abs/2410.05229), *ICLR 2025*).
- **Procedural task generators.** ARC-AGI-style tasks (Chollet, ["On the Measure of Intelligence"](https://arxiv.org/abs/1911.01547), *2019*) whose per-item generator is held private cannot be memorised in aggregate.

The trade-off: fresh generation reduces contamination but changes what you are measuring — the model's competence on the *generator's distribution*, not the original benchmark's fixed items.

### 4. Private test sets

Withhold the test set entirely; require submissions through a leaderboard that runs eval server-side and returns only the aggregate. FLORES `test` (chapter 06) works this way; Kaggle competitions do; some enterprise-facing benchmarks do.

Weaknesses:

- Iterating requires submitting many times, which itself leaks information about the test distribution (leaderboard-overfitting). Cap per-team submission rate.
- The organiser must be trusted not to disclose the test set — and to keep it off the training corpora of the models they later evaluate.

### 5. Adversarial and perturbed variants

Some tasks let you check whether the model's answer depends on the *content* of the question or on surface-form memorisation. Modify the question in a way that changes the answer but is superficially similar; the memorising model will still emit the original answer. Applied by MMLU-Redux (Gema et al., ["Are We Done with MMLU?"](https://arxiv.org/abs/2406.04127), *NAACL 2025 Findings*) and by contaminated-benchmark re-releases with paraphrased inputs.

Adversarial variants also stress robustness (see mod-113 chapter on robustness evaluation).

## Decontamination: cleaning training data

If you *train* a model and want its evaluation numbers to be honest, decontaminate the training data:

- **Extract all n-grams from your target eval sets** (SQuAD, MMLU, ARC, HumanEval, ...).
- **Remove documents from training** that overlap with those n-grams above a threshold — either drop the whole document (LLaMA 2 approach) or mask the overlapping spans.
- **Report the decontamination protocol** in the model card: which evals were decontaminated against, what the n-gram threshold was, how many training documents were dropped.

The LLaMA 2 paper (Touvron et al., ["Llama 2: Open Foundation and Fine-Tuned Chat Models"](https://arxiv.org/abs/2307.09288), *2023*) and OLMo (Groeneveld et al., ["OLMo: Accelerating the Science of Language Models"](https://arxiv.org/abs/2402.00838), *2024*) publish their decontamination pipelines — read them as reference protocols.

Two decontamination pitfalls:

- **False negatives from paraphrase.** N-gram decontamination catches verbatim overlap but not paraphrase; a paraphrased SQuAD question in training data slips through and still leaks answer knowledge. Semantic decontamination (embed both, remove near-duplicates by cosine similarity) is heavier but stronger.
- **False positives from common phrasing.** "What is the capital of" appears in millions of non-benchmark documents. Threshold carefully or use per-example checks rather than blanket n-gram matching. Aim for high-precision matching on rare 13-grams, not high-recall on frequent ones.

## Reporting: what a contamination audit looks like in a model card

The IMDB-of-model-cards convention emerging in 2024–2025 (published in HELM, in the model cards for OLMo / Pythia / LLaMA 3) includes a **contamination table**:

- **Per benchmark**: the contamination detection method (13-gram, semantic, Min-K%), the contamination rate, the metric on the clean subset, the metric on the contaminated subset, and the delta.

A model that scores 84 % on MMLU with 3 % contamination and equal clean/dirty performance is more credible than a model that scores 87 % on MMLU with 15 % contamination and a 10-point clean/dirty gap. Report both numbers; let the reader judge.

## Common failure modes

- **"Contamination is not our problem — we used the standard benchmark."** The standard benchmark is exactly the one most likely to be contaminated. Check.
- **Zero contamination reported without describing the method.** Meaningless — the strictness of the check drives the number. Cite the tool and threshold.
- **Averaging clean and dirty subsets.** Report them separately; the delta is the interesting number.
- **Assuming publication-cutoff protects.** Cutoffs are approximate and closed-model providers often re-scrape between releases without publishing the new cutoff. Time-boxing is a partial defence, not a proof.
- **Trusting a private test set that has been iterated on for years.** Every submission reveals bits about the test distribution. Rotate the test set periodically.
- **Publishing a canary without instructions to exclude.** Model providers cannot exclude a canary they do not know about.

## Chapter summary

- Contamination is the presence of eval items (or their answers) in pretraining data. It inflates the eval metric with memorisation and is nearly universal on popular benchmarks.
- **Detection.** For open-corpus models, run a 13-gram or MinHash-LSH overlap check between the test set and the pretraining data. Report the contamination rate and the clean-vs-dirty subset metrics. For closed models, use behavioural proxies (verbatim continuation, Min-K% Prob) — evidence, not proof.
- **Design contamination-resistant evals.** Prefer time-boxed data (LiveBench pattern), canary-instrumented private evals, procedurally-generated fresh items (GSM-Symbolic pattern), private test sets, or adversarial paraphrases (MMLU-Redux pattern).
- **Decontaminate training data.** Extract eval n-grams, drop training documents that overlap; report the protocol in the model card (LLaMA 2 / OLMo reference protocols).
- A contamination audit is now table stakes for an honest model card: contamination rate, method, and clean-vs-dirty delta per benchmark.
- The most reliable long-term move is data that postdates the model plus a canary — everything else is a defence in depth.
