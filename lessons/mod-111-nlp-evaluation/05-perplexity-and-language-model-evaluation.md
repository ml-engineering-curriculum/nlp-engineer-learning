# Perplexity and Language-Model Evaluation

## Motivation

Perplexity is the default intrinsic evaluation of language models — the number every LM paper reports, the metric every base-model training run tracks, and one of the metrics most likely to be reported incorrectly. Two LMs with different tokenisers cannot be compared on perplexity. Perplexity on the test set an LM saw during pretraining is a training loss, not an evaluation. Sliding-window perplexity computed the wrong way over long documents can be off by an order of magnitude.

This chapter defines perplexity crisply, walks the cross-tokeniser normalisation (bits-per-byte / bits-per-character), covers correct sliding-window evaluation, and finishes with when to reach for perplexity, when to reach for downstream task metrics, and when perplexity is straight-up not the right instrument.

## Perplexity: the metric

For a token sequence $x = (x_1, \dots, x_T)$ and a language model $p_\theta$, the average negative log-likelihood per token is:

$$\text{NLL}(x) = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

**Perplexity** is the exponent of that:

$$\text{PPL}(x) = \exp\left( -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}) \right)$$

Intuitively: the geometric mean of $1 / p_\theta(x_t \mid x_{<t})$ over the sequence. PPL = 1 is perfect prediction; PPL = |vocab| is chance.

Corpus-level PPL is computed over the concatenated token stream (or equivalently, the exponentiated *token-count-weighted* mean of per-sequence NLLs). Do not average per-sequence PPLs directly — that over-weights short sequences.

```python
import torch, math
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2").eval().cuda()
tok   = AutoTokenizer.from_pretrained("gpt2")

def corpus_ppl(texts):
    total_nll, total_tokens = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.cuda()
        with torch.inference_mode():
            out = model(ids, labels=ids)
        # HF returns mean NLL per token; multiply by token count to get sum
        n = ids.numel()
        total_nll    += out.loss.item() * n
        total_tokens += n
    return math.exp(total_nll / total_tokens)
```

## The cross-tokeniser problem

Perplexity is defined *over tokens* — and tokens depend on the tokeniser. A model with a byte-level BPE tokeniser splits English into more tokens per word than a model with a large SentencePiece vocab; the byte-BPE model can achieve a *lower* PPL because each next-token prediction is easier (smaller local branching factor) even if the two models are equally good at modelling English.

**Two models with different tokenisers cannot be compared on perplexity.** This is the single most-common LM-evaluation mistake in blog posts and casual benchmarks. The fixes:

### Bits per byte (BPB)

Normalise the total log-loss by the number of *bytes* in the raw text, not the number of tokens:

$$\text{BPB}(x) = \frac{1}{|x|_\text{bytes}} \sum_{t=1}^{T} -\log_2 p_\theta(x_t \mid x_{<t})$$

BPB is tokeniser-invariant — any model that assigns a probability distribution to the same byte string produces a comparable number. The Pile (Gao et al., ["The Pile: An 800GB Dataset of Diverse Text for Language Modeling"](https://arxiv.org/abs/2101.00027), *2020*) established BPB as the standard cross-model reporting metric for pretraining evaluation.

```python
def bits_per_byte(nll_sum_nats, text):
    return nll_sum_nats / (len(text.encode("utf-8")) * math.log(2))
```

Report BPB whenever you compare LMs with different tokenisers. Report PPL alongside for same-tokeniser continuity with prior work.

### Bits per character (BPC)

$$\text{BPC}(x) = \frac{1}{|x|_\text{chars}} \sum_{t=1}^{T} -\log_2 p_\theta(x_t \mid x_{<t})$$

Character-level analog of BPB. Used historically on character-level LMs (enwik8, text8 benchmarks); on modern byte-level tokenisers BPB is equivalent up to a constant per-language byte-to-character ratio.

### Word-level perplexity

For text that is naturally word-segmented (English, most European languages), word-level PPL normalises per whitespace word instead of per subword token. Historically the metric for n-gram LMs on Penn Treebank and WikiText. Report BPB in modern work; word-level PPL is a legacy metric.

## Long-document perplexity: sliding-window evaluation

Modern LMs have context limits (2K, 4K, 8K, 32K, ...). To score perplexity on a document longer than the context, you slide a window across the document and score the *newly-visible* tokens in each window. Naïve non-overlapping chunking underestimates PPL because the first token of every chunk has no context.

The correct **strided evaluation** (as used in the GPT-2 paper — Radford et al., ["Language Models are Unsupervised Multitask Learners"](https://openai.com/research/better-language-models), *OpenAI 2019* — and codified in the Hugging Face `perplexity of fixed-length models` docs):

- Choose a stride `s` (smaller than the context length `L`).
- Slide by `s` at each step. At each window, score only the last `s` tokens (the "new" ones), conditioning on the preceding `L - s` tokens.
- Sum log-losses across all "scored" tokens, divide by that count, exponentiate.

```python
def strided_ppl(text, model, tok, stride=512, max_length=1024, device="cuda"):
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    nll_sum, n = 0.0, 0
    prev_end = 0
    for begin in range(0, ids.size(1), stride):
        end = min(begin + max_length, ids.size(1))
        target_len = end - prev_end
        input_ids  = ids[:, begin:end]
        target_ids = input_ids.clone()
        target_ids[:, :-target_len] = -100     # ignore already-scored context
        with torch.inference_mode():
            out = model(input_ids, labels=target_ids)
        nll_sum += out.loss.item() * target_len
        n       += target_len
        prev_end = end
        if end == ids.size(1):
            break
    return math.exp(nll_sum / n)
```

Two configuration decisions:

- **Stride vs. context length.** Smaller stride = more context per predicted token = lower (better) PPL but more compute. Report the stride. The GPT-2 evaluation uses `stride=512, max_length=1024` on WikiText-103.
- **The first `max_length - stride` tokens.** These predict with truncated context; either exclude them or acknowledge the bias. Most protocols include them and accept the small upward bias for the sake of comparability across implementations.

## Standard LM benchmarks

The perplexity benchmarks that appear in every LM paper:

- **WikiText-103** (Merity et al., ["Pointer Sentinel Mixture Models"](https://arxiv.org/abs/1609.07843), *ICLR 2017*) — 103M tokens of Wikipedia articles, kept intact (long documents). The standard word-level PPL benchmark and the standard sliding-window PPL benchmark.
- **The Pile validation set** — 26 sub-domains covering Common Crawl, GitHub, arXiv, PubMed, Books, StackExchange, HackerNews, etc. Report per-domain BPB (the head domains — Common Crawl, GitHub — dominate the aggregate; the tail domains are where domain-shift shows).
- **C4 validation** (Raffel et al., ["Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"](https://arxiv.org/abs/1910.10683), *JMLR 2020*) — the cleaned Common Crawl derivative used to train T5 and related models.
- **PG-19** (Rae et al., ["Compressive Transformers for Long-Range Sequence Modelling"](https://arxiv.org/abs/1911.05507), *ICLR 2020*) — long books, tests long-context modelling.
- **LAMBADA** (Paperno et al., ["The LAMBADA dataset: Word prediction requiring a broad discourse context"](https://arxiv.org/abs/1606.06031), *ACL 2016*) — narrative passages where the last-word target is unpredictable from local context, testing broad discourse modelling. Reported as *last-word accuracy* rather than PPL.

The `lm-evaluation-harness` (chapter 06) ships these with canonical implementations; use it and cite the harness version.

## Perplexity and downstream capability — the fraught relationship

Lower PPL on a pretraining-style validation set correlates with better downstream task performance, but the correlation is imperfect and gets worse as the base model gets better. Two well-documented divergences:

- **Instruction-tuned or RLHF'd models often have higher raw PPL** on generic held-out text than their base models — they concentrated probability mass on instruction-conforming outputs at the cost of some breadth. PPL is a poor proxy for chat-model quality.
- **Two models with similar PPL can have very different downstream capabilities** — architecture, data mix, and training dynamics all shift downstream capability at fixed PPL.

**Rule of thumb:** perplexity is the right metric for *base language model training progress* (are you making the loss go down?) and for *comparing base LMs on the same held-out corpus with the same tokeniser*. It is not the right metric for capability evaluation of finetuned or instruction-following models — for that, reach for task benchmarks (chapter 06) and human evaluation (chapter 09).

## Common failure modes

- **Comparing PPL across tokenisers.** Report BPB.
- **Averaging per-sequence PPL directly.** Averages the geometric means over sequences, which is not the corpus PPL. Weight by token count or compute over the concatenated stream.
- **Sliding-window PPL with the wrong stride policy.** Naïve non-overlapping chunks underestimate; overlapping-and-scoring-everything double-counts. Use the strided-eval pattern above and report the stride.
- **PPL on data the model was trained on.** That is the training loss, not evaluation. Chapter 08 covers contamination detection; for LMs, held-out means held out from *pretraining*, not just from finetuning.
- **PPL as a proxy for instruction-following quality.** It is not. Use downstream task benchmarks and human eval.
- **PPL on numeric or code text without acknowledging it is a different distribution.** Report per-domain BPB; the C4-validation aggregate hides code-vs-prose gaps.
- **Reporting PPL without the corpus.** "PPL of 12.3" means nothing without the test set. WikiText-103? The Pile validation? Your team's held-out slice? Name it.

## Adjacent intrinsic metrics

- **Bits per character (BPC)** — legacy for character LMs; equivalent up to a language-specific constant to BPB.
- **Zero-shot task loss.** For LMs evaluated as prompt-conditioned classifiers (e.g., BoolQ zero-shot), the standard metric is task accuracy, not PPL — see chapter 06's coverage of the `lm-evaluation-harness`.
- **Cloze accuracy.** LAMBADA's last-word accuracy is a cloze test scored as EM on the final token. Correlates better with downstream capability than raw PPL for larger models.
- **Calibration on multiple-choice.** For lm-eval-harness benchmarks that use multiple-choice framing (ARC, HellaSwag, MMLU), some ship both `acc` and `acc_norm` — the latter normalises by the length of each answer choice to correct for length bias.

## Chapter summary

- Perplexity is $\exp(-\frac{1}{T} \sum \log p_\theta(x_t | x_{<t}))$; corpus PPL is computed over the concatenated token stream (or token-count-weighted, not per-sequence-averaged).
- **PPL is not comparable across tokenisers.** Report **bits per byte (BPB)** for cross-tokeniser comparison. The Pile paper established BPB as the standard.
- Long-document PPL requires strided evaluation: slide by `stride`, score only the last `stride` tokens, condition on the preceding `context - stride`. Report the stride and context length.
- Standard benchmarks: WikiText-103, The Pile validation (with per-domain breakdown), C4 validation, PG-19 for long-context, LAMBADA for broad discourse. Use `lm-evaluation-harness` and cite the harness version.
- PPL is the right metric for base-LM training progress and same-tokeniser same-corpus comparisons. It is a poor proxy for instruction-tuned or RLHF'd model quality — use downstream benchmarks and human eval.
- Common failure modes are all boundary conditions: cross-tokeniser comparison, wrong sliding-window stride, per-sequence averaging, testing on training data, and reporting PPL without naming the corpus.
