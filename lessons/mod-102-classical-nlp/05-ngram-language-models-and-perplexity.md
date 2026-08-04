# n-gram Language Models and Perplexity

## Motivation

An n-gram language model estimates the probability of the next token given a short window of previous tokens. It is, by 2026, a dated model for open-domain generation — a transformer decoder is strictly better at that. It is still, in 2026, the model that quietly runs inside production ASR beam-search rescoring, keyboard next-word prediction on low-end devices, spelling correctors, statistical machine translation lattices, and every research paper that reports a "perplexity" number for a language model.

You should be able to train one, smooth it, evaluate it, and reason about when it beats a neural LM on latency, memory, or determinism. This chapter is that curriculum.

## The maximum-likelihood estimate and why it fails

For a sequence `w₁ w₂ … w_T`, a bigram model factors:

```
P(w₁, w₂, …, w_T) = ∏ P(wᵢ | wᵢ₋₁)
```

with the boundary trick `P(w₁ | <s>)` and a stop token `</s>` at the end so probabilities normalise. The maximum-likelihood estimate is just counting:

```
P_MLE(wᵢ | wᵢ₋₁) = count(wᵢ₋₁, wᵢ) / count(wᵢ₋₁)
```

Two failure modes appear immediately:

- **Zero-probability n-grams.** Any bigram not observed in training gets probability zero, which drives sequence probability to zero (and log-likelihood to `-∞`). One unseen bigram poisons the entire evaluation.
- **Data sparsity grows with `n`.** A trigram model over a 50k vocabulary has 1.25 × 10¹⁴ possible trigrams; you observe a few billion. Almost every trigram at inference is unseen.

Smoothing is the collective name for the fixes. All classical LM smoothing techniques trade off predicted probability on frequent n-grams against reserved probability mass for unseen ones.

## Smoothing, in order of sophistication

### Add-`k` (Laplace / Lidstone)

Add a constant `k` (often `1`, sometimes fractional) to every count:

```
P(w | h) = (count(h, w) + k) / (count(h) + k · V)
```

Simple, always positive, and *almost never the right choice for language modelling.* Add-1 systematically over-smooths high-frequency words. Useful mostly for teaching, for small-V toy examples, and for Naive Bayes text classifiers.

### Katz backoff

If the higher-order n-gram was observed, use its discounted probability. Otherwise, "back off" to the shorter context, scaled by a normalisation constant so the total mass sums to 1:

```
P_katz(wᵢ | wᵢ₋ₙ₊₁, …, wᵢ₋₁) =
    d(counts) · P_ML(wᵢ | …)          if count(wᵢ₋ₙ₊₁, …, wᵢ) > 0
    α(context) · P_katz(wᵢ | wᵢ₋ₙ₊₂, …, wᵢ₋₁)  otherwise
```

Reference: Slava Katz, "Estimation of Probabilities from Sparse Data for the Language Model Component of a Speech Recognizer", *IEEE ASSP*, 1987.

### Kneser-Ney (and modified Kneser-Ney)

The state-of-the-art smoothing for n-gram LMs. The insight of Kneser and Ney (1995) is that a lower-order model should count the number of *distinct contexts* a word appears in, not its raw frequency. A word like `Francisco` may be frequent in the corpus, but only after `San`; it should get a small backoff probability because it is unlikely to appear in a novel context.

Modified Kneser-Ney (Chen & Goodman, ["An empirical study of smoothing techniques for language modeling"](https://www.cs.cmu.edu/~roni/11761/PreviousYearsHandouts/chen-goodman-99.pdf), *Computer Speech & Language*, 1999) uses three discount parameters (for singleton, doubleton, and 3+ counts) and remains the dominant classical LM smoothing.

Practically, do not implement smoothing yourself for production. `KenLM` (Kenneth Heafield's C++/Python library, <https://github.com/kpu/kenlm>) implements modified Kneser-Ney at web scale, produces binary models that memory-map in constant time, and is the standard for ASR/MT rescoring. `SRILM` (SRI's older toolkit) implements a wider selection of smoothing algorithms and is still the reference for research work: <http://www.speech.sri.com/projects/srilm/>.

Reference: Reinhard Kneser & Hermann Ney, "Improved backing-off for M-gram language modeling", *ICASSP 1995*.

### Stupid backoff

At web scale (billions of n-grams), even Kneser-Ney is expensive. Brants et al. ("Large Language Models in Machine Translation", *EMNLP 2007*) showed that at sufficient scale, a heuristic — use the MLE if the n-gram was observed, otherwise back off with a fixed multiplicative penalty — matches Kneser-Ney on translation BLEU. Not a probability distribution (does not normalise), but that does not matter for lattice rescoring.

## Perplexity, the standard evaluation

Perplexity is the reported metric for language models, classical and neural:

```
PPL(W) = exp( - (1/N) · Σᵢ log P(wᵢ | context) )
```

Interpretations:

- **Geometric mean of the inverse per-token probability.** Perplexity 100 means the model, on average, is as uncertain about the next token as if it were choosing uniformly among 100 options.
- **Base-`e` exponentiation of cross-entropy** in nats; base-2 gives bits per token (also called BPC when tokens are characters).
- **Cannot be compared across tokenisers.** A per-character model, a per-word model, and a BPE model produce different `N`. You can only compare perplexities *at the same tokenisation and vocabulary*.

Gotchas that catch people:

- **Open-vocabulary evaluation.** If the vocabulary is closed and OOV tokens are mapped to `<unk>`, perplexity is understated (the model gets to "know" its own unknown token). Report OOV rate alongside perplexity. Reference: Alex Graves et al., "Speech Recognition with Deep Recurrent Neural Networks", *ICASSP 2013*, remains the canonical explanation.
- **End-of-sequence tokens.** Include or exclude `</s>` from `N` consistently across models being compared.
- **Held-out set contamination.** If your training corpus contains the test set (very easy on Wikipedia/CommonCrawl), perplexity is meaningless.

## Building an n-gram LM in practice

```python
# Using KenLM to train + evaluate
# 1. Train a 4-gram model (shell)
#   lmplz -o 4 --interpolate_unigrams 0 < corpus.txt > model.arpa
#   build_binary -q 8 model.arpa model.bin
#
# 2. Query from Python

import kenlm
model = kenlm.Model("model.bin")

score = model.score("the cat sat on the mat", bos=True, eos=True)   # log10 P
# Perplexity over a held-out corpus:
def perplexity(model, sentences):
    logp, ntok = 0.0, 0
    for s in sentences:
        logp += model.score(s, bos=True, eos=True)
        ntok += len(s.split()) + 1   # +1 for </s>; adjust to match training
    return 10 ** (-logp / ntok)
```

Design choices to think about:

- **Order.** 3-4 grams for keyboard prediction; 5-grams for large-corpus ASR rescoring. Higher orders explode memory quickly (KenLM's binary uses ~5-20 bytes per n-gram at typical scales).
- **Vocabulary.** Closed vs. open; usually apply a frequency threshold and treat everything below as `<unk>`.
- **Character-level.** For low-resource languages or noisy text, character-level (or BPE-level) n-grams sidestep the OOV problem entirely.

## Where n-gram LMs still win in 2026

- **ASR / TTS beam rescoring.** Every hypothesis in a lattice is scored by a small LM inside the decoder loop. Cheap, deterministic, and easy to swap. See KenLM's use inside `Kaldi`, `wav2letter`, `Espresso`, and the CTC decoders shipped with `torchaudio`.
- **Domain-adapted rescoring on top of a neural LM.** A generic neural LM plus a domain-specific n-gram LM (interpolated on log-probs) is a widely-used pattern for medical, legal, and enterprise chat where the tail distribution matters.
- **Spelling and typing.** GBoard-style keyboards ship a small n-gram LM on-device for candidate ranking.
- **Cheap classification via LM scoring.** Language identification and dialect ID (see chapter 09) are often implemented as per-language character n-gram LMs; the one with highest per-token log-probability wins.
- **Anomaly detection on structured text.** Web logs, DNS queries, code — anywhere the "language" is narrow, an n-gram LM's perplexity flags rare inputs cheaply. Reference: Freeman et al., "Identifying Suspicious URLs: An Application of Large-Scale Online Learning", *ICML 2009*, and the DNS-tunneling literature that followed.

## What n-gram LMs cannot do

- Model long-range dependencies. A 5-gram LM cannot know that a sentence started with "In summary" and should conclude, not begin.
- Generalise to unseen phrasings. Smoothing hedges probability, but there is no representation to interpolate between.
- Produce fluent open-domain text. This is where neural LMs took over and are not coming back.

Treat n-gram LMs as *baselines and complements*, not as replacements for a transformer. The engineering payoff is where they run alongside the neural model on the same lattice.

## Reading n-gram LM output like a debugger

Two habits that pay off:

- **Score sentence by sentence, then token by token.** KenLM's `.full_scores(...)` returns `(prob, ngram_length, oov_flag)` per token. A sudden drop in `ngram_length` (a backoff) signals a rare context; a run of `oov_flag=True` says your vocabulary does not fit the domain.
- **Compare a `-i` interpolated model vs. a `--no-prune` model.** If pruning cost you material perplexity, the tail matters and you should retrain with more data or less pruning.

## Chapter summary

- n-gram LMs estimate `P(wᵢ | wᵢ₋ₙ₊₁, …, wᵢ₋₁)` from counts; smoothing (Kneser-Ney or its modified variant) is the difference between a toy model and a production one.
- Perplexity is the standard evaluation but is tokeniser-dependent and undermined by OOV handling; report OOV rate alongside.
- KenLM is the current production toolkit; SRILM is the research reference.
- n-gram LMs still ship in ASR/TTS rescoring, on-device typing, spelling correction, and low-cost language identification. They complement, not replace, transformer LMs in serious pipelines.
- The next chapter turns from generative sequence models to discriminative sequence *tagging* — POS taggers and named-entity recognisers built with HMMs, CRFs, and averaged perceptrons.
