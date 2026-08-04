# Byte Pair Encoding and Byte-Level BPE

## Motivation

Byte Pair Encoding (BPE) is the most-deployed sub-word algorithm in production NLP. GPT-2, GPT-3, GPT-4, RoBERTa, LLaMA, Mistral, and most open-weights decoder-only LLMs use a byte-level BPE variant. Even when a paper says "we use SentencePiece", it is often a BPE model trained through the SentencePiece library. Understanding BPE end-to-end is the single highest-leverage tokenizer skill an NLP engineer can build.

BPE has two lives in the literature: a 1994 compression algorithm by Philip Gage, and a 2016 NMT tokenizer by Sennrich, Haddow, and Birch. This chapter walks the tokenizer variant, then covers the byte-level twist introduced with GPT-2 that eliminates the `<unk>` token entirely.

## The training algorithm

Given a training corpus and a target vocabulary size `V`:

1. Split the corpus into words (whitespace-delimited, or via a pre-tokeniser). Represent each word as a sequence of characters plus a special end-of-word marker (e.g. `</w>`) so the algorithm can distinguish word-internal from word-final pieces.
2. Initialise the vocabulary with every unique character that appears.
3. Count all adjacent symbol pairs across the corpus, weighted by word frequency.
4. Merge the most frequent pair into a single new symbol, add it to the vocabulary, and record the merge rule.
5. Repeat step 3-4 until the vocabulary reaches `V`, or no pair occurs more than once.

The output is (a) a vocabulary of symbols and (b) an ordered list of merge rules.

### A worked micro-example

Corpus (with counts):

```
"low"     : 5
"lower"   : 2
"newest"  : 6
"widest"  : 3
```

Initial character-level representation (using `</w>` for end-of-word):

```
l o w </w>       x5
l o w e r </w>   x2
n e w e s t </w> x6
w i d e s t </w> x3
```

Pair counts include `(e, s) = 9`, `(s, t) = 9`, `(l, o) = 7`, `(o, w) = 7`, `(n, e) = 6`, `(s, t</w>) = 9` — the exact tie-breaking depends on implementation, but suppose `(e, s)` wins.

After one merge of `(e, s) → es`:

```
l o w </w>        x5
l o w e r </w>    x2
n e w es t </w>   x6
w i d es t </w>   x3
```

Next iteration `(es, t)` becomes the top pair (9), giving symbol `est`. Then `(est, </w>) → est</w>` (9). Continue for `V - |base characters|` steps.

At the end, `"newest"` may be a single token, `"newer"` splits into `new`, `er</w>`, and a novel word like `"widen"` decomposes into learned sub-word pieces without ever producing `<unk>`.

## The inference algorithm

At encode time, BPE is *deterministic*:

1. Represent the input word as its character sequence.
2. Apply the ordered merge rules; at each step, replace the *earliest-learned* mergeable pair anywhere in the sequence.
3. Stop when no merge rule applies.

Because the merges are applied in learning order, the segmentation of a given word is fixed by the training corpus — the same word always produces the same tokens, in every context. This determinism makes BPE cheap to run and easy to cache, but it also means BPE cannot express uncertainty about segmentation (contrast Unigram LM in chapter 03).

## Byte-level BPE (GPT-2 and beyond)

Character-level BPE has three practical problems:

1. The base "characters" are Unicode code points. If a code point never appears in training, encoding fails or falls back to `<unk>`.
2. Multi-byte code points (CJK, emoji, mathematical symbols) inflate the base vocabulary.
3. Whitespace, punctuation, and casing interact awkwardly with the pre-tokeniser.

The GPT-2 paper (Radford et al., 2019) resolves this by running BPE over **UTF-8 bytes** rather than Unicode code points. The base vocabulary is exactly 256 items — every possible byte — so no input can ever be `<unk>`. Merges then compose byte sequences into sub-word units.

Two implementation details are worth knowing:

- **Reversible byte-to-unicode map.** UTF-8 bytes include control characters that libraries choke on. GPT-2 defines a bijection from raw bytes to printable Unicode code points, so tokens can be stored and displayed as strings while still round-tripping to bytes.
- **Whitespace is part of the token.** A leading space is encoded as `Ġ` (byte 0x20 mapped through the bijection). This is why `"Hello"` and `" Hello"` tokenise differently in GPT-family models and why leading-space bugs are a recurring source of prompt regressions.

Byte-level BPE is the default for Hugging Face `ByteLevelBPETokenizer` and for the tokenizers of GPT-2, GPT-3, GPT-4 (via `tiktoken`), RoBERTa, and LLaMA-family models.

## Practical BPE with `tokenizers`

The Hugging Face `tokenizers` library exposes each pipeline stage separately. Training a byte-level BPE from scratch looks like this:

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

tokenizer = Tokenizer(models.BPE(unk_token=None))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=32_000,
    min_frequency=2,
    special_tokens=["<|endoftext|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 base bytes
)
tokenizer.train(files=["corpus.txt"], trainer=trainer)
tokenizer.save("bpe-32k.json")
```

Key knobs an NLP engineer should be able to explain from first principles:

- **`vocab_size`** — trades embedding-matrix cost against fertility. Common sizes: 32k (BERT/RoBERTa), 50k (GPT-2), 100-200k (multilingual LLMs).
- **`min_frequency`** — pairs below this frequency are ignored. Too low → noisy merges memorising typos; too high → under-training the tail.
- **`initial_alphabet`** — force the base vocabulary to include every byte (or every character you care about) so nothing is unrepresentable.
- **`special_tokens`** — reserved IDs that the trainer will not decompose. Add these *before* training; adding them later shifts every ID.

## Debugging BPE in the wild

When downstream metrics look off, BPE-specific things to check:

- **Fertility (tokens per word or per byte)** on your domain vs. a reference corpus. A tokenizer that averages 1.5 tokens/word on English news but 6 tokens/word on your medical notes is telling you the vocabulary does not cover your domain.
- **Leading-space asymmetries.** `tokenizer.encode(" Hello")` vs. `tokenizer.encode("Hello")` should produce different IDs in byte-level BPE. Prompt templates that concatenate strings without accounting for this leak tokens.
- **Merge-rule provenance.** If a merge produced a garbled token (e.g. a mid-word HTML fragment), it usually means your training corpus contained un-cleaned scraped text. Fix upstream, then retrain.
- **`tiktoken` compatibility.** OpenAI models use `tiktoken` — a fast Rust implementation of byte-level BPE with pre-tokeniser regexes baked in. It is not bit-identical to `tokenizers` ByteLevelBPE; if you need OpenAI-compatible IDs, use `tiktoken` directly.

## Chapter summary

- BPE learns a fixed, ordered list of merges by repeatedly combining the most frequent adjacent pair.
- Encoding is deterministic — the same word always yields the same tokens.
- Byte-level BPE (GPT-2 style) runs the algorithm over UTF-8 bytes with a 256-item base vocabulary, eliminating `<unk>` entirely.
- Vocabulary size, `min_frequency`, initial alphabet, and pre-tokeniser choice are the dials that shape downstream fertility and cost.
- BPE is the workhorse of modern decoder-only LLMs; the next chapter shows what changes when we replace greedy merges with likelihood-driven ones.
