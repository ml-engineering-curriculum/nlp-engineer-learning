# exercise-01: BPE From Scratch and Against Hugging Face `tokenizers`

**Estimated effort:** 3 hours

## Objective

Implement Byte Pair Encoding end-to-end from scratch — training, encoding, and decoding — then reproduce (or explain the divergence from) a Hugging Face `ByteLevelBPETokenizer` trained on the same corpus. The point is to demystify BPE: after this exercise, no BPE-related bug in production should feel like magic.

## Prerequisites

- Chapters [01](../01-why-tokenization-shapes-every-later-decision.md), [02](../02-byte-pair-encoding-and-byte-level-bpe.md), and [06](../06-training-a-domain-adapted-tokenizer.md) of this module.
- Python 3.10+, `tokenizers ≥ 0.15`, `datasets` (optional, for corpus loading).
- Comfort reading small algorithmic Python; no ML training required.

## Problem statement

Build two artefacts and compare them.

### Part A — BPE from scratch

Implement a `train_bpe(corpus_iter, vocab_size, min_frequency=2, end_of_word="</w>")` function that returns:

- a `vocab: dict[str, int]` mapping token strings to IDs;
- an ordered `merges: list[tuple[str, str]]` (the merge rules in learning order).

Constraints:

- Operate on a whitespace-pre-tokenised corpus (word-level for now — byte-level is Part C).
- Use the classic algorithm: initialise with characters + end-of-word marker, count pairs, greedily merge the most-frequent pair, tie-break deterministically (e.g. lexicographically) and document your choice.
- Do not use `collections.Counter` blindly — think about how you keep pair counts up-to-date incrementally as merges happen. A pure recomputation each iteration is acceptable for a small corpus but should be flagged in your write-up as an `O(V · corpus)` cost.

Then implement `encode(word, vocab, merges)` and `decode(ids, vocab)` and verify round-trip on a held-out sample.

### Part B — Reproduce with Hugging Face `tokenizers`

Train a `BpeTrainer` (character-level, matching your Part A pipeline as closely as possible) on the same corpus with the same `vocab_size` and `min_frequency`. Compare:

- vocabulary size and overlap of the two token sets;
- fertility (avg tokens/word) on a held-out slice;
- token-by-token encoding agreement on the held-out slice — record the top 20 disagreements and explain each.

### Part C — Byte-level BPE

Re-run Part B with `pre_tokenizers.ByteLevel()` and `models.BPE()` seeded with `initial_alphabet=ByteLevel.alphabet()`. Compare against the character-level run:

- What happens to vocabulary size at the same target?
- How does fertility differ on ASCII vs. non-ASCII text?
- Encode a string containing an emoji or Han character with both tokenizers. Which produces `<unk>` (or its fallback)? Why?

## Recommended corpus

Any UTF-8 text with at least 5-10 MiB and some morphological variety. Suggested public options:

- A single language slice of Wikipedia dump (e.g. the `en` or `fi` snapshot on [dumps.wikimedia.org](https://dumps.wikimedia.org/)).
- The [`wikitext-2` or `wikitext-103`](https://huggingface.co/datasets/wikitext) dataset via `datasets`.
- A domain-specific corpus of your own choosing (medical abstracts, code, chat logs) — but hold Parts A and B on the *same* corpus for fair comparison.

Do not use a corpus larger than ~200 MiB unless you explicitly want to benchmark performance; the from-scratch trainer will be slow.

## Starter guidance

- Represent each unique word by a tuple of symbols (`("l", "o", "w", "</w>")`) with a frequency count. Store the corpus as `dict[tuple[str, ...], int]`. Merging is then a rewrite of tuple keys.
- Log the top 20 merges learned by each implementation side-by-side. That single log usually reveals most disagreements.
- When comparing to `tokenizers`, disable normalisation (`tokenizer.normalizer = None`) so the only variable is the algorithm itself.
- Use `tokenizer.get_vocab()` and `tokenizer.model.get_merges()` (if available) or export the JSON with `tokenizer.save(...)` and inspect the `model.merges` array directly.
- Byte-level BPE in `tokenizers` maps bytes through a printable-unicode bijection (see chapter 02). Do not be surprised when your vocabulary contains `Ġ`, `Ĉ`, `Ċ`, etc. — they are legitimate byte encodings, not corruption.

## Acceptance criteria

You should be able to demonstrate all of the following:

- [ ] `train_bpe` produces a vocabulary and ordered merges deterministically for a fixed seed corpus.
- [ ] `encode` / `decode` round-trip a held-out sample with 100% string equality on the whitespace-tokenised text.
- [ ] Fertility measured on the held-out slice is within ~5% between your scratch implementation and the HF character-level BPE (small differences due to tie-breaking are expected — explain them).
- [ ] You can point at the top disagreements between the two tokenizers and explain each in terms of merge order or pre-tokenisation.
- [ ] Byte-level BPE handles multi-byte input (emoji, CJK) without `<unk>`; character-level BPE does not. You can produce a runnable example demonstrating both.
- [ ] A short write-up (`README.md` in your solution folder) explains: your tie-breaking rule, the observed fertility numbers, and the reason for each of the top disagreements.

## Stretch goals

- **Merge-count optimisation.** Replace full-corpus pair recomputation with an incremental update: when merging `(a, b)`, only the words that contained the pair need their pair counts touched. Measure the speedup on a 100-MiB corpus.
- **BPE-dropout at inference.** Implement Provilkov, Emelianenko & Voita's BPE-dropout (ACL 2020) — probabilistically skip each merge with probability `p` during encoding. Show how sampled segmentations vary for a chosen word.
- **`tiktoken` parity.** Load OpenAI's `cl100k_base` tokenizer via `tiktoken` and encode the same held-out slice. It is *not* bit-identical to `tokenizers`' byte-level BPE; identify at least one class of disagreement and explain it (hint: pre-tokeniser regex).
- **Multilingual stress test.** Train the same 32k-vocab byte-level BPE on an English-only corpus and on a 50/50 English/Japanese corpus. Report fertility on each language separately for both tokenizers, and discuss the trade-off.
