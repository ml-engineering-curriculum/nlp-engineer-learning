# Why Tokenization Shapes Every Later Decision

## Motivation

A transformer does not read text. It reads a sequence of integer IDs whose meaning is entirely defined by the tokenizer that produced them. Every downstream property you care about — vocabulary size, embedding-matrix cost, sequence length, context-window utilisation, cross-lingual transfer, arithmetic and code fidelity, safety filters, and even how you count tokens for a bill — is determined at the tokenizer boundary.

If you get the tokenizer wrong, later modules cannot recover it. Fine-tuning a decoder on Japanese text with a tokenizer trained on English C4 will burn most of your context window on single-byte fragments, blow up your fertility (tokens per word), and quietly degrade every metric downstream. Choosing between BPE, WordPiece, Unigram LM, and SentencePiece is therefore an architectural decision, not a pre-processing detail.

This chapter frames tokenization as the API between raw text and a transformer, and sets up the rest of the module.

## The API contract of a tokenizer

A modern tokenizer is a bidirectional pipeline:

```
raw text ──► normalise ──► pre-tokenise ──► model (BPE/WP/Unigram) ──► post-process ──► ids
                                                                                      │
raw text ◄── decode ◄────────────── join ◄──── lookup vocab ◄────────────────────────┘
```

Each stage is separately configurable in libraries like Hugging Face `tokenizers` and Google `sentencepiece`:

- **Normalisation** — Unicode normalisation form, lower-casing, accent stripping, control-character removal.
- **Pre-tokenisation** — coarse splitting into pieces that the sub-word model will never merge across. Whitespace splitting is the classic choice; byte-level BPE uses a per-byte pre-tokeniser; SentencePiece uses a "meta-space" scheme that keeps whitespace as a first-class character.
- **Model** — the learned sub-word algorithm (BPE, WordPiece, Unigram LM).
- **Post-processing** — adding special tokens (`[CLS]`, `[SEP]`, `<s>`, `</s>`, `<|endoftext|>`), building attention masks, and templating for pair inputs.
- **Decoding** — reversing the process, ideally losslessly, to recover text from IDs.

Any bug you can push earlier in the pipeline (normalisation, pre-tokenisation) is a bug the model does not have to learn around.

## Four invariants worth memorising

1. **Vocabulary size is a hyperparameter with cost.** Every extra token adds a row to the input embedding matrix and (for tied embeddings) the output projection. At hidden size 4096 and float16 storage, a 32k-token vocabulary already costs 256 MiB just for the embedding tables. Doubling the vocabulary doubles that cost.
2. **Sequence length is measured in tokens, not characters.** The context window advertised by a model (e.g. 8k, 32k, 200k) is a token count. A tokenizer that fragments your domain — legal Latin, chemistry SMILES, Korean, source code — silently shrinks the *effective* context you paid for.
3. **Attention is quadratic in sequence length.** Standard self-attention is O(L²) in compute and memory for the attention matrix, so a tokenizer that doubles fertility on your corpus quadruples attention cost.
4. **Detokenisation must be lossless enough for your task.** Generation, translation, and code tasks require the decoded text to match what a human or a compiler expects. Classification is more forgiving. SentencePiece was designed specifically to guarantee round-trippable detokenisation across scripts that lack whitespace boundaries.

## What we mean by "sub-word"

A sub-word tokenizer sits between two extremes:

- **Word-level.** Small sequences, but out-of-vocabulary (OOV) words become `<unk>`, morphology is invisible, and vocabularies explode on inflected languages.
- **Character-level (or byte-level).** No OOV, but sequences are long and semantics are diluted across many steps.

Sub-word schemes learn a vocabulary of frequent character sequences from a training corpus, so common words stay one token while rare or novel words decompose into re-usable pieces. The three sub-word algorithms we will study — BPE, WordPiece, and Unigram LM — differ in *how* they choose those pieces:

| Algorithm     | Vocabulary construction   | Segmentation at inference | Introduced by                                 |
|---------------|---------------------------|---------------------------|-----------------------------------------------|
| BPE           | Greedy merges of most frequent adjacent pair | Deterministic merge replay | Sennrich, Haddow & Birch, ACL 2016 (based on Gage 1994 compression scheme) |
| WordPiece     | Greedy merges that maximise training-corpus likelihood | Longest-match-first from left | Schuster & Nakajima, ICASSP 2012; popularised by BERT |
| Unigram LM    | Prune from a large seed vocabulary using a unigram LM | Viterbi over token log-probabilities | Kudo, ACL 2018 |

`SentencePiece` is a *library* that implements BPE and Unigram LM with a script-agnostic pre-tokeniser; it is not a fourth algorithm. Chapters 02-04 dig into each in turn.

## Where this module sits in the track

- **Module 101 (this module)** builds the tokenizer + transformer foundation every later module relies on.
- **Module 102 (classical NLP)** picks up rule-based text-normalisation and morphology in more depth.
- **Modules 103-107 (task-specific NLP)** all consume tokens produced by the tools you learn here; if you have not internalised the tokenizer API, task decisions later will feel arbitrary.
- **Module 108 (embeddings)** reuses the same vocabulary discussion for representation learning.
- **Modules 111-112 (evaluation, production)** revisit tokenization when we measure fertility, coverage, and cost.

## Chapter summary

- A tokenizer is a bidirectional pipeline (normalise → pre-tokenise → model → post-process; and back).
- Vocabulary size, sequence length, attention cost, and detokenisation fidelity are all set at the tokenizer boundary and cannot be repaired downstream.
- Sub-word tokenizers (BPE, WordPiece, Unigram LM) are the industry default; SentencePiece is a library, not a fourth algorithm.
- The rest of this module works down that pipeline — algorithms, Unicode-correct text handling, training your own tokenizer, and then the transformer families that consume its output.
