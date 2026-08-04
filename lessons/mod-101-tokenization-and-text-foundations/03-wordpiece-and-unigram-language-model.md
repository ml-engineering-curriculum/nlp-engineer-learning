# WordPiece and Unigram LM

## Motivation

BPE picks merges by *frequency*. Two other algorithms shipped with production models pick pieces by *likelihood*:

- **WordPiece** — the tokenizer of BERT, DistilBERT, ELECTRA, and MobileBERT. Its greedy merges maximise the likelihood of the training corpus under a bigram-ish model instead of just counting frequencies.
- **Unigram LM** — the algorithm of Kudo (2018) that ships with SentencePiece and powers T5, mT5, ALBERT, XLNet, and many multilingual encoders. It is the only sub-word algorithm that models segmentation *probabilistically*, which enables sub-word regularisation.

Both are important because they surface in models you will fine-tune. If you cannot reason about their vocabularies you cannot reason about their fine-tuning behaviour.

## WordPiece

### Training

WordPiece was introduced for Japanese/Korean voice search (Schuster & Nakajima, ICASSP 2012). Like BPE, it starts with a base vocabulary of characters and greedily grows a target vocabulary by merging adjacent pairs, but the merge criterion is different:

Instead of choosing the most frequent pair, WordPiece chooses the pair `(x, y)` that maximises

```
score(x, y) = count(xy) / (count(x) * count(y))
```

Intuitively, this promotes pairs that co-occur more than their marginal frequencies would predict — a bigram-mutual-information style score. Merges that just cement two already-common tokens (e.g. common letters) are penalised, so the algorithm prefers merges that capture genuine morphology.

The BERT implementation prefixes continuation pieces with `##`:

```
"tokenizing" → ["token", "##izing"]
"unaffordable" → ["una", "##ffor", "##dable"]   # depends on the trained vocab
```

The `##` marker means "attach to the previous token with no space". A token *without* `##` is a word-initial piece.

### Inference: longest-match-first

At encode time, BERT-style WordPiece runs a **greedy longest-match-first** scan from the left of each word:

```
def wordpiece_encode(word, vocab, unk="[UNK]"):
    tokens = []
    start = 0
    while start < len(word):
        end = len(word)
        cur_substr = None
        while start < end:
            substr = word[start:end]
            if start > 0:
                substr = "##" + substr
            if substr in vocab:
                cur_substr = substr
                break
            end -= 1
        if cur_substr is None:
            return [unk]
        tokens.append(cur_substr)
        start = end
    return tokens
```

Two consequences:

1. If any substring cannot be represented, the entire word maps to `[UNK]` — WordPiece is not byte-level.
2. Longest-match-first can be pathological for compound-heavy languages if the vocabulary lacks morpheme boundaries; the algorithm cannot back-track once a long piece is chosen.

Modern BERT tokenizers add a normaliser (`BertNormalizer`) that handles casing, strips accents (unless `strip_accents=False` is set), and splits punctuation.

## Unigram Language Model

### The idea

The Unigram LM tokenizer of Kudo (2018) inverts the BPE workflow. Instead of *building up* a vocabulary from characters, it *prunes down* from a large seed vocabulary:

1. Seed a candidate vocabulary of size ~10× target (e.g. from BPE, suffix arrays, or all substrings up to a max length).
2. Fit unigram token probabilities on the corpus, using the EM algorithm and Viterbi to find the best segmentation of every sentence.
3. Compute each token's *loss contribution* — how much corpus log-likelihood would drop if it were removed.
4. Remove the bottom `p%` of tokens (usually 10-20%) whose removal costs the least.
5. Repeat 2-4 until the vocabulary reaches the target size.

The result is a vocabulary `V` and a unigram distribution `p(t)` over its tokens.

### Segmentation at inference: Viterbi

Unlike BPE and WordPiece, Unigram LM segmentation is a **probabilistic decoding problem**. Given a sentence, the tokenizer finds the segmentation that maximises

```
∑_i log p(t_i)
```

using Viterbi (dynamic programming) over the lattice of possible segmentations that use only tokens in `V`. This has two big consequences:

1. **The same word can segment differently depending on the tokens around it.** In practice this is rare (unigram probabilities are context-free) but the machinery is there.
2. **The tokenizer can sample alternative segmentations** by treating the lattice as a distribution — Kudo calls this *sub-word regularisation*, and it acts as data augmentation for translation, spelling-robustness, and multilingual training.

### Sub-word regularisation

For a given sentence, Unigram LM can output not just the argmax segmentation but the *k*-best segmentations weighted by score, or a temperature-scaled sample. Training a model on such varied segmentations of the same text improves robustness — the model no longer memorises "the one true segmentation" of every training example. This is why Unigram LM shows up in translation models (mBART's variants, mT5) and multilingual encoders where morphological ambiguity is common.

BPE has no equivalent; BPE-dropout (Provilkov, Emelianenko, & Voita, ACL 2020) is a later workaround that stochastically skips merge rules during encoding to simulate the same effect.

## Choosing between BPE, WordPiece, and Unigram

Order-of-magnitude guidance (all should be validated empirically on your corpus with the metrics from chapter 06):

| Property                              | BPE                | WordPiece         | Unigram LM        |
|---------------------------------------|--------------------|-------------------|-------------------|
| Segmentation determinism              | Deterministic      | Deterministic     | Probabilistic (argmax or sampled) |
| Handles unseen characters             | Byte-level: yes; char-level: no | No (`[UNK]`) | Depends on seed; SentencePiece adds byte-fallback |
| Sub-word regularisation               | Via BPE-dropout only | No               | Native            |
| Training cost                         | Cheap greedy pass  | Cheap greedy pass | Expensive EM      |
| Typical production users              | GPT-2/3/4, RoBERTa, LLaMA, Mistral | BERT, ELECTRA, DistilBERT | T5, mT5, ALBERT, XLNet |
| Default in Hugging Face `AutoTokenizer` | Wherever the source model uses BPE | Wherever the source model uses WordPiece | Wherever the source model uses Unigram (usually SentencePiece) |

### First-order heuristics

- **Match the pretrained model.** If you are fine-tuning BERT, you use its WordPiece vocabulary; if you are fine-tuning T5, you use its Unigram. This is non-negotiable.
- **Training a new model from scratch, English or Latin-script mixture, decoder-only** → byte-level BPE. Every open-weights decoder does this.
- **Training a new model from scratch, heavily multilingual (many scripts) or morphologically rich (Turkish, Finnish, Korean, Japanese)** → Unigram LM (usually via SentencePiece). Sub-word regularisation and script-agnostic pre-tokenisation both help.
- **Constrained-latency inference on device** → BPE. Deterministic encoding is trivially cacheable; Viterbi is heavier.
- **Domain-specialised encoder for classification** → WordPiece if you want to be BERT-compatible, byte-level BPE if you have no such constraint.

## Chapter summary

- WordPiece is BPE with a mutual-information-flavoured merge score and a longest-match-first encoder; it is the BERT-family default.
- Unigram LM prunes a seed vocabulary using EM + Viterbi and can output probabilistic (sampled) segmentations, enabling sub-word regularisation.
- Choosing among BPE, WordPiece, and Unigram is usually forced by the pretrained checkpoint you are extending; when training from scratch, the rules of thumb above apply.
- Chapter 04 covers SentencePiece — the library that operationalises Unigram (and BPE) for scripts without whitespace.
