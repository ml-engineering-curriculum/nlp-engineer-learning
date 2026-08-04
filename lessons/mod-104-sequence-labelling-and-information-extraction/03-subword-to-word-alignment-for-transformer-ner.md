# Subword-to-Word Alignment for Transformer NER

## Motivation

Your NER annotations are on words. Your transformer consumes subword pieces. The two do not line up: `hemorrhage` is a single labelled word in your corpus but three or four pieces in a BERT tokenizer. Every training example needs a step that answers: *given a sequence of subword IDs and a sequence of word-level labels, what label goes on each subword?* Get this wrong and you either train against garbage labels or discard 30 % of the signal in your dataset.

This chapter is a walkthrough of the alignment recipe — the one every `run_ner.py` script in the Hugging Face ecosystem uses, why it exists, the four common variants, and the pitfalls that only surface when you audit the aligned examples yourself.

## The problem, concretely

Take the CoNLL-2003 example from chapter 02:

```
Word:   John    Smith   works   at    Bank    of      America  .
Label:  B-PER   I-PER   O       O     B-ORG   I-ORG   I-ORG    O
```

Tokenised with `bert-base-cased`:

```
Subword: [CLS]  John  Smith  works  at  Bank  of  America  .  [SEP]
Word-id: None   0     1      2      3   4     5   6        7  None
```

Here every word happens to be a single subword. Now consider a biomedical example:

```
Word:   Patient  reported  hemorrhage  .
Label:  O        O         B-DIS       O

Subword: [CLS]  Patient  reported  hem  ##or  ##rha  ##ge  .  [SEP]
Word-id: None   0        1         2    2     2      2     3  None
```

Word `hemorrhage` is one word (index 2) but four subwords. Which of those subwords gets the `B-DIS` label? What do the other three get?

## The standard recipe: first-subword-only

The Hugging Face convention — used in every official token-classification example — is:

1. Tokenise with `is_split_into_words=True` (input is already a list of words).
2. Call `tokenized.word_ids(batch_index=i)` to get the word index for each subword position (`None` for special tokens).
3. For each subword: if it is the *first* subword of a word, label it with that word's tag. Otherwise, label it `-100`.
4. Loss and metrics ignore any position labelled `-100`.

Code:

```python
def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=False):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=256,
    )
    all_labels = []
    for i, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(word_labels[word_id])
            else:
                label_ids.append(word_labels[word_id] if label_all_tokens else -100)
            previous_word_id = word_id
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized
```

Why `-100`? PyTorch's `CrossEntropyLoss` has an `ignore_index=-100` default, so any subword you label `-100` contributes nothing to the loss or accuracy. Chapter 02's `compute_metrics` filter drops those positions before handing predictions to `seqeval`.

## Why only the first subword?

Two arguments, both empirical:

1. **The first subword carries the "word" identity.** In BPE and WordPiece, the leading piece is what you would look up as the word's stem; the trailing `##`-prefixed pieces are morphological continuations. Assigning the label there matches the annotation semantics.
2. **Labelling every subword drifts.** If you label every subword `I-DIS` for a four-subword entity, your model learns that `##or`, `##rha`, `##ge` are `I-DIS` even in words that are not diseases. It biases the model toward false positives.

Devlin et al.'s original BERT NER experiments used first-subword-only, and every subsequent Hugging Face example followed. Empirical comparisons (below) generally find this variant equal to or better than the alternatives.

## The four alignment variants — pick with numbers

| Variant | First subword | Continuation subwords | Notes |
| --- | --- | --- | --- |
| **A. First-only** (default) | word label | `-100` | HF default; safest baseline |
| **B. Label-all-tokens** | word label | word label (or converted) | Every subword scored; more supervision signal |
| **C. Label-all-with-I-conversion** | word label | `B-X` → `I-X` for continuations | Cleaner than B: an entity's first subword keeps `B-`, later subwords are `I-` even if the word's label was `B-` |
| **D. Word-restore decode** | word label | `-100` | Same as A at train time, but at decode time you *only* look at the first-subword predictions and re-emit word-level tags; span extraction is done on words, not subwords |

Findings from the literature:

- Nguyen et al., ["Distilling BERT for Token Classification"](https://arxiv.org/abs/2110.15078), *arXiv 2021*, and multiple reproductions on CoNLL-2003 report **A ≈ D > B** by ~0.3–0.6 F1 on standard NER benchmarks. **C** is usually within noise of A but avoids the "I-tag at position that started as B-" pathology.
- Domain corpora with heavy subword splitting (biomedical, source code) benefit slightly from **C** because more training signal is preserved without the `B-`/`I-` corruption.
- **D** matters for downstream span extraction — you want to emit *word-level* spans anyway, so decoding on words removes an entire class of subword-boundary bugs.

Recommendation: start with **A** (the default) and switch to **C** with **D** decoding if the domain has many multi-subword entities. Always audit 50 aligned examples by hand before training.

## The label-all-tokens with `B-` → `I-` conversion

If you choose variant C, you need to be careful: a naïve "label every subword with the word's label" turns

```
Word:  Bank(B-ORG)  of(I-ORG)  America(I-ORG)
```

into subwords like

```
Bank(B-ORG) of(I-ORG) Amer(I-ORG) ##ica(I-ORG)
```

which is fine, but a word whose label is `B-` gets `B-` on all its subwords, producing an invalid tag sequence at decode time. The convention is:

```python
def convert_b_to_i(label_id, id2label, label2id):
    label = id2label[label_id]
    if label.startswith("B-"):
        return label2id["I-" + label[2:]]
    return label_id

# ... inside the loop
else:
    label_ids.append(convert_b_to_i(word_labels[word_id], id2label, label2id))
```

Only the *first* subword of the word keeps the `B-`; continuation subwords get the corresponding `I-`. This preserves valid BIO structure at every position.

## Alignment at inference time

You have two decoding paths, and they matter more than the training-time choice.

**Subword-level decoding.** Take argmax at every subword position, then walk the sequence with a BIO parser that emits spans. Requires the tokenizer's `offset_mapping` so you can map back to character positions in the original text. Standard `Trainer` + `seqeval` pipelines do this implicitly through the `-100` mask.

**Word-level decoding.** For each word, look only at the label predicted for its first subword. Emit spans from the word-level tag sequence. Cleaner and matches how your annotations were produced. Recommended when you need to hand span offsets to a downstream consumer that thinks in words or characters.

```python
def decode_word_level(tokens, subword_preds, word_ids, id2label):
    word_tags = []
    seen = set()
    for subword_idx, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen:
            continue
        seen.add(word_id)
        word_tags.append(id2label[subword_preds[subword_idx]])
    return list(zip(tokens, word_tags))
```

For character-level span offsets (the standard payload to downstream systems), use `tokenizer(..., return_offsets_mapping=True)` and reconstruct spans from consecutive `B-X`/`I-X` runs. This is what Hugging Face's `TokenClassificationPipeline(aggregation_strategy="first")` does out of the box.

## Aggregation strategies in `pipeline("ner")`

Hugging Face's inference pipeline gives you four aggregation strategies for turning per-subword predictions into character-level spans:

- **`"none"`** — raw subword labels, no aggregation. For debugging only.
- **`"simple"`** — group consecutive same-label subwords; the emitted label is the label of the first subword in the group. Ignores `B-`/`I-` structure.
- **`"first"`** — like `simple` but the emitted label comes from the *first subword of each word*. Matches training if you used variant A/C above.
- **`"average"`** — average logits across subwords of a word before argmax. Occasionally better on noisy models; almost identical to `"first"` on well-trained ones.
- **`"max"`** — take the max logit across subwords. Rarely used in production.

Use `"first"` when you trained with variant A or C. Never leave it at the default (`"none"`) in production — the raw subword tags will not match your character-offset consumer.

## Long-document truncation and the offset trap

`tokenizer(..., truncation=True, max_length=256)` drops subwords past position 256. Your labels list is silently truncated to match. If your entities cluster in the tail of long documents (e.g., contract signatures, discharge-summary diagnoses at the end), you can lose 5–20 % of the entities without noticing.

Two mitigations:

1. **Log the truncation rate** on the training set (`sum(len(input_ids) == max_length) / N`). If > 5 %, either bump `max_length` or move to sliding-window / long-context encoders (chapter 07).
2. **Sliding-window with `return_overflowing_tokens=True`**. The tokenizer emits multiple chunks per document with configurable stride; each chunk carries its own aligned labels. Downstream, you deduplicate span predictions across overlapping windows. This is the standard workaround before jumping to Longformer / BigBird.

```python
tokenized = tokenizer(
    examples["tokens"],
    is_split_into_words=True,
    truncation=True,
    max_length=256,
    stride=64,
    return_overflowing_tokens=True,
    padding=False,
)
```

You now have to re-derive `word_ids` per chunk and duplicate the source label list for each overflow chunk. `datasets` and Hugging Face's example scripts have reference implementations; do not roll your own without reading them first.

## Pitfalls that only show up when you audit

- **Wrong tokenizer.** Loading `bert-base-cased`'s model with `bert-base-uncased`'s tokenizer silently lowercases your entities and destroys casing signal for PER/ORG. Instantiate both from the same repo revision.
- **`is_split_into_words=False`** on a corpus of already-tokenised words. Symptom: entities span the wrong pieces, alignment silently misassigns labels. This is the #1 source of "my NER is 20 F1 points below the paper" bugs.
- **Whitespace-only tokens** in some corpora (BigBIO, CROSSNER) — `tokenizer` may drop these silently, shifting word indices. Filter or repair before alignment.
- **Special tokens counted in `word_ids`.** They come back as `None`; `None != previous_word_id` is `True`, so a naïve implementation labels `[CLS]` with the label of word 0. Always check the `if word_id is None:` branch is present.
- **CJK and other zero-space languages.** `is_split_into_words=True` assumes whitespace-separated words. For Chinese/Japanese NER, treat each character as a "word" and use a character-preserving tokenizer, or move to a token-level scheme (chapter 07 covers this in the multilingual section).
- **Mixed-case labels.** `B-PER` and `B-per` become different label IDs. Uppercase in your label map; lowercase in your gold data; the confusion matrix is a mess. Normalise at load time.

## The audit checklist

Before you train, print 20 aligned examples and confirm:

- [ ] `[CLS]` and `[SEP]` are labelled `-100`.
- [ ] The first subword of every multi-subword entity has the right `B-`/`I-` tag.
- [ ] Continuation subwords are `-100` (variant A) or valid `I-X` (variant C).
- [ ] No `B-X` on a continuation subword (invalid BIO).
- [ ] The re-decoded word tags match the original word labels for a random sample of 20.

If any of these fail, the model will not train. `seqeval` numbers on a broken alignment can look plausible while the model has learned nothing useful — the metric averages over `-100`-filtered positions, and broken alignment often leaves those positions correct.

## Chapter summary

- Subword-to-word alignment is the tax every transformer NER pays: annotations are on words, models consume subwords, and the alignment step is where 30 % of first-time bugs live.
- The default recipe — first-subword-only, continuations set to `-100` — is safe and matches the Hugging Face examples. Label-all-with-`B-`-to-`I-`-conversion is a defensible upgrade for domains with heavy subword splitting.
- At inference, decode word-level or use `pipeline(aggregation_strategy="first")` for character-level spans; never trust raw subword labels.
- Long-document truncation silently drops entities in the tail; log the truncation rate and switch to sliding-window before switching to long-context encoders.
- Always audit 20 aligned examples by hand before training. Alignment bugs pass linting and often pass evaluation until they hit production.
