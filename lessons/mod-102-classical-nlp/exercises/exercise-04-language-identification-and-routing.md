# exercise-04: Language Identification and Routing

**Estimated effort:** 2 hours

## Objective

Build and evaluate a real language-identification and routing layer for a multilingual corpus. Learn where off-the-shelf LID succeeds, where it fails, and how to design confidence thresholds and script-based fast paths that keep coverage high without misrouting.

## Prerequisites

- Chapter [09](../09-language-identification-and-routing.md).
- Chapter 05 of module 101 (BCP-47 tags, Unicode scripts) as background.
- Python 3.10+; `fasttext` (`pip install fasttext`); `pycld3` or `gcld3` (`pip install gcld3`); optionally `langid`, `langdetect`.
- Download fastText's `lid.176.ftz` from <https://fasttext.cc/docs/en/language-identification.html>.

## Problem statement

### Part A — Baseline: single-model LID

Assemble a labelled multilingual evaluation set. Options:

- A public multilingual dataset with language labels — e.g., a stratified sample of [Tatoeba sentences](https://tatoeba.org/en/downloads), the [`papluca/language-identification`](https://huggingface.co/datasets/papluca/language-identification) dataset on Hugging Face, or your own collection with confident labels.
- At minimum 10 languages including: two Latin-script confusables (e.g. Spanish + Portuguese), one non-Latin script (Cyrillic, Arabic, Devanagari, Han), and one language with two scripts (Serbian or Hindi transliterated).
- Aim for at least 500 sentences per language, stratified across short (<50 chars), medium (50-200), and long (200+) buckets.

Evaluate `lid.176.ftz` and one of {`cld3` / `pycld3` / `gcld3`, `langid.py`, `langdetect`} on this set. Report:

- Overall accuracy.
- Per-language precision and recall.
- Per-length-bucket accuracy.
- Latency per prediction (median, p95, p99) and model size on disk.

### Part B — Script-based fast path + threshold policy

Build a `identify(text) -> (bcp47_tag, confidence, decision)` function:

1. Compute a Unicode-script histogram (via `unicodedata.name` or PyICU). If a single script dominates and uniquely identifies a language (e.g. only Han → route to the CJK sub-router), return that tag with `decision="script"`.
2. Otherwise, call fastText / CLD3 as in Part A.
3. Apply confidence and delta thresholds (chapter 09). Return `"und"` for anything below.

Re-evaluate on your labelled set. Report:

- Accuracy on documents where `decision != "und"`.
- Coverage (fraction routed to a concrete language).
- The trade-off curve: for confidence thresholds `[0.5, 0.65, 0.8, 0.9]`, plot coverage vs. accuracy.

### Part C — Short-text handling

Concatenate consecutive short messages "per user" to simulate chat aggregation. Compare LID accuracy on:

- Single 20-character message.
- The 3-message running concatenation.
- The 5-message running concatenation.

For at least three of the most-confused language pairs, quantify how much aggregation helps. Discuss when aggregation is not available (e.g. no user session) and what your fallback would be.

### Part D — Confusable analysis

For at least three confusable pairs (choose from Norwegian/Danish, Spanish/Portuguese/Galician, Malay/Indonesian, Serbian-Cyrillic/Bulgarian/Macedonian, Farsi/Arabic/Urdu):

- Compute the confusion matrix entries for both directions.
- Read 10 misclassified examples per pair and categorise the errors (proper-noun-heavy content? Loanwords? Numerals? Short greeting phrases?).
- Propose one concrete mitigation per pair (character-n-gram LM tiebreaker, prior from user metadata, delta threshold change).

## Starter guidance

- Normalise text before scoring: strip URLs, mentions, and pure-numeric segments; they leak signal into most LID models. Log the normaliser you used.
- Use `sklearn.metrics.classification_report` and `confusion_matrix` for aggregate stats.
- For latency, warm the model with a few calls before timing (skips first-call JIT/loading cost).
- Store the exact fastText model hash in your write-up; multiple `lid.176.ftz` snapshots exist.
- The `unicodedata` module's `script` info in stdlib is limited — for real script inspection use `pyicu` (`icu.Char.getPropertyValueName(icu.UProperty.SCRIPT, ord(ch))`).

## Acceptance criteria

- [ ] Labelled evaluation set with the required diversity (≥10 languages, ≥3 length buckets, two same-script confusables, one two-script language, one non-Latin script).
- [ ] Baseline accuracy table for at least two LID libraries, including per-language, per-length-bucket, latency, model size.
- [ ] Script-based fast path implemented; threshold sweep table or plot with coverage vs. accuracy.
- [ ] Short-text aggregation experiment with quantified improvement for three confusable pairs.
- [ ] Confusable-error analysis with 10 examples per pair and one proposed mitigation each.
- [ ] Write-up (`README.md`) recommends a concrete routing policy — confidence and delta thresholds, script fast path, aggregation rule — and states which of your evaluation metrics it optimises.

## Stretch goals

- **Character-n-gram LM tiebreaker.** Train per-language character KenLM models (from exercise 02) on your confusable languages; use the LM with the lowest per-character perplexity to break ties from the LID model.
- **Prior from metadata.** Simulate user metadata (browser `Accept-Language`) as a Dirichlet prior over language and combine with the LID posterior. Report accuracy improvement on the short-text bucket.
- **Token-level LID.** Pick a code-switched corpus (LinCE has one for English-Spanish tweets) and train or evaluate a token-level LID (BiLSTM tagger over character embeddings, or a rule-based script-plus-dictionary classifier). Report token-level F1.
- **Model diff.** Compare fastText's predictions to CLD3 on the same test set; report where they disagree and which is right more often on each disagreement class.
