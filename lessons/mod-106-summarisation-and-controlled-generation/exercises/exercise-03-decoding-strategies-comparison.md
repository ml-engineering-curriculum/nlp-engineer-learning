# exercise-03: Decoding Strategies Comparison

**Estimated effort:** 3 hours

## Objective

Take a single fixed summariser and turn the entire decoding zoo from chapter 06 loose on it. Measure **quality**, **diversity**, and **stability** for greedy, beam, diverse beam, nucleus, typical, contrastive search, and MBR. End with a decision matrix that maps decoding strategy to product ask (deterministic summary, "regenerate" button, faithfulness-reranked pipeline). This is the exercise that stops "the model got worse when I changed `num_beams`" from being a mystery.

## Prerequisites

- Chapter [06](../06-decoding-strategies.md) — the decoding zoo. Optionally [07](../07-generation-controls-length-repetition-calibration.md) for length/repetition knobs.
- A fine-tuned summariser from exercise-01 **or** a zero-shot public checkpoint (`facebook/bart-large-cnn`, `google/pegasus-xsum`, `google/pegasus-cnn_dailymail`). Do not use a different model per strategy — the point of the exercise is to hold the model fixed.
- Python 3.10+; `transformers`, `datasets`, `evaluate`, `rouge_score`, `bert_score`, `sacrebleu` (for self-BLEU diversity).
- GPU strongly recommended for MBR (many samples per input).

## Dataset

Pick 500 examples from **one** of:

- **CNN/DailyMail** dev set — extractive-friendly; exposes the beam-search sweet spot.
- **XSum** dev set — abstractive; exposes hallucination-under-beam-width the fastest.
- **SAMSum** dev set — dialogue; different length distribution.

Recommendation: **CNN/DailyMail** for the primary run and **XSum** for the stretch "faithfulness × decoding" cell.

## Problem statement

### Part A — The strategy grid

Generate summaries for the same 500 dev inputs under each of the following configurations. **Use the same `GenerationConfig` base** (same `max_new_tokens`, `min_length`, `length_penalty`) — only vary the strategy-specific parameters listed below.

| # | Strategy                    | Key parameters                                                             |
|---|-----------------------------|----------------------------------------------------------------------------|
| 1 | Greedy                      | `num_beams=1, do_sample=False`                                             |
| 2 | Beam-4                      | `num_beams=4, no_repeat_ngram_size=3, early_stopping=True`                 |
| 3 | Beam-8                      | `num_beams=8, no_repeat_ngram_size=3, early_stopping=True`                 |
| 4 | Beam-16                     | `num_beams=16, no_repeat_ngram_size=3, early_stopping=True`                |
| 5 | Diverse beam (3 groups × 2) | `num_beams=6, num_beam_groups=3, diversity_penalty=1.0`                    |
| 6 | Nucleus (p=0.9, T=0.7)      | `do_sample=True, top_p=0.9, temperature=0.7`                               |
| 7 | Nucleus (p=0.95, T=1.0)     | `do_sample=True, top_p=0.95, temperature=1.0`                              |
| 8 | Typical (p=0.95)            | `do_sample=True, typical_p=0.95`                                           |
| 9 | Contrastive search          | `penalty_alpha=0.6, top_k=4`                                               |

Save all 9 × 500 outputs to disk (one JSONL file per strategy) so subsequent parts can rescore without regeneration.

### Part B — Quality panel

For each strategy, on the same 500 outputs, report:

- **ROUGE-1 / ROUGE-2 / ROUGE-L (F1)** via `evaluate.load("rouge")` with `use_stemmer=True`.
- **BERTScore-F1** with `model_type="microsoft/deberta-xlarge-mnli"`, `rescale_with_baseline=True`. If runtime is tight, restrict to a 200-example subset for BERTScore and note the subset in your write-up.
- **Mean and 90th-percentile output length** (tokens under the same tokeniser).
- **Repetition rate** — fraction of outputs that contain the same tri-gram twice. This surfaces the classic decoding pathology.

Present as one Markdown table (rows = strategies, columns = metrics) with 95 % bootstrap CIs (1000 resamples) on ROUGE-L.

### Part C — Diversity panel

Diversity matters for user-facing "regenerate" buttons, MBR, and creative rewriting. Measure it two ways:

- **Self-BLEU across seeds.** For strategies 6, 7, 8 (sampling), run each *three times* with different seeds. Compute mean pairwise BLEU across the three outputs per input. Lower = more diverse.
- **Distinct-n.** For all 9 strategies, compute the fraction of unique bi-grams and tri-grams across the 500 outputs. Higher = more lexical variety in the population.

Include a per-strategy row in a diversity table.

### Part D — Stability under seed / re-run

For strategies 6, 7, 8 (the sampling ones), quantify seed variance:

- Regenerate the same 500 inputs 3 times with seeds `[42, 43, 44]`.
- Report mean and standard deviation of ROUGE-L across the three runs.
- Report the fraction of inputs where **all three runs agree** (all three ROUGE-L within ± 2 pts).

Deterministic strategies (1–5, 9) should report `std = 0` and `agreement = 100 %` — include them in the table to make the contrast visible.

### Part E — MBR decoding

Implement MBR from chapter 06:

1. For each of a 200-example subset, sample $N = 16$ candidates with strategy 6 (nucleus).
2. Score each candidate `c` by $\sum_{c' \ne c} \text{sim}(c, c')$ where `sim` is BERTScore-F1. This is the MBR utility.
3. Return the candidate with the highest MBR score.

Report ROUGE-1/2/L and BERTScore-F1 on the MBR outputs. Compare against beam-4 (row 2) and greedy (row 1) on the *same* 200-example subset. Also report per-input latency (ideally seconds/example on your hardware).

### Part F — Beam-width sweep (the beam-search curse)

Plot or table: ROUGE-L vs. `num_beams ∈ {1, 2, 4, 8, 16, 32}` on the full 500-example set. This is the empirical version of the Stahlberg & Byrne / Meister & Cotterell result cited in chapter 06.

Expected shape: ROUGE-L climbs from 1 → 4/5, then flattens or declines. If yours climbs monotonically, either your model is unusual or you have a bug — investigate before trusting the number.

### Part G — Failure-mode catalogue

Pull *the same* 5 inputs across all 9 strategies (plus MBR). For each input, paste the 10 outputs side by side and annotate which pathologies from chapter 06's "failure modes" checklist appear:

- Repetition (n-gram loops).
- Truncation / mid-sentence stop.
- Empty output.
- Length drift (much longer / shorter than reference).
- High seed variance (for sampling strategies).
- Beam-width-amplified confidence in a wrong summary.

You do not need to annotate every input, but you should be able to identify at least three distinct pathologies across the sample.

### Part H — Decision matrix and write-up

A 500–800 word `README.md` covering:

- Model and dataset used.
- The quality table (Part B), diversity table (Part C), stability table (Part D), and MBR row (Part E).
- The beam-width sweep plot / table (Part F).
- The failure-mode catalogue (Part G) with 3+ annotated pathologies.
- A **decision matrix** with rows = product asks and columns = recommended strategy, e.g.:
  - "Deterministic single summary" → beam-4 (or 5, based on your data).
  - "'Regenerate' button that gives different-but-good options" → diverse beam or nucleus.
  - "Best-effort quality at premium latency budget" → MBR or diverse-beam + faithfulness rerank.
  - "Creative rewrite / paraphrase for a UX feature" → typical or contrastive.
- One "what next" idea (a strategy you did not test but would try in production).

## Starter guidance

- **Save `GenerationConfig`s to disk.** One YAML/JSON per strategy in `configs/`. This is how production ships decoding — as a versioned artefact — and doing it here builds the habit.
- **`min_length` matters.** Without it, some models emit near-empty outputs. Set `min_length=15` for CNN/DailyMail, `min_length=8` for XSum.
- **`do_sample=True` does not silently disable `num_beams`.** If you set both, the model warns and uses `num_beams` × sampling (very slow, rarely what you want). Set exactly one of the decoding families.
- **Seed the samplers.** `transformers.set_seed(seed)` before each sampling run, or the "seed variance" in Part D is measuring machine noise instead of sampler noise.
- **BERTScore is expensive.** Batch it (`batch_size=32`). If your BERTScore runtime exceeds your budget, restrict Part B to a 200-example subset and note it.
- **MBR utility choice.** BERTScore is a reasonable default; BLEU is faster but less well-aligned with human judgement for summarisation. Note which you used.
- **Contrastive search is decoder-only-friendly.** On encoder-decoder BART/PEGASUS it works but is less well-studied. If contrastive-search results look degenerate, note it and move on rather than tuning it into shape.

## Acceptance criteria

- [ ] All 9 strategies (Part A) generated on the same 500 inputs with the same base `GenerationConfig`.
- [ ] Quality panel (ROUGE, BERTScore, length, repetition) reported with bootstrap CIs on ROUGE-L.
- [ ] Diversity panel (self-BLEU across seeds, distinct-n) reported.
- [ ] Stability panel (seed variance for sampling strategies) reported.
- [ ] MBR-16 implemented and compared against greedy / beam-4.
- [ ] Beam-width sweep at `num_beams ∈ {1, 2, 4, 8, 16, 32}`; plotted or tabulated.
- [ ] Failure-mode catalogue with 5 inputs × 10 outputs and ≥ 3 distinct pathologies annotated.
- [ ] 500–800 word write-up with a decision matrix mapping product asks to decoding strategies.

## Stretch goals

- **Faithfulness × decoding.** Repeat Part B on XSum with a sentence-level NLI faithfulness score (chapter 11, `microsoft/deberta-v3-large-mnli`). Does beam-16 hallucinate *more* than beam-4? Does nucleus hallucinate more than beam? Rank strategies by faithfulness × ROUGE-L, not ROUGE-L alone.
- **Constrained + sampled.** Attach a `no_repeat_ngram_size` and a `min_length` constraint to nucleus (row 6). Re-measure quality and diversity. Are constraints as effective for sampling as for beam?
- **Speculative decoding.** If your inference stack supports it (vLLM, TGI), measure latency for beam-4 with and without speculative decoding against a smaller draft model. Report tokens/sec and any quality delta.
- **Length-penalty sweep.** Fix beam-4; sweep `length_penalty ∈ {0.6, 0.8, 1.0, 1.2, 2.0}`. Plot ROUGE-L and mean length. Find the setting where the mean output length matches the reference distribution most closely.
- **Custom `LogitsProcessor`.** Write and plug in a custom `LogitsProcessor` that penalises tokens outside the source vocabulary (a soft-copy prior). Measure ROUGE-L and hallucination change; discuss whether it is worth the complexity.
