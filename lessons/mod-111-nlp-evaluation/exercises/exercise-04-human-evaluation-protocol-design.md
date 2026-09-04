# exercise-04: Human Evaluation Protocol Design

**Estimated effort:** 3 hours

## Objective

Design and pilot a **human-evaluation protocol** for one of three consequential NLP tasks — summarisation, translation, or open-ended chat generation. Deliver a fully-specified rubric, a screening/calibration set, a position-bias-controlled rating interface (or spec + fixtures), a small pilot's-worth of ratings, and an analysis that includes per-rater z-standardisation, paired-bootstrap significance, inter-annotator agreement, and — critically — an **LLM-as-judge cross-check** and a **demographic / dialect slice** where either applies.

## Prerequisites

- Chapter [09](../09-human-evaluation-protocol-design.md); mod-107 chapter 11 (MT-specific human eval) or mod-106 chapter 13 (summarisation human eval) if you pick those tracks.
- Python 3.10+; `numpy`, `pandas`, `scipy`, `matplotlib`. For LLM-as-judge: an accessible frontier LLM API of your choice.
- At least 2 humans who will rate (colleagues, coursemates, or hired raters). If you cannot find 2, run the human portion on yourself + LLM-as-judge and be explicit in the write-up.

## Problem statement

### Part A — Pick the task and system pair

Choose one of the following task-and-system-pair setups:

- **Summarisation.** Two summarisers on 30 news articles from a *time-boxed* source (recent XSum-style headlines, or your choice). Two systems to compare: e.g., `facebook/bart-large-cnn` vs. `google/pegasus-xsum` (or your favourite two). Rubric is the SummEval-style 5-dimension Likert (coherence, consistency, fluency, relevance, faithfulness) — see Fabbri et al., 2021.
- **Machine translation.** Two MT systems on 30 sentences from FLORES-200 devtest, one high-resource pair (eng→deu recommended for accessibility). Rubric is source-based DA (0–100) plus a light MQM error-tag pass (accuracy / fluency, minor / major).
- **Open-ended chat.** Two LLMs on 30 prompts sampled from a *time-boxed* source (recent StackExchange questions after your model's cutoff, or MT-Bench-style prompts you write). Rubric is pairwise ("which response is better") plus a 3-dimension Likert (helpfulness, factuality, safety).

Document the pick, the two systems, and why — one paragraph in `setup.md`.

### Part B — Write the rubric and screening set

Produce `rubric.md` with:

- **Task definition.** What is being rated; what "quality" means for this task.
- **Rubric shape.** Which of the shapes from chapter 09 you chose and why.
- **Per-dimension anchors.** For each scale value or error tag, a short verbal anchor *and* at least one worked example (positive + negative where applicable). Anchors + examples are the single biggest driver of inter-annotator agreement.
- **Screening set.** 10 items of known difficulty with an "answer key" — the outputs you (the protocol designer) think a competent rater should produce. Raters run this before real rating; you filter based on their agreement with the key.
- **Rater instructions.** Step-by-step: "you will see X, do Y, then Z." Include position-bias-control text ("do not assume the order reflects any preference"), fatigue guidance ("no more than 90 minutes per session"), and content-sensitivity warning if applicable.

Total: 400–800 words plus the screening set. Version the guideline (`v1.md`); if you iterate after the pilot, keep both versions and note the diff.

### Part C — Build the rating fixture

Produce `rating_fixture/` — a folder containing:

- **`items.jsonl`** — 30 evaluation items, each with `{"item_id", "source", "output_a", "output_b", "order_seed"}`. Randomise the presentation order per rater (either flip A/B randomly per item, or generate two "presentation variants" of the batch — one for order AB, one for BA — and hand each rater a *different* variant so the aggregate is order-balanced).
- **`screening.jsonl`** — the 10 screening items from Part B.
- **A rating interface.** Either (a) a minimal Streamlit / Gradio app that shows items one at a time and captures ratings, (b) a well-designed spreadsheet template with instructions and validation, or (c) a Jupyter notebook the rater fills in. Which one is fine — pick what your raters will actually use.
- **`README.md`** — how a rater starts, saves progress, and submits.

### Part D — Run the pilot

Recruit **at least 2 raters** (if possible 3+; if not, self + LLM-judge). For each:

- Run the screening set first; record scores.
- If a rater fails screening (< 0.6 agreement with the answer key), retrain or exclude and document.
- Run the full 30-item batch with randomised order.
- Record ratings in `pilot_ratings.csv` (columns: `item_id`, `rater_id`, `system_a_score`, `system_b_score` for Likert; `winner`, `tie_flag` for pairwise; `error_tag`, `severity` for MQM).

Anonymise rater IDs before analysis; publish only aggregated numbers.

### Part E — Analysis

Produce `analysis.md` with:

- **Per-rater z-standardised scores.** For scalar rubrics, standardise per rater before aggregating; do not average raw scores.
- **System means per dimension.** Table with per-rubric-dimension mean, per system, with bootstrap 95 % CIs.
- **Paired-bootstrap significance** on the system-vs-system differences per dimension (import `sigtest` from exercise-02 if you completed it, otherwise implement inline).
- **Inter-annotator agreement.** Krippendorff's alpha for categorical / ordinal rubrics; Pearson + Spearman for continuous; per-dimension. Report both raw and standardised. Interpret the alpha with the standard cutoffs (0.667 acceptable, 0.8 solid — see Krippendorff, 2004).
- **Position-bias diagnostic.** Compare the win rate for whichever system was shown first vs. shown second (for pairwise) or the mean score for first-presented vs. second-presented (for Likert). If bias exceeds a few percent, discuss.
- **LLM-as-judge cross-check.** Use a frontier LLM (Claude / GPT-4-class / open-weight instruct) prompted with your rubric to rate the same 30 items. Report the same table (per-dimension mean) and the Spearman correlation between LLM and human aggregated ratings. Interpret whether the LLM judge tracks the humans on this task.
- **Demographic / dialect slice discussion.** If your task and data admit a demographic axis (dialect for text-to-text; region for chat; native-language for MT), report per-group scores. If it does not — say so and explain what a bigger-budget version would slice on.

### Part F — The one-page protocol summary

`protocol_summary.md` — one page (≤ 500 words) that a manager or ethics reviewer can read to decide whether to fund the full-scale (300-item, 5-rater, per-language) version. Include:

- Task, systems, rubric shape, rater pool composition (with demographics you feel comfortable publishing at the aggregate level), IAA range across dimensions, headline system-vs-system delta with CI and $p$-value, position-bias check result, LLM-judge Spearman, one honest caveat about what the pilot does *not* measure.

## Starter guidance

- **Anchors + examples make or break IAA.** Do not ship a rubric with "5 = flawless" and no example.
- **Randomise order per item, not per rater.** Half the items should be AB, half BA — otherwise a rater who reads carefully at the start and tires by the end biases the aggregate.
- **Cap the session.** 30 items should fit in an hour; do not run one 3-hour session.
- **Anonymise before analysis.** Rater IDs in the CSV should be `r1`, `r2`, `r3` — not names.
- **The LLM-as-judge is *cross-check*, not primary.** Use it to test whether your protocol correlates with LLM signal, not to replace humans. If Spearman is < 0.4, refine the LLM prompt or fall back to human-only.
- **Publish rater-pool composition** at the aggregate — "3 raters, all native English speakers, 2 with prior MT rating experience" or similar. Do not name individuals.
- **If you have to skip the demographic slice** (small pool, no dialect variation available), state that plainly in Part F's caveat.
- **Recycle exercise-02's toolkit** for CIs and paired significance if you have it; otherwise ship the minimum inline.

## Acceptance criteria

- [ ] `setup.md` — task, system pair, and justification.
- [ ] `rubric.md` + 10-item `screening.jsonl` — rubric with per-value anchors and worked examples; screening keys included.
- [ ] `rating_fixture/` — 30-item evaluation batch with per-item order randomisation, a runnable / usable rating interface, and a README for raters.
- [ ] `pilot_ratings.csv` — anonymised ratings from ≥ 2 raters on the 30-item batch (screening filter applied and documented).
- [ ] `analysis.md` — per-rater z-standardised means, per-system + per-dimension table with bootstrap CIs, paired-bootstrap significance, Krippendorff's alpha per dimension, position-bias diagnostic, LLM-judge cross-check with Spearman correlation to human aggregate, demographic / dialect discussion.
- [ ] `protocol_summary.md` — one-page summary suitable for a manager or ethics reviewer, with headline delta + CI + $p$-value, IAA range, and one caveat.
- [ ] All rater identities anonymised; aggregate rater-pool composition disclosed.

## Stretch goals

- **Three-rater agreement.** Add a third human rater; report Fleiss' kappa (categorical rubrics) or Krippendorff's alpha across three raters.
- **Full MQM pass.** If you chose MT, layer a proper MQM span-level annotation on top of the DA scores. Report per-error-category counts and severity-weighted MQM scores.
- **Cross-order agreement.** Present the same items in both orders to the same rater (separated in time); measure within-rater test-retest reliability.
- **Rater diversity stress test.** Split your raters into two demographic sub-pools (native / non-native, or expert / non-expert) and report per-pool aggregates. Discuss any systematic differences.
- **Full pre-registration.** Write a pre-registration document (before running the pilot) stating the effect size you expect and the sample size that would give 80 % power for the paired bootstrap. Compare pre-registered predictions to post-hoc findings.
- **LLM-judge prompt search.** Try 3 different LLM-judge prompt formulations and report which correlates best with human aggregate. Do not "prompt-tune to your target" — declare the prompt choice before running the final comparison.

## Deliverables

```
setup.md
rubric.md
screening.jsonl
rating_fixture/
    items.jsonl
    screening.jsonl
    interface/   # streamlit_app.py OR ratings_template.xlsx OR rate.ipynb
    README.md
pilot_ratings.csv
analysis.md
protocol_summary.md
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
