# mod-111 · NLP-Specific Evaluation: Metrics, Methodology, Benchmarks

The evaluation discipline that lets you decide whether a system is actually better — pick the right metric per task, pick the right benchmark suite, prove or disprove a system-vs-system delta with paired significance, check that the eval set is not contaminated by the model's pretraining data, and design a human-evaluation protocol whose numbers a stakeholder can act on. The chapters walk the NLP-specific metric catalogue (BLEU / chrF / ROUGE / METEOR / COMET / BLEURT / BERTScore / seqeval entity-F1 / SQuAD F1 & EM / perplexity), the six benchmark suites that dominate reporting (GLUE / SuperGLUE / XTREME / FLORES / MTEB / lm-evaluation-harness), and the methodology layer that sits on top of both.

**Estimated effort:** 12 hours

## Learning objectives

- Apply the NLP-specific metric catalogue (BLEU, chrF, ROUGE, METEOR, COMET, BLEURT, BERTScore, seqeval entity F1, SQuAD F1 / EM, perplexity) correctly per task.
- Pick the right benchmark suite (GLUE / SuperGLUE / XTREME / FLORES / MTEB / lm-evaluation-harness) and reason about its limitations.
- Run statistical significance tests for NLP (bootstrap resampling, paired permutation, system-pair comparisons).
- Detect contamination against pretraining data and design contamination-resistant held-out evaluation.
- Design human-evaluation protocols with calibrated rubrics, position-bias controls, and demographic / dialect slicing.

## Chapters

1. [The NLP metric catalogue and task fit](01-metric-taxonomy-and-task-fit.md)
2. [Classification and sequence-labelling metrics](02-classification-and-sequence-labelling-metrics.md)
3. [Extractive QA metrics: SQuAD F1 and EM](03-extractive-qa-metrics.md)
4. [Generation metrics: BLEU, chrF, ROUGE, METEOR, BERTScore, BLEURT, COMET](04-generation-metrics-bleu-rouge-meteor-bertscore-comet.md)
5. [Perplexity and language-model evaluation](05-perplexity-and-language-model-evaluation.md)
6. [Benchmark suites: GLUE, SuperGLUE, XTREME, FLORES, MTEB, lm-evaluation-harness](06-benchmark-suites-glue-superglue-xtreme-flores-mteb-lm-eval-harness.md)
7. [Statistical significance for NLP: bootstrap, paired permutation, and how not to fool yourself](07-statistical-significance-for-nlp.md)
8. [Contamination detection and decontamination](08-contamination-detection-and-decontamination.md)
9. [Human evaluation: rubrics, position bias, and demographic slicing](09-human-evaluation-protocol-design.md)

## Exercises

- [exercise-01 · Task-aware metric selection rubric](exercises/exercise-01-task-aware-metric-selection-rubric.md)
- [exercise-02 · Bootstrap significance for NLP](exercises/exercise-02-bootstrap-significance-for-nlp.md)
- [exercise-03 · Contamination detection and decontamination](exercises/exercise-03-contamination-detection-and-decontamination.md)
- [exercise-04 · Human evaluation protocol design](exercises/exercise-04-human-evaluation-protocol-design.md)

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.

## Also in this module

- `labs/` — long-form hands-on labs (added on a later authoring cycle).
- `quizzes/` — knowledge checks (added on a later authoring cycle).
- [`resources.md`](resources.md) — primary sources, standards, and further reading.
