# Cross-Lingual Transfer and Zero-Shot

## Motivation

You have labelled data in English (or Chinese, or French) and you want a classifier / tagger / extractor to work in a language for which you have *no* labels. The three patterns that make this possible — **translate-train**, **translate-test**, and **zero-shot transfer** — are the workhorses of multilingual NLP in production. Which one to reach for depends on the direction of the language pair, whether MT quality is good enough, and how many target-language examples you can *cheaply* create. This chapter walks the trade-offs and gives you a decision guide.

## The three patterns

Say the source (training) language is $L_s$ (usually English) and the target (inference) language is $L_t$. You have labelled data $\{(x_i^s, y_i)\}$ in $L_s$ and unlabelled inference-time inputs $\{x_j^t\}$ in $L_t$.

- **Zero-shot transfer.** Fine-tune a multilingual encoder (XLM-R, mDeBERTa, mT5) on $\{(x_i^s, y_i)\}$; at inference, feed $x_j^t$ directly. No translation at either training or inference time. Cheap; quality varies with how well the encoder covers $L_t$.
- **Translate-train.** Translate the training data $x_i^s \to \tilde{x}_i^t$ using an NMT model; fine-tune the multilingual encoder on $\{(\tilde{x}_i^t, y_i)\}$. Adds one training-time translation pass per training example; usually beats zero-shot when MT quality is decent and enough training data exists to overcome MT noise.
- **Translate-test.** Fine-tune the multilingual encoder (or even a monolingual $L_s$ encoder) on $\{(x_i^s, y_i)\}$; at inference, translate each $x_j^t \to \tilde{x}_j^s$ and predict on $\tilde{x}_j^s$. Adds one inference-time translation pass per request; benefits from a strong source-language encoder.

Two additional patterns are sometimes named:

- **Translate-both.** Translate training data to $L_t$ *and* translate test inputs to $L_s$; predict with both, ensemble. Higher cost, occasionally higher quality.
- **Multi-source / joint multilingual.** Fine-tune the multilingual encoder on labelled data across *many* source languages simultaneously. When you have labels in several languages already, this is almost always better than any single-source approach.

## When each wins

Rough guidance, backed by the XTREME (Hu et al., ["XTREME: A Massively Multilingual Multi-task Benchmark"](https://arxiv.org/abs/2003.11080), *ICML 2020*) and XTREME-R (Ruder et al., ["XTREME-R: Towards More Challenging and Nuanced Multilingual Evaluation"](https://arxiv.org/abs/2104.07412), *EMNLP 2021*) empirical studies:

| Situation                                                             | Prefer                                                          |
|----------------------------------------------------------------------|-----------------------------------------------------------------|
| Small classifier, latency-critical, target language covered well by XLM-R | Zero-shot on XLM-R-large.                                        |
| Sentence-level classification, MT quality good on the pair              | Translate-train on the target language.                          |
| Structured tasks (NER, span extraction) where labels are hard to align   | Zero-shot (avoid alignment-preservation issues from MT).         |
| High-quality reference translations available (professional TM)          | Translate-train with the professional translations.              |
| Low-resource target language, MT quality unknown                        | Zero-shot as baseline; translate-train if MT is > ~30 chrF.       |
| Ensemble budget available                                                | Translate-both, then ensemble by confidence.                     |

## Translate-train, concretely

Take your English training set and translate to Swahili with NLLB:

```python
translator = build_translator("facebook/nllb-200-distilled-1.3B",
                              src_lang="eng_Latn", tgt_lang="swh_Latn")

def translate_split(dataset, batch_size=32):
    swahili_texts = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset["text"][i : i + batch_size]
        swahili_texts.extend(translator(batch, num_beams=4, max_new_tokens=256))
    return dataset.add_column("text_sw", swahili_texts)

sw_train = translate_split(en_train)
```

Then fine-tune XLM-R on the Swahili-translated data with the labels unchanged. Two things to watch:

- **Label preservation on structured tasks.** For classification, the label is a categorical `y` — it survives translation trivially. For **NER or span extraction**, the label is a *span in the source string*. MT rewrites the string; the span offsets no longer point at anything. Options: use a *span-preserving* MT model with alignment (usually degrades MT quality), use span projection via word alignment (fastalign, awesome-align), or just skip translate-train for span tasks and use zero-shot.
- **Filter low-quality translations.** Score each translated example with COMET-QE (chapter 10) and drop the bottom decile. Translate-train is only as good as the translations you feed it.

## Translate-test, concretely

Fine-tune (or take pretrained) on the source-language task, then translate at inference:

```python
# Fine-tune: standard English classifier on XLM-R.
classifier = pipeline("text-classification", model="my/finetuned-xlmr-en")

# Inference: translate each incoming target-language input to English.
translator = build_translator("facebook/nllb-200-distilled-1.3B",
                              src_lang="swh_Latn", tgt_lang="eng_Latn")

def predict(text_sw):
    text_en = translator([text_sw], num_beams=4, max_new_tokens=256)[0]
    return classifier(text_en)[0]
```

Trade-offs:

- **Adds a translation call per request.** Latency and cost matter. Pre-translate at ingestion if possible.
- **Loses information the target-language original carried.** MT is lossy; sarcasm, code-switching, and register cues often do not survive.
- **Simplifies the classifier stack.** If your English classifier is already excellent and MT is decent, translate-test is often the "just make it work" fastest path.

## True zero-shot transfer

Fine-tune XLM-R on English, evaluate on Swahili — literally as-is. Chapter 07 already covered the mechanics; the only additional consideration here is **language-invariant fine-tuning tricks** that improve transfer quality:

- **Adversarial language classifiers.** Train the encoder to *not* be able to tell what language its input is in (via a gradient-reversal head), forcing more language-neutral representations. Chen et al., ["Adversarial Deep Averaging Networks for Cross-Lingual Sentiment Classification"](https://arxiv.org/abs/1606.01614) (*TACL 2018*).
- **Task-specific multilingual fine-tuning schedules.** Warm up on the multilingual pretraining objective in the target language before switching to the source-language task. `MAD-X` (Pfeiffer et al., ["MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer"](https://arxiv.org/abs/2005.00052), *EMNLP 2020*) is the standard reference.
- **Consistency-regularised training with translated pairs.** Force the model to produce the same output on the English and translated Swahili input via a KL loss.

For most production deployments, none of the above beats *just have more languages in the fine-tune*. If you can pay for labels in even 2–3 target languages, the joint multilingual fine-tune usually crosses translate-train quality.

## When zero-shot silently fails: tokenisation coverage

Zero-shot's failure mode is not "the model produces garbage" — it is "the model produces confident garbage." Two diagnostics to run before trusting a zero-shot deployment:

- **Tokenisation shatter rate.** For each target language, tokenise a held-out corpus of a few thousand sentences and count the average subword pieces per word. If this is > 3, the encoder is representing the language mostly through byte-fallback and zero-shot will be poor. Consider XLM-V (larger vocab) or ByT5.
- **Language-ID probe on the encoder output.** Add a linear language-ID head on the frozen encoder; if it easily distinguishes your target language from the source, the representations are *not* language-neutral and cross-lingual transfer will lose ground. Chen et al. (2018) is the classical treatment.

## NER and structured tagging: the special case

NER, chunking, POS, and other token-level tasks are the hardest transfer setting because the labels are span-anchored. Three patterns:

- **Zero-shot on XLM-R with the source subword-alignment convention.** Works — chapter 01 of mod-104 covers the subword-alignment mechanics. This is the pragmatic default.
- **Span projection via word alignment.** Use a word aligner (`awesome-align`, `simalign`, `fastalign`) to project source-language spans through the translation. Quality depends heavily on the aligner; error compounds with MT noise.
- **Silver labels from a large teacher model.** Use a very-strong multilingual (or a language-specific) model to auto-label the target-language corpus; fine-tune your production model on the silver labels. This is the recipe most competition submissions use.

## Multi-source joint fine-tuning

When you have labelled data in more than one language, mix them into a single training set. Two useful tricks:

- **Language upsampling with temperature.** Sample from each language with $p(L) \propto n_L^{1/\tau}$ (temperature $\tau = 3–5$). Prevents high-resource languages from dominating.
- **Language-tag prepend.** Prepend a `[LANG=en]` / `[LANG=sw]` token to each input. Lets the model condition on language when useful without forcing it.

Joint multilingual fine-tuning is now the strongest published pattern on XTREME-style benchmarks whenever multi-language labels exist.

## Retrieval-augmented cross-lingual transfer

For classification / extraction tasks where you have a small labelled pool and a large multilingual encoder:

1. Encode both the labelled pool and the target-language input with a multilingual sentence-encoder (LaBSE, multilingual-mpnet).
2. Retrieve the top-$k$ labelled neighbours from the pool for the target-language input.
3. Predict the majority label, or feed the neighbours as few-shot examples to an mT5 / instruction-tuned LLM.

This is a cross-lingual variant of kNN classification and is the workhorse for cold-start deployments where you can label a few hundred pool examples in each language you care about.

## Evaluating cross-lingual transfer

The multilingual analogues of the standard task benchmarks:

- **XNLI** (Conneau et al., 2018) — NLI in 15 languages. The canonical benchmark for classification transfer.
- **XTREME** — 40 languages, 9 tasks. The most-cited transfer benchmark.
- **XTREME-R** — the harder successor.
- **TyDiQA** (Clark et al., ["TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages"](https://arxiv.org/abs/2003.05002), *TACL 2020*) — QA across 11 typologically diverse languages.
- **WikiANN / PAN-X** — NER across 176 languages.
- **XCOPA**, **XStoryCloze**, **X-FACTR** — commonsense, story completion, factual probes.

Report *per-language* numbers. A single averaged score hides a factor-of-two performance range across the language set — the flatter your per-language distribution, the better your deployment will survive contact with your actual user base.

## Common failure modes and their fixes

- **Zero-shot works on high-resource languages, fails on low-resource.** Tokenisation shatter is likely — check the per-word subword count. XLM-V or a language-specific fine-tune usually fixes it.
- **Translate-train collapses.** Translation quality is too low. Filter with COMET-QE, or use a stronger translator (bigger NLLB tier).
- **NER translate-train produces nonsense labels.** Span offsets did not survive MT. Use zero-shot instead, or silver-label with a teacher model.
- **Cross-lingual model performs worse than a from-scratch monolingual baseline.** Sometimes the target language has enough data of its own that monolingual wins. Always run the monolingual baseline as a sanity check.
- **Translate-test loses cues from the original.** Sarcasm, code-switching, and dialect signals do not survive round-trip. Consider zero-shot or translate-both.

## Chapter summary

- Three patterns cover most cross-lingual transfer: zero-shot (fine-tune on source, infer on target), translate-train (translate the training data), and translate-test (translate the inference input).
- Zero-shot is the cheapest baseline; translate-train usually beats it when MT quality is decent and labels are plentiful; translate-test is the "make it work now" fallback when the source-language pipeline is already strong.
- Structured / span-anchored tasks (NER, extraction) prefer zero-shot because MT rewrites the string and breaks the offsets.
- If labels exist in multiple languages, joint multilingual fine-tuning with temperature-sampled language balancing is the strongest pattern.
- Always report *per-language* numbers on the target language set, not just an average — the tail of the distribution decides whether the deployment survives real users.
- Zero-shot's silent failure mode is *tokenisation shatter*; diagnose it with average subword pieces per word before trusting the model.
