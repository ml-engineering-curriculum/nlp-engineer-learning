# Resources for mod-111 · NLP-Specific Evaluation: Metrics, Methodology, Benchmarks

Prefer primary sources: papers, standards, and official documentation. Blog posts and library docs are included only where they are the canonical reference for a tool or idea.

## Metric taxonomy and task fit

- **`evaluate` (metric hub + canonical implementations):** <https://huggingface.co/docs/evaluate/index>. Metric catalogue: <https://huggingface.co/evaluate-metric>.
- **`scikit-learn` classification metrics (`sklearn.metrics`):** <https://scikit-learn.org/stable/modules/model_evaluation.html>.
- **`sacrebleu` (BLEU / chrF / TER with pinned signatures):** <https://github.com/mjpost/sacrebleu>.
- **`seqeval` (span-level NER / chunking F1):** <https://github.com/chakki-works/seqeval>.
- **`rouge-score` (Google's reference ROUGE):** <https://github.com/google-research/google-research/tree/master/rouge>.
- **`bert-score`:** <https://github.com/Tiiiger/bert_score>.
- **`unbabel-comet`:** <https://github.com/Unbabel/COMET>.
- **BLEURT:** <https://github.com/google-research/bleurt>.

## Classification and sequence-labelling metrics

- **MCC on CoLA / linguistic acceptability:** Alex Warstadt, Amanpreet Singh, Samuel R. Bowman, "Neural Network Acceptability Judgments", *TACL 2019*. [arXiv:1805.12471](https://arxiv.org/abs/1805.12471).
- **Expected Calibration Error (ECE) + temperature scaling:** Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger, "On Calibration of Modern Neural Networks", *ICML 2017*. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599).
- **Reliability diagrams (origin):** Allan H. Murphy, Robert L. Winkler, "Reliability of Subjective Probability Forecasts of Precipitation and Temperature", *Applied Statistics 1977*.
- **`seqeval` (strict mode and schemes):** Hiroki Nakayama, "seqeval: A Python framework for sequence labeling evaluation". Repo and documentation: <https://github.com/chakki-works/seqeval>.
- **CoNLL-2003 NER (defines the span-level F1 convention):** Erik F. Tjong Kim Sang, Fien De Meulder, "Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition", *CoNLL 2003*. [aclanthology.org/W03-0419](https://aclanthology.org/W03-0419/).
- **OntoNotes NER:** Sameer Pradhan et al., "Towards Robust Linguistic Analysis using OntoNotes", *CoNLL 2013*. [aclanthology.org/W13-3516](https://aclanthology.org/W13-3516/).
- **Multi-label metrics (coverage, LRAP, ranking-loss definitions):** Grigorios Tsoumakas, Ioannis Katakis, "Multi-Label Classification: An Overview", *IJDWM 2007*. <https://www.igi-global.com/article/multi-label-classification/1786>.
- **Matthews, "Comparison of the predicted and observed secondary structure of T4 phage lysozyme":** B. W. Matthews, *Biochimica et Biophysica Acta 1975* (origin of MCC).

## Extractive QA metrics (SQuAD family)

- **SQuAD 1.1:** Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang, "SQuAD: 100,000+ Questions for Machine Comprehension of Text", *EMNLP 2016*. [arXiv:1606.05250](https://arxiv.org/abs/1606.05250). Evaluator (official normaliser): <https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py>.
- **SQuAD 2.0 (unanswerable + best-threshold F1):** Pranav Rajpurkar, Robin Jia, Percy Liang, "Know What You Don't Know: Unanswerable Questions for SQuAD", *ACL 2018*. [arXiv:1806.03822](https://arxiv.org/abs/1806.03822).
- **XQuAD:** Mikel Artetxe, Sebastian Ruder, Dani Yogatama, "On the Cross-lingual Transferability of Monolingual Representations", *ACL 2020*. [arXiv:1910.11856](https://arxiv.org/abs/1910.11856). Dataset: <https://github.com/deepmind/xquad>.
- **MLQA:** Patrick Lewis, Barlas Oğuz, Ruty Rinott, Sebastian Riedel, Holger Schwenk, "MLQA: Evaluating Cross-lingual Extractive Question Answering", *ACL 2020*. [arXiv:1910.07475](https://arxiv.org/abs/1910.07475). Evaluator with per-language normalisers: <https://github.com/facebookresearch/MLQA>.
- **TyDi QA:** Jonathan H. Clark et al., "TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages", *TACL 2020*. [arXiv:2003.05002](https://arxiv.org/abs/2003.05002). Scorer: <https://github.com/google-research-datasets/tydiqa>.
- **MRQA 2019 shared task:** Adam Fisch et al., "MRQA 2019 Shared Task: Evaluating Generalization in Reading Comprehension", *MRQA Workshop 2019*. [arXiv:1910.09753](https://arxiv.org/abs/1910.09753).
- **Answer-equivalence classifier:** Jannis Bulian et al., "Tomayto, Tomahto. Beyond Token-level Answer Equivalence for Question Answering Evaluation", *EMNLP 2022*. [arXiv:2202.07654](https://arxiv.org/abs/2202.07654).

## Generation metrics (BLEU, chrF, ROUGE, METEOR, BERTScore, BLEURT, COMET)

- **BLEU:** Kishore Papineni, Salim Roukos, Todd Ward, Wei-Jing Zhu, "BLEU: A Method for Automatic Evaluation of Machine Translation", *ACL 2002*. [aclanthology.org/P02-1040](https://aclanthology.org/P02-1040/).
- **SacreBLEU (reporting signatures and pinned tokeniser):** Matt Post, "A Call for Clarity in Reporting BLEU Scores", *WMT 2018*. [arXiv:1804.08771](https://arxiv.org/abs/1804.08771).
- **chrF:** Maja Popović, "chrF: character n-gram F-score for automatic MT evaluation", *WMT 2015*. [aclanthology.org/W15-3049](https://aclanthology.org/W15-3049/).
- **chrF++:** Maja Popović, "chrF++: words helping character n-grams", *WMT 2017*. [aclanthology.org/W17-4770](https://aclanthology.org/W17-4770/).
- **ROUGE:** Chin-Yew Lin, "ROUGE: A Package for Automatic Evaluation of Summaries", *ACL 2004 Workshop*. [aclanthology.org/W04-1013](https://aclanthology.org/W04-1013/).
- **METEOR:** Satanjeev Banerjee, Alon Lavie, "METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments", *ACL 2005 Workshop*. [aclanthology.org/W05-0909](https://aclanthology.org/W05-0909/).
- **TER:** Matthew Snover et al., "A Study of Translation Edit Rate with Targeted Human Annotation", *AMTA 2006*. [aclanthology.org/2006.amta-papers.25](https://aclanthology.org/2006.amta-papers.25/).
- **BERTScore:** Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, Yoav Artzi, "BERTScore: Evaluating Text Generation with BERT", *ICLR 2020*. [arXiv:1904.09675](https://arxiv.org/abs/1904.09675).
- **BLEURT:** Thibault Sellam, Dipanjan Das, Ankur P. Parikh, "BLEURT: Learning Robust Metrics for Text Generation", *ACL 2020*. [arXiv:2004.04696](https://arxiv.org/abs/2004.04696).
- **BLEURT-20 (compact checkpoint):** Amy Pu, Hyung Won Chung, Ankur P. Parikh, Sebastian Gehrmann, Thibault Sellam, "Learning Compact Metrics for MT", *EMNLP 2021*. [arXiv:2110.06341](https://arxiv.org/abs/2110.06341).
- **COMET:** Ricardo Rei, Craig Stewart, Ana C. Farinha, Alon Lavie, "COMET: A Neural Framework for MT Evaluation", *EMNLP 2020*. [arXiv:2009.09025](https://arxiv.org/abs/2009.09025). Checkpoint: [`Unbabel/wmt22-comet-da`](https://huggingface.co/Unbabel/wmt22-comet-da).
- **COMET-Kiwi (reference-free):** Ricardo Rei et al., "CometKiwi: IST-Unbabel 2022 Submission for the Quality Estimation Shared Task", *WMT 2022*. [aclanthology.org/2022.wmt-1.60](https://aclanthology.org/2022.wmt-1.60/). Checkpoint: [`Unbabel/wmt22-cometkiwi-da`](https://huggingface.co/Unbabel/wmt22-cometkiwi-da).
- **xCOMET (span-level, MQM-style):** Nuno M. Guerreiro et al., "xCOMET: Transparent Machine Translation Evaluation through Fine-grained Error Detection", *TACL 2024*. [arXiv:2310.10482](https://arxiv.org/abs/2310.10482).
- **GEMBA (LLM-as-judge for MT):** Tom Kocmi, Christian Federmann, "Large Language Models Are State-of-the-Art Evaluators of Translation Quality", *EAMT 2023*. [arXiv:2302.14520](https://arxiv.org/abs/2302.14520). MQM variant: Tom Kocmi, Christian Federmann, "GEMBA-MQM: Detecting Translation Quality Error Spans with GPT-4", *WMT 2023*. [aclanthology.org/2023.wmt-1.64](https://aclanthology.org/2023.wmt-1.64/).
- **WMT22 Metrics Shared Task ("Stop Using BLEU"):** Markus Freitag et al., "Results of the WMT22 Metrics Shared Task", *WMT 2022*. [aclanthology.org/2022.wmt-1.2](https://aclanthology.org/2022.wmt-1.2/). WMT23: [aclanthology.org/2023.wmt-1.51](https://aclanthology.org/2023.wmt-1.51/).
- **MBR decoding (motivates reference-free metrics at decode time):** Bryan Eikema, Wilker Aziz, "Is MAP Decoding All You Need? The Inadequacy of the Mode in Neural Machine Translation", *COLING 2020*. [arXiv:2005.10283](https://arxiv.org/abs/2005.10283).
- **Summarisation faithfulness metrics overview (SummEval):** Alexander R. Fabbri et al., "SummEval: Re-evaluating Summarization Evaluation", *TACL 2021*. [arXiv:2007.12626](https://arxiv.org/abs/2007.12626). See mod-106 chapters 10–12 for full faithfulness coverage.

## Perplexity and language-model evaluation

- **Perplexity (foundational):** Peter F. Brown et al., "An Estimate of an Upper Bound for the Entropy of English", *Computational Linguistics 1992*. [aclanthology.org/J92-1002](https://aclanthology.org/J92-1002/).
- **WikiText-103:** Stephen Merity, Caiming Xiong, James Bradbury, Richard Socher, "Pointer Sentinel Mixture Models", *ICLR 2017*. [arXiv:1609.07843](https://arxiv.org/abs/1609.07843).
- **Strided-eval PPL (GPT-2):** Alec Radford et al., "Language Models are Unsupervised Multitask Learners", *OpenAI 2019*. <https://openai.com/research/better-language-models>. Hugging Face long-form walkthrough: <https://huggingface.co/docs/transformers/perplexity>.
- **The Pile + Bits-per-Byte convention:** Leo Gao et al., "The Pile: An 800GB Dataset of Diverse Text for Language Modeling", 2020. [arXiv:2101.00027](https://arxiv.org/abs/2101.00027).
- **C4 validation:** Colin Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", *JMLR 2020*. [arXiv:1910.10683](https://arxiv.org/abs/1910.10683).
- **PG-19 (long-context PPL):** Jack W. Rae et al., "Compressive Transformers for Long-Range Sequence Modelling", *ICLR 2020*. [arXiv:1911.05507](https://arxiv.org/abs/1911.05507).
- **LAMBADA (last-word accuracy):** Denis Paperno et al., "The LAMBADA dataset: Word prediction requiring a broad discourse context", *ACL 2016*. [arXiv:1606.06031](https://arxiv.org/abs/1606.06031).
- **Enwik8 / text8 (character-level BPC benchmarks):** Marcus Hutter, "The Hutter Prize" / Matt Mahoney, "Large Text Compression Benchmark". <http://mattmahoney.net/dc/text.html>.

## Benchmark suites

- **GLUE:** Alex Wang et al., "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding", *ICLR 2019*. [arXiv:1804.07461](https://arxiv.org/abs/1804.07461). Leaderboard: <https://gluebenchmark.com/>.
- **SuperGLUE:** Alex Wang et al., "SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems", *NeurIPS 2019*. [arXiv:1905.00537](https://arxiv.org/abs/1905.00537). Leaderboard: <https://super.gluebenchmark.com/>.
- **XTREME:** Junjie Hu et al., "XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalisation", *ICML 2020*. [arXiv:2003.11080](https://arxiv.org/abs/2003.11080).
- **XTREME-R:** Sebastian Ruder et al., "XTREME-R: Towards More Challenging and Nuanced Multilingual Evaluation", *EMNLP 2021*. [arXiv:2104.07412](https://arxiv.org/abs/2104.07412).
- **FLORES-101:** Naman Goyal et al., "The FLORES-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation", *TACL 2022*. [arXiv:2106.03193](https://arxiv.org/abs/2106.03193).
- **FLORES-200 (NLLB release):** NLLB Team et al., "No Language Left Behind: Scaling Human-Centered Machine Translation", 2022. [arXiv:2207.04672](https://arxiv.org/abs/2207.04672). Dataset: <https://huggingface.co/datasets/facebook/flores>.
- **MTEB:** Niklas Muennighoff, Nouamane Tazi, Loïc Magne, Nils Reimers, "MTEB: Massive Text Embedding Benchmark", *EACL 2023*. [arXiv:2210.07316](https://arxiv.org/abs/2210.07316). Leaderboard: <https://huggingface.co/spaces/mteb/leaderboard>.
- **MMTEB (multilingual extension):** Kenneth Enevoldsen et al., "MMTEB: Massive Multilingual Text Embedding Benchmark", 2025. [arXiv:2502.13595](https://arxiv.org/abs/2502.13595).
- **BEIR (zero-shot retrieval):** Nandan Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models", *NeurIPS 2021 D&B*. [arXiv:2104.08663](https://arxiv.org/abs/2104.08663).
- **`lm-evaluation-harness`:** EleutherAI, <https://github.com/EleutherAI/lm-evaluation-harness>. Reference: Leo Gao et al., "A framework for few-shot language model evaluation", 2021 onward, Zenodo DOI <https://zenodo.org/records/10256836>.
- **HELM:** Percy Liang et al., "Holistic Evaluation of Language Models", *2022, ongoing*. [arXiv:2211.09110](https://arxiv.org/abs/2211.09110). Live site: <https://crfm.stanford.edu/helm/>.
- **BIG-Bench:** Aarohi Srivastava et al., "Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models", 2022. [arXiv:2206.04615](https://arxiv.org/abs/2206.04615).
- **BBH (BIG-Bench Hard):** Mirac Suzgun et al., "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them", *ACL 2023 Findings*. [arXiv:2210.09261](https://arxiv.org/abs/2210.09261).
- **MMLU:** Dan Hendrycks et al., "Measuring Massive Multitask Language Understanding", *ICLR 2021*. [arXiv:2009.03300](https://arxiv.org/abs/2009.03300).
- **MMLU-Redux / MMLU-Pro (saturation + contamination response):** Aryo Pradipta Gema et al., "Are We Done with MMLU?", *NAACL 2025 Findings*. [arXiv:2406.04127](https://arxiv.org/abs/2406.04127). Yubo Wang et al., "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark", *NeurIPS 2024 D&B*. [arXiv:2406.01574](https://arxiv.org/abs/2406.01574).
- **MT-Bench and Chatbot Arena:** Lianmin Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", *NeurIPS 2023 D&B*. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685).
- **Arena-Hard:** Tianle Li et al., "From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline", 2024. [arXiv:2406.11939](https://arxiv.org/abs/2406.11939).
- **AlpacaEval:** Xuechen Li et al., "AlpacaEval: An Automatic Evaluator of Instruction-following Models", 2023. <https://github.com/tatsu-lab/alpaca_eval>.
- **ARC:** Peter Clark et al., "Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge", 2018. [arXiv:1803.05457](https://arxiv.org/abs/1803.05457).
- **HellaSwag:** Rowan Zellers et al., "HellaSwag: Can a Machine Really Finish Your Sentence?", *ACL 2019*. [arXiv:1905.07830](https://arxiv.org/abs/1905.07830).
- **TruthfulQA:** Stephanie Lin, Jacob Hilton, Owain Evans, "TruthfulQA: Measuring How Models Mimic Human Falsehoods", *ACL 2022*. [arXiv:2109.07958](https://arxiv.org/abs/2109.07958).
- **GSM8K:** Karl Cobbe et al., "Training Verifiers to Solve Math Word Problems", 2021. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168).
- **HumanEval:** Mark Chen et al., "Evaluating Large Language Models Trained on Code", 2021. [arXiv:2107.03374](https://arxiv.org/abs/2107.03374).
- **WinoGrande:** Keisuke Sakaguchi et al., "WinoGrande: An Adversarial Winograd Schema Challenge at Scale", *AAAI 2020*. [arXiv:1907.10641](https://arxiv.org/abs/1907.10641).
- **PIQA:** Yonatan Bisk et al., "PIQA: Reasoning about Physical Commonsense in Natural Language", *AAAI 2020*. [arXiv:1911.11641](https://arxiv.org/abs/1911.11641).
- **BoolQ:** Christopher Clark et al., "BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions", *NAACL 2019*. [arXiv:1905.10044](https://arxiv.org/abs/1905.10044).

## Statistical significance for NLP

- **Bootstrap (foundational):** Bradley Efron, "Bootstrap Methods: Another Look at the Jackknife", *Annals of Statistics 1979*. Reprinted in Efron & Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall 1993.
- **Paired bootstrap for MT:** Philipp Koehn, "Statistical Significance Tests for Machine Translation Evaluation", *EMNLP 2004*. [aclanthology.org/W04-3250](https://aclanthology.org/W04-3250/).
- **Approximate randomisation (permutation test for NLP):** Alexander Yeh, "More Accurate Tests for the Statistical Significance of Result Differences", *COLING 2000*. [aclanthology.org/C00-2137](https://aclanthology.org/C00-2137/).
- **Pitfalls in MT significance:** Stefan Riezler, John T. Maxwell III, "On Some Pitfalls in Automatic Evaluation and Significance Testing for MT", *ACL 2005 Workshop*. [aclanthology.org/W05-0908](https://aclanthology.org/W05-0908/).
- **McNemar for classifier comparison:** Thomas G. Dietterich, "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms", *Neural Computation 1998*. <https://direct.mit.edu/neco/article/10/7/1895/6224>.
- **Hitchhiker's guide (survey of NLP sig-testing practice):** Rotem Dror, Gili Baumer, Segev Shlomov, Roi Reichart, "The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing", *ACL 2018*. [aclanthology.org/P18-1128](https://aclanthology.org/P18-1128/).
- **Multiple testing in NLP:** Rotem Dror, Gili Baumer, Marina Bogomolov, Roi Reichart, "Replicability Analysis for Natural Language Processing: Testing Significance with Multiple Datasets", *TACL 2017*. [arXiv:1709.09500](https://arxiv.org/abs/1709.09500).
- **Benjamini-Hochberg FDR:** Yoav Benjamini, Yosef Hochberg, "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing", *JRSS-B 1995*.
- **Seed variance in NLP experiments:** Nils Reimers, Iryna Gurevych, "Reporting Score Distributions Makes a Difference: Performance Study of LSTM-networks for Sequence Tagging", *EMNLP 2017*. [aclanthology.org/D17-1035](https://aclanthology.org/D17-1035/).
- **"Show Your Work" (best-of-k and multi-seed reporting):** Jesse Dodge et al., "Show Your Work: Improved Reporting of Experimental Results", *EMNLP 2019*. [arXiv:1909.03004](https://arxiv.org/abs/1909.03004).
- **Equivalence testing (TOST):** Daniel Lakens, "Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses", *Social Psychological and Personality Science 2017*. <https://journals.sagepub.com/doi/10.1177/1948550617697177>.
- **SacreBLEU `--paired-bs` (paired bootstrap CLI):** <https://github.com/mjpost/sacrebleu#significance-testing>.

## Contamination detection and decontamination

- **GPT-3 13-gram contamination check:** Tom B. Brown et al., "Language Models are Few-Shot Learners", *NeurIPS 2020*. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165). See appendix on train/test overlap.
- **PaLM decontamination:** Aakanksha Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways", 2022. [arXiv:2204.02311](https://arxiv.org/abs/2204.02311).
- **LLaMA 1 / 2 decontamination protocols:** Hugo Touvron et al., "LLaMA: Open and Efficient Foundation Language Models", 2023. [arXiv:2302.13971](https://arxiv.org/abs/2302.13971). Hugo Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models", 2023. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288).
- **OLMo decontamination:** Dirk Groeneveld et al., "OLMo: Accelerating the Science of Language Models", *ACL 2024*. [arXiv:2402.00838](https://arxiv.org/abs/2402.00838).
- **BIG-Bench canary strings and decontamination harness:** <https://github.com/google/BIG-bench> and Srivastava et al., [arXiv:2206.04615](https://arxiv.org/abs/2206.04615) (canary discussion).
- **Deduplication across benchmarks (training data):** Katherine Lee et al., "Deduplicating Training Data Makes Language Models Better", *ACL 2022*. [arXiv:2107.06499](https://arxiv.org/abs/2107.06499). Repo: <https://github.com/google-research/deduplicate-text-datasets>.
- **`datasketch` (MinHash / LSH in Python):** <https://github.com/ekzhu/datasketch>.
- **Time Travel in LLMs (guided-instruction probe):** Shahriar Golchin, Mihai Surdeanu, "Time Travel in LLMs: Tracing Data Contamination in Large Language Models", *ICLR 2024*. [arXiv:2308.08493](https://arxiv.org/abs/2308.08493).
- **Min-K% Prob membership inference:** Weijia Shi et al., "Detecting Pretraining Data from Large Language Models", *ICLR 2024*. [arXiv:2310.16789](https://arxiv.org/abs/2310.16789).
- **Data Contamination Through the Lens of Time (substring exact match):** Marc Marone, Benjamin Van Durme, "Data Contamination Through the Lens of Time", 2023. [arXiv:2310.10628](https://arxiv.org/abs/2310.10628).
- **LiveBench (rolling, contamination-resistant benchmark):** Colin White et al., "LiveBench: A Challenging, Contamination-Free LLM Benchmark", *ICLR 2025*. [arXiv:2406.19314](https://arxiv.org/abs/2406.19314).
- **GSM-Symbolic (procedural math reasoning):** Iman Mirzadeh et al., "GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models", *ICLR 2025*. [arXiv:2410.05229](https://arxiv.org/abs/2410.05229).
- **On the Measure of Intelligence (procedural task generators):** François Chollet, "On the Measure of Intelligence", 2019. [arXiv:1911.01547](https://arxiv.org/abs/1911.01547). ARC-AGI dataset: <https://github.com/fchollet/ARC-AGI>.
- **BigScience / BLOOM data preparation (decontamination code):** <https://github.com/bigscience-workshop/data-preparation>.
- **MMLU-Redux (contaminated-benchmark re-release with corrections):** Aryo Pradipta Gema et al., "Are We Done with MMLU?", *NAACL 2025 Findings*. [arXiv:2406.04127](https://arxiv.org/abs/2406.04127).

## Human evaluation protocol design

- **Direct Assessment (continuous 0–100 scale):** Yvette Graham, Timothy Baldwin, Alistair Moffat, Justin Zobel, "Continuous Measurement Scales in Human Evaluation of Machine Translation", *ACL 2013 Linguistic Annotation Workshop*. [aclanthology.org/W13-2305](https://aclanthology.org/W13-2305/).
- **MQM framework:** Arle Lommel, Hans Uszkoreit, Aljoscha Burchardt, "Multidimensional Quality Metrics (MQM): A Framework for Declaring and Describing Translation Quality Metrics", *Tradumàtica 2014*. <https://themqm.info/>.
- **MQM at scale (SQM 0–6, expert raters):** Markus Freitag et al., "Experts, Errors, and Context: A Large-Scale Study of Human Evaluation for Machine Translation", *TACL 2021*. [arXiv:2104.14478](https://arxiv.org/abs/2104.14478).
- **SummEval (summarisation human-eval rubric + benchmark):** Alexander R. Fabbri et al., "SummEval: Re-evaluating Summarization Evaluation", *TACL 2021*. [arXiv:2007.12626](https://arxiv.org/abs/2007.12626).
- **LLM-as-judge (MT-Bench + Chatbot Arena):** Lianmin Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", *NeurIPS 2023 D&B*. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685).
- **LLM-judge bias audit:** Guiming Hardy Chen et al., "Humans or LLMs as the Judge? A Study on Judgement Bias", *EMNLP 2024*. [arXiv:2402.10669](https://arxiv.org/abs/2402.10669).
- **GEMBA (LLM-as-judge for MT, DA and MQM variants):** Tom Kocmi, Christian Federmann, "Large Language Models Are State-of-the-Art Evaluators of Translation Quality", *EAMT 2023*. [arXiv:2302.14520](https://arxiv.org/abs/2302.14520).
- **Krippendorff's alpha:** Klaus Krippendorff, *Content Analysis: An Introduction to Its Methodology*, Sage 2004 (2nd ed.). Short reference: Klaus Krippendorff, "Computing Krippendorff's Alpha-Reliability", 2011. <https://repository.upenn.edu/asc_papers/43/>.
- **Cohen's kappa:** Jacob Cohen, "A Coefficient of Agreement for Nominal Scales", *Educational and Psychological Measurement 1960*.
- **Fleiss' kappa (for ≥ 3 raters):** Joseph L. Fleiss, "Measuring Nominal Scale Agreement among Many Raters", *Psychological Bulletin 1971*.
- **Model Cards (per-slice reporting convention):** Margaret Mitchell et al., "Model Cards for Model Reporting", *FAT\* 2019*. [arXiv:1810.03993](https://arxiv.org/abs/1810.03993).
- **Datasheets for Datasets:** Timnit Gebru et al., "Datasheets for Datasets", *Communications of the ACM 2021*. [arXiv:1803.09010](https://arxiv.org/abs/1803.09010).
- **Dialect bias in NLP (TwitterAAE):** Su Lin Blodgett, Lisa Green, Brendan O'Connor, "Demographic Dialectal Variation in Social Media: A Case Study of African-American English", *EMNLP 2016*. [arXiv:1608.08868](https://arxiv.org/abs/1608.08868).
- **Racial bias in hate-speech detection:** Thomas Davidson, Debasmita Bhattacharya, Ingmar Weber, "Racial Bias in Hate Speech and Abusive Language Detection Datasets", *ACL 2019 Workshop*. [arXiv:1905.12516](https://arxiv.org/abs/1905.12516).
- **HolisticBias (13-axis demographic descriptor dataset):** Eric Michael Smith et al., "I'm sorry to hear that: Finding New Biases in Language Models with a Holistic Descriptor Dataset", *EMNLP 2022*. [arXiv:2205.09209](https://arxiv.org/abs/2205.09209).
- **ANLI / adversarial NLI (crowd-worker protocol as reference):** Yixin Nie et al., "Adversarial NLI: A New Benchmark for Natural Language Understanding", *ACL 2020*. [arXiv:1910.14599](https://arxiv.org/abs/1910.14599).
- **Ethical considerations in crowd rating:** Mary L. Gray, Siddharth Suri, *Ghost Work: How to Stop Silicon Valley from Building a New Global Underclass*, Houghton Mifflin Harcourt 2019 (background reading on crowd-worker labour practices).

## Libraries and reference implementations

- **Hugging Face `evaluate`** — <https://huggingface.co/docs/evaluate/index>
- **Hugging Face `datasets`** — <https://huggingface.co/docs/datasets/index>
- **Hugging Face `transformers`** — <https://huggingface.co/docs/transformers/index>
- **SacreBLEU** — <https://github.com/mjpost/sacrebleu>
- **Unbabel COMET** — <https://github.com/Unbabel/COMET>
- **BLEURT** — <https://github.com/google-research/bleurt>
- **`bert-score`** — <https://github.com/Tiiiger/bert_score>
- **`rouge-score` (Google reference)** — <https://github.com/google-research/google-research/tree/master/rouge>
- **`seqeval`** — <https://github.com/chakki-works/seqeval>
- **`scikit-learn` metrics** — <https://scikit-learn.org/stable/modules/model_evaluation.html>
- **`scipy.stats`** — <https://docs.scipy.org/doc/scipy/reference/stats.html>
- **`statsmodels` (McNemar, multiple-testing)** — <https://www.statsmodels.org/>
- **`lm-evaluation-harness`** — <https://github.com/EleutherAI/lm-evaluation-harness>
- **HELM Classic + HELM Lite** — <https://github.com/stanford-crfm/helm>
- **MTEB** — <https://github.com/embeddings-benchmark/mteb>
- **`datasketch` (MinHash / LSH)** — <https://github.com/ekzhu/datasketch>
- **`deduplicate-text-datasets` (Google)** — <https://github.com/google-research/deduplicate-text-datasets>
- **BIG-bench** — <https://github.com/google/BIG-bench>
- **GEMBA reference prompts** — <https://github.com/MicrosoftTranslator/GEMBA>
- **`krippendorff` (Python, Krippendorff's alpha)** — <https://github.com/pln-fing-udelar/fast-krippendorff>
- **`alpaca_eval`** — <https://github.com/tatsu-lab/alpaca_eval>
- **FastChat / MT-Bench** — <https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge>

## Companion tracks and modules

- **`mod-101` (this track)** — tokenisation and text foundations. Owns the BPE / SentencePiece mechanics that drive the cross-tokeniser perplexity problem in chapter 05.
- **`mod-102` (this track)** — classical NLP. Historical home of n-gram LM perplexity and precision/recall against structured outputs.
- **`mod-103` (this track)** — text classification. Monolingual classification metrics (chapter 02) are applied in depth.
- **`mod-104` (this track)** — sequence labelling and information extraction. `seqeval` and span-level F1 (chapter 02) are the evaluation core for NER / chunking training.
- **`mod-105` (this track)** — question answering and machine reading. SQuAD F1 / EM (chapter 03) is the evaluation side of mod-105's training coverage.
- **`mod-106` (this track)** — summarisation and controlled generation. ROUGE / BERTScore / faithfulness metrics (chapter 04) are detailed per-task in mod-106 chapters 10–12.
- **`mod-107` (this track)** — machine translation and multilingual NLP. SacreBLEU / chrF / COMET / BLEURT and MT human eval (chapter 04 and 09) sit alongside mod-107 chapters 10–11.
- **`mod-108` (this track)** — embeddings and representation learning. MTEB (chapter 06) is the evaluation layer for mod-108's models.
- **`mod-110` (this track)** — NLP data engineering. Deduplication and contamination overlap: the MinHash / LSH stack from mod-110 chapter 02 is the foundation for chapter 08's contamination detection.
- **`mod-112` (this track)** — production NLP pipelines. Production-time quality monitoring consumes reference-free metrics (chapter 04) and extends the significance toolkit (chapter 07) to streaming evaluation.
- **`mod-113` (this track)** — NLP systems design and responsible release. Robustness, fairness, and model-card contamination tables (chapters 08–09) feed the responsible-release checklist.
- **`llm-engineer` track** — `lm-evaluation-harness`, MT-Bench, AlpacaEval, and MMLU variants (chapter 06) are the primary evaluation surface for LLM training work.
- **`rag-engineer` track** — MTEB / BEIR retrieval metrics (chapter 06) are the evaluation layer for retriever and reranker tuning.
