# Resources for mod-107 · Machine Translation & Multilingual NLP

Prefer primary sources: papers, standards, and official documentation. Blog posts and library docs are included only where they are the canonical reference for a tool or idea.

## Encoder-decoder NMT architectures

- **Transformer:** Ashish Vaswani et al., "Attention Is All You Need", *NeurIPS 2017*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
- **Seq2seq origins:** Ilya Sutskever, Oriol Vinyals, Quoc V. Le, "Sequence to Sequence Learning with Neural Networks", *NeurIPS 2014*. [arXiv:1409.3215](https://arxiv.org/abs/1409.3215).
- **Attention:** Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate", *ICLR 2015*. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473).
- **Marian NMT (framework + OPUS-MT):** Marcin Junczys-Dowmunt et al., "Marian: Fast Neural Machine Translation in C++", *ACL 2018 demo*. [aclanthology.org/P18-4020](https://aclanthology.org/P18-4020/). OPUS-MT model index: <https://huggingface.co/Helsinki-NLP>. OPUS-MT dashboard: <https://opus.nlpl.eu/dashboard/>.
- **mBART:** Yinhan Liu et al., "Multilingual Denoising Pre-training for Neural Machine Translation", *TACL 2020*. [arXiv:2001.08210](https://arxiv.org/abs/2001.08210).
- **mBART-50:** Yuqing Tang et al., "Multilingual Translation with Extensible Multilingual Pretraining and Finetuning", 2020. [arXiv:2008.00401](https://arxiv.org/abs/2008.00401).
- **M2M-100:** Angela Fan et al., "Beyond English-Centric Multilingual Machine Translation", *JMLR 2021*. [arXiv:2010.11125](https://arxiv.org/abs/2010.11125).
- **NLLB-200:** NLLB Team et al., "No Language Left Behind: Scaling Human-Centered Machine Translation", 2022. [arXiv:2207.04672](https://arxiv.org/abs/2207.04672). Model card: [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M), [`facebook/nllb-200-1.3B`](https://huggingface.co/facebook/nllb-200-1.3B), [`facebook/nllb-200-3.3B`](https://huggingface.co/facebook/nllb-200-3.3B).
- **Google's NMT (historical):** Yonghui Wu et al., "Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation", 2016. [arXiv:1609.08144](https://arxiv.org/abs/1609.08144).
- **Attention-free NMT — RETVec / character-level:** Colin Cherry et al., "Revisiting Character-Based Neural Machine Translation with Capacity and Compression", *EMNLP 2018*. [arXiv:1808.09943](https://arxiv.org/abs/1808.09943).

## Tokenisation for multilingual models

- **SentencePiece:** Taku Kudo, John Richardson, "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing", *EMNLP 2018 demo*. [arXiv:1808.06226](https://arxiv.org/abs/1808.06226). Repo: <https://github.com/google/sentencepiece>.
- **BPE (subword units):** Rico Sennrich, Barry Haddow, Alexandra Birch, "Neural Machine Translation of Rare Words with Subword Units", *ACL 2016*. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909).
- **Unigram LM tokenisation:** Taku Kudo, "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates", *ACL 2018*. [arXiv:1804.10959](https://arxiv.org/abs/1804.10959).
- **BPE-dropout:** Ivan Provilkov, Dmitrii Emelianenko, Elena Voita, "BPE-Dropout: Simple and Effective Subword Regularization", *ACL 2020*. [arXiv:1910.13267](https://arxiv.org/abs/1910.13267).
- **Multilingual vocabulary bottleneck:** Davis Liang et al., "XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models", 2023. [arXiv:2301.10472](https://arxiv.org/abs/2301.10472).

## Multilingual encoders and cross-lingual transfer

- **XLM:** Guillaume Lample, Alexis Conneau, "Cross-lingual Language Model Pretraining", *NeurIPS 2019*. [arXiv:1901.07291](https://arxiv.org/abs/1901.07291).
- **XLM-RoBERTa:** Alexis Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale", *ACL 2020*. [arXiv:1911.02116](https://arxiv.org/abs/1911.02116). Models: [`FacebookAI/xlm-roberta-base`](https://huggingface.co/FacebookAI/xlm-roberta-base), [`FacebookAI/xlm-roberta-large`](https://huggingface.co/FacebookAI/xlm-roberta-large).
- **mDeBERTa-v3:** Pengcheng He et al., "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing", *ICLR 2023*. [arXiv:2111.09543](https://arxiv.org/abs/2111.09543). Model: [`microsoft/mdeberta-v3-base`](https://huggingface.co/microsoft/mdeberta-v3-base).
- **mT5:** Linting Xue et al., "mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer", *NAACL 2021*. [arXiv:2010.11934](https://arxiv.org/abs/2010.11934).
- **ByT5:** Linting Xue et al., "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models", *TACL 2022*. [arXiv:2105.13626](https://arxiv.org/abs/2105.13626).
- **LaBSE:** Fangxiaoyu Feng et al., "Language-agnostic BERT Sentence Embedding", *ACL 2022*. [arXiv:2007.01852](https://arxiv.org/abs/2007.01852).
- **Adapter-based transfer — MAD-X:** Jonas Pfeiffer et al., "MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer", *EMNLP 2020*. [arXiv:2005.00052](https://arxiv.org/abs/2005.00052).
- **Adversarial language-invariant training:** Xilun Chen et al., "Adversarial Deep Averaging Networks for Cross-Lingual Sentiment Classification", *TACL 2018*. [arXiv:1606.01614](https://arxiv.org/abs/1606.01614).
- **Multilingual RoBERTa scaling:** Naman Goyal et al., "Larger-Scale Transformers for Multilingual Masked Language Modeling", *RepL4NLP 2021*. [arXiv:2105.00572](https://arxiv.org/abs/2105.00572).

## Fine-tuning NMT

- **Regularisation for NMT:** Marcin Junczys-Dowmunt, Roman Grundkiewicz, "The Alexa Prize Socialbot Grand Challenge and NMT recipes" — see also Ott et al., "Scaling Neural Machine Translation", *WMT 2018*. [arXiv:1806.00187](https://arxiv.org/abs/1806.00187).
- **Label smoothing:** Christian Szegedy et al., "Rethinking the Inception Architecture for Computer Vision", *CVPR 2016*. [arXiv:1512.00567](https://arxiv.org/abs/1512.00567). Sec 7.
- **LoRA:** Edward J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", *ICLR 2022*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685). Library: [`peft`](https://github.com/huggingface/peft).
- **Adapter modules for MT:** Ankur Bapna, Orhan Firat, "Simple, Scalable Adaptation for Neural Machine Translation", *EMNLP 2019*. [arXiv:1909.08478](https://arxiv.org/abs/1909.08478).
- **Domain adaptation for NMT (survey):** Chenhui Chu, Rui Wang, "A Survey of Domain Adaptation for Neural Machine Translation", *COLING 2018*. [arXiv:1806.00258](https://arxiv.org/abs/1806.00258).

## Low-resource and unsupervised NMT

- **Back-translation (NMT formalisation):** Rico Sennrich, Barry Haddow, Alexandra Birch, "Improving Neural Machine Translation Models with Monolingual Data", *ACL 2016*. [arXiv:1511.06709](https://arxiv.org/abs/1511.06709).
- **Sampled vs. beam back-translation at scale:** Sergey Edunov, Myle Ott, Michael Auli, David Grangier, "Understanding Back-Translation at Scale", *EMNLP 2018*. [arXiv:1808.09381](https://arxiv.org/abs/1808.09381).
- **Tagged back-translation:** Isaac Caswell, Ciprian Chelba, David Grangier, "Tagged Back-Translation", *WMT 2019*. [arXiv:1906.06442](https://arxiv.org/abs/1906.06442).
- **Iterative back-translation for unsupervised NMT:** Guillaume Lample et al., "Phrase-Based & Neural Unsupervised Machine Translation", *EMNLP 2018*. [arXiv:1804.07755](https://arxiv.org/abs/1804.07755).
- **Unsupervised NMT (encoder-decoder):** Mikel Artetxe et al., "Unsupervised Neural Machine Translation", *ICLR 2018*. [arXiv:1710.11041](https://arxiv.org/abs/1710.11041). Also Guillaume Lample et al., "Unsupervised Machine Translation Using Monolingual Corpora Only", *ICLR 2018*. [arXiv:1711.00043](https://arxiv.org/abs/1711.00043).
- **Trivial transfer learning:** Tom Kocmi, Ondřej Bojar, "Trivial Transfer Learning for Low-Resource Neural Machine Translation", *WMT 2018*. [arXiv:1809.00357](https://arxiv.org/abs/1809.00357).
- **Rapid adaptation to new languages:** Graham Neubig, Junjie Hu, "Rapid Adaptation of Neural Machine Translation to New Languages", *EMNLP 2018*. [arXiv:1808.04189](https://arxiv.org/abs/1808.04189).
- **Massively multilingual training with temperature sampling:** Naveen Arivazhagan et al., "Massively Multilingual Neural Machine Translation in the Wild: Findings and Challenges", 2019. [arXiv:1907.05019](https://arxiv.org/abs/1907.05019).
- **Bicleaner AI (parallel-corpus quality classifier):** Jaume Zaragoza-Bernabeu, Gema Ramírez-Sánchez, Marta Bañón, Sergio Ortiz Rojas, "Bicleaner AI: Bicleaner Goes Neural", *LREC 2022*. Repo: <https://github.com/bitextor/bicleaner-ai>.
- **Vecalign (sentence alignment):** Brian Thompson, Philipp Koehn, "Vecalign: Improved Sentence Alignment in Linear Time and Space", *EMNLP 2019*. [aclanthology.org/D19-1136](https://aclanthology.org/D19-1136/).

## Terminology-constrained decoding

- **Grid Beam Search:** Chris Hokamp, Qun Liu, "Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search", *ACL 2017*. [arXiv:1704.07138](https://arxiv.org/abs/1704.07138).
- **Dynamic Beam Allocation:** Matt Post, David Vilar, "Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation", *NAACL 2018*. [arXiv:1804.06609](https://arxiv.org/abs/1804.06609).
- **DiCE (Data augmentation for terminology):** Georgiana Dinu, Prashant Mathur, Marcello Federico, Yaser Al-Onaizan, "Training Neural Machine Translation to Apply Terminology Constraints", *ACL 2019*. [arXiv:1906.01105](https://arxiv.org/abs/1906.01105).
- **Personalised NMT (extreme adaptation):** Paul Michel, Graham Neubig, "Extreme Adaptation for Personalized Neural Machine Translation", *ACL 2018*. [arXiv:1805.01817](https://arxiv.org/abs/1805.01817).
- **WMT Terminology Task (2023):** <https://www2.statmt.org/wmt23/terminology-task.html>.
- **`PrefixConstrainedLogitsProcessor`, `PhrasalConstraint`, `DisjunctiveConstraint` (Hugging Face):** <https://huggingface.co/docs/transformers/generation_strategies#constrained-beam-search>.

## Domain adaptation

- **Continued pre-training / domain adaptation:** Chenhui Chu, Raj Dabre, Sadao Kurohashi, "An Empirical Comparison of Domain Adaptation Methods for Neural Machine Translation", *ACL 2017 short*. [aclanthology.org/P17-2061](https://aclanthology.org/P17-2061/).
- **Fine-tuning + regularisation (elastic weight consolidation for MT):** Barret Zoph et al., "Transfer Learning for Low-Resource Neural Machine Translation", *EMNLP 2016*. [aclanthology.org/D16-1163](https://aclanthology.org/D16-1163/).
- **Data selection (Moore-Lewis):** Robert Moore, William Lewis, "Intelligent Selection of Language Model Training Data", *ACL 2010*. [aclanthology.org/P10-2041](https://aclanthology.org/P10-2041/).

## Scripts, transliteration, and locale

- **Unicode Standard Annex #15 (Normalisation):** <https://unicode.org/reports/tr15/>.
- **UAX #9 (Bidirectional Algorithm):** <https://unicode.org/reports/tr9/>.
- **BCP-47 (language tags):** IETF RFC 5646. <https://datatracker.ietf.org/doc/html/rfc5646>.
- **ISO 639-3 (languages) and ISO 15924 (scripts):** <https://iso639-3.sil.org/> and <https://unicode.org/iso15924/>.
- **`langcodes` (Python):** <https://github.com/rspeer/langcodes>.
- **`opencc` (Traditional ↔ Simplified Chinese):** <https://github.com/BYVoid/OpenCC>.
- **`fugashi` / `mecab-python3` (Japanese morphology):** <https://github.com/polm/fugashi>.
- **`indic-nlp-library` (Indic scripts + transliteration):** <https://github.com/anoopkunchukuttan/indic_nlp_library>.
- **`icu4x` / PyICU (Unicode + locale services):** <https://pypi.org/project/PyICU/>.
- **Anthony Aristar et al., "Language Tags in HTML and XML" (BCP-47 practical guide, W3C):** <https://www.w3.org/International/articles/language-tags/>.
- **Uroman (universal romanisation):** Ulf Hermjakob, Jonathan May, Kevin Knight, "Out-of-the-box Universal Romanization Tool uroman", *ACL 2018 demo*. [aclanthology.org/P18-4003](https://aclanthology.org/P18-4003/).

## Automatic MT evaluation

- **BLEU:** Kishore Papineni, Salim Roukos, Todd Ward, Wei-Jing Zhu, "BLEU: A Method for Automatic Evaluation of Machine Translation", *ACL 2002*. [aclanthology.org/P02-1040](https://aclanthology.org/P02-1040/).
- **SacreBLEU:** Matt Post, "A Call for Clarity in Reporting BLEU Scores", *WMT 2018*. [arXiv:1804.08771](https://arxiv.org/abs/1804.08771). Repo: <https://github.com/mjpost/sacrebleu>.
- **chrF:** Maja Popović, "chrF: character n-gram F-score for automatic MT evaluation", *WMT 2015*. [aclanthology.org/W15-3049](https://aclanthology.org/W15-3049/).
- **chrF++:** Maja Popović, "chrF++: words helping character n-grams", *WMT 2017*. [aclanthology.org/W17-4770](https://aclanthology.org/W17-4770/).
- **METEOR:** Satanjeev Banerjee, Alon Lavie, "METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments", *ACL 2005 Workshop*. [aclanthology.org/W05-0909](https://aclanthology.org/W05-0909/).
- **TER:** Matthew Snover et al., "A Study of Translation Edit Rate with Targeted Human Annotation", *AMTA 2006*. [aclanthology.org/2006.amta-papers.25](https://aclanthology.org/2006.amta-papers.25/).
- **BLEURT:** Thibault Sellam, Dipanjan Das, Ankur P. Parikh, "BLEURT: Learning Robust Metrics for Text Generation", *ACL 2020*. [arXiv:2004.04696](https://arxiv.org/abs/2004.04696). BLEURT-20: Amy Pu et al., "Learning Compact Metrics for MT", *EMNLP 2021*. [arXiv:2110.06341](https://arxiv.org/abs/2110.06341).
- **COMET:** Ricardo Rei et al., "COMET: A Neural Framework for MT Evaluation", *EMNLP 2020*. [arXiv:2009.09025](https://arxiv.org/abs/2009.09025). Repo: <https://github.com/Unbabel/COMET>.
- **COMET-Kiwi (reference-free):** Ricardo Rei et al., "CometKiwi: IST-Unbabel 2022 Submission for the Quality Estimation Shared Task", *WMT 2022*. [aclanthology.org/2022.wmt-1.60](https://aclanthology.org/2022.wmt-1.60/).
- **XCOMET (span-level):** Nuno Guerreiro et al., "xCOMET: Transparent Machine Translation Evaluation through Fine-grained Error Detection", *TACL 2024*. [arXiv:2310.10482](https://arxiv.org/abs/2310.10482).
- **YiSi / BERTScore for MT:** Chi-kiu Lo, "YiSi — a Unified Semantic MT Quality Evaluation and Estimation Metric", *WMT 2019*. [aclanthology.org/W19-5358](https://aclanthology.org/W19-5358/); Tianyi Zhang et al., "BERTScore: Evaluating Text Generation with BERT", *ICLR 2020*. [arXiv:1904.09675](https://arxiv.org/abs/1904.09675).
- **MBR decoding:** Bryan Eikema, Wilker Aziz, "Is MAP Decoding All You Need? The Inadequacy of the Mode in Neural Machine Translation", *COLING 2020*. [arXiv:2005.10283](https://arxiv.org/abs/2005.10283).
- **GEMBA (LLM-as-judge for MT):** Tom Kocmi, Christian Federmann, "Large Language Models Are State-of-the-Art Evaluators of Translation Quality", *EAMT 2023*. [arXiv:2302.14520](https://arxiv.org/abs/2302.14520).
- **Statistical significance (paired bootstrap):** Philipp Koehn, "Statistical Significance Tests for Machine Translation Evaluation", *EMNLP 2004*. [aclanthology.org/W04-3250](https://aclanthology.org/W04-3250/).
- **WMT Metrics Shared Task (annual):** e.g. Markus Freitag et al., "Results of the WMT22 Metrics Shared Task: Stop Using BLEU – Neural Metrics Are Better and More Robust", *WMT 2022*. [aclanthology.org/2022.wmt-1.2](https://aclanthology.org/2022.wmt-1.2/).

## Human evaluation for MT

- **Direct Assessment:** Yvette Graham, Timothy Baldwin, Alistair Moffat, Justin Zobel, "Continuous Measurement Scales in Human Evaluation of Machine Translation", *ACL 2013 Linguistic Annotation Workshop*. [aclanthology.org/W13-2305](https://aclanthology.org/W13-2305/).
- **MQM framework:** Arle Lommel, Hans Uszkoreit, Aljoscha Burchardt, "Multidimensional Quality Metrics (MQM): A Framework for Declaring and Describing Translation Quality Metrics", *Tradumàtica 2014*. <https://themqm.info/>.
- **Experts, errors, and context (MQM at scale):** Markus Freitag et al., "Experts, Errors, and Context: A Large-Scale Study of Human Evaluation for Machine Translation", *TACL 2021*. [arXiv:2104.14478](https://arxiv.org/abs/2104.14478).
- **SQM (0–6 scalar):** ibid. Freitag et al., 2021.
- **Post-editing effort / TER-based:** Lucia Specia, Marco Turchi, Nicola Cancedda, Marc Dymetman, Nello Cristianini, "Estimating the Sentence-Level Quality of Machine Translation Systems", *EAMT 2009*.
- **Inter-annotator agreement:** Klaus Krippendorff, "Content Analysis: An Introduction to Its Methodology", Sage 2004 (Krippendorff's alpha). Jacob Cohen, "A Coefficient of Agreement for Nominal Scales", *Educational and Psychological Measurement 1960*.

## Benchmarks and evaluation sets

### Translation

- **FLORES-101:** Naman Goyal et al., "The FLORES-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation", *TACL 2022*. [arXiv:2106.03193](https://arxiv.org/abs/2106.03193).
- **FLORES-200:** part of the NLLB release; see NLLB paper [arXiv:2207.04672](https://arxiv.org/abs/2207.04672). Dataset card: <https://huggingface.co/datasets/facebook/flores>.
- **NTREX-128:** Christian Federmann et al., "NTREX-128 – News Test References for MT Evaluation of 128 Languages", *SUMEVAL 2022*. [aclanthology.org/2022.sumeval-1.4](https://aclanthology.org/2022.sumeval-1.4/).
- **TICO-19:** Antonios Anastasopoulos et al., "TICO-19: the Translation Initiative for COvid-19", 2020. [arXiv:2007.01788](https://arxiv.org/abs/2007.01788).
- **WMT conference proceedings:** <https://www2.statmt.org/> (news test sets, biomedical, chat, terminology sub-tracks, metrics shared task).
- **OPUS corpora:** Jörg Tiedemann, "Parallel Data, Tools and Interfaces in OPUS", *LREC 2012*. <https://opus.nlpl.eu/>.
- **CCMatrix:** Angela Fan et al., "CCMatrix: Mining Billions of High-Quality Parallel Sentences on the Web", *ACL 2021*. [arXiv:1911.04944](https://arxiv.org/abs/1911.04944).
- **CCAligned:** Ahmed El-Kishky et al., "CCAligned: A Massive Collection of Cross-Lingual Web-Document Pairs", *EMNLP 2020*. [aclanthology.org/2020.emnlp-main.480](https://aclanthology.org/2020.emnlp-main.480/).
- **ParaCrawl:** Marta Bañón et al., "ParaCrawl: Web-Scale Acquisition of Parallel Corpora", *ACL 2020*. [aclanthology.org/2020.acl-main.417](https://aclanthology.org/2020.acl-main.417/).
- **Samanantar (Indic):** Gowtham Ramesh et al., "Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages", *TACL 2022*. [arXiv:2104.05596](https://arxiv.org/abs/2104.05596).
- **MAFAND-MT (African):** David I. Adelani et al., "A Few Thousand Translations Go a Long Way! Leveraging Pre-trained Models for African News Translation", *NAACL 2022*. [aclanthology.org/2022.naacl-main.223](https://aclanthology.org/2022.naacl-main.223/).

### Cross-lingual understanding

- **XNLI:** Alexis Conneau et al., "XNLI: Evaluating Cross-lingual Sentence Representations", *EMNLP 2018*. [arXiv:1809.05053](https://arxiv.org/abs/1809.05053).
- **XTREME:** Junjie Hu et al., "XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalisation", *ICML 2020*. [arXiv:2003.11080](https://arxiv.org/abs/2003.11080).
- **XTREME-R:** Sebastian Ruder et al., "XTREME-R: Towards More Challenging and Nuanced Multilingual Evaluation", *EMNLP 2021*. [arXiv:2104.07412](https://arxiv.org/abs/2104.07412).
- **TyDiQA:** Jonathan H. Clark et al., "TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages", *TACL 2020*. [arXiv:2003.05002](https://arxiv.org/abs/2003.05002).
- **MLQA:** Patrick Lewis et al., "MLQA: Evaluating Cross-lingual Extractive Question Answering", *ACL 2020*. [arXiv:1910.07475](https://arxiv.org/abs/1910.07475).
- **XQuAD:** Mikel Artetxe, Sebastian Ruder, Dani Yogatama, "On the Cross-lingual Transferability of Monolingual Representations", *ACL 2020*. [arXiv:1910.11856](https://arxiv.org/abs/1910.11856).
- **WikiANN / PAN-X:** Xiaoman Pan et al., "Cross-lingual Name Tagging and Linking for 282 Languages", *ACL 2017*. [aclanthology.org/P17-1178](https://aclanthology.org/P17-1178/).
- **MasakhaNER 1.0 and 2.0:** David I. Adelani et al., "MasakhaNER: Named Entity Recognition for African Languages", *TACL 2021*. [arXiv:2103.11811](https://arxiv.org/abs/2103.11811). "MasakhaNER 2.0", *EMNLP 2022*. [aclanthology.org/2022.emnlp-main.298](https://aclanthology.org/2022.emnlp-main.298/).
- **MLSum:** Thomas Scialom et al., "MLSUM: The Multilingual Summarization Corpus", *EMNLP 2020*. [arXiv:2004.14900](https://arxiv.org/abs/2004.14900).
- **XL-Sum:** Tahmid Hasan et al., "XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages", *ACL 2021 Findings*. [arXiv:2106.13822](https://arxiv.org/abs/2106.13822).
- **XCOPA:** Edoardo M. Ponti et al., "XCOPA: A Multilingual Dataset for Causal Commonsense Reasoning", *EMNLP 2020*. [arXiv:2005.00333](https://arxiv.org/abs/2005.00333).
- **AfriSenti:** Shamsuddeen Hassan Muhammad et al., "AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages", *EMNLP 2023*. [arXiv:2302.08956](https://arxiv.org/abs/2302.08956).
- **IndicNLP Suite:** Divyanshu Kakwani et al., "IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages", *EMNLP 2020 Findings*. [aclanthology.org/2020.findings-emnlp.445](https://aclanthology.org/2020.findings-emnlp.445/).
- **SEACrowd:** Holy Lovenia et al., "SEACrowd: A Multilingual Multimodal Data Hub and Benchmark Suite for Southeast Asian Languages", *EMNLP 2024*. [aclanthology.org/2024.emnlp-main.296](https://aclanthology.org/2024.emnlp-main.296/).
- **C-Eval:** Yuzhen Huang et al., "C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models", *NeurIPS 2023*. [arXiv:2305.08322](https://arxiv.org/abs/2305.08322).
- **JGLUE:** Kentaro Kurihara et al., "JGLUE: Japanese General Language Understanding Evaluation", *LREC 2022*. [aclanthology.org/2022.lrec-1.317](https://aclanthology.org/2022.lrec-1.317/).
- **KLUE:** Sungjoon Park et al., "KLUE: Korean Language Understanding Evaluation", *NeurIPS 2021 D&B*. [arXiv:2105.09680](https://arxiv.org/abs/2105.09680).

## Alignment and mining tools

- **LASER (multilingual sentence embeddings for mining):** Mikel Artetxe, Holger Schwenk, "Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond", *TACL 2019*. [arXiv:1812.10464](https://arxiv.org/abs/1812.10464). Repo: <https://github.com/facebookresearch/LASER>.
- **`fast_align`:** Chris Dyer, Victor Chahuneau, Noah A. Smith, "A Simple, Fast, and Effective Reparameterization of IBM Model 2", *NAACL 2013*. [aclanthology.org/N13-1073](https://aclanthology.org/N13-1073/). Repo: <https://github.com/clab/fast_align>.
- **`awesome-align`:** Zi-Yi Dou, Graham Neubig, "Word Alignment by Fine-tuning Embeddings on Parallel Corpora", *EACL 2021*. [arXiv:2101.08231](https://arxiv.org/abs/2101.08231).
- **`simalign`:** Masoud Jalili Sabet et al., "SimAlign: High Quality Word Alignments Without Parallel Training Data Using Static and Contextualized Embeddings", *EMNLP 2020 Findings*. [arXiv:2004.08728](https://arxiv.org/abs/2004.08728).
- **`hunalign`:** Dániel Varga et al., "Parallel corpora for medium density languages", *Recent Advances in Natural Language Processing 2005*.

## Libraries and reference implementations

- **Hugging Face `transformers`** — <https://huggingface.co/docs/transformers/index>
- **Hugging Face `run_translation.py` example** — <https://github.com/huggingface/transformers/tree/main/examples/pytorch/translation>
- **Hugging Face `datasets`** — <https://huggingface.co/docs/datasets/index>
- **Hugging Face `evaluate`** — <https://huggingface.co/docs/evaluate/index>
- **Unbabel COMET** — <https://github.com/Unbabel/COMET>
- **SacreBLEU** — <https://github.com/mjpost/sacrebleu>
- **BLEURT** — <https://github.com/google-research/bleurt>
- **`peft` (LoRA and adapters)** — <https://github.com/huggingface/peft>
- **`adapters` (successor of adapter-transformers)** — <https://github.com/adapter-hub/adapters>
- **Marian NMT (C++ training framework)** — <https://marian-nmt.github.io/>
- **fairseq (Meta's seq2seq training framework, NLLB reference impl)** — <https://github.com/facebookresearch/fairseq>
- **OpenNMT** — <https://opennmt.net/>
- **Bicleaner AI** — <https://github.com/bitextor/bicleaner-ai>
- **OpenCC (Simplified ↔ Traditional Chinese)** — <https://github.com/BYVoid/OpenCC>
- **`fugashi` / MeCab (Japanese tokenisation)** — <https://github.com/polm/fugashi>
- **`indic-nlp-library`** — <https://github.com/anoopkunchukuttan/indic_nlp_library>
- **`langcodes` (BCP-47 handling)** — <https://github.com/rspeer/langcodes>
- **`sentencepiece`** — <https://github.com/google/sentencepiece>
- **CAMeL Tools (Arabic NLP)** — <https://github.com/CAMeL-Lab/camel_tools>

## Companion tracks

- **`mod-101` (this track)** — subword tokenisation and text foundations. Owns SentencePiece / BPE at the level of detail this module builds on.
- **`mod-103` (this track)** — text classification. The monolingual counterpart to the multilingual-classification arm of chapters 07–08.
- **`mod-104` (this track)** — sequence labelling and information extraction. Where the subword-alignment mechanics for NER zero-shot come from.
- **`mod-106` (this track)** — summarisation and controlled generation. Shares the encoder-decoder recipe and the constrained-decoding techniques generalised for translation in chapter 06.
- **`mod-108` (this track)** — embeddings and representation learning. LaBSE / multilingual sentence encoders overlap with chapter 07's retrieval defaults.
- **`mod-111` (this track)** — NLP evaluation methodology. Statistical significance, annotator agreement, and dataset governance beyond MT.
- **`mod-112` (this track)** — production NLP pipelines. Deployment concerns for MT and multilingual classifiers.
- **`rag-engineer` track** — cross-lingual retrieval; LaBSE and multilingual dense retrievers as ingest-time components.
- **`speech-engineer` track** — MT sits downstream of ASR in the speech-translation pipeline; upstream from TTS in speech-to-speech.
