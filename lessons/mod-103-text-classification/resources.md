# Resources for mod-103 · Text Classification End-to-End

Prefer primary sources: papers, standards, and official documentation. Blog posts and tutorials are included only where they are the canonical or most-referenced explanation.

## Textbooks and canonical references

- **Jurafsky & Martin, *Speech and Language Processing*, 3rd ed. (in progress)** — <https://web.stanford.edu/~jurafsky/slp3/>. Chapters 4 (Naive Bayes / Sentiment Classification), 5 (Logistic Regression), 6 (Vector semantics) provide the classical grounding for this module.
- **Manning, Raghavan & Schütze, *Introduction to Information Retrieval*, Cambridge University Press, 2008** — <https://nlp.stanford.edu/IR-book/>. Chapters 6 (Scoring, term weighting, vector-space model) and 13 (Text classification and Naive Bayes) for TF-IDF and vector-space fundamentals.
- **Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, 2nd ed., Springer, 2009** — <https://hastie.su.domains/ElemStatLearn/>. Reference for logistic regression, SVMs, ensembles, calibration.

## Linear baselines

- **Joachims, ["Text Categorization with Support Vector Machines: Learning with Many Relevant Features"](https://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf), *ECML 1998*.** The paper that established linear SVMs as the strong text-classification baseline.
- **Weinberger, Dasgupta, Langford, Smola & Attenberg, ["Feature Hashing for Large Scale Multitask Learning"](https://arxiv.org/abs/0902.2206), *ICML 2009*.** Motivation for `HashingVectorizer`.
- **`scikit-learn` text feature extraction** — <https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction>.
- **`scikit-learn` linear models** — <https://scikit-learn.org/stable/modules/linear_model.html>.

## fastText

- **Bojanowski, Grave, Joulin & Mikolov, ["Enriching Word Vectors with Subword Information"](https://arxiv.org/abs/1607.04606), *TACL 2017*.** The subword embedding side of fastText.
- **Joulin, Grave, Bojanowski & Mikolov, ["Bag of Tricks for Efficient Text Classification"](https://arxiv.org/abs/1607.01759), *EACL 2017*.** The classifier side.
- **Joulin, Grave, Bojanowski, Douze, Jégou & Mikolov, ["FastText.zip: Compressing text classification models"](https://arxiv.org/abs/1612.03651), *arXiv 2016*.** Quantisation and pruning for deployment.
- **fastText project site** — <https://fasttext.cc/>.
- **fastText source** — <https://github.com/facebookresearch/fastText>.
- **Morin & Bengio, ["Hierarchical Probabilistic Neural Network Language Model"](https://proceedings.mlr.press/r5/morin05a.html), *AISTATS 2005*.** Origin of hierarchical softmax.
- **Jégou, Douze & Schmid, ["Product Quantization for Nearest Neighbor Search"](https://ieeexplore.ieee.org/document/5432202), *TPAMI 2011*.** The quantisation method fastText uses.

## Encoder architectures and fine-tuning

- **Devlin, Chang, Lee & Toutanova, ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805), *NAACL 2019*.**
- **Liu, Ott, Goyal, Du, Joshi, Chen, Levy, Lewis, Zettlemoyer & Stoyanov, ["RoBERTa: A Robustly Optimized BERT Pretraining Approach"](https://arxiv.org/abs/1907.11692), *arXiv 2019*.**
- **He, Liu, Gao & Chen, ["DeBERTa: Decoding-enhanced BERT with Disentangled Attention"](https://arxiv.org/abs/2006.03654), *ICLR 2021*.**
- **He, Gao & Chen, ["DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing"](https://arxiv.org/abs/2111.09543), *ICLR 2023*.**
- **Sanh, Debut, Chaumond & Wolf, ["DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"](https://arxiv.org/abs/1910.01108), *NeurIPS Workshop 2019*.**
- **Sun, Qiu, Xu & Huang, ["How to Fine-Tune BERT for Text Classification?"](https://arxiv.org/abs/1905.05583), *CCL 2019*.** Pooling, layer-wise LR decay, further pretraining tricks.
- **Mosbach, Andriushchenko & Klakow, ["On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines"](https://arxiv.org/abs/2006.04884), *ICLR 2021*.** Seed variance and stability recipes.
- **Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), *NeurIPS 2017*.** The transformer architecture.
- **Beltagy, Peters & Cohan, ["Longformer: The Long-Document Transformer"](https://arxiv.org/abs/2004.05150), *arXiv 2020*.** Long-context option for long documents.

## Multilingual encoders

- **Conneau, Khandelwal, Goyal, Chaudhary, Wenzek, Guzmán, Grave, Ott, Zettlemoyer & Stoyanov, ["Unsupervised Cross-lingual Representation Learning at Scale"](https://arxiv.org/abs/1911.02116), *ACL 2020*.** XLM-R.
- **Wenzek et al., ["CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data"](https://arxiv.org/abs/1911.00359), *LREC 2020*.** The CC-100 corpus.
- **Liang et al., ["XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models"](https://arxiv.org/abs/2301.10472), *NAACL 2023*.**
- **Chung, Garrette, Tan & Riesa, ["Improving Multilingual Models with Language-Clustered Vocabularies"](https://arxiv.org/abs/2010.12777), *EMNLP 2020*.** Curse of multilinguality and cluster-level vocabularies.
- **Conneau & Lample, ["Cross-lingual Language Model Pretraining"](https://arxiv.org/abs/1901.07291), *NeurIPS 2019*.** XLM.

## Multi-label and hierarchical classification

- **Read, Pfahringer, Holmes & Frank, ["Classifier Chains for Multi-label Classification"](https://link.springer.com/chapter/10.1007/978-3-642-04174-7_17), *ECML-PKDD 2009*.**
- **Xiao, Huang, Chen & Jing, ["Label-Specific Document Representation for Multi-Label Text Classification"](https://aclanthology.org/D19-1044/), *EMNLP-IJCNLP 2019*.**
- **Silla & Freitas, ["A Survey of Hierarchical Classification across Different Application Domains"](https://link.springer.com/article/10.1007/s10618-010-0175-9), *Data Mining and Knowledge Discovery*, 2011.** Canonical taxonomy of flat / local / global approaches.
- **Kiritchenko, Matwin, Nock & Famili, ["Learning and Evaluation in the Presence of Class Hierarchies: Application to Text Categorization"](https://link.springer.com/chapter/10.1007/11766247_34), *AI 2006*.** Hierarchical F1 construction.
- **Kosmopoulos, Partalas, Gaussier, Paliouras & Androutsopoulos, ["Evaluation Measures for Hierarchical Classification: a unified view and novel approaches"](https://arxiv.org/abs/1306.6802), *Data Mining and Knowledge Discovery 2015*.** Unified evaluation framework with Python code.
- **Zhou, Ma, Long, Xu, Ding, Zhang, Xie & Liu, ["Hierarchy-Aware Global Model for Hierarchical Text Classification"](https://aclanthology.org/2020.acl-main.104/), *ACL 2020*.** HiAGM.
- **Frank & Hall, ["A Simple Approach to Ordinal Classification"](https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf), *ECML 2001*.** Cumulative-link ordinal classification.

## Class imbalance

- **Chawla, Bowyer, Hall & Kegelmeyer, ["SMOTE: Synthetic Minority Over-sampling Technique"](https://arxiv.org/abs/1106.1813), *JAIR 2002*.** Synthetic oversampling (tabular; not applicable to raw text).
- **Cui, Jia, Lin, Song & Belongie, ["Class-Balanced Loss Based on Effective Number of Samples"](https://arxiv.org/abs/1901.05555), *CVPR 2019*.** Reweighting long-tail distributions.
- **Lin, Goyal, Girshick, He & Dollár, ["Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002), *ICCV 2017*.**
- **Elkan, ["The Foundations of Cost-Sensitive Learning"](https://cseweb.ucsd.edu/~elkan/rescale.pdf), *IJCAI 2001*.** Base-rate correction and Bayes-optimal thresholds.
- **`imbalanced-learn`** — <https://imbalanced-learn.org/>. `scikit-learn`-compatible resampling toolkit.

## Calibration and threshold tuning

- **Guo, Pleiss, Sun & Weinberger, ["On Calibration of Modern Neural Networks"](https://arxiv.org/abs/1706.04599), *ICML 2017*.** Temperature scaling and modern-network miscalibration.
- **Platt, ["Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods"](https://home.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf), *1999*.** Platt scaling.
- **Zadrozny & Elkan, ["Transforming classifier scores into accurate multiclass probability estimates"](https://cseweb.ucsd.edu/~elkan/calibrated.pdf), *KDD 2002*.** Isotonic regression for calibration.
- **Kull, Filho & Flach, ["Beta Calibration"](https://proceedings.mlr.press/v54/kull17a.html), *AISTATS 2017*.** Alternative to Platt scaling.
- **Brier, "Verification of forecasts expressed in terms of probability", *Monthly Weather Review*, 1950.** Brier score.
- **Saito & Rehmsmeier, ["The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets"](https://doi.org/10.1371/journal.pone.0118432), *PLoS ONE 2015*.**
- **Chicco & Jurman, ["The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation"](https://doi.org/10.1186/s12864-019-6413-7), *BMC Genomics 2020*.**
- **`netcal`** — <https://github.com/EFS-OpenSource/calibration-framework>. Reliability diagrams, ECE, temperature/isotonic/Platt calibration.
- **`sklearn.calibration.CalibratedClassifierCV`** — <https://scikit-learn.org/stable/modules/calibration.html>.

## Datasets and benchmarks

### English classification

- **Zhang, Zhao & LeCun, ["Character-level Convolutional Networks for Text Classification"](https://arxiv.org/abs/1509.01626), *NeurIPS 2015*.** AG News, DBpedia-14, Yelp/Amazon polarity, Yahoo Answers.
- **Maas, Daly, Pham, Huang, Ng & Potts, ["Learning Word Vectors for Sentiment Analysis"](https://aclanthology.org/P11-1015/), *ACL 2011*.** IMDb reviews.
- **Socher, Perelygin, Wu, Chuang, Manning, Ng & Potts, ["Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank"](https://aclanthology.org/D13-1170/), *EMNLP 2013*.** SST-2.
- **Wang et al., ["GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding"](https://arxiv.org/abs/1804.07461), *ICLR 2019*.** GLUE benchmark.
- **Wang et al., ["SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems"](https://arxiv.org/abs/1905.00537), *NeurIPS 2019*.**

### Multi-label / hierarchical

- **Lewis, Yang, Rose & Li, ["RCV1: A New Benchmark Collection for Text Categorization Research"](https://www.jmlr.org/papers/v5/lewis04a.html), *JMLR 2004*.** Reuters RCV1 hierarchical multi-label.
- **Kowsari, Brown, Heidarysafa, Meimandi, Gerber & Barnes, ["HDLTex: Hierarchical Deep Learning for Text Classification"](https://arxiv.org/abs/1709.08267), *ICMLA 2017*.** Web of Science WoS-46985.
- **Aly, Remus & Biemann, ["Hierarchical Multi-label Classification of Text with Capsule Networks"](https://aclanthology.org/P19-2045/), *ACL SRW 2019*.** Blurb Genre Collection.
- **Demszky, Movshovitz-Attias, Ko, Cowen, Nemade & Ravi, ["GoEmotions: A Dataset of Fine-Grained Emotions"](https://arxiv.org/abs/2005.00547), *ACL 2020*.**

### Multilingual

- **Conneau et al., ["XNLI: Evaluating Cross-lingual Sentence Representations"](https://arxiv.org/abs/1809.05053), *EMNLP 2018*.**
- **Yang, Zhang, Tar & Baldridge, ["PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification"](https://arxiv.org/abs/1908.11828), *EMNLP-IJCNLP 2019*.**
- **Keung, Lu, Szarvas & Smith, ["The Multilingual Amazon Reviews Corpus"](https://arxiv.org/abs/2010.02573), *EMNLP 2020*.**
- **FitzGerald et al., ["MASSIVE: A 1M-Example Multilingual NLU Dataset"](https://arxiv.org/abs/2204.08582), *ACL 2023*.**
- **Hu, Ruder, Siddhant, Neubig, Firat & Johnson, ["XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization"](https://arxiv.org/abs/2003.11080), *ICML 2020*.**
- **Ruder et al., ["XTREME-R: Towards More Challenging and Nuanced Multilingual Evaluation"](https://arxiv.org/abs/2104.07412), *EMNLP 2021*.**
- **NLLB Team, ["No Language Left Behind: Scaling Human-Centered Machine Translation"](https://arxiv.org/abs/2207.04672), *arXiv 2022*.** FLORES-200 benchmark for multilingual coverage assessment.

### Toxicity and safety

- **Davidson, Warmsley, Macy & Weber, ["Automated Hate Speech Detection and the Problem of Offensive Language"](https://arxiv.org/abs/1703.04009), *ICWSM 2017*.**
- **Jigsaw / Google, ["Toxic Comment Classification Challenge"](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge), *Kaggle 2018*.**

### Domain-specific

- **Malo, Sinha, Korhonen, Wallenius & Takala, ["Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts"](https://arxiv.org/abs/1307.5336), *JASIST 2014*.** Financial PhraseBank.
- **`scispaCy` biomedical datasets** — <https://allenai.github.io/scispacy/>.

## Toolkits and libraries

- **Hugging Face `transformers`** — <https://github.com/huggingface/transformers>.
- **Hugging Face `datasets`** — <https://github.com/huggingface/datasets>.
- **Hugging Face `evaluate`** — <https://github.com/huggingface/evaluate>. Standard metric implementations.
- **Hugging Face `Trainer` documentation** — <https://huggingface.co/docs/transformers/main/en/main_classes/trainer>.
- **`accelerate`** — <https://github.com/huggingface/accelerate>. Multi-GPU / mixed-precision helper.
- **`scikit-learn`** — <https://scikit-learn.org/>. Linear models, calibration, metrics.
- **`fasttext`** — <https://github.com/facebookresearch/fastText>.
- **`imbalanced-learn`** — <https://imbalanced-learn.org/>.
- **`skmultilearn`** — <http://scikit.ml/>. Multi-label extensions on top of scikit-learn.
- **`netcal`** — <https://github.com/EFS-OpenSource/calibration-framework>. Calibration toolkit.

## Model cards worth reading in full

- **`microsoft/deberta-v3-base`** — <https://huggingface.co/microsoft/deberta-v3-base>.
- **`microsoft/mdeberta-v3-base`** — <https://huggingface.co/microsoft/mdeberta-v3-base>.
- **`xlm-roberta-base`** — <https://huggingface.co/FacebookAI/xlm-roberta-base>.
- **`roberta-base`** — <https://huggingface.co/FacebookAI/roberta-base>.
- **`distilbert-base-uncased`** — <https://huggingface.co/distilbert-base-uncased>.
- **`facebook/xlm-v-base`** — <https://huggingface.co/facebook/xlm-v-base>.
