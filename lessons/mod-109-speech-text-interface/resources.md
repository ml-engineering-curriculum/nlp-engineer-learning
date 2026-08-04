# Resources for mod-109-speech-text-interface (Speech / Text Interface for NLP Engineers)

Primary sources, standards, and tooling used across the chapters and exercises. Follow the primary source before secondary summaries.

## ASR models and toolkits

- Radford, Kim, Xu, Brockman, McLeavey & Sutskever, ["Robust Speech Recognition via Large-Scale Weak Supervision"](https://arxiv.org/abs/2212.04356), 2022 — the Whisper paper: architecture, task tokens, LID, translation head, and training-data recipe.
- OpenAI, [`whisper` GitHub repository](https://github.com/openai/whisper) — reference implementation, model cards, and the reference English text-normaliser used for WER on LibriSpeech / TED-LIUM.
- Baevski, Zhou, Mohamed & Auli, ["wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"](https://arxiv.org/abs/2006.11477), NeurIPS 2020 — the self-supervised CTC-fine-tunable acoustic model family.
- Hsu et al., ["HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units"](https://arxiv.org/abs/2106.07447), 2021 — sibling of wav2vec 2.0; frequently a stronger CTC starting point.
- Chen et al., ["WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing"](https://arxiv.org/abs/2110.13900), 2021 — SSL model with a strong speaker-verification / diarisation head.
- Pratap et al., ["Scaling Speech Technology to 1,000+ Languages"](https://arxiv.org/abs/2305.13516), 2023 — Meta MMS: wav2vec-family ASR for 1 100+ languages, useful reference for low-resource fallbacks.
- NVIDIA NeMo — [ASR docs](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/intro.html) and [Canary / Parakeet model cards](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html).
- Hugging Face Transformers, [`AutomaticSpeechRecognitionPipeline`](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline) and the [Whisper model docs](https://huggingface.co/docs/transformers/en/model_doc/whisper) — chunking, `return_timestamps`, batching for long-form audio.
- SYSTRAN, [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-backed Whisper with word-level timestamps and VAD-filter chunking.

## Timestamp alignment and forced alignment

- Bain, Huh, Han & Zisserman, ["WhisperX: Time-Accurate Speech Transcription of Long-Form Audio"](https://arxiv.org/abs/2303.00747), Interspeech 2023 — VAD chunking, wav2vec 2.0 forced alignment on top of Whisper.
- Zhang, Lu, Ma, Zhu, Zhang, Lin, Zhang, Xu, Zhu & Meng, ["Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages"](https://arxiv.org/abs/2303.01037), 2023 — long-form and streaming ASR discussion.
- MFA (Montreal Forced Aligner) — [documentation](https://montreal-forced-aligner.readthedocs.io/) and [MFA repo](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner).
- W3C, [WebVTT: The Web Video Text Tracks Format](https://www.w3.org/TR/webvtt1/) — the caption format that most timestamped ASR output eventually ships as.
- MPEG / ITU-T, [SRT (SubRip Subtitle)](https://en.wikipedia.org/wiki/SubRip#Format) — the older subtitle format; profile it carefully because there is no formal spec.

## Text normalisation and inverse text normalisation

- OpenAI Whisper, [`whisper/normalizers/english.py`](https://github.com/openai/whisper/blob/main/whisper/normalizers/english.py) and [`basic.py`](https://github.com/openai/whisper/blob/main/whisper/normalizers/basic.py) — the reference English / basic normalisers used to compute Whisper-paper WER.
- Zhang, Bakhturina, Gorman & Ginsburg, ["NeMo Inverse Text Normalization: From Development to Production"](https://arxiv.org/abs/2104.05055), Interspeech 2021 — WFST-based ITN in NVIDIA NeMo.
- NVIDIA NeMo Text Processing, [`nemo_text_processing`](https://github.com/NVIDIA/NeMo-text-processing) — TN and ITN grammars for many languages.
- Zhang, Sproat, Ng, Stahlberg, Peng, Gorman & Roark, ["Neural Models of Text Normalization for Speech Applications"](https://direct.mit.edu/coli/article/45/2/293/95508/Neural-Models-of-Text-Normalization-for-Speech), *Computational Linguistics* 45(2), 2019 — the seq2seq TN / ITN problem framing and error mode ("silly errors" from neural TN).
- Sproat & Jaitly, ["RNN Approaches to Text Normalization: A Challenge"](https://arxiv.org/abs/1611.00068), 2016 — the Google TN challenge and dataset.
- Ebden & Sproat, ["The Kestrel TTS text normalization system"](https://www.cambridge.org/core/journals/natural-language-engineering/article/kestrel-tts-text-normalization-system/F0C18A3F596B75D83B75C479E23795DA), *Natural Language Engineering*, 2015 — the reference production TN architecture.
- Unicode Consortium, [Unicode Technical Standard #35: Locale Data Markup Language — Numbers / Dates / Units](https://www.unicode.org/reports/tr35/tr35-numbers.html) — the standards you must respect when writing ITN for a new locale; also underpins ICU.

## Punctuation restoration and truecasing

- Alam, Khan & Alam, ["Punctuation Restoration using Transformer Models for High- and Low-Resource Languages"](https://aclanthology.org/2020.wnut-1.18/), W-NUT 2020 — the widely reused reference recipe on IWSLT 2011 ref-transcripts.
- Che, Wang & Ma, ["Punctuation Prediction for Unsegmented Transcript Based on Word Vector"](https://aclanthology.org/L16-1103/), LREC 2016 — early neural sequence-labelling framing.
- Guerreiro, Zerva, van Stigt, Rei & Martins, ["Multilingual punctuation restoration for streaming speech"](https://arxiv.org/abs/2203.14976), 2022 — streaming-aware punctuation.
- Kim, Kim, Han, Lee, Shon & Choi, ["StreamAtt: Direct Streaming Speech-to-Text Translation with Attention-based Audio History Selection"](https://arxiv.org/abs/2406.06097), 2024 — streaming ASR/ST latency framing that applies to streaming punctuation as well.
- ["Neural Machine Translation of Rare Words with Subword Units"](https://arxiv.org/abs/1508.07909) (Sennrich, Haddow & Birch, 2016) — background on the subword models the punctuation restoration models are built on.
- Hugging Face, [`oliverguhr/fullstop-punctuation-multilang-large`](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large) and [`felflare/bert-restore-punctuation`](https://huggingface.co/felflare/bert-restore-punctuation) — reference multilingual punctuation-restoration checkpoints.
- Susanto, Chieu & Lu, ["The Impact of Truecasing on Neural Machine Translation"](https://aclanthology.org/W16-2308/), WMT 2016 — background on truecasing and its downstream impact.
- Sanchez, ["Word-Casing: A New Look at the Automatic Case Restoration Problem"](https://arxiv.org/abs/1903.11222), 2019 — approaches to truecasing (rule / statistical / neural).

## Diarisation

- Bredin et al., ["pyannote.audio: neural building blocks for speaker diarization"](https://arxiv.org/abs/1911.01255), ICASSP 2020 — the reference open-source diarisation toolkit; also see the [pyannote.audio 3.x pipeline](https://github.com/pyannote/pyannote-audio).
- Bredin & Laurent, ["End-to-end speaker segmentation for overlap-aware resegmentation"](https://arxiv.org/abs/2104.04045), Interspeech 2021 — overlap-aware segmentation used in pyannote v3.
- Fujita, Kanda, Horiguchi, Nagamatsu & Watanabe, ["End-to-End Neural Speaker Diarization with Self-Attention"](https://arxiv.org/abs/1909.06247), ASRU 2019 — the EEND family.
- Park, Kanda, Dimitriadis, Han, Watanabe & Narayanan, ["A Review of Speaker Diarization: Recent Advances with Deep Learning"](https://arxiv.org/abs/2101.09624), *Computer Speech & Language*, 2022 — a good, dense survey.
- LDC, [Rich Transcription Time Marked (RTTM) format](https://catalog.ldc.upenn.edu/docs/LDC2004T12/RTTM-format-v13.pdf) — the standard file format for time-marked speaker information.
- Anguera, Bozonnet, Evans, Fredouille, Friedland & Vinyals, ["Speaker Diarization: A Review of Recent Research"](https://ieeexplore.ieee.org/document/6135543), IEEE TASLP 20(2), 2012 — the foundational review.
- Watanabe, Mandel, Barker, Vincent et al., ["The Fifth 'CHiME' Speech Separation and Recognition Challenge"](https://arxiv.org/abs/2004.09249), 2020 — canonical dinner-party overlap dataset used for evaluation.
- NIST, [Rich Transcription Evaluation](https://www.nist.gov/itl/iad/mig/rich-transcription-evaluation) — origin of DER (Diarization Error Rate) and the associated `md-eval` scoring tool.
- pyannote-metrics, [documentation](https://pyannote.github.io/pyannote-metrics/) — Python implementation of DER, JER, purity/coverage.

## Voice activity detection and streaming

- Silero VAD, [`snakers4/silero-vad`](https://github.com/snakers4/silero-vad) — the small, fast VAD used by `faster-whisper` and many streaming stacks.
- Ryant, Church, Cieri, Cristia, Du, Ganapathy & Liberman, ["The Second DIHARD Diarization Challenge"](https://arxiv.org/abs/1906.07839), Interspeech 2019 — a widely used benchmark for diarisation over noisy audio.
- Google WebRTC VAD (integration reference), [`py-webrtcvad`](https://github.com/wiseman/py-webrtcvad) — the classic energy/GMM VAD you will occasionally still see in production.

## Locale, language codes, and text conventions

- IETF, [BCP-47: Tags for Identifying Languages](https://www.rfc-editor.org/info/bcp47) — the language-tag standard you must round-trip.
- ISO 639-1 / 639-2 / 639-3 — the language codes underpinning BCP-47; useful when Whisper's `iso-639-1` codes and diarisation tooling disagree.
- Unicode Consortium, [Unicode Technical Standard #10: Unicode Collation Algorithm](https://www.unicode.org/reports/tr10/) — case folding and locale-aware casing correctness references.
- CLDR (Unicode Common Locale Data Repository), [Numbers, Dates, Units data files](https://cldr.unicode.org/) — the data source you should pull from before hand-writing ITN grammars.

## Evaluation

- Morris, Maier & Green, ["From WER and RIL to MER and WIL"](https://www.researchgate.net/publication/221489635_From_WER_and_RIL_to_MER_and_WIL_improved_evaluation_measures_for_connected_speech_recognition), Interspeech 2004 — the reference for MER / WIL alongside WER.
- Kilgour, Kim, Watanabe & Lugosch, ["A Comparison of Post-Processing Approaches for Improving ASR"](https://arxiv.org/abs/2004.09765), 2020 — where each post-processing stage helps or hurts WER.
- `jiwer` — [documentation](https://jitsi.github.io/jiwer/) — the standard Python library for WER / CER / MER computation with alignment.
- NIST SCLite (`sctk`) — [documentation](https://github.com/usnistgov/SCTK) — the canonical scoring toolkit for WER and confusion matrices in ASR research.

## Datasets frequently used at the text-side

- Panayotov, Chen, Povey & Khudanpur, ["LibriSpeech: an ASR corpus based on public domain audio books"](https://ieeexplore.ieee.org/document/7178964), ICASSP 2015 — the reference clean English benchmark.
- Ardila et al., ["Common Voice: A Massively-Multilingual Speech Corpus"](https://arxiv.org/abs/1912.06670), LREC 2020 — Mozilla Common Voice; wide multilingual coverage.
- TED-LIUM release 3, [Rousseau et al., 2018](https://arxiv.org/abs/1805.04699) — talk-length audio; useful for evaluating long-form Whisper + WhisperX pipelines.
- Wang, Chen, Yoshioka et al., ["VoxPopuli: A Large-Scale Multilingual Speech Corpus for Representation Learning, Semi-Supervised Learning and Interpretation"](https://arxiv.org/abs/2101.00390), ACL 2021 — European Parliament recordings; strong on multilingual and code-switching.
- Cieri, Miller & Walker, ["The Fisher Corpus: A Resource for the Next Generations of Speech-to-Text"](https://aclanthology.org/L04-1400/), LREC 2004 — conversational telephony; the historic diarisation training / evaluation set.
- IWSLT (International Conference on Spoken Language Translation) — [annual proceedings](https://aclanthology.org/venues/iwslt/) — punctuation restoration references almost always use the IWSLT TED transcripts.
