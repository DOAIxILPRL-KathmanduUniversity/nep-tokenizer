# Industry Findings, Baselines, and Tokenizer Adaptation

This document synthesizes key architectural findings across modern Indic and Devanagari tokenization research, specifies the standard 64k and 128k evaluation protocols, and details how standalone tokenizers relate to foundation model pretraining versus vocabulary expansion for existing models like Google Gemma or Meta LLaMA.

---

## 1. Evaluation Standards: The 64k and 128k Vocab Frontier

In modern Large Language Model (LLM) engineering for multilingual and Indic languages, vocabulary size selection represents a fundamental trade-off:
- **Compression Efficiency vs. Softmax Latency**: Larger vocabularies compress text into fewer sequence tokens (reducing attention complexity $\mathcal{O}(N^2)$ and inference prefill/decode passes), but increase the parameter count of the input embedding matrix ($W_e \in \mathbb{R}^{V \times d}$) and the final output LM projection head ($W_{\text{lm}} \in \mathbb{R}^{d \times V}$).
- **The Standard Evaluation Anchors**:
  - **64,000 (64k)**: The standard efficient frontier for regional and bilingual models (e.g. Param-1 / BharatGen 64k baseline, Yak 64k, Himalaya AI v1 64k).
  - **128,000 (128k)**: The standard frontier for high-capacity multilingual and omni models (e.g. Param-1 / BharatGen 128k baseline, Yak 128k, LLaMA-3).

All evaluation configurations in this repository prioritize **64,000** and **128,000** vocabulary sizes to ensure direct comparability with published baselines.

---

## 2. Key Industry Findings & Architectural Insights

Recent independent studies across Indic tokenization (Param-1/BharatGen, KrutrimAI, Arkios, Yak, and Sarvam) highlight several critical linguistic and computational considerations:

### A. The Pre-tokenization Ceiling Bug (Akshara Splitting)
Standard Western LLMs (GPT-2, GPT-4, LLaMA-3, Mistral) define words in their pre-tokenizers strictly as sequences of Unicode letters (`\p{L}+`):
- In Latin scripts, both vowels and consonants are letters (`\p{L}`).
- In **Devanagari and Indic abugidas**, consonants are letters (`\p{L}`), but dependent vowel signs (*matras*: `ा, ि, ी, ु, ू, े, ै, ो, ौ`) and virama (`्`) belong to Unicode categories `Mn` (non-spacing combining mark) or `Mc` (spacing combining mark).
- Consequently, `\p{L}+` forcefully splits Devanagari words at every vowel sign:
  - `नेपाली` under `\p{L}+` splits into `['न', 'े', 'प', 'ा', 'ल', 'ी']` before BPE merges even begin.
- **Mark-Aware Regex (`[\p{L}\p{M}]+`)**: Admitting Unicode marks (`\p{M}`) into the word pattern prevents artificial orthographic syllable splitting and allows words to remain whole.

### B. The Severed Matra Anomaly (Kathmandu Bug)
When words are severed at vowel signs, subsequent subword tokens often begin with a bare combining mark (e.g. `काठमाडौं` $\rightarrow$ `'क' + 'ाठ' + 'म' + 'ाड' + 'ौं'`, where `'ाठ'` and `'ाड'` start with an isolated matra `ा`).
- In broken regex tokenizers, 30% to 63% of Nepali tokens begin with an isolated combining mark.
- Robust tokenizers (Yak, Arkios, and SentencePiece Metaspace) ensure that dependent marks remain bound to consonant stems, eliminating or drastically reducing this anomaly.

### C. Unicode Character Completeness & Silent Byte-Fallback
Byte-level BPE tokenizers avoid unknown token errors (`<unk>`) by falling back to UTF-8 raw bytes. However:
- If an everyday Devanagari character lacks a dedicated single-token ID in the vocabulary, it silently decomposes into **3 to 4 raw byte tokens**.
- Models trained with insufficient character coverage (< 0.9995) or lacking explicit Devanagari block initialization frequently suffer from this silent expansion on less frequent letters (e.g., `ऋ, ः, ऽ, ॐ`).
- **Himalayan Tokenizer v2** sets `character_coverage: 0.9999` to guarantee that all 73 everyday Nepali characters have dedicated token representations.

---

## 3. From-Scratch Pretraining vs. Adapting Pretrained Models (Gemma, LLaMA)

A crucial engineering distinction must be made regarding how this standalone tokenizer is applied:

### A. Direct Use: Training Foundation Models From Scratch
The tokenizers produced by `train_sp_bpe.py` (and the `v2` configurations) are standalone SentencePiece BPE models. They are designed as the native tokenizer for **training Himalayan foundation models from scratch**:
- The model architecture is initialized with $V = 64{,}000$ or $V = 128{,}000$.
- The embedding matrix and LM head match the tokenizer's vocabulary index from step 0.

### B. Adapting Existing Open-Weights Models (Gemma, LLaMA, Qwen)
**A standalone tokenizer cannot simply replace the tokenizer of an existing pretrained model.**
In models like Google Gemma (`google/gemma-3-27b-it`) or Meta LLaMA (`meta-llama/Llama-3.1-8B-Instruct`):
- The model weights ($W_e$ and $W_{\text{lm}}$) are rigidly tied to the base tokenizer's exact token IDs.
- Directly swapping tokenizers scrambles all token mappings, completely destroying pretrained representations.

### C. The Vocabulary Expansion Protocol for Gemma / LLaMA
To utilize the high-quality Nepali subwords learned by `nep-tokenizer` within an existing foundation model, practitioners must follow a **Vocabulary Expansion & Continual Pretraining** pipeline:

1. **Vocabulary Differencing ($\Delta V$)**:
   Extract high-frequency Nepali subwords from `nep-tokenizer` that do not exist in the base model's vocabulary:
   $$\Delta V = V_{\text{nep-tokenizer}} \setminus V_{\text{base}}$$

2. **Resize Embedding Matrix**:
   Extend the model's token embeddings from $V_{\text{base}}$ to $V_{\text{base}} + |\Delta V|$:
   ```python
   model.resize_token_embeddings(len(extended_tokenizer))
   ```

3. **Subword Decomposition Initialization (Piece-Pooling)**:
   Instead of initializing new token embeddings with random Gaussian noise (which destabilizes training and degrades language generation), initialize each new token embedding as the mean of its constituent subwords in the base tokenizer:
   $$E(w_{\text{new}}) = \frac{1}{|P_{\text{base}}(w_{\text{new}})|} \sum_{t \in P_{\text{base}}(w_{\text{new}})} E_{\text{base}}(t)$$

4. **Targeted Continual Pretraining**:
   - Freeze the majority of the transformer backbone (or apply LoRA/QLoRA).
   - Train the newly added token embeddings on high-quality Nepali text to calibrate representation vectors before unfreezing attention layers.
   - This prevents catastrophic forgetting of base reasoning and multilingual capabilities while drastically reducing sequence length on Nepali text.

---

## 4. Evaluated Baselines & Reference Panel

The evaluation pipeline in `scripts/eval_tokenizer_hf.py`, `scripts/benchmark_all_models.py`, and `configs/eval_baselines.yml` supports evaluating standalone SentencePiece models and Hugging Face repositories across 17 distinct models and configurations:

| Model Config Name | Hub Repository / Path | Target Vocab | Architecture / Paradigm |
| :--- | :--- | :---: | :--- |
| **`nep-tokenizer-v2`** | `DOAIxILPRL-KU/nep-tokenizer` (64k, 100k, 128k) | 64k / 100k / 128k | DOAI x ILPRL (KU) v2 with 0.9999 character coverage |
| **`himalaya-bpe-v1`** | `himalaya-ai/nepali-tokenizer-bpe-*-5b` | 64k / 100k / 128k | Himalayan AI previous baseline (SentencePiece BPE, 0.997 coverage) |
| **`yak-64k` / `yak-128k`** | `ampixa/yak-tokenizer` (`v2.0/64k`, `128k`) | 64k / 128k | Lossless whole-word BPE, zero normalizer (Ampixa 2026) |
| **`arkios-65k`** | `sajalregmi4/arkios-1b-base` | 65k | Mark-aware ByteLevel BPE (Regmi et al. 2026) |
| **`qwen3-nepali-166k`** | `sidskarki/qwen3-nepali-tokenizer` | 166k | Extended Qwen2 tokenizer with 15k Devanagari subwords |
| **`nepalibpe-50k`** | `Aananda-giri/NepaliBPE` | 50k | Open Nepali BPE subword baseline |
| **`nepalibert-50k`** | `Rajan/NepaliBERT` | 50k | WordPiece Devanagari MLM baseline |
| **`bert-nepali-25k`** | `raygx/BERT_Nepali_Tokenizer` | 25k | WordPiece Nepali tokenizer |
| **`bharatgen-param1`** | `bharatgenai/Param-1-5B` | 256k | Param-1 Indic multilingual reference baseline |
| **`sarvam-1`** | `sarvamai/sarvam-1` | ~68k | Sarvam Indic 10-language foundation baseline |
| **`llama-3.1-8b`** | `meta-llama/Llama-3.1-8B-Instruct` | 128k | Frontier multilingual LLaMA-3 byte-level BPE baseline |
| **`gemma-2-9b`** | `google/gemma-2-9b` | 256k | Google Gemma SentencePiece Unigram baseline |

---

## 5. Running the Evaluation Pipeline

### A. Run Evaluation Across All Baselines

```bash
python scripts/eval_tokenizer_hf.py \
  --config configs/eval_baselines.yml \
  --eval-dir data/eval \
  --output-dir data/eval_results_hf
```

### B. Evaluate Local SentencePiece Models

```bash
python scripts/eval_tokenizer.py \
  --models-dir data/tokenizers \
  --eval-dir data/eval \
  --output-dir data/eval_results
```

### C. Render Unified Comparison Table

```bash
python scripts/render_eval_table.py \
  --inputs data/eval_results_hf,data/eval_results \
  --metric fertility \
  --lang-order ne,hi,en,roman_ne,roman_hi
```

> **Note on Benchmark Scores**: All empirical evaluation scores are generated locally on the held-out evaluation corpus produced by `scripts/build_eval_corpus.py`. Precomputed benchmark tables are omitted from the repository to ensure independent, reproducible evaluation on Himalayan AI's training and evaluation datasets.

---

## 6. References & Citations

Foundational research and references for Nepali and Indic tokenization:

```bibtex
@article{regmi2026vowel,
  title={Vowel Signs Are Not Letters: A Pre-tokenization Ceiling on Multilingual Tokenizer Fertility},
  author={Regmi, Sajal and Pudasaini, Siddhartha and Pun, Chetan Phakami},
  journal={arXiv preprint arXiv:2608.26449},
  year={2026},
  url={https://arxiv.org/abs/2608.26449}
}

@misc{ampixa2026yak,
  title={Yak: An Open, Lossless, Nepali-First Tokenizer},
  author={{Ampixa Labs}},
  year={2026},
  howpublished={\url{https://ampixa.com/blog/yak-an-open-lossless-nepali-first-tokenizer/}},
  note={Hugging Face: \url{https://huggingface.co/ampixa/yak-tokenizer}}
}

@article{himalaya2024nepali,
  title={Nepali Tokenizer Pipeline: Scalable Multi-Source SentencePiece Sampling for Devanagari},
  author={{Himalaya AI}},
  year={2024},
  note={Hugging Face: \url{https://huggingface.co/himalaya-ai/nepali-tokenizer-bpe-64k-5b}}
}

@article{chaudhary2026merge,
  title={Merge-Based Vocabulary Extension Stalls at the Pre-Tokenizer: An English-Preserving Repair for Nepali, Tested in Continued Pretraining},
  author={Chaudhary, Aarjan},
  journal={Naamche Labs Research / arXiv preprint},
  year={2026}
}

@article{alexandre2024papaya,
  title={Preserving Morphemes: Morphology-Guided Pre-Tokenization for Nepali},
  author={Shrestha, Kalash and Pradhan, Nikhil},
  year={2024}
}

@inproceedings{brahma2024morphtok,
  title={MorphTok: Morphologically Grounded Tokenization for Indic Languages},
  author={Brahma, Maharaj and Karthika, NJ and Singh, Atul and Adiga, Devaraj and Bhate, Smruti and Ramakrishnan, Ganesh and Saluja, Rohit and Desarkar, Maunendra Sankar},
  booktitle={Proceedings of EMNLP / IndicNLP},
  year={2024}
}

@phdthesis{prasain2011nepali,
  title={A Study on Morphology of Nepali},
  author={Prasain, Balaram},
  school={Tribhuvan University, Kathmandu, Nepal},
  year={2011}
}

@inproceedings{sennrich2016neural,
  title={Neural Machine Translation of Rare Words with Subword Units},
  author={Sennrich, Rico and Haddow, Barry and Birch, Alexandra},
  booktitle={Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL)},
  pages={1715--1725},
  year={2016}
}

@inproceedings{kudo2018subword,
  title={Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates},
  author={Kudo, Taku},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)},
  pages={66--75},
  year={2018}
}
```

