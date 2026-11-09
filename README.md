# Nepali Tokenizer Pipeline

Standalone pipeline for sampling tokenizer corpora, training SentencePiece BPE tokenizers, and evaluating fertility/token-to-word ratios across Nepali, Hindi, English, and Romanized corpora.

## Setup

```bash
pip install -r requirements.txt
```

## 1) Sample tokenizer corpus

```bash
python scripts/run_parallel_sampling.py \
  --config configs/tokenizer_sources.yml \
  --groups configs/tokenizer_parallel_groups.yml \
  --target-tokens 5000000000 \
  --scale-targets \
  --upload-repo himalaya-ai/nepali-tokenizer-corpus \
  --shard-rows 150000 \
  --max-parallel 4 \
  --skip-local
```

## 2) Train SentencePiece BPE

Train with standard configuration (64k / 128k):
```bash
python scripts/train_sp_bpe.py \
  --source-repo himalaya-ai/nepali-tokenizer-corpus \
  --split train \
  --train-config configs/train_config.yml \
  --output-dir data/tokenizers
```

Or train with Himalayan Tokenizer v2 (0.9999 character coverage):
```bash
python scripts/train_sp_bpe.py \
  --source-repo himalaya-ai/nepali-tokenizer-corpus \
  --split train \
  --train-config configs/train_config_v2.yml \
  --output-dir data/tokenizers
```

## 3) Build evaluation corpus

```bash
python scripts/build_eval_corpus.py \
  --config configs/eval_sources.yml
```

## 4) Evaluate tokenizers & reference baselines

### 4a) Evaluate local SentencePiece models:
```bash
python scripts/eval_tokenizer.py \
  --models-dir data/tokenizers \
  --eval-dir data/eval \
  --output-dir data/eval_results
```

### 4b) Evaluate pretrained & reference baselines:
Evaluates previous baseline (`himalaya-bpe-64k-v1`), peer tokenizers (`arkios-65k`, `yak-64k`, `yak-128k`), and foundation models (`bharatgen-param2`, `sarvam-105b`, `llama-3.1-8b`, `gemma-3-27b`):
```bash
python scripts/eval_tokenizer_hf.py \
  --config configs/eval_baselines.yml \
  --eval-dir data/eval \
  --output-dir data/eval_results_hf
```

### 4c) Render comparison table:
```bash
python scripts/render_eval_table.py \
  --inputs data/eval_results,data/eval_results_hf \
  --metric fertility \
  --lang-order ne,hi,en,roman_ne,roman_hi
```

## 5) Tokenization integrity & test cases

Run validation suite checking 73 everyday Devanagari characters, matra cohesion (Kathmandu bug check), and lossless round-trip reconstruction:
```bash
python experiments/test_tokenization_integrity.py \
  --model himalaya-ai/nepali-tokenizer-bpe-64k-5b
```

## 6) Extract paper notes

```bash
python scripts/extract_paper_notes.py \
  --param /Users/interstellarninja/Desktop/HimalayaGPT/papers/2407.12481v2.pdf \
  --krutrim /Users/interstellarninja/Desktop/HimalayaGPT/papers/2507.13390v1.pdf
```

## 7) Standardized Benchmark & Plot Generation

Run empirical evaluation across all 17 models on the consistent benchmark and generate publication plots:
```bash
# Run benchmark across all dedicated Nepali, Indic, and frontier LLM tokenizers
python scripts/benchmark_all_models.py

# Generate publication-grade figures into docs/figures/ and hf_push/figures/
python scripts/generate_plots.py --output-dir docs/figures
```

## 8) Stage & push tokenizers to Hugging Face

Stage and publish multi-tier models (64k, 100k, 128k) to Hugging Face Hub (`DOAIxILPRL-KU/nep-tokenizer`):
```bash
# Local staging only into hf_push/ (gitignored)
python scripts/push_to_hf.py

# Explicit upload to Hugging Face repository (reads HF_TOKEN and HF_REPO_ID from .env)
python scripts/push_to_hf.py --upload
```

## Notes & Documentation
- Fertility is reported as tokens/word (KrutrimAI proxy).
- BharatGen baselines referenced: 1.87 @64k, 1.60 @128k.
- Comprehensive analysis, industry findings across baselines, and vocabulary expansion protocols for Gemma/LLaMA are detailed in [docs/industry_findings_and_baselines.md](docs/industry_findings_and_baselines.md).
- BibTeX citations for underlying papers and reference architectures are available in [docs/industry_findings_and_baselines.md#6-references--citations](docs/industry_findings_and_baselines.md#6-references--citations).
