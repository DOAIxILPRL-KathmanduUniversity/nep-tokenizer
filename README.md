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

```bash
python scripts/train_sp_bpe.py \
  --source-repo himalaya-ai/nepali-tokenizer-corpus \
  --split train \
  --train-config configs/train_config.yml \
  --output-dir data/tokenizers
```

## 3) Build evaluation corpus

```bash
python scripts/build_eval_corpus.py \
  --config configs/eval_sources.yml
```

## 4) Evaluate tokenizers

```bash
python scripts/eval_tokenizer.py \
  --models-dir data/tokenizers \
  --eval-dir data/eval
```

## 5) Extract paper notes

```bash
python scripts/extract_paper_notes.py \
  --param /Users/interstellarninja/Desktop/HimalayaGPT/papers/2407.12481v2.pdf \
  --krutrim /Users/interstellarninja/Desktop/HimalayaGPT/papers/2507.13390v1.pdf
```

## Notes
- Fertility is reported as tokens/word (KrutrimAI proxy).
- BharatGen baselines referenced: 1.87 @64k, 1.60 @128k.
