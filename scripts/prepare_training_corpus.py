#!/usr/bin/env python3
"""
scripts/prepare_training_corpus.py

Builds a balanced, high-diversity training corpus for Nep-Tokenizer v2:
- 70% Nepali (FineWeb-2 npi_Deva + Himalaya-AI pretrain corpus)
- 20% English (FineWeb-Edu sample-10BT)
- 10% Code (Python / standard code snippets)
Total target: ~50-80 MB of high-diversity, cleaned text.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from datasets import load_dataset

# Set HF token
env_file = Path(__file__).resolve().parents[1] / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "tokenizer_train_corpus.txt"
TARGET_MB = 60.0  # 60MB target

SOURCES = [
    {
        "name": "nepali_fineweb2",
        "dataset": "HuggingFaceFW/fineweb-2",
        "config": "npi_Deva",
        "split": "train",
        "field": "text",
        "target_mb": TARGET_MB * 0.40,
        "min_len": 50,
    },
    {
        "name": "nepali_pretrain",
        "dataset": "himalaya-ai/nepali-pretrain-corpus",
        "config": None,
        "split": "train",
        "field": "text",
        "target_mb": TARGET_MB * 0.30,
        "min_len": 50,
    },
    {
        "name": "english_fineweb_edu",
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "field": "text",
        "target_mb": TARGET_MB * 0.20,
        "min_len": 100,
    },
]

CODE_SNIPPETS = """
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

class TokenizerEvaluator:
    def __init__(self, vocab_size: int = 64000):
        self.vocab_size = vocab_size
        self.results = {}

    def evaluate(self, dataset, batch_size: int = 32):
        for batch in dataset:
            tokens = self.encode(batch)
            self.results.append(tokens)
        return self.results

import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        return x + self.mlp(x)
""" * 500


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size >= TARGET_MB * 0.8 * 1e6:
        print(f"Corpus already exists at {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size/1e6:.1f} MB). Skipping download.")
        return

    print(f"Building tokenizer training corpus (~{TARGET_MB} MB) to {OUTPUT_PATH}...")
    t0 = time.time()
    total_bytes = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        # 1. First write synthetic Devanagari seed block to ensure 100% Devanagari representation
        print("Adding Devanagari phoneme and character seed sequences...")
        vowels = "अ आ इ ई उ ऊ ऋ ॠ ऌ ॡ ए ऐ ओ औ अं अः"
        matras = "ा ि ी ु ू ृ ॄ ॅ ॆ े ै ॉ ॊ ो ौ ्"
        consonants = "क ख ग घ ङ च छ ज झ ञ ट ठ ड ढ ण त थ द ध न प फ ब भ म य र ल व श ष स ह"
        nukta_consonants = "क़ ख़ ग़ ज़ ड़ ढ़ फ़ य़ ऱ ऴ"
        conjuncts = "क्ष त्र ज्ञ श्र द्य द्व द्म त्त द्ध ङ्ग ङ्ख ङ्क ह्म ह्न ह्ल"
        punctuation = "। ॥ ॐ ऽ ० १ २ ३ ४ ५ ६ ७ ८ ९"
        seed_line = f"{vowels} {matras} {consonants} {nukta_consonants} {conjuncts} {punctuation}\n"
        for _ in range(200):
            out.write(seed_line)
            total_bytes += len(seed_line.encode("utf-8"))

        # 2. Write Code Snippets
        out.write(CODE_SNIPPETS)
        total_bytes += len(CODE_SNIPPETS.encode("utf-8"))
        print(f"Code and seed written: {total_bytes/1e6:.2f} MB")

        # 3. Stream from HF sources
        for src in SOURCES:
            print(f"Streaming from {src['name']} ({src['dataset']})... target: {src['target_mb']:.1f} MB")
            src_bytes = 0
            t_src = time.time()
            try:
                if src["config"]:
                    ds = load_dataset(src["dataset"], src["config"], split=src["split"], streaming=True)
                else:
                    ds = load_dataset(src["dataset"], split=src["split"], streaming=True)

                for item in ds:
                    text = (item.get(src["field"]) or "").strip()
                    if len(text) < src["min_len"]:
                        continue
                    line = text.replace("\n", " ") + "\n"
                    b_len = len(line.encode("utf-8"))
                    out.write(line)
                    src_bytes += b_len
                    total_bytes += b_len
                    if src_bytes >= src["target_mb"] * 1e6:
                        break

                print(f"  -> Got {src_bytes/1e6:.1f} MB in {time.time()-t_src:.1f}s")
            except Exception as e:
                print(f"  Warning: failed streaming from {src['name']}: {e}")

    print(f"Total corpus built: {OUTPUT_PATH.stat().st_size/1e6:.2f} MB in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

