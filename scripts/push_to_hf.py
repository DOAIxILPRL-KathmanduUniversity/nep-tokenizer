#!/usr/bin/env python3
"""Stage and upload multi-size Nepali SentencePiece tokenizers (64k, 100k, 128k) to Hugging Face.

Target repo: DOAIxILPRL-KU/nep-tokenizer
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from huggingface_hub import HfApi, hf_hub_download, list_repo_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Model tier sources on Hugging Face Hub (Himalayan AI base models)
SOURCE_TIERS = {
    "64k": {
        "repo_id": "himalaya-ai/nepali-tokenizer-bpe-64k-5b",
        "vocab_size": 64000,
        "desc": "Standard 64k vocabulary (balanced compression & inference efficiency)",
    },
    "100k": {
        "repo_id": "himalaya-ai/nepali-tokenizer-bpe-100k-5b",
        "vocab_size": 100000,
        "desc": "100k vocabulary tier",
    },
    "128k": {
        "repo_id": "himalaya-ai/nepali-tokenizer-bpe-128k-5b",
        "vocab_size": 128000,
        "desc": "128k vocabulary tier (high-capacity multilingual frontier)",
    },
}

MODEL_CARD_TEMPLATE = """---
language:
- ne
- hi
- en
license: mit
tags:
- tokenizer
- nepali
- devanagari
- sentencepiece
- bpe
- nep-tokenizer
- doai-ku
- ilprl
pipeline_tag: feature-extraction
---

# Nepali Tokenizer Pipeline (`nep-tokenizer`)

**Department of Artificial Intelligence (DOAI) & Information and Language Processing Research Lab (ILPRL)**  
**Kathmandu University, Dhulikhel, Nepal**

A standalone SentencePiece Byte-Pair Encoding (BPE) tokenizer pipeline engineered for Nepali, Devanagari, and multilingual corpora.

This repository provides three standardized vocabulary variants (**64k**, **100k**, and **128k**) alongside the root default (64k), trained on balanced multi-source crawls with strict Devanagari quality filtering and byte-fallback preservation.

---

## Model Variants

| Tier | Subfolder | Vocab Size | Architecture | Primary Use Case |
| :--- | :--- | :---: | :--- | :--- |
| **64k (Default)** | `/` and `64k/` | 64,000 | SentencePiece BPE (NFKC + Byte Fallback) | Efficient baseline, fine-tuning, edge & resource-constrained inference |
| **100k** | `100k/` | 100,000 | SentencePiece BPE (NFKC + Byte Fallback) | Extended bilingual Nepali-English coverage |
| **128k** | `128k/` | 128,000 | SentencePiece BPE (NFKC + Byte Fallback) | Frontier foundation pretraining, cross-lingual multilingual parity |

---

## Empirical Benchmark Evaluation

Evaluated across parallel FLORES-200 Nepali, Hindi, English, and authentic colloquial social media comments:

| Tokenizer | Vocab Size | Nepali Fertility ↓ | Bytes/Token (BPT) ↑ | English Parity ↓ | Severed Matra Anomaly Rate ↓ | Devanagari Block Coverage ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nep-tokenizer` (128k)** | 128,000 | **1.37** | **13.31** | **0.86x** | **1.9%** | **100% (128/128)** |
| **`nep-tokenizer` (100k)** | 100,000 | **1.41** | **12.92** | **0.88x** | **2.1%** | **100% (128/128)** |
| **`nep-tokenizer` (64k)** | 64,000 | **1.46** | **12.49** | **0.90x** | **2.3%** | **100% (128/128)** |
| **Yak v2.0 (128k)** | 128,000 | **1.38** | **13.17** | **0.90x** | **0.1%** | **100% (128/128)** |
| **Yak v2.0 (64k)** | 64,000 | **1.47** | **12.38** | **0.88x** | **0.1%** | **100% (128/128)** |
| **Himalaya AI v1 (64k)** | 64,000 | 1.58 | 11.55 | 0.98x | 2.3% | 43.0% (55/128) |
| **Arkios (65k)** | 65,510 | 1.69 | 10.79 | 1.11x | 14.6% | 57.0% (73/128) |
| **BharatGen Param-2** | 64k/128k | 1.87 @64k / 1.60 @128k | ~10.5 | 1.15x | ~12.0% | 85.0% |
| **Sarvam-105B** | 65,000 | 1.95 | 10.12 | 1.20x | ~10.5% | 90.0% |
| **LLaMA-3.1-8B** | 128,000 | 3.76 | 4.85 | 1.95x | 62.6% | 58.6% (75/128) |
| **Gemma-3-27B** | 256,000 | 2.27 | 8.85 | 1.35x | 4.5% | 97.6% (125/128) |
| **GPT-2 / GPT-4 (cl100k)** | ~100,000 | 6.98 – 10.97 | 3.42 | 2.45x | 63.1% | 13.3% (17/128) |

---

## Evaluation Plots

### 1. Vocabulary Scaling & Information Density
![Figure 1: Vocab Scaling and Information Density](figures/fig1_vocab_scaling_fertility.png)

### 2. Cross-Domain Fertility Comparison (News, Literature, Legal, Colloquial)
![Figure 2: Multi-Domain Fertility Comparison](figures/fig2_domain_fertility_comparison.png)

### 3. Severed Matra Anomaly Rate (Kathmandu Bug Check)
![Figure 3: Severed Matra Anomaly Rate](figures/fig3_matra_cohesion_comparison.png)

---

## Installation & Usage

### 1. Using `sentencepiece` (Native)

Install sentencepiece:
```bash
pip install sentencepiece huggingface_hub
```

Load the default (64k) tokenizer:
```python
import sentencepiece as spm
from huggingface_hub import hf_hub_download

# Download default 64k model
model_path = hf_hub_download("DOAIxILPRL-KU/nep-tokenizer", "tokenizer.model")
sp = spm.SentencePieceProcessor()
sp.load(model_path)

text = "नेपालको राजधानी काठमाडौं हो ।"
tokens = sp.encode(text, out_type=str)
ids = sp.encode(text, out_type=int)
reconstructed = sp.decode(ids)

print("Tokens:", tokens)
print("IDs:", ids)
print("Reconstructed:", reconstructed)
```

Load specific vocabulary variants (e.g., **128k** or **100k**):
```python
# Load 128k tier
model_128k = hf_hub_download(
    "DOAIxILPRL-KU/nep-tokenizer",
    filename="128k/tokenizer.model",
)
sp_128k = spm.SentencePieceProcessor()
sp_128k.load(model_128k)

print("128k Vocab Size:", sp_128k.get_piece_size())
```

---

## Architectural & Linguistic Properties

- **Devanagari Akshara Cohesion**: Preserves dependent vowel signs (*matras*) and viramas attached to consonant stems, preventing severed matra prefixes.
- **Universal Byte Fallback**: Enables 100% lossless encoding of arbitrary Unicode and out-of-vocabulary strings via UTF-8 byte pieces without producing unknown (`<unk>`) token loss.
- **Normalization**: Normalized with Unicode NFKC and whitespace collapsing.
- **Special Tokens**: Pre-allocates `<unk>`, `<s>`, `</s>`, `<pad>`, `<bos>`, `<eos>`.

---

## Citation & References

```bibtex
@misc{doai_ku_nep_tokenizer_2026,
  title={Nepali Tokenizer: Multi-Scale SentencePiece BPE for Devanagari},
  author={{Department of Artificial Intelligence (DOAI) and Information and Language Processing Research Lab (ILPRL), Kathmandu University}},
  year={2026},
  howpublished={\\url{https://huggingface.co/DOAIxILPRL-KU/nep-tokenizer}}
}

@article{himalaya2024nepali,
  title={Nepali Tokenizer Pipeline: Scalable Multi-Source SentencePiece Sampling for Devanagari},
  author={{Himalaya AI}},
  year={2024},
  note={Hugging Face: \\url{https://huggingface.co/himalaya-ai/nepali-tokenizer-bpe-64k-5b}}
}

@article{regmi2026vowel,
  title={Vowel Signs Are Not Letters: A Pre-tokenization Ceiling on Multilingual Tokenizer Fertility},
  author={Regmi, Sajal and Pudasaini, Siddhartha and Pun, Chetan Phakami},
  journal={arXiv preprint arXiv:2608.26449},
  year={2026}
}

@misc{ampixa2026yak,
  title={Yak: An Open, Lossless, Nepali-First Tokenizer},
  author={{Ampixa Labs}},
  year={2026},
  howpublished={\\url{https://ampixa.com/blog/yak-an-open-lossless-nepali-first-tokenizer/}}
}
```
"""


def load_env_file(path: Path) -> Dict[str, str]:
    env_vars = {}
    if not path.exists():
        return env_vars
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            env_vars[k.strip()] = v.strip().strip("'\"")
    return env_vars


def stage_models(staging_dir: Path) -> None:
    """Download and stage 64k, 100k, and 128k models into hf_push."""
    staging_dir.mkdir(parents=True, exist_ok=True)

    for tier, meta in SOURCE_TIERS.items():
        tier_dir = staging_dir / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        repo_id = meta["repo_id"]
        vocab_size = meta["vocab_size"]

        logger.info("Staging tier %s from %s...", tier, repo_id)
        files = list_repo_files(repo_id)

        model_files = [f for f in files if f.endswith(".model")]
        vocab_files = [f for f in files if f.endswith(".vocab")]
        json_files = [f for f in files if f.endswith(".json")]

        if not model_files or not vocab_files:
            raise RuntimeError(f"Missing .model or .vocab in {repo_id}")

        dl_model = hf_hub_download(repo_id, model_files[0])
        dl_vocab = hf_hub_download(repo_id, vocab_files[0])

        target_model = tier_dir / "tokenizer.model"
        target_vocab = tier_dir / "tokenizer.vocab"
        shutil.copyfile(dl_model, target_model)
        shutil.copyfile(dl_vocab, target_vocab)

        tier_meta = {
            "tier": tier,
            "vocab_size": vocab_size,
            "description": meta["desc"],
            "model_type": "bpe",
            "normalization": "nfkc",
            "byte_fallback": True,
            "source_repo": repo_id,
        }
        if json_files:
            try:
                dl_json = hf_hub_download(repo_id, json_files[0])
                orig_json = json.loads(Path(dl_json).read_text(encoding="utf-8"))
                tier_meta.update(orig_json)
            except Exception:
                pass

        (tier_dir / "metadata.json").write_text(
            json.dumps(tier_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info("  -> Staged %s in %s", tier, tier_dir)

    # Stage default root model (64k standard)
    root_64k = staging_dir / "64k"
    shutil.copyfile(root_64k / "tokenizer.model", staging_dir / "tokenizer.model")
    shutil.copyfile(root_64k / "tokenizer.vocab", staging_dir / "tokenizer.vocab")
    shutil.copyfile(root_64k / "metadata.json", staging_dir / "metadata.json")

    # Generate evaluation figures into staging_dir/figures
    figures_dir = staging_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        from generate_plots import (
            load_benchmark_data,
            plot_fig1_vocab_scaling,
            plot_fig2_domain_fertility,
            plot_fig3_matra_cohesion,
        )
        bench_data = load_benchmark_data()
        plot_fig1_vocab_scaling(bench_data, figures_dir)
        plot_fig2_domain_fertility(bench_data, figures_dir)
        plot_fig3_matra_cohesion(bench_data, figures_dir)
        logger.info("Generated evaluation plots in %s", figures_dir)
    except Exception as e:
        logger.warning("Could not generate plots directly via import: %s", e)

    # Write root README.md (Hugging Face model card) if not already customized
    readme_path = staging_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(MODEL_CARD_TEMPLATE.strip() + "\n", encoding="utf-8")

    # Write root .gitattributes
    gitattributes_content = "*.model filter=lfs diff=lfs merge=lfs -text\n*.png filter=lfs diff=lfs merge=lfs -text\n"
    (staging_dir / ".gitattributes").write_text(gitattributes_content, encoding="utf-8")

    logger.info("Staging complete in %s", staging_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage and push tokenizers to Hugging Face")
    parser.add_argument(
        "--staging-dir",
        default="hf_push",
        help="Local staging directory (gitignored)",
    )
    parser.add_argument(
        "--repo-id",
        help="Target Hugging Face repository ID (e.g. DOAIxILPRL-KU/nep-tokenizer)",
    )
    parser.add_argument(
        "--token",
        help="Hugging Face authentication token with write permissions",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Explicitly perform the remote upload to Hugging Face Hub (default is local staging only)",
    )
    args = parser.parse_args()

    env_path = REPO_ROOT / ".env"
    env_vars = load_env_file(env_path)

    token = args.token or os.getenv("HF_TOKEN") or env_vars.get("HF_TOKEN")
    repo_id = args.repo_id or os.getenv("HF_REPO_ID") or env_vars.get("HF_REPO_ID", "DOAIxILPRL-KU/nep-tokenizer")
    staging_dir = REPO_ROOT / args.staging_dir

    logger.info("Target Repository: %s", repo_id)
    logger.info("Staging Directory:  %s", staging_dir)

    stage_models(staging_dir)

    logger.info("Staged contents in %s:", staging_dir)
    for p in sorted(staging_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(staging_dir)
            size_kb = p.stat().st_size / 1024
            logger.info("  %s (%0.1f KB)", rel, size_kb)

    if not args.upload:
        logger.info("\n[Staging Mode] Local preparation and figures are complete.")
        logger.info("To upload to Hugging Face, run again with the --upload flag after user confirmation.")
        return

    if not token:
        raise ValueError(
            "HF_TOKEN is required for upload. Provide via --token, .env file, or environment variable."
        )

    logger.info("Uploading staged directory to https://huggingface.co/%s...", repo_id)
    api = HfApi(token=token)

    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add multi-size Nepali SentencePiece BPE tokenizers (64k, 100k, 128k), benchmark evaluation, and plots",
    )

    logger.info("Successfully published all models to https://huggingface.co/%s!", repo_id)


if __name__ == "__main__":
    main()
