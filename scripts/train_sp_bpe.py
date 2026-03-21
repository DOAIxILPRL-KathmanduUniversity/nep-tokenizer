#!/usr/bin/env python3
"""Train SentencePiece BPE tokenizers from a streaming HF dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import yaml
from datasets import load_dataset

try:
    import sentencepiece as spm
except Exception as exc:  # pragma: no cover
    raise SystemExit("sentencepiece is required. Install via pip install sentencepiece") from exc

# Ensure repo root is on path
import sys

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nepali_tokenizer.hf_utils import get_field_value
from nepali_tokenizer.quality_filters import FilterSpec, normalize_text, passes_quality

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def iter_texts(
    repo_id: str,
    split: str,
    field: str,
    config: Optional[str],
    max_samples: Optional[int],
    shuffle_buffer: Optional[int],
    seed: int,
    filter_spec: Optional[FilterSpec],
) -> Iterator[str]:
    if config:
        ds = load_dataset(repo_id, name=config, split=split, streaming=True)
    else:
        ds = load_dataset(repo_id, split=split, streaming=True)
    if shuffle_buffer and shuffle_buffer > 0:
        try:
            ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)
        except Exception:
            logger.warning("Shuffle failed; continuing without shuffle")
    seen = 0
    for row in ds:
        text_raw = get_field_value(row, field)
        if not isinstance(text_raw, str):
            continue
        text_norm = normalize_text(text_raw)
        if not text_norm:
            continue
        if filter_spec and not passes_quality(text_norm, filter_spec):
            continue
        yield text_norm
        seen += 1
        if max_samples and seen >= max_samples:
            break


def config_hash(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SentencePiece BPE tokenizers")
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--config", help="HF dataset config name")
    parser.add_argument("--field", default="text")
    parser.add_argument(
        "--train-config",
        default="configs/train_config.yml",
        help="YAML config with vocab sizes and SP options",
    )
    parser.add_argument("--output-dir", default="data/tokenizers")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--shuffle-buffer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.train_config)
    vocab_sizes = cfg.get("vocab_sizes", [64000])
    model_type = cfg.get("model_type", "bpe")
    normalization = cfg.get("normalization", "nfkc")
    character_coverage = float(cfg.get("character_coverage", 0.9995))
    byte_fallback = bool(cfg.get("byte_fallback", True))
    user_defined_symbols = cfg.get("user_defined_symbols") or []

    filter_spec = None
    if isinstance(cfg.get("filters"), dict):
        filter_spec = FilterSpec.from_dict(cfg["filters"])

    os.makedirs(args.output_dir, exist_ok=True)
    slug = args.source_repo.replace("/", "_")
    cfg_hash = config_hash(cfg)

    for vocab_size in vocab_sizes:
        model_prefix = Path(args.output_dir) / f"{slug}-{vocab_size}"
        logger.info("Training SentencePiece BPE vocab_size=%s", vocab_size)

        iterator = iter_texts(
            repo_id=args.source_repo,
            split=args.split,
            field=args.field,
            config=args.config,
            max_samples=args.max_samples,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed,
            filter_spec=filter_spec,
        )

        spm.SentencePieceTrainer.train(
            sentence_iterator=iterator,
            model_prefix=str(model_prefix),
            vocab_size=int(vocab_size),
            model_type=model_type,
            normalization_rule_name=normalization,
            character_coverage=character_coverage,
            byte_fallback=byte_fallback,
            user_defined_symbols=user_defined_symbols,
        )

        meta = {
            "source_repo": args.source_repo,
            "split": args.split,
            "config": args.config,
            "field": args.field,
            "vocab_size": int(vocab_size),
            "model_type": model_type,
            "normalization": normalization,
            "character_coverage": character_coverage,
            "byte_fallback": byte_fallback,
            "user_defined_symbols": user_defined_symbols,
            "max_samples": args.max_samples,
            "shuffle_buffer": args.shuffle_buffer,
            "seed": args.seed,
            "config_hash": cfg_hash,
        }
        meta_path = Path(f"{model_prefix}.json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved model + metadata: %s", model_prefix)


if __name__ == "__main__":
    main()
