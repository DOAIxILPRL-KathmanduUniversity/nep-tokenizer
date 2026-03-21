#!/usr/bin/env python3
"""Evaluate HF tokenizers (from model repos) on a fixed eval corpus."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    from transformers import AutoTokenizer
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "transformers is required. Install via pip install transformers"
    ) from exc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


DEFAULT_MODELS = [
    "bharatgenai/Param2-17B-A2.4B-Thinking",
    "sarvamai/sarvam-105b",
    "openai/gpt-oss-120b",
    "meta-llama/Llama-3.1-8B-Instruct",
]


def list_eval_files(eval_dir: str) -> Dict[str, Path]:
    eval_path = Path(eval_dir)
    files = {}
    for path in eval_path.glob("eval_*.jsonl"):
        lang = path.stem.replace("eval_", "")
        files[lang] = path
    return files


def encode_ids(tokenizer, text: str) -> List[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        return tokenizer(text, add_special_tokens=False).input_ids


def eval_file(tokenizer, path: Path) -> Dict[str, Any]:
    total_tokens = 0
    total_words = 0
    total_chars = 0
    total_docs = 0
    unk_tokens = 0
    unk_id = getattr(tokenizer, "unk_token_id", None)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            total_docs += 1
            total_chars += len(text)
            words = len(text.split())
            total_words += words
            ids = encode_ids(tokenizer, text)
            total_tokens += len(ids)
            if unk_id is not None and unk_id >= 0:
                unk_tokens += sum(1 for i in ids if i == unk_id)

    fertility = (total_tokens / total_words) if total_words else 0.0
    token_word_ratio = fertility
    tokens_per_char = (total_tokens / total_chars) if total_chars else 0.0
    unk_rate = (unk_tokens / total_tokens) if total_tokens else 0.0

    return {
        "docs": total_docs,
        "tokens": total_tokens,
        "words": total_words,
        "chars": total_chars,
        "fertility": fertility,
        "token_word_ratio": token_word_ratio,
        "tokens_per_char": tokens_per_char,
        "unk_rate": unk_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HF model tokenizers")
    parser.add_argument(
        "--eval-dir",
        default="data/eval",
        help="Directory with eval_*.jsonl files",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated list of HF model repo IDs (optional)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/eval_results_hf",
        help="Output directory for JSON/CSV results",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom tokenizer code from model repos",
    )
    parser.add_argument(
        "--use-slow",
        action="store_true",
        help="Use slow tokenizers instead of fast ones",
    )
    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_list:
        model_list = DEFAULT_MODELS

    eval_files = list_eval_files(args.eval_dir)
    if not eval_files:
        raise SystemExit("No eval_*.jsonl files found in eval dir")

    os.makedirs(args.output_dir, exist_ok=True)
    results: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []

    for repo_id in model_list:
        logger.info("Loading tokenizer: %s", repo_id)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                trust_remote_code=args.trust_remote_code,
                use_fast=not args.use_slow,
            )
        except Exception as exc:
            logger.warning("Failed to load %s: %s", repo_id, exc)
            continue

        model_key = repo_id.replace("/", "__")
        results[model_key] = {}

        for lang, path in eval_files.items():
            metrics = eval_file(tokenizer, path)
            results[model_key][lang] = metrics
            csv_rows.append(
                {
                    "model": model_key,
                    "lang": lang,
                    **metrics,
                }
            )
            logger.info("%s %s fertility=%.4f", model_key, lang, metrics["fertility"])

    if not csv_rows:
        raise SystemExit("No results were produced (all models failed to load?)")

    json_path = Path(args.output_dir) / "tokenizer_eval.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = Path(args.output_dir) / "tokenizer_eval.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    logger.info("Saved eval results to %s and %s", json_path, csv_path)


if __name__ == "__main__":
    main()
