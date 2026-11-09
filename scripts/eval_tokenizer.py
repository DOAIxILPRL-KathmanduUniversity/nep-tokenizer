#!/usr/bin/env python3
"""Evaluate SentencePiece tokenizer fertility and token-to-word ratios."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    import sentencepiece as spm
except Exception as exc:  # pragma: no cover
    raise SystemExit("sentencepiece is required. Install via pip install sentencepiece") from exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def list_eval_files(eval_dir: str) -> Dict[str, Path]:
    eval_path = Path(eval_dir)
    files = {}
    for path in eval_path.glob("eval_*.jsonl"):
        lang = path.stem.replace("eval_", "")
        files[lang] = path
    return files


def load_models(models: List[str], models_dir: str | None) -> List[Path]:
    result: List[Path] = []
    if models_dir:
        for path in Path(models_dir).glob("*.model"):
            result.append(path)
    for item in models:
        path = Path(item)
        if path.is_dir():
            result.extend(path.glob("*.model"))
        else:
            result.append(path)
    return result


def resolve_model_key(model_path: Path) -> str:
    model_key = model_path.stem
    meta_path = model_path.with_suffix(".json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            corpus_tag = meta.get("corpus_tag")
            vocab_size = meta.get("vocab_size")
            source_repo = meta.get("source_repo")
            if corpus_tag and vocab_size:
                return f"nep-tokenizer-{corpus_tag}-{int(vocab_size)}"
            if source_repo == "himalaya-ai/nepali-tokenizer-corpus" and vocab_size:
                return f"nep-tokenizer-5b-{int(vocab_size)}"
        except Exception:
            pass
    return model_key


def eval_file(sp: spm.SentencePieceProcessor, path: Path) -> Dict[str, Any]:
    total_tokens = 0
    total_words = 0
    total_chars = 0
    total_bytes = 0
    total_docs = 0
    unk_tokens = 0
    unk_id = sp.unk_id()

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
            total_bytes += len(text.encode("utf-8"))
            words = len(text.split())
            total_words += words
            ids = sp.encode(text, out_type=int)
            total_tokens += len(ids)
            if unk_id >= 0:
                unk_tokens += sum(1 for i in ids if i == unk_id)

    fertility = (total_tokens / total_words) if total_words else 0.0
    token_word_ratio = fertility
    tokens_per_char = (total_tokens / total_chars) if total_chars else 0.0
    bytes_per_token = (total_bytes / total_tokens) if total_tokens else 0.0
    unk_rate = (unk_tokens / total_tokens) if total_tokens else 0.0

    return {
        "docs": total_docs,
        "tokens": total_tokens,
        "words": total_words,
        "chars": total_chars,
        "bytes": total_bytes,
        "fertility": fertility,
        "token_word_ratio": token_word_ratio,
        "tokens_per_char": tokens_per_char,
        "bytes_per_token": bytes_per_token,
        "unk_rate": unk_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SentencePiece tokenizer")
    parser.add_argument("--eval-dir", default="data/eval")
    parser.add_argument("--models-dir", help="Directory containing .model files")
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated list of .model files (optional)",
    )
    parser.add_argument("--output-dir", default="data/eval_results")
    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    model_paths = load_models(model_list, args.models_dir)
    if not model_paths:
        raise SystemExit("No models found. Use --models or --models-dir.")

    eval_files = list_eval_files(args.eval_dir)
    if not eval_files:
        raise SystemExit("No eval_*.jsonl files found in eval dir")

    os.makedirs(args.output_dir, exist_ok=True)
    results: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []

    for model_path in model_paths:
        sp = spm.SentencePieceProcessor()
        sp.load(str(model_path))
        model_key = resolve_model_key(model_path)
        results[model_key] = {}

        for lang, path in eval_files.items():
            metrics = eval_file(sp, path)
            results[model_key][lang] = metrics
            csv_rows.append(
                {
                    "model": model_key,
                    "lang": lang,
                    **metrics,
                }
            )
            logger.info("%s %s fertility=%.4f", model_key, lang, metrics["fertility"])

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
