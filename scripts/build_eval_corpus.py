#!/usr/bin/env python3
"""Build fixed evaluation corpora for tokenizer evaluation."""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import yaml
from datasets import load_dataset

# Ensure repo root on path
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


def iter_source_rows(source: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    repo_id = source["id"]
    split = source.get("split", "train")
    config = source.get("config")
    if config:
        ds = load_dataset(repo_id, name=config, split=split, streaming=True)
    else:
        ds = load_dataset(repo_id, split=split, streaming=True)
    for row in ds:
        yield row


def allocate_targets(total: int, sources: List[Dict[str, Any]]) -> List[int]:
    weights = [float(src.get("weight", 0.0)) for src in sources]
    if any(w > 0 for w in weights):
        weight_sum = sum(weights)
        targets = [int(total * (w / weight_sum)) for w in weights]
    else:
        per = int(total / max(len(sources), 1))
        targets = [per for _ in sources]
    # adjust remainder to last source
    if targets:
        remainder = total - sum(targets)
        targets[-1] += remainder
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tokenizer eval corpus")
    parser.add_argument(
        "--config",
        default="configs/eval_sources.yml",
        help="YAML file with eval sources",
    )
    parser.add_argument("--rows-per-lang", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = args.output_dir or cfg.get("output_dir", "data/eval")
    rows_per_lang = args.rows_per_lang or int(cfg.get("rows_per_lang", 5000))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))

    global_filters = None
    if isinstance(cfg.get("filters"), dict):
        global_filters = FilterSpec.from_dict(cfg["filters"])

    os.makedirs(output_dir, exist_ok=True)
    manifest: Dict[str, Any] = {
        "rows_per_lang": rows_per_lang,
        "output_dir": output_dir,
        "languages": {},
    }

    for lang_entry in cfg.get("languages", []):
        lang = lang_entry["lang"]
        sources = lang_entry.get("sources", [])
        if not sources:
            logger.warning("No sources for lang %s", lang)
            continue

        total_rows = int(lang_entry.get("rows", rows_per_lang))
        targets = allocate_targets(total_rows, sources)
        rng = random.Random(seed)
        out_path = Path(output_dir) / f"eval_{lang}.jsonl"
        kept_total = 0
        per_source_counts: Dict[str, int] = {}

        with open(out_path, "w", encoding="utf-8") as out_f:
            for src, target in zip(sources, targets):
                if target <= 0:
                    continue
                source_id = src["id"]
                source_key = source_id
                if src.get("config"):
                    source_key = f"{source_id}:{src['config']}"
                if src.get("split"):
                    source_key = f"{source_key}:{src['split']}"

                filter_spec = global_filters
                if isinstance(src.get("filters"), dict):
                    filter_spec = (
                        filter_spec.merge(src["filters"]) if filter_spec else FilterSpec.from_dict(src["filters"])
                    )

                count = 0
                for row_idx, row in enumerate(iter_source_rows(src)):
                    text_raw = get_field_value(row, src.get("field", "text"))
                    if not isinstance(text_raw, str):
                        continue
                    text_norm = normalize_text(text_raw)
                    if not text_norm:
                        continue
                    if filter_spec and not passes_quality(text_norm, filter_spec):
                        continue

                    doc_id = None
                    if src.get("doc_id_field"):
                        doc_id = get_field_value(row, src.get("doc_id_field"))
                    if not doc_id:
                        doc_id = f"{source_key}:{row_idx}"

                    url = None
                    if src.get("url_field"):
                        url = get_field_value(row, src.get("url_field"))

                    out_f.write(
                        json.dumps(
                            {
                                "text": text_norm,
                                "source": source_key,
                                "lang": lang,
                                "doc_id": str(doc_id),
                                "url": str(url) if url is not None else None,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    count += 1
                    kept_total += 1
                    if count >= target:
                        break

                per_source_counts[source_key] = count
                logger.info("Lang %s: %s rows from %s", lang, count, source_key)

        manifest["languages"][lang] = {
            "rows": kept_total,
            "sources": per_source_counts,
            "output": str(out_path),
        }

    manifest_path = Path(output_dir) / "eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
