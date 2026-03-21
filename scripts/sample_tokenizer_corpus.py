#!/usr/bin/env python3
"""
Weighted tokenizer-corpus sampler.

Streams from a shortlist of HF datasets, applies quality filters, and writes
JSONL until per-source token budgets are met.

Output JSONL schema (one doc per line):
  text, source, lang, doc_id, url, tokens, row_id
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Pattern

import yaml
from datasets import load_dataset
from datasets import Dataset, Features, Value
from huggingface_hub import HfApi, get_token, login

# Ensure project root is on sys.path for scripts.* imports
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
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def make_doc_id(source_key: str, row_id: int, text_norm: str) -> str:
    raw = f"{source_key}|{row_id}|{text_norm}"
    return hashlib.blake2b(raw.encode("utf-8", errors="ignore"), digest_size=16).hexdigest()


def load_encoder(name: str):
    if name == "tiktoken":
        try:
            import tiktoken
        except Exception as exc:  # pragma: no cover
            raise SystemExit(
                "tiktoken not available. Install with: pip install tiktoken"
            ) from exc
        return tiktoken.get_encoding("cl100k_base")
    raise ValueError(f"Unsupported token estimator: {name}")


def encode_len(encoder, text: str) -> int:
    try:
        return len(encoder.encode(text, disallowed_special=()))
    except Exception:
        return 0


@dataclass
class SourcePlan:
    source_id: str
    source_key: str
    kind: str
    split: str
    config: Optional[str]
    field: Any
    url_field: Optional[Any]
    doc_id_field: Optional[Any]
    lang: str
    target_tokens: int
    weight: float
    explicit_target: bool
    filters: Optional[FilterSpec]
    shuffle_buffer: int


def build_sources(
    cfg: Dict[str, Any], total_target: int, scale_targets: bool = False
) -> List[SourcePlan]:
    sources_cfg = cfg.get("sources") or []
    if not sources_cfg:
        raise SystemExit("No sources defined in config")

    default_filters = None
    if isinstance(cfg.get("filters"), dict):
        default_filters = FilterSpec.from_dict(cfg["filters"])

    if scale_targets:
        base_target_sum = 0
        missing_targets = False
        for src in sources_cfg:
            if src.get("target_tokens") is None:
                missing_targets = True
                continue
            base_target_sum += int(src.get("target_tokens"))
        if missing_targets or base_target_sum <= 0:
            logger.warning(
                "scale-targets enabled but some sources lack target_tokens; "
                "falling back to weight-based allocation"
            )
            scale_targets = False

    plans: List[SourcePlan] = []
    for src in sources_cfg:
        source_id = src.get("id")
        kind = src.get("type", "hf")
        split = src.get("split", "train")
        config = src.get("config")
        field = src.get("field", "text")
        url_field = src.get("url_field")
        doc_id_field = src.get("doc_id_field")
        lang = src.get("lang", "unknown")
        shuffle_buffer = int(src.get("shuffle_buffer", 10000))
        weight = float(src.get("weight", 0.0))

        if not source_id:
            continue

        if scale_targets and src.get("target_tokens") is not None:
            target_tokens = int(
                total_target * (int(src.get("target_tokens")) / base_target_sum)
            )
            explicit_target = True
        elif src.get("target_tokens") is not None:
            target_tokens = int(src.get("target_tokens"))
            explicit_target = True
        else:
            target_tokens = 0
            explicit_target = False

        if src.get("max_tokens") is not None:
            target_tokens = min(target_tokens, int(src["max_tokens"]))

        filters = default_filters
        if isinstance(src.get("filters"), dict):
            filters = filters.merge(src["filters"]) if filters else FilterSpec.from_dict(
                src["filters"]
            )

        source_key = f"{source_id}:{config or 'default'}:{split}"
        plans.append(
            SourcePlan(
                source_id=source_id,
                source_key=source_key,
                kind=kind,
                split=split,
                config=config,
                field=field,
                url_field=url_field,
                doc_id_field=doc_id_field,
                lang=lang,
                target_tokens=target_tokens,
                weight=weight,
                explicit_target=explicit_target,
                filters=filters,
                shuffle_buffer=shuffle_buffer,
            )
        )
    return plans


def compile_regex(pattern: Optional[str]) -> Optional[Pattern[str]]:
    if not pattern:
        return None
    return re.compile(pattern)


def filter_sources(
    plans: List[SourcePlan],
    include_re: Optional[Pattern[str]],
    exclude_re: Optional[Pattern[str]],
    include_sources: Optional[List[str]] = None,
) -> List[SourcePlan]:
    include_set = {s.strip() for s in include_sources or [] if s.strip()}
    if not include_re and not exclude_re and not include_set:
        return plans
    kept: List[SourcePlan] = []
    for plan in plans:
        key = plan.source_key
        if include_set:
            if plan.source_id not in include_set and key not in include_set:
                continue
        if include_re and not include_re.search(key):
            continue
        if exclude_re and exclude_re.search(key):
            continue
        kept.append(plan)
    return kept


def load_checkpoint(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {"sources": {}}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {"sources": {}}


def write_checkpoint(path: Optional[str], data: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def iter_hf_rows(plan: SourcePlan) -> Iterator[Dict[str, Any]]:
    try:
        if plan.config:
            ds = load_dataset(
                plan.source_id, name=plan.config, split=plan.split, streaming=True
            )
        else:
            ds = load_dataset(plan.source_id, split=plan.split, streaming=True)
        if plan.shuffle_buffer and plan.shuffle_buffer > 0:
            try:
                ds = ds.shuffle(buffer_size=plan.shuffle_buffer, seed=42)
            except Exception:
                logger.warning("Shuffle failed for %s; continuing without shuffle", plan.source_key)
        for row in ds:
            yield row
    except Exception as exc:
        logger.warning("Failed to load %s: %s", plan.source_key, exc)
        return


def ensure_repo(api: HfApi, repo_id: str, token: str) -> None:
    try:
        api.repo_info(repo_id, repo_type="dataset", token=token)
    except Exception:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)


def upload_shard(
    api: HfApi,
    repo_id: str,
    token: str,
    rows: List[Dict[str, Any]],
    shard_name: str,
    max_retries: int,
    backoff: float,
    backoff_max: float,
    failed_dir: Optional[str],
) -> None:
    data_dict = {
        "text": [row.get("text") for row in rows],
        "source": [row.get("source") for row in rows],
        "lang": [row.get("lang") for row in rows],
        "doc_id": [row.get("doc_id") for row in rows],
        "url": [row.get("url") for row in rows],
        "tokens": [row.get("tokens") for row in rows],
        "row_id": [row.get("row_id") for row in rows],
    }
    features = Features(
        {
            "text": Value("string"),
            "source": Value("string"),
            "lang": Value("string"),
            "doc_id": Value("string"),
            "url": Value("string"),
            "tokens": Value("int64"),
            "row_id": Value("int64"),
        }
    )
    dataset = Dataset.from_dict(data_dict, features=features)
    os.makedirs("data/tokenizer_shards", exist_ok=True)
    parquet_path = f"data/tokenizer_shards/{shard_name}"
    dataset.to_parquet(parquet_path)

    attempt = 0
    while True:
        try:
            api.upload_file(
                path_or_fileobj=parquet_path,
                path_in_repo=f"data/{shard_name}",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            os.remove(parquet_path)
            return
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                logger.warning(
                    "Upload failed after %s attempts for %s: %s", max_retries, shard_name, exc
                )
                if failed_dir:
                    os.makedirs(failed_dir, exist_ok=True)
                    failed_path = os.path.join(failed_dir, shard_name)
                    os.replace(parquet_path, failed_path)
                    logger.warning("Saved failed shard to %s", failed_path)
                else:
                    os.remove(parquet_path)
                return
            sleep_s = min(backoff * (2 ** (attempt - 1)), backoff_max)
            sleep_s = sleep_s * (0.8 + random.random() * 0.4)
            logger.warning(
                "Upload error for %s (attempt %s/%s). Sleeping %.1fs",
                shard_name,
                attempt,
                max_retries,
                sleep_s,
            )
            time.sleep(sleep_s)


def find_repo_shard_start(api: HfApi, repo_id: str, shard_prefix: str, lang: str) -> int:
    """Find next shard index by scanning existing repo files."""
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as exc:
        logger.warning("Failed to list repo files for %s: %s", repo_id, exc)
        return 0

    pattern = re.compile(
        rf"^{re.escape(shard_prefix)}-{re.escape(lang)}-(\\d+)\\.parquet$"
    )
    max_idx = -1
    for path in files:
        name = os.path.basename(path)
        match = pattern.match(name)
        if not match:
            continue
        try:
            idx = int(match.group(1))
        except ValueError:
            continue
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Weighted tokenizer corpus sampler")
    parser.add_argument("--config", required=True, help="Path to tokenizer source config")
    parser.add_argument("--target-tokens", type=int, help="Override total target tokens")
    parser.add_argument(
        "--scale-targets",
        action="store_true",
        help="Scale per-source target_tokens to match --target-tokens (proportional)",
    )
    parser.add_argument("--output", help="Override output JSONL path")
    parser.add_argument("--stats", help="Override stats JSON path")
    parser.add_argument("--include-regex", help="Only include sources matching regex")
    parser.add_argument("--exclude-regex", help="Exclude sources matching regex")
    parser.add_argument(
        "--include-sources",
        help="Comma-separated source ids (or source_key) to include explicitly",
    )
    parser.add_argument(
        "--equal-split",
        action="store_true",
        help="Allocate target tokens evenly across included sources",
    )
    parser.add_argument(
        "--checkpoint",
        default="data/tokenizer_sampling_checkpoint.json",
        help="Checkpoint JSON path",
    )
    parser.add_argument("--upload-repo", help="Upload shards to HF dataset repo")
    parser.add_argument(
        "--shard-prefix",
        default="tokenizer",
        help="Prefix for shard filenames when uploading",
    )
    parser.add_argument(
        "--shard-rows",
        type=int,
        default=100000,
        help="Rows per shard when uploading (default 100k)",
    )
    parser.add_argument(
        "--max-upload-retries",
        type=int,
        default=8,
        help="Max retries for shard upload before skipping",
    )
    parser.add_argument(
        "--upload-backoff",
        type=float,
        default=5.0,
        help="Initial backoff seconds for upload retries",
    )
    parser.add_argument(
        "--upload-backoff-max",
        type=float,
        default=120.0,
        help="Max backoff seconds for upload retries",
    )
    parser.add_argument(
        "--failed-shard-dir",
        default="data/tokenizer_failed_shards",
        help="Where to store shards that exceeded retry limit",
    )
    parser.add_argument(
        "--skip-local",
        action="store_true",
        help="Do not write local JSONL output (upload-only mode)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    total_target = int(args.target_tokens or cfg.get("target_tokens", 50_000_000_000))
    output_path = args.output or cfg.get("output_path", "data/tokenizer_corpus.jsonl")
    stats_path = args.stats or cfg.get("stats_path", output_path + ".stats.json")

    encoder_name = cfg.get("token_estimator", "tiktoken")
    encoder = load_encoder(encoder_name)

    scale_targets = bool(args.scale_targets or cfg.get("scale_targets", False))
    plans = build_sources(cfg, total_target, scale_targets=scale_targets)
    include_sources = (
        [s.strip() for s in args.include_sources.split(",")]
        if args.include_sources
        else None
    )
    plans = filter_sources(
        plans,
        compile_regex(args.include_regex),
        compile_regex(args.exclude_regex),
        include_sources=include_sources,
    )
    if not plans:
        raise SystemExit("No valid sources resolved from config")

    if args.equal_split:
        per_source = int(total_target / max(len(plans), 1))
        for plan in plans:
            plan.target_tokens = per_source
            plan.explicit_target = True
        scale_targets = False

    # Allocate weight-based targets after filtering to ensure correct proportions
    pending = [p for p in plans if not p.explicit_target]
    if pending:
        explicit_sum = sum(p.target_tokens for p in plans if p.explicit_target)
        remaining = max(total_target - explicit_sum, 0)
        weight_sum = sum(p.weight for p in pending)
        if weight_sum <= 0:
            per_source = int(remaining / max(len(pending), 1))
            for plan in pending:
                plan.target_tokens = per_source
        else:
            for plan in pending:
                plan.target_tokens = int(remaining * (plan.weight / weight_sum))

    if not args.skip_local:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint)
    ck_sources = checkpoint.setdefault("sources", {})
    checkpoint["target_tokens"] = total_target

    total_written = 0
    total_tokens = 0
    stats: Dict[str, Any] = {"sources": {}, "total": {}}

    if args.upload_repo:
        token = get_token() or os.getenv("HF_TOKEN")
        if not token:
            login()
            token = get_token()
        api = HfApi()
        ensure_repo(api, args.upload_repo, token)
    else:
        api = None
        token = None

    out_f = None
    if not args.skip_local:
        out_f = open(output_path, "a", encoding="utf-8")

    shard_index_by_lang: Dict[str, int] = {}
    if args.upload_repo and api:
        existing = checkpoint.get("shard_index_by_lang")
        if isinstance(existing, dict):
            shard_index_by_lang.update(existing)
        for lang in {p.lang for p in plans}:
            if lang not in shard_index_by_lang:
                start_idx = find_repo_shard_start(api, args.upload_repo, args.shard_prefix, lang)
                shard_index_by_lang[lang] = start_idx
        checkpoint["shard_index_by_lang"] = shard_index_by_lang
        write_checkpoint(args.checkpoint, checkpoint)

    try:
        for plan in plans:
            src_stat = ck_sources.get(plan.source_key, {})
            if src_stat.get("done"):
                logger.info("Skipping completed source: %s", plan.source_key)
                stats["sources"][plan.source_key] = src_stat
                continue

            logger.info(
                "Sampling %s target_tokens=%s", plan.source_key, plan.target_tokens
            )
            rows_seen = 0
            kept = 0
            dropped = 0
            empty = 0
            token_sum = 0
            encode_errors = 0
            shard_rows: List[Dict[str, Any]] = []
            shard_tokens = 0
            shard_index = int(shard_index_by_lang.get(plan.lang, 0))

            for row_idx, row in enumerate(iter_hf_rows(plan)):
                rows_seen += 1
                text_raw = get_field_value(row, plan.field)
                if not isinstance(text_raw, str):
                    empty += 1
                    continue
                text_norm = normalize_text(text_raw)
                if not text_norm:
                    empty += 1
                    continue

                if plan.filters and not passes_quality(text_norm, plan.filters):
                    dropped += 1
                    continue

                tok_count = encode_len(encoder, text_norm)
                if tok_count <= 0:
                    encode_errors += 1
                    continue

                doc_id = None
                if plan.doc_id_field:
                    doc_id = get_field_value(row, plan.doc_id_field)
                if not doc_id:
                    doc_id = make_doc_id(plan.source_key, row_idx, text_norm)

                url = None
                if plan.url_field:
                    url = get_field_value(row, plan.url_field)

                out_row = {
                    "text": text_norm,
                    "source": plan.source_key,
                    "lang": plan.lang,
                    "doc_id": str(doc_id),
                    "url": str(url) if url is not None else None,
                    "tokens": tok_count,
                    "row_id": row_idx,
                }
                if out_f:
                    out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                kept += 1
                token_sum += tok_count
                total_written += 1
                total_tokens += tok_count
                shard_rows.append(out_row)
                shard_tokens += tok_count

                if args.upload_repo and shard_rows and len(shard_rows) >= args.shard_rows:
                    shard_name = f"{args.shard_prefix}-{plan.lang}-{shard_index:06d}.parquet"
                    upload_shard(
                        api,
                        args.upload_repo,
                        token,
                        shard_rows,
                        shard_name,
                        args.max_upload_retries,
                        args.upload_backoff,
                        args.upload_backoff_max,
                        args.failed_shard_dir,
                    )
                    shard_index += 1
                    if args.upload_repo:
                        shard_index_by_lang[plan.lang] = shard_index
                        checkpoint["shard_index_by_lang"] = shard_index_by_lang
                        write_checkpoint(args.checkpoint, checkpoint)
                    shard_rows = []
                    shard_tokens = 0

                if token_sum >= plan.target_tokens:
                    break

                if kept % 1000 == 0:
                    if out_f:
                        out_f.flush()
                    ck_sources[plan.source_key] = {
                        "tokens": token_sum,
                        "rows": kept,
                        "rows_seen": rows_seen,
                        "dropped": dropped,
                        "empty": empty,
                        "encode_errors": encode_errors,
                        "kept_rate": (kept / rows_seen) if rows_seen else 0.0,
                        "drop_rate": (dropped / rows_seen) if rows_seen else 0.0,
                        "shard_index": shard_index,
                        "done": False,
                    }
                    write_checkpoint(args.checkpoint, checkpoint)

            if args.upload_repo and shard_rows:
                shard_name = f"{args.shard_prefix}-{plan.lang}-{shard_index:06d}.parquet"
                upload_shard(
                    api,
                    args.upload_repo,
                    token,
                    shard_rows,
                    shard_name,
                    args.max_upload_retries,
                    args.upload_backoff,
                    args.upload_backoff_max,
                    args.failed_shard_dir,
                )
                shard_index += 1
                shard_index_by_lang[plan.lang] = shard_index
                checkpoint["shard_index_by_lang"] = shard_index_by_lang

            src_stat = {
                "tokens": token_sum,
                "rows": kept,
                "rows_seen": rows_seen,
                "dropped": dropped,
                "empty": empty,
                "encode_errors": encode_errors,
                "target_tokens": plan.target_tokens,
                "kept_rate": (kept / rows_seen) if rows_seen else 0.0,
                "drop_rate": (dropped / rows_seen) if rows_seen else 0.0,
                "shard_index": shard_index,
                "done": True,
            }
            ck_sources[plan.source_key] = src_stat
            stats["sources"][plan.source_key] = src_stat
            write_checkpoint(args.checkpoint, checkpoint)

    finally:
        if out_f:
            out_f.close()

    stats["total"] = {
        "tokens": total_tokens,
        "rows": total_written,
        "target_tokens": total_target,
    }
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("Sampling complete. Wrote %s rows, %s tokens.", total_written, total_tokens)
    logger.info("Stats saved to %s", stats_path)


if __name__ == "__main__":
    main()
