#!/usr/bin/env python3
"""
scripts/push_v2_to_hf.py

Uploads Nep-Tokenizer v2 (64k Native Hugging Face BPE) to Hugging Face Hub:
  Target: DOAIxILPRL-KU/nep-tokenizer

Maintains a clean 2-commit history (initial commit + v2 release):
- tokenizer.json (Native Rust BPE)
- tokenizer_config.json
- special_tokens_map.json
- metadata.json
- README.md (Comprehensive model card & benchmark leaderboard)
- figures/ (Publication figures 1, 2, 3)
- .gitattributes
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_REPO = "DOAIxILPRL-KU/nep-tokenizer"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                HF_TOKEN = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set. Please set it in .env or export HF_TOKEN.")


def main():
    print(f"Staging and pushing Nep-Tokenizer v2 (64k) to https://huggingface.co/{HF_REPO}...")

    with tempfile.TemporaryDirectory() as tmpdir:
        clone_url = f"https://user:{HF_TOKEN}@huggingface.co/{HF_REPO}"
        repo_dir = Path(tmpdir) / "repo"

        # 1. Clone remote repo
        print("Cloning repository...")
        subprocess.run(["git", "clone", clone_url, str(repo_dir)], check=True)

        # 2. Reset to initial commit (HEAD~1) to maintain exactly 2 commits
        subprocess.run(["git", "reset", "--hard", "HEAD~1"], cwd=repo_dir, check=True)

        # 3. Clean working tree (except .git)
        for item in repo_dir.iterdir():
            if item.name == ".git":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        # 4. Copy 64k native HF files from hf_push
        hf_push = REPO_ROOT / "hf_push"
        shutil.copy(hf_push / "64k" / "tokenizer.json", repo_dir / "tokenizer.json")
        shutil.copy(hf_push / "64k" / "tokenizer_config.json", repo_dir / "tokenizer_config.json")
        shutil.copy(hf_push / "64k" / "special_tokens_map.json", repo_dir / "special_tokens_map.json")
        shutil.copy(hf_push / "64k" / "metadata.json", repo_dir / "metadata.json")
        shutil.copy(hf_push / "README.md", repo_dir / "README.md")

        # 5. Copy figures
        fig_dest = repo_dir / "figures"
        fig_dest.mkdir(parents=True, exist_ok=True)
        for f in ["fig1_vocab_scaling_fertility.png", "fig2_domain_fertility_comparison.png", "fig3_matra_cohesion_comparison.png"]:
            shutil.copy(REPO_ROOT / "figures" / f, fig_dest / f)

        # 6. Gitattributes for LFS
        (repo_dir / ".gitattributes").write_text("*.png filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")

        # 7. Git add and commit
        subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True)
        commit_msg = (
            "Release Nep-Tokenizer v2 (64k): Native Hugging Face BPE with Mark-Aware Pre-tokenization "
            "and 100% Devanagari Block Coverage\n\n"
            "- Architecture: Native Rust BPE (PreTrainedTokenizerFast)\n"
            "- Pre-tokenization: Mark-Aware Regex [\\p{L}\\p{M}]+ & Digit Grouping [0-9०-९]{1,3}\n"
            "- Full 17-Model Benchmark Evaluation and Publication Figures\n"
            "- Affiliation: DOAI & ILPRL, Kathmandu University"
        )
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=repo_dir, check=True)

        # 8. Force push to maintain clean 2-commit history
        print("Force-pushing clean release to Hugging Face...")
        subprocess.run(["git", "push", "--force", "origin", "main"], cwd=repo_dir, check=True)
        print("Successfully published to Hugging Face Hub!")


if __name__ == "__main__":
    main()
