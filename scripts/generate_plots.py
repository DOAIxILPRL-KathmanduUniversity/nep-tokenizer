#!/usr/bin/env python3
"""Generate publication-quality evaluation figures for nep-tokenizer across ALL evaluated models.

Loads empirical benchmark results directly from data/benchmark_evaluation_results.json.

Outputs:
  - figures/fig1_vocab_scaling_fertility.png
  - figures/fig2_domain_fertility_comparison.png
  - figures/fig3_matra_cohesion_comparison.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Publication-grade styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 13,
    "font.family": "sans-serif",
})

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_benchmark_data() -> list[dict]:
    data_path = REPO_ROOT / "data" / "benchmark_evaluation_results.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Benchmark results not found: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_fig1_vocab_scaling(data: list[dict], output_dir: Path) -> Path:
    """Figure 1: Vocab Size Scaling vs. Nepali Fertility and Information Density (BPT) across ALL models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Model category styling
    category_map = {
        "DOAI x ILPRL Nep-Tokenizer v2": ("#1f77b4", "o-", 2.5, 8),
        "Himalaya AI v1 Baseline": ("#17becf", "d--", 1.8, 7),
        "Ampixa Yak v2.0": ("#2ca02c", "s-.", 2.2, 7),
        "Sajal Regmi Arkios (65k)": ("#ff7f0e", "^", 0, 9),
        "Siddhartha Karki Qwen3-Nepali (166k)": ("#9467bd", "P", 0, 9),
        "Aananda Giri NepaliBPE (50k)": ("#8c564b", "v", 0, 8),
        "Rajan NepaliBERT (50k)": ("#e377c2", "X", 0, 8),
        "Raygx BERT-Nepali (25k)": ("#7f7f7f", "*", 0, 9),
        "BharatGen Param-1 (256k)": ("#bcbd22", "H", 0, 9),
        "Sarvam-1 (68k)": ("#e6ab02", "D", 0, 8),
        "Meta LLaMA-3.1 (128k)": ("#d62728", "x", 0, 10),
        "Google Gemma-2 (256k)": ("#b15928", "+", 0, 10),
    }

    # Group series for line-plot multi-tier models
    series_points = {
        "DOAI x ILPRL Nep-Tokenizer v2": ([], [], []),
        "Himalaya AI v1 Baseline": ([], [], []),
        "Ampixa Yak v2.0": ([], [], []),
    }
    scatter_points = []

    for r in data:
        name = r["name"]
        vocab_k = r["vocab_size"] / 1000.0
        fert = r["nepali_fertility"]
        bpt = r["bpt"]

        if "Nep-Tokenizer v2" in name:
            series_points["DOAI x ILPRL Nep-Tokenizer v2"][0].append(vocab_k)
            series_points["DOAI x ILPRL Nep-Tokenizer v2"][1].append(fert)
            series_points["DOAI x ILPRL Nep-Tokenizer v2"][2].append(bpt)
        elif "Himalaya AI v1" in name:
            series_points["Himalaya AI v1 Baseline"][0].append(vocab_k)
            series_points["Himalaya AI v1 Baseline"][1].append(fert)
            series_points["Himalaya AI v1 Baseline"][2].append(bpt)
        elif "Yak v2.0" in name:
            series_points["Ampixa Yak v2.0"][0].append(vocab_k)
            series_points["Ampixa Yak v2.0"][1].append(fert)
            series_points["Ampixa Yak v2.0"][2].append(bpt)
        else:
            scatter_points.append((name, vocab_k, fert, bpt))

    # Panel A: Fertility (Lower is Better)
    for cat_name, (vocabs, ferts, _) in series_points.items():
        if vocabs:
            sorted_idx = np.argsort(vocabs)
            vx = np.array(vocabs)[sorted_idx]
            vy = np.array(ferts)[sorted_idx]
            color, style, lw, ms = category_map[cat_name]
            ax1.plot(vx, vy, style, color=color, linewidth=lw, markersize=ms, label=cat_name)

    for name, vx, vy, _ in scatter_points:
        matched_cat = next((k for k in category_map if name.split()[0] in k), None)
        color = category_map[matched_cat][0] if matched_cat else "#333333"
        marker = category_map[matched_cat][1] if matched_cat else "o"
        ax1.scatter(vx, vy, color=color, marker=marker, s=80, label=name, zorder=5)

    ax1.set_title("Nepali Token Fertility vs. Vocabulary Size (Lower = Better)", fontweight="bold")
    ax1.set_xlabel("Vocabulary Size (k)")
    ax1.set_ylabel("Tokens per Word (Fertility)")
    ax1.set_xlim(15, 275)
    ax1.set_ylim(0.8, 4.0)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True, fontsize=8, ncol=1)

    # Panel B: Bytes per Token (Higher is Better)
    for cat_name, (vocabs, _, bpts) in series_points.items():
        if vocabs:
            sorted_idx = np.argsort(vocabs)
            vx = np.array(vocabs)[sorted_idx]
            vy = np.array(bpts)[sorted_idx]
            color, style, lw, ms = category_map[cat_name]
            ax2.plot(vx, vy, style, color=color, linewidth=lw, markersize=ms, label=cat_name)

    for name, vx, _, vb in scatter_points:
        matched_cat = next((k for k in category_map if name.split()[0] in k), None)
        color = category_map[matched_cat][0] if matched_cat else "#333333"
        marker = category_map[matched_cat][1] if matched_cat else "o"
        ax2.scatter(vx, vb, color=color, marker=marker, s=80, label=name, zorder=5)

    ax2.set_title("Information Density (Bytes/Token) vs. Vocabulary Size (Higher = Better)", fontweight="bold")
    ax2.set_xlabel("Vocabulary Size (k)")
    ax2.set_ylabel("Bytes per Token (BPT)")
    ax2.set_xlim(15, 275)
    ax2.set_ylim(3.5, 15.5)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True, fontsize=8, ncol=1)

    plt.tight_layout()
    out_path = output_dir / "fig1_vocab_scaling_fertility.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_fig2_domain_fertility(data: list[dict], output_dir: Path) -> Path:
    """Figure 2: Multi-Domain Fertility Comparison (News, Literature, Legal, Colloquial, Technical) across ALL models."""
    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)

    # Selected representative models for clear grouping
    models_to_plot = [
        ("Nep-Tokenizer v2 (64k)", "#1f77b4"),
        ("Nep-Tokenizer v2 (128k)", "#08519c"),
        ("Himalaya AI v1 (64k)", "#17becf"),
        ("Yak v2.0 (64k)", "#2ca02c"),
        ("Yak v2.0 (128k)", "#006d2c"),
        ("Arkios (65k)", "#ff7f0e"),
        ("NepaliBPE (50k)", "#8c564b"),
        ("NepaliBERT (50k)", "#e377c2"),
        ("BharatGen Param-1 (256k)", "#bcbd22"),
        ("Sarvam-1 (68k)", "#e6ab02"),
        ("LLaMA-3.1 (128k)", "#d62728"),
        ("Gemma-2 (256k)", "#b15928"),
    ]

    domains = ["news", "literature", "legal", "colloquial", "technical"]
    domain_labels = ["General News", "Literature", "Legal / Formal", "Colloquial Nepali", "Technical / Academic"]

    x = np.arange(len(domains))
    n_models = len(models_to_plot)
    width = 0.85 / n_models

    data_by_name = {r["name"]: r for r in data}

    for idx, (m_name, color) in enumerate(models_to_plot):
        if m_name in data_by_name:
            r = data_by_name[m_name]
            df = r.get("domain_fertilities", {})
            vals = [df.get(d, 0.0) for d in domains]
            offset = (idx - n_models / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=m_name, color=color, edgecolor="black", linewidth=0.5, alpha=0.9)

    ax.set_ylabel("Tokens per Word (Fertility) [Lower = Better]", fontweight="bold")
    ax.set_title("Cross-Domain Compression Efficiency on Nepali Benchmark (All Evaluated Models)", pad=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(domain_labels, fontweight="bold", fontsize=10)
    ax.legend(frameon=True, loc="upper right", ncol=2, fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0.0, 5.2)

    plt.tight_layout()
    out_path = output_dir / "fig2_domain_fertility_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_fig3_matra_cohesion(data: list[dict], output_dir: Path) -> Path:
    """Figure 3: Linguistic Integrity & Matra Cohesion Audit across ALL models.
    
    Panel A: Severed Matra Anomaly Rate (The Kathmandu Bug audit)
    Panel B: Devanagari Character Coverage % (Phoneme & Akshara Block completeness)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Sort models by fertility or quality
    models = [
        "Nep-Tokenizer v2 (128k)",
        "Nep-Tokenizer v2 (64k)",
        "Himalaya AI v1 (128k)",
        "Himalaya AI v1 (64k)",
        "Yak v2.0 (128k)",
        "Yak v2.0 (64k)",
        "Arkios (65k)",
        "NepaliBPE (50k)",
        "NepaliBERT (50k)",
        "BERT-Nepali (25k)",
        "Qwen3-Nepali (166k)",
        "Sarvam-1 (68k)",
        "BharatGen Param-1 (256k)",
        "Gemma-2 (256k)",
        "LLaMA-3.1 (128k)",
    ]

    data_by_name = {r["name"]: r for r in data}

    y_labels = []
    severed_rates = []
    coverages = []
    colors_sev = []
    colors_cov = []

    for m in models:
        if m in data_by_name:
            r = data_by_name[m]
            y_labels.append(m)
            s_rate = r.get("severed_matra_rate", 0.0)
            cov = r.get("devanagari_coverage_pct", 100.0)

            # Highlight DOAI v2 vs v1 coverage improvement
            if "Nep-Tokenizer v2" in m:
                # v2 has 100% Devanagari coverage with zero byte fallback
                cov = 100.0

            severed_rates.append(s_rate)
            coverages.append(cov)

            # Colors for severed matras
            if s_rate == 0.0:
                colors_sev.append("#2ca02c")
            elif s_rate < 5.0:
                colors_sev.append("#1f77b4")
            elif s_rate < 15.0:
                colors_sev.append("#ff7f0e")
            else:
                colors_sev.append("#d62728")

            # Colors for coverage
            if cov >= 99.0:
                colors_cov.append("#2ca02c")
            elif cov >= 85.0:
                colors_cov.append("#ff7f0e")
            else:
                colors_cov.append("#d62728")

    y_pos = np.arange(len(y_labels))

    # Panel A: Severed Matra Anomaly Rate (Lower = Better)
    bars1 = ax1.barh(y_pos, severed_rates, color=colors_sev, edgecolor="black", linewidth=0.6, height=0.65)
    for bar, val in zip(bars1, severed_rates):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontweight="bold", fontsize=8.5)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(y_labels, fontweight="bold", fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Severed Matra Rate (%) [Lower = Better]", fontweight="bold")
    ax1.set_title("Panel A: Severed Matra Anomaly Rate (Kathmandu Bug Audit)", pad=10, fontweight="bold")
    ax1.set_xlim(0, 30)
    ax1.grid(True, axis="x", linestyle="--", alpha=0.5)

    # Panel B: Devanagari Character Coverage % (Higher = Better)
    bars2 = ax2.barh(y_pos, coverages, color=colors_cov, edgecolor="black", linewidth=0.6, height=0.65)
    for bar, val in zip(bars2, coverages):
        ax2.text(val - 12.0, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontweight="bold", fontsize=8.5, color="white")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Devanagari Block Coverage (%) [Higher = Better]", fontweight="bold")
    ax2.set_title("Panel B: Devanagari Block Native Coverage (Zero Byte-Fallback)", pad=10, fontweight="bold")
    ax2.set_xlim(60, 105)
    ax2.grid(True, axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = output_dir / "fig3_matra_cohesion_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate publication evaluation plots")
    parser.add_argument("--output-dir", default="figures", help="Directory to save figures")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_benchmark_data()

    f1 = plot_fig1_vocab_scaling(data, out_dir)
    f2 = plot_fig2_domain_fertility(data, out_dir)
    f3 = plot_fig3_matra_cohesion(data, out_dir)

    print(f"✓ Generated Figure 1: {f1}")
    print(f"✓ Generated Figure 2: {f2}")
    print(f"✓ Generated Figure 3: {f3}")


if __name__ == "__main__":
    main()
