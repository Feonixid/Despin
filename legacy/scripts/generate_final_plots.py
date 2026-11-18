import os
import json
import shutil
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

def generate_all_publication_plots():
    os.makedirs('figures_v2', exist_ok=True)

    with open('results_v2/comprehensive_comparison.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    summary = data['summary']
    methods = [row["Method"] for row in summary]
    accuracies = [row["Accuracy (%)"] for row in summary]
    tokens = [row["Avg Tokens"] for row in summary]
    tes = [row["Token Efficiency Score (TES)"] for row in summary]

    colors = ["#4C72B0", "#55A868", "#E69F00", "#C44E52", "#8B5CF6"]
    markers = ["o", "s", "D", "^", "*"]

    # --- Plot 1: Pareto Frontier ---
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for i, m in enumerate(methods):
        sz = 260 if "DESPIN" in m else 130
        ax.scatter(tokens[i], accuracies[i], color=colors[i % len(colors)], marker=markers[i % len(markers)], s=sz, label=m, zorder=5)
        
        offset_y = 2.0
        offset_x = 0
        if "Passive" in m:
            offset_y = 2.2
            offset_x = 30
        elif "Best-of-2" in m:
            offset_y = -3.0
            offset_x = -30
        elif "Greedy" in m:
            offset_y = -3.0
            offset_x = 0
        elif "Sampling" in m:
            offset_y = 2.2
            offset_x = -35
        elif "DESPIN" in m:
            offset_y = 2.2
            offset_x = 35

        ax.annotate(
            m,
            (tokens[i] + offset_x, accuracies[i] + offset_y),
            fontsize=9,
            ha="center",
            weight="bold" if "DESPIN" in m else "normal"
        )

    ax.set_xlabel("Average Generated Tokens per Problem (Compute Footprint)")
    ax.set_ylabel("Pass@1 Accuracy (%)")
    ax.set_title("Compute vs. Accuracy Pareto Frontier (DeepSeek-R1-Distill-1.5B)")
    ax.set_ylim(-2, 75)
    ax.set_xlim(0, 2600)
    ax.grid(True, linestyle="--", alpha=0.5)

    out1 = 'figures_v2/fig1_pareto_v2.png'
    plt.tight_layout()
    plt.savefig(out1, dpi=300)
    plt.close()
    print(f"Saved {out1}")

    # --- Plot 2: Token Efficiency Bar Chart ---
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.barh(methods, tes, color=colors, alpha=0.88, edgecolor="black")
    ax.set_xlabel("Token Efficiency Score: TES = Pass@1 / (Tokens / 1000)")
    ax.set_title("Token Efficiency Comparison across Inference Strategies")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            f"{w:.2f}",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9.5,
            weight="bold"
        )

    out2 = 'figures_v2/fig2_tes_v2.png'
    plt.tight_layout()
    plt.savefig(out2, dpi=300)
    plt.close()
    print(f"Saved {out2}")

    # Copy to artifact directory
    artifact_dir = "C:/Users/User/.gemini/antigravity/brain/275d4ee5-d34a-4a1f-acdc-dcbf60c53518/figures_v2"
    os.makedirs(artifact_dir, exist_ok=True)
    shutil.copy(out1, os.path.join(artifact_dir, "fig1_pareto_v2.png"))
    shutil.copy(out2, os.path.join(artifact_dir, "fig2_tes_v2.png"))
    print(f"Copied plots to {artifact_dir}")

if __name__ == '__main__':
    generate_all_publication_plots()
