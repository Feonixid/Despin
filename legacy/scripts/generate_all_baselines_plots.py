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
    "legend.fontsize": 9.5,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

def generate_all():
    os.makedirs('figures_v2', exist_ok=True)
    with open('results_v2/massive_sweep_all_baselines.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    summary = data['summary']
    methods = [r["Method"] for r in summary]
    accuracies = [r["Pass@1 Acc (%)"] for r in summary]
    tokens = [r["Avg Tokens"] for r in summary]
    tes = [r["Token Efficiency Score (TES)"] for r in summary]

    colors = ["#4C72B0", "#55A868", "#D97706", "#DC2626", "#8B5CF6"]
    markers = ["o", "s", "D", "^", "*"]

    # --- Plot 1: Pareto Frontier across all 5 baselines ---
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for i, m in enumerate(methods):
        sz = 300 if "DESPIN" in m else 150
        ax.scatter(tokens[i], accuracies[i], color=colors[i], marker=markers[i], s=sz, label=m, zorder=5)

        offset_y = 1.8
        offset_x = 0
        if "Early-Exit" in m:
            offset_y = 2.5
            offset_x = 40
        elif "Repetition" in m:
            offset_y = -3.2
            offset_x = 0
        elif "Sampling" in m:
            offset_y = 2.2
            offset_x = 0
        elif "Greedy" in m:
            offset_y = -3.0
            offset_x = -25
        elif "DESPIN" in m:
            offset_y = 2.0
            offset_x = -35

        ax.annotate(
            m,
            (tokens[i] + offset_x, accuracies[i] + offset_y),
            fontsize=8.5,
            ha="center",
            weight="bold" if "DESPIN" in m else "normal"
        )

    ax.set_xlabel("Average Generated Tokens per Problem (Compute Footprint)")
    ax.set_ylabel("Pass@1 Accuracy (%)")
    ax.set_title("Compute vs. Accuracy Pareto Frontier across 5 Baselines (N = 60)")
    ax.set_ylim(0, 76)
    ax.set_xlim(100, 1350)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    out1 = 'figures_v2/fig1_pareto_all_baselines.png'
    plt.tight_layout()
    plt.savefig(out1, dpi=300)
    plt.close()
    print(f"Saved {out1}")

    # --- Plot 2: Token Efficiency Bar Chart ---
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    bars = ax.barh(methods, tes, color=colors, alpha=0.88, edgecolor="black")
    ax.set_xlabel("Token Efficiency Score: TES = Pass@1 / (Tokens / 1000)")
    ax.set_title("Token Efficiency Score (TES) Comparison across 5 Decoding Strategies (N = 60)")
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

    out2 = 'figures_v2/fig2_tes_all_baselines.png'
    plt.tight_layout()
    plt.savefig(out2, dpi=300)
    plt.close()
    print(f"Saved {out2}")

    # Copy to artifact directory
    artifact_dir = "C:/Users/User/.gemini/antigravity/brain/275d4ee5-d34a-4a1f-acdc-dcbf60c53518/figures_v2"
    os.makedirs(artifact_dir, exist_ok=True)
    shutil.copy(out1, os.path.join(artifact_dir, "fig1_pareto_all_baselines.png"))
    shutil.copy(out2, os.path.join(artifact_dir, "fig2_tes_all_baselines.png"))
    print(f"Copied plots to {artifact_dir}")

if __name__ == '__main__':
    generate_all()

