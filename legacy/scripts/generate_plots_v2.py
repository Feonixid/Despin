import os
import json
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


def plot_benchmark_v2(results_path: str = "results_v2/benchmark_v2_results.json", out_dir: str = "figures_v2"):
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(results_path):
        print(f"Results file {results_path} not found.")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data["summary"]
    methods = [row["Method"] for row in summary]
    accuracies = [row["Accuracy (%)"] for row in summary]
    tokens = [row["Avg Tokens"] for row in summary]
    tes = [row["Token Efficiency Score (TES)"] for row in summary]

    # Figure 1: Pareto Curve (Accuracy vs Tokens)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    colors = ["#4C72B0", "#55A868", "#E69F00", "#C44E52", "#8172D5"]
    markers = ["o", "s", "D", "^", "*"]

    for i, m in enumerate(methods):
        sz = 200 if "DESPIN" in m else 120
        ax.scatter(tokens[i], accuracies[i], color=colors[i % len(colors)], marker=markers[i % len(markers)], s=sz, label=m, zorder=5)
        offset_y = 1.2 if "DESPIN" in m else -1.8
        ax.annotate(m, (tokens[i], accuracies[i] + offset_y), fontsize=8.5, ha="center", weight="bold" if "DESPIN" in m else "normal")

    ax.set_xlabel("Average Generated Tokens per Problem (Compute Footprint)")
    ax.set_ylabel("Pass@1 Accuracy (%)")
    ax.set_title("Compute vs. Accuracy Pareto Frontier (MATH-500 & GSM8K)")
    ax.grid(True, linestyle="--", alpha=0.5)

    out_path1 = os.path.join(out_dir, "fig1_pareto_v2.png")
    plt.tight_layout()
    plt.savefig(out_path1, dpi=300)
    plt.close()
    print(f"Saved {out_path1}")

    # Figure 2: Token Efficiency Score (TES)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.barh(methods, tes, color=colors[:len(methods)], alpha=0.85, edgecolor="black")
    ax.set_xlabel("Token Efficiency Score (Accuracy / [Tokens/1000])")
    ax.set_title("Token Efficiency Comparison (Higher is Better)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.2f}", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9, weight="bold")

    out_path2 = os.path.join(out_dir, "fig2_tes_v2.png")
    plt.tight_layout()
    plt.savefig(out_path2, dpi=300)
    plt.close()
    print(f"Saved {out_path2}")


def plot_phase_space_diagram(out_path: str = "figures_v2/fig3_phase_space_dynamics.png"):
    os.makedirs("figures_v2", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=300)

    # Synthetic 2D phase-space trajectory illustration
    # Normal trajectory
    t = np.linspace(0, 3, 50)
    x_clean = 0.5 * t
    y_clean = 0.8 * t + 0.1 * np.sin(3 * t)
    ax.plot(x_clean, y_clean, color="#16A34A", linewidth=2.5, label="Clean Reasoning Trajectory")

    # Bifurcation point (Commitment Pivot)
    pivot_x, pivot_y = x_clean[-1], y_clean[-1]
    ax.scatter([pivot_x], [pivot_y], color="#D97706", s=180, zorder=6, label="Commitment Pivot $t^*$ (Origin of Cycle)")
    ax.annotate("Commitment Pivot $t^*$\n(Recurrence Origin)", (pivot_x, pivot_y),
                textcoords="offset points", xytext=(-20, 20), ha="right", fontsize=9, weight="bold", color="#B45309")

    # Attractor Limit Cycle (Looping Trajectory)
    theta = np.linspace(0, 4 * np.pi, 100)
    r = 0.45
    x_loop = pivot_x + r * np.cos(theta) + 0.3 * (theta / (4 * np.pi))
    y_loop = pivot_y + r * np.sin(theta)
    ax.plot(x_loop, y_loop, color="#DC2626", linestyle="--", linewidth=2.0, label="Attractor Limit Cycle (Spinning Loop)")

    # Vector u_attractor
    ax.annotate("", xy=(pivot_x + 0.6, pivot_y + 0.1), xytext=(pivot_x, pivot_y),
                arrowprops=dict(arrowstyle="->", color="#DC2626", lw=2.0))
    ax.text(pivot_x + 0.65, pivot_y + 0.1, "$u_{\\mathrm{attractor}}$", color="#DC2626", fontsize=10, weight="bold")

    # Deflation & Orthogonal Branch
    branch_x = np.linspace(pivot_x, pivot_x - 0.7, 40)
    branch_y = np.linspace(pivot_y, pivot_y + 1.2, 40)
    ax.plot(branch_x, branch_y, color="#2563EB", linewidth=2.5, label="DESPIN Deflated Trajectory (Rescued)")
    ax.scatter([branch_x[-1]], [branch_y[-1]], color="#16A34A", marker="*", s=220, zorder=6)
    ax.annotate("Correct Proof Solved\n(No Re-computation)", (branch_x[-1], branch_y[-1]),
                textcoords="offset points", xytext=(-10, 15), fontsize=9, weight="bold", color="#15803D")

    ax.set_title("Phase-Space Recurrence and Attractor Subspace Deflation (ASD)")
    ax.set_xlabel("Latent Activation Dimension 1")
    ax.set_ylabel("Latent Activation Dimension 2")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_benchmark_v2()
    plot_phase_space_diagram()

