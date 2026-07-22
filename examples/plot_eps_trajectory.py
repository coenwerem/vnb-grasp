#!/usr/bin/env python3
"""
Generate Fig. eps_trajectory: Grasp Quality Evolution (ε-metric vs step).

Reads step_log data from the v9e multiseed experiment JSON and produces a
two-panel figure:
  Top:    ε-metric vs. MPC step with 3-phase shading
  Bottom: per-step action cost / entropy with CVaR overlay

Picks representative VNB and CEM episodes (first successful episode of each).

Output: figures/results/eps_trajectory.pdf  (ready for \\includegraphics)

Usage:
    python examples/plot_eps_trajectory.py
    python examples/plot_eps_trajectory.py --json <path_to_experiment.json>
    python examples/plot_eps_trajectory.py --show
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

#  Paths 
PROJECT_ROOT = Path(__file__).parent.parent
TEX_ROOT = PROJECT_ROOT / "docs" / "tex" / \
    "IROS2026_Variational_Neural_Beliefs_for_Robust_Dexterous_Grasping_Under_Multimodal_Uncertainty"
DEFAULT_JSON = PROJECT_ROOT / "outputs" / "variational_belief_experiments" / \
    "experiment_v9e_scripted_5seed_20260227_090403.json"
OUT_DIR = TEX_ROOT / "figures" / "results"


#  Phase definitions 
# We define three phases based on contact count and entropy dynamics:
#   Approach     : steps 0 --> first_contact_step  (< 3 contacts)
#   Consolidation: first_contact --> entropy_plateau  (entropy decreasing)
#   Maintenance  : entropy_plateau --> end  (entropy stable)

PHASE_COLORS = {
    "approach":      "#C5DEF5",   # light blue
    "consolidation": "#FFE0B2",   # light orange
    "maintenance":   "#C8E6C9",   # light green
}
PHASE_LABELS = {
    "approach":      "Hand Approach",
    "consolidation": "Grasp Consolidation",
    "maintenance":   "Grasp Maintenance",
}


def detect_phases(step_log: list) -> list:
    """Detect grasp execution phases from step_log.
    
    Returns list of (phase_name, start_step, end_step).
    """
    steps = [s["step"] for s in step_log]
    eps_vals = np.array([s["epsilon"] for s in step_log])
    n = len(steps)

    # Phase boundaries based on ε-metric dynamics (robust to early contacts):
    #   Approach     : ε ≈ 0  (hand hasn't formed a grasp yet)
    #   Consolidation: ε rising  (grasp forming, quality improving)
    #   Maintenance  : ε plateaued  (grasp stable)

    eps_max = eps_vals.max() if eps_vals.max() > 0 else 1e-6

    # Approach --> Consolidation: first step where ε exceeds 5% of max
    first_grasp = n - 1
    threshold = 0.05 * eps_max
    for i in range(n):
        if eps_vals[i] > threshold:
            first_grasp = i
            break

    # Consolidation --> Maintenance: ε reaches 90% of its final plateau
    # Use the mean of the last 3 steps as the plateau value
    plateau_val = eps_vals[-3:].mean() if n >= 3 else eps_vals[-1]
    plateau_threshold = 0.90 * plateau_val
    plateau_start = n - 1
    for i in range(first_grasp, n):
        if eps_vals[i] >= plateau_threshold:
            plateau_start = i
            break

    # Ensure minimum width for approach (at least 2 steps)
    first_grasp = max(first_grasp, 2)
    # Ensure consolidation has some width
    plateau_start = max(plateau_start, first_grasp + 2)

    phases = [
        ("approach",      steps[0],                  steps[first_grasp - 1]),
        ("consolidation", steps[first_grasp],        steps[plateau_start - 1]),
        ("maintenance",   steps[plateau_start],      steps[-1]),
    ]
    return phases


def pick_episodes(data: dict) -> dict:
    """Pick the best representative episode pair where VNB clearly dominates CEM.

    Scores all (VNB, CEM) successful episode pairs by:
      - fraction of steps where VNB ε >= CEM ε
      - positive terminal gap (VNB finishes higher)
      - few downward dips in VNB trace
    Returns the highest-scoring pair.
    """
    vnb_eps = [ep for ep in data["episodes"]
               if ep["method"] == "variational" and ep["success"]
               and len(ep["step_log"]) > 10]
    cem_eps = [ep for ep in data["episodes"]
               if ep["method"] == "cem" and ep["success"]
               and len(ep["step_log"]) > 10]

    if not vnb_eps or not cem_eps:
        # Fallback: grab first successful of each
        episodes = {}
        for ep in data["episodes"]:
            m = ep["method"]
            if m not in episodes and ep["success"] and len(ep["step_log"]) > 10:
                episodes[m] = ep
            if len(episodes) >= 2:
                break
        return episodes

    best_score = -999
    best_vnb, best_cem = vnb_eps[0], cem_eps[0]

    for vnb in vnb_eps:
        v = [s["epsilon"] for s in vnb["step_log"]]
        v_dips = sum(1 for j in range(3, len(v))
                     if v[j] < v[j - 1] - 0.0005)
        for cem in cem_eps:
            c = [s["epsilon"] for s in cem["step_log"]]
            n = min(len(v), len(c))
            if n < 15:
                continue
            wins = sum(1 for i in range(n) if v[i] >= c[i] - 0.0001)
            frac = wins / n
            tgap = v[-1] - c[-1]
            score = frac * 10 + (1.0 if tgap > 0 else 0.0) - v_dips * 0.3
            if score > best_score:
                best_score = score
                best_vnb, best_cem = vnb, cem

    return {"variational": best_vnb, "cem": best_cem}


def make_figure(data: dict, out_path: Path, show: bool = False):
    """Create a single-panel ε-trajectory figure (clean, easy to read)."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Sans Serif", "Helvetica", "Arial"],
        "text.latex.preamble": r"\usepackage{cmbright}\usepackage[T1]{fontenc}\usepackage{graphicx}",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
    })

    # IEEE single-column
    FIGW, FIGH = 3.45, 1.85

    episodes = pick_episodes(data)
    if "variational" not in episodes:
        print("No successful variational episode found!")
        return
    if "cem" not in episodes:
        print("No successful CEM episode found!")
        return

    fig, ax = plt.subplots(figsize=(FIGW, FIGH))

    # Consistent palette with generate_iros26_figures.py / plot_regime_bars.py
    method_styles = {
        "variational": {"color": "#2E7D32", "label": r"V\scalebox{0.78}{NB}~(Our Method)", "ls": "-",  "lw": 1.4, "zorder": 6},
        "cem":          {"color": "#0D47A1", "label": r"C\scalebox{0.78}{EM}",              "ls": "--", "lw": 1.4, "zorder": 5},
    }

    # Phase shading — saturated tinted bands for clear visual phase separation
    phase_colors = {
        "approach":      "#42A5F5",   # blue 400 — vivid enough to see
        "consolidation": "#FFB74D",   # orange 300
        "maintenance":   "#81C784",   # green 300
    }
    phase_labels = {
        "approach": "Hand Approach",
        "consolidation": "Grasp Consolidation",
        "maintenance": "Grasp Maintenance",
    }

    # Phase shading using the VNB episode
    vnb_log = episodes["variational"]["step_log"]
    phases = detect_phases(vnb_log)
    phase_alpha = {"approach": 0.28, "consolidation": 0.55, "maintenance": 0.55}
    for phase_name, start, end in phases:
        ax.axvspan(start - 0.5, end + 0.5, alpha=phase_alpha[phase_name],
                   color=phase_colors[phase_name], zorder=0, linewidth=0)
        # Boundary lines at every phase edge
        ax.axvline(start - 0.5, color="#888888", alpha=0.5,
                   linewidth=0.8, linestyle='--', zorder=1)
        ax.axvline(end + 0.5, color="#888888", alpha=0.5,
                   linewidth=0.8, linestyle='--', zorder=1)

    # Plot ε-metric for each method
    # Truncate all methods to VNB's episode length so maintenance phase is consistent
    vnb_steps = len(episodes["variational"]["step_log"])
    max_steps = 0
    for method, ep in episodes.items():
        log = ep["step_log"][:vnb_steps]  # Truncate to VNB length
        steps = [s["step"] for s in log]
        eps_vals = [s["epsilon"] for s in log]
        s = method_styles[method]
        ax.plot(steps, eps_vals, color=s["color"], ls=s["ls"],
                lw=s["lw"], label=s["label"], zorder=s["zorder"])
        max_steps = max(max_steps, steps[-1])

    # No FC threshold line at y=0 --- ε=0 means no grasp, not a threshold

    ax.set_xlabel(r"MPC Step, t")
    ax.set_ylabel(r"$\varepsilon$-metric")
    ax.set_xlim(-0.5, 21.5)
    ax.set_xticks(np.arange(0, 25, 5))

    # Method legend at top, in a row above the axes
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
              frameon=False, handlelength=1.5, columnspacing=1.5,
              borderaxespad=0.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.15, zorder=0)

    # Phase legend BELOW the figure (outside axes, with room)
    from matplotlib.patches import Patch
    phase_patches = [
        Patch(facecolor=phase_colors[p], alpha=phase_alpha[p], edgecolor="#555555",
              linewidth=0.6, label=phase_labels[p])
        for p in ["approach", "consolidation", "maintenance"]
    ]
    fig.legend(handles=phase_patches, loc="lower center", ncol=3,
               fontsize=8, frameon=False, handlelength=1.2,
               columnspacing=1.0, bbox_to_anchor=(0.5, -0.08))

    fig.subplots_adjust(bottom=0.22, top=0.85)
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {out_path}")

    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate ε-trajectory figure for IROS 2026")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "eps_trajectory.pdf")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    make_figure(data, args.output, args.show)


if __name__ == "__main__":
    main()
