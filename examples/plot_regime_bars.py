#!/usr/bin/env python3
"""
Generate Fig. regime_bars: Per-Regime Performance Comparison (CEM vs VNB).

Reads the v9e multiseed hero experiment JSON and produces a grouped bar chart
with 4 friction-regime clusters (nominal, adversarial, wide, bimodal), each
showing CEM (blue) and VNB (orange) bars for three metrics:
  - Success Rate (%)
  - Robust Success (%)
  - Perturbation Survival (%)

Output: figures/results/regime_bars.pdf  (ready for \\includegraphics)

Usage:
    python examples/plot_regime_bars.py
    python examples/plot_regime_bars.py --json <path_to_experiment.json>
    python examples/plot_regime_bars.py --show   # display interactively
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

#  Paths 
PROJECT_ROOT  = Path(__file__).parent.parent
TEX_ROOT = PROJECT_ROOT / "docs" / "tex" / \
    "IROS2026_Variational_Neural_Beliefs_for_Robust_Dexterous_Grasping_Under_Multimodal_Uncertainty"
DEFAULT_JSON = PROJECT_ROOT / "outputs" / "variational_belief_experiments" / \
    "experiment_hero_final_20260301_194004.json"
OUT_DIR = TEX_ROOT / "figures" / "results"


#  Style 
# IEEE double-column width
FIGW, FIGH = 7.16, 3.6  # inches  (3 regimes, full double-column width)
COLORS = {
    "cem":         "#1565C0",  # deep blue  (consistent with generate_iros26_figures.py)
    "variational": "#2E7D32",  # forest green
}
LABELS = {
    "cem":         r"C\scalebox{0.78}{EM}",
    "variational": r"V\scalebox{0.78}{NB} (Ours)",
}
REGIMES_ORDER = ["nominal", "wide", "bimodal"]
REGIMES_DISPLAY = {
    "nominal":     "Nominal",
    "adversarial": "Adversarial",
    "wide":        "Wide",
    "bimodal":     "Bimodal",
}
METRICS = [
    ("SR",    "success",                    "Success Rate"),
    ("Rob",   "robust_success",             "Robust Success"),
    ("Pert",  "perturbation_survival_rate", "Pert. Survival"),
]


def load_data(json_path, beta_filter: float = None) -> dict:
    """Group episodes by (method, regime). Accepts single path or list of paths.
    
    Args:
        json_path: Path to JSON file(s)
        beta_filter: Only include episodes with this beta value (default None = all betas)
    """
    if isinstance(json_path, (list, tuple)):
        paths = json_path
    else:
        paths = [json_path]
    
    groups = defaultdict(list)
    seen = set()
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        for ep in data["episodes"]:
            # Filter to specific beta value (VNB advantage is at beta=0.95)
            if beta_filter is not None and ep.get("beta") != beta_filter:
                continue
            key = (ep["method"], ep.get("object_name",""), ep["beta"],
                   ep["seed"], ep["friction_regime"])
            if key not in seen:
                seen.add(key)
                method = ep["method"]
                regime = ep["friction_regime"]
                groups[(method, regime)].append(ep)
    return groups


def compute_metric(episodes: list, metric_key: str) -> float:
    """Compute percentage for a metric across episodes."""
    if metric_key == "perturbation_survival_rate":
        return float(np.mean([ep[metric_key] for ep in episodes])) * 100
    else:
        return sum(1 for ep in episodes if ep[metric_key]) / len(episodes) * 100


def make_figure(groups: dict, out_path: Path, show: bool = False):
    """Create the grouped bar chart."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Sans Serif", "Helvetica", "Arial"],
        "text.latex.preamble": r"\usepackage{cmbright}\usepackage[T1]{fontenc}\usepackage{graphicx}",
        "font.size": 17,
        "axes.titlesize": 19,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "figure.dpi": 300,
    })

    methods = ["cem", "variational"]
    n_regimes = len(REGIMES_ORDER)
    n_methods = len(methods)
    n_metrics = len(METRICS)

    fig, ax = plt.subplots(figsize=(FIGW, FIGH))

    # Bar geometry — spread regimes across wider x-range so bars are fat
    bar_width = 0.10                # absolute bar width in data units
    gap_between_metrics = 0.04      # gap between metric sub-groups
    metric_block = n_methods * bar_width  # width of one metric sub-group
    group_width = n_metrics * metric_block + (n_metrics - 1) * gap_between_metrics
    # Space regimes apart so clusters don't overlap
    x_spacing = group_width + 0.35
    x_centers = np.arange(n_regimes) * x_spacing

    # Within each regime cluster, we have n_metrics sub-groups of n_methods bars
    # Layout: [SR_CEM, SR_VNB, gap, Rob_CEM, Rob_VNB, gap, Pert_CEM, Pert_VNB]
    hatches = ["", "//", "xx"]  # different hatch per metric for grayscale readability

    handles = []
    legend_labels = []

    for mi, (metric_short, metric_key, metric_label) in enumerate(METRICS):
        for ji, method in enumerate(methods):
            vals = []
            for regime in REGIMES_ORDER:
                eps = groups.get((method, regime), [])
                if eps:
                    vals.append(compute_metric(eps, metric_key))
                else:
                    vals.append(0)

            # Position offset: metric group offset + method offset
            metric_offset = mi * (metric_block + gap_between_metrics)
            method_offset = ji * bar_width
            total_offset = metric_offset + method_offset - group_width / 2 + bar_width / 2

            bars = ax.bar(
                x_centers + total_offset,
                vals,
                width=bar_width * 0.92,
                color=COLORS[method],
                edgecolor="black",
                linewidth=0.4,
                alpha=0.92,
                hatch=hatches[mi],
                zorder=3,
            )

            # Only add legend entries for first regime pass
            if mi == 0:
                handles.append(bars[0])
                legend_labels.append(LABELS[method])

    # Metric group labels (above the clusters)
    for mi, (metric_short, _, _) in enumerate(METRICS):
        metric_offset = mi * (metric_block + gap_between_metrics)
        center = metric_offset - group_width / 2 + metric_block / 2
        for ri in range(n_regimes):
            if ri == 0:  # label once at top
                pass

    ax.set_xticks(x_centers)
    ax.set_xticklabels([REGIMES_DISPLAY[r] for r in REGIMES_ORDER])
    ax.set_ylabel(r"Percentage (\%)")
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.axhline(y=100, color="gray", linewidth=0.3, linestyle=":", zorder=1)

    # Add metric sub-labels as a secondary guide
    from matplotlib.patches import Patch

    # Build a combined legend: method colors + metric hatches
    method_handles = list(handles)
    method_labels  = list(legend_labels)

    hatch_styles = ["", "//", "xx"]
    metric_patches = [
        Patch(facecolor="gray", edgecolor="black", hatch=hatch_styles[i] * 2,
              linewidth=0.5, label=METRICS[i][2])
        for i in range(n_metrics)
    ]

    # Method legend at top — two rows: methods on top, metric hatches below
    ax.legend(
        method_handles + metric_patches,
        method_labels + [m[2] for m in METRICS],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=False,
        ncol=3,
        fontsize=13,
        columnspacing=1.0,
        handlelength=2.0,
        handleheight=1.4,
        borderaxespad=0.0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, zorder=0)

    plt.tight_layout(rect=[0, 0, 1, 0.82])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved: {out_path}")

    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate per-regime bar chart for IROS 2026")
    parser.add_argument("--json", type=Path, nargs="+", default=[DEFAULT_JSON])
    parser.add_argument("--output", type=Path, default=OUT_DIR / "regime_bars.pdf")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    groups = load_data(args.json)
    make_figure(groups, args.output, args.show)


if __name__ == "__main__":
    main()
