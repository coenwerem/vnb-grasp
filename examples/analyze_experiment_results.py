#!/usr/bin/env python3
"""
Analyze and summarize variational belief experiment results for IROS 2026 paper.

Reads the JSON output from run_variational_belief_experiments.py and generates:
  1. Summary statistics tables (LaTeX-ready)
  2. Per-condition comparisons
  3. Key findings for Results section

Usage:
    python examples/analyze_experiment_results.py [results.json]
    # If no file given, uses the latest results file in outputs/
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_results(path: str = None):
    if path:
        with open(path) as f:
            return json.load(f)

    # Find latest results file
    out_dir = Path("outputs/variational_belief_experiments")
    files = sorted(out_dir.glob("experiment_results_iros26_*.json"))
    if not files:
        files = sorted(out_dir.glob("experiment_results_full_*.json"))
    if not files:
        print("No results files found in", out_dir)
        sys.exit(1)
    path = files[-1]
    print(f"Loading: {path}")
    with open(path) as f:
        return json.load(f)


def group_results(data):
    """Group episodes by (method, object, beta)"""
    groups = defaultdict(list)
    for ep in data["episodes"]:
        key = (ep["method"], ep["object_name"], ep["beta"])
        groups[key] = groups.get(key, [])
        groups[key].append(ep)
    return dict(groups)


def compute_stats(episodes, key):
    """Compute mean ± std for a given key across episodes"""
    vals = [ep[key] for ep in episodes]
    return np.mean(vals), np.std(vals)


def print_main_table(groups):
    """Print the main results table"""
    print("\n" + "=" * 120)
    print("TABLE I: Main Results: Grasp Acquisition Under Friction Uncertainty")
    print("=" * 120)

    header = (
        f"{'Method':<12} {'Object':<20} {'β':>4} | "
        f"{'Success':>7} {'ε':>8} {'Quality':>8} {'GWS Vol':>8} | "
        f"{'Lift':>5} {'Shear':>5} | "
        f"{'Steps':>5} {'Time(s)':>7} | {'Grads':>5}"
    )
    print(header)
    print("-" * 120)

    for (method, obj, beta), eps in sorted(groups.items()):
        n = len(eps)
        succ = sum(1 for e in eps if e["success"]) / n * 100
        eps_m, eps_s = compute_stats(eps, "final_epsilon")
        q_m, q_s = compute_stats(eps, "final_contact_quality")
        v_m, v_s = compute_stats(eps, "final_gws_volume")
        lift = sum(1 for e in eps if e["lift_success"]) / n * 100
        shear = sum(1 for e in eps if e["shear_success"]) / n * 100
        steps_m, steps_s = compute_stats(eps, "n_steps")
        time_m, time_s = compute_stats(eps, "runtime_s")

        if method == "variational":
            grads = sum(1 for e in eps if e["has_exact_grads"]) / n * 100
            grads_str = f"{grads:4.0f}%"
        else:
            grads_str = "  n/a"

        print(
            f"{method:<12} {obj:<20} {beta:>4.2f} | "
            f"{succ:>6.0f}% {eps_m:>7.4f} {q_m:>7.3f}  {v_m:>7.2f}  | "
            f"{lift:>4.0f}% {shear:>4.0f}% | "
            f"{steps_m:>5.1f} {time_m:>6.1f}  | {grads_str}"
        )

    print("=" * 120)


def print_belief_quality_table(groups):
    """Print belief quality comparison (CVaR, entropy, gradient norms)"""
    print("\n" + "=" * 100)
    print("TABLE II: Belief Quality Metrics")
    print("=" * 100)

    header = (
        f"{'Method':<12} {'Object':<20} {'β':>4} | "
        f"{'CVaR':>8} {'CostMean':>8} {'CostStd':>8} | "
        f"{'Entropy':>8} | "
        f"{'∇μ norm':>8} {'∇π norm':>8}"
    )
    print(header)
    print("-" * 100)

    for (method, obj, beta), eps in sorted(groups.items()):
        if method == "variational":
            cvar_m = np.mean([e["belief_cvar"] for e in eps])
            cost_m = np.mean([e["belief_cost_mean"] for e in eps])
            cost_s = np.mean([e["belief_cost_std"] for e in eps])
            ent_m = np.mean([e["final_entropy"] for e in eps])
            gn_means = np.mean([e["grad_norm_means"] for e in eps])
            gn_logits = np.mean([e["grad_norm_logits"] for e in eps])
        else:
            # Particle method: extract from step_log
            cvar_m = np.mean([e["step_log"][-1]["cvar"] if e["step_log"] else 0 for e in eps])
            cost_m = np.mean([e["step_log"][-1]["cost"] if e["step_log"] else 0 for e in eps])
            cost_s = 0.0
            ent_m = np.mean([e["final_entropy"] for e in eps])
            gn_means = 0.0
            gn_logits = 0.0

        gn_str = f"{gn_means:>7.3f}  {gn_logits:>7.3f}" if method == "variational" else "     n/a      n/a"
        print(
            f"{method:<12} {obj:<20} {beta:>4.2f} | "
            f"{cvar_m:>7.3f}  {cost_m:>7.3f}  {cost_s:>7.3f}  | "
            f"{ent_m:>7.3f}  | {gn_str}"
        )

    print("=" * 100)


def print_speed_comparison(groups):
    """Print speed comparison between methods"""
    print("\n" + "=" * 80)
    print("TABLE III: Computational Efficiency")
    print("=" * 80)

    # Group by (object, beta) and compare methods
    conditions = defaultdict(dict)
    for (method, obj, beta), eps in groups.items():
        time_m = np.mean([e["runtime_s"] for e in eps])
        conditions[(obj, beta)][method] = time_m

    header = f"{'Object':<20} {'β':>4} | {'Particle(s)':>11} {'Variational(s)':>14} {'Speedup':>8}"
    print(header)
    print("-" * 80)

    speedups = []
    for (obj, beta), methods in sorted(conditions.items()):
        if "particle" in methods and "variational" in methods:
            pt = methods["particle"]
            vt = methods["variational"]
            speedup = pt / vt if vt > 0 else float("inf")
            speedups.append(speedup)
            print(f"{obj:<20} {beta:>4.2f} | {pt:>10.1f}  {vt:>13.1f}  {speedup:>7.0f}x")

    if speedups:
        print(f"\nMean speedup: {np.mean(speedups):.0f}x  (range: {min(speedups):.0f}x - {max(speedups):.0f}x)")
    print("=" * 80)


def print_beta_sensitivity(groups):
    """Analyze effect of β on grasp quality and CVaR"""
    print("\n" + "=" * 90)
    print("TABLE IV: Effect of Risk Level β on Grasp Outcomes")
    print("=" * 90)

    for method in ["particle", "variational"]:
        print(f"\n  {method.upper()}:")
        header = f"  {'Object':<20} {'β':>4} | {'Success':>7} {'ε':>8} {'Quality':>8} {'CVaR':>8} {'Steps':>5}"
        print(header)
        print("  " + "-" * 80)

        for obj in sorted(set(k[1] for k in groups.keys())):
            for beta in sorted(set(k[2] for k in groups.keys())):
                key = (method, obj, beta)
                if key not in groups:
                    continue
                eps = groups[key]
                n = len(eps)
                succ = sum(1 for e in eps if e["success"]) / n * 100
                eps_m = np.mean([e["final_epsilon"] for e in eps])
                q_m = np.mean([e["final_contact_quality"] for e in eps])
                if method == "variational":
                    cvar_m = np.mean([e["belief_cvar"] for e in eps])
                else:
                    cvar_m = np.mean([e["step_log"][-1]["cvar"] if e["step_log"] else 0 for e in eps])
                steps_m = np.mean([e["n_steps"] for e in eps])

                print(f"  {obj:<20} {beta:>4.2f} | {succ:>6.0f}% {eps_m:>7.4f} {q_m:>7.3f}  {cvar_m:>7.3f} {steps_m:>5.1f}")

    print("=" * 90)


def print_latex_table(groups):
    """Generate a LaTeX-ready results table"""
    print("\n%  LaTeX Table for IROS 2026 Paper ")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Comparison of particle-filter and variational neural beliefs across objects and risk levels.}")
    print("\\label{tab:main_results}")
    print("\\small")
    print("\\begin{tabular}{@{}llc|cccc|cc|c@{}}")
    print("\\toprule")
    print("Method & Object & $\\beta$ & Succ & $\\epsilon$ & Quality & GWS Vol & Lift & Shear & Time (s) \\\\")
    print("\\midrule")

    prev_obj = None
    for (method, obj, beta), eps in sorted(groups.items()):
        n = len(eps)
        if obj != prev_obj and prev_obj is not None:
            print("\\midrule")
        prev_obj = obj

        succ = sum(1 for e in eps if e["success"]) / n * 100
        eps_m = np.mean([e["final_epsilon"] for e in eps])
        q_m = np.mean([e["final_contact_quality"] for e in eps])
        v_m = np.mean([e["final_gws_volume"] for e in eps])
        lift = sum(1 for e in eps if e["lift_success"]) / n * 100
        shear = sum(1 for e in eps if e["shear_success"]) / n * 100
        time_m = np.mean([e["runtime_s"] for e in eps])

        m_short = "PF" if method == "particle" else "\\textbf{Ours}"
        obj_short = obj.replace("graspit_", "").replace("_", " ").title()

        print(
            f"{m_short} & {obj_short} & {beta:.2f} & "
            f"{succ:.0f}\\% & {eps_m:.4f} & {q_m:.3f} & {v_m:.1f} & "
            f"{lift:.0f}\\% & {shear:.0f}\\% & {time_m:.1f} \\\\"
        )

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def print_key_findings(groups):
    """Summarize key findings for the paper"""
    print("\n" + "=" * 80)
    print("KEY FINDINGS FOR PAPER")
    print("=" * 80)

    # Overall success rates
    particle_eps = [ep for eps in groups.values() for ep in eps if eps[0]["method"] == "particle"]
    variational_eps = [ep for eps in groups.values() for ep in eps if eps[0]["method"] == "variational"]

    # Actually reconstruct properly
    p_all, v_all = [], []
    for (method, _, _), eps in groups.items():
        if method == "particle":
            p_all.extend(eps)
        else:
            v_all.extend(eps)

    if p_all:
        p_succ = sum(1 for e in p_all if e["success"]) / len(p_all) * 100
        p_time = np.mean([e["runtime_s"] for e in p_all])
        print(f"\n1. Particle Filter (baseline):")
        print(f"   Success rate: {p_succ:.0f}%  ({sum(1 for e in p_all if e['success'])}/{len(p_all)})")
        print(f"   Mean time:    {p_time:.1f}s per episode")
        print(f"   Exact grads:  No (sampling-based)")

    if v_all:
        v_succ = sum(1 for e in v_all if e["success"]) / len(v_all) * 100
        v_time = np.mean([e["runtime_s"] for e in v_all])
        v_grads = sum(1 for e in v_all if e["has_exact_grads"]) / len(v_all) * 100
        print(f"\n2. Variational Neural Belief (ours):")
        print(f"   Success rate: {v_succ:.0f}%  ({sum(1 for e in v_all if e['success'])}/{len(v_all)})")
        print(f"   Mean time:    {v_time:.1f}s per episode")
        print(f"   Exact grads:  {v_grads:.0f}%")

    if p_all and v_all:
        speedup = p_time / v_time if v_time > 0 else float("inf")
        print(f"\n3. Speed comparison:")
        print(f"   Speedup: {speedup:.0f}x ({p_time:.1f}s --> {v_time:.1f}s)")

        # Per-beta analysis
        print(f"\n4. β sensitivity:")
        for beta in sorted(set(e["beta"] for e in p_all)):
            p_b = [e for e in p_all if e["beta"] == beta]
            v_b = [e for e in v_all if e["beta"] == beta]
            if p_b and v_b:
                p_s = sum(1 for e in p_b if e["success"]) / len(p_b) * 100
                v_s = sum(1 for e in v_b if e["success"]) / len(v_b) * 100
                p_q = np.mean([e["final_contact_quality"] for e in p_b])
                v_q = np.mean([e["final_contact_quality"] for e in v_b])
                print(f"   β={beta:.2f}: PF success={p_s:.0f}% quality={p_q:.3f} | "
                      f"Ours success={v_s:.0f}% quality={v_q:.3f}")

    print("=" * 80)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    data = load_results(path)
    print(f"Loaded {data['n_episodes']} episodes from {data['timestamp']}")

    groups = group_results(data)
    print(f"Grouped into {len(groups)} conditions")

    print_main_table(groups)
    print_belief_quality_table(groups)
    print_speed_comparison(groups)
    print_beta_sensitivity(groups)
    print_latex_table(groups)
    print_key_findings(groups)


if __name__ == "__main__":
    main()
