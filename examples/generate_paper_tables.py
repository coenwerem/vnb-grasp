#!/usr/bin/env python3
"""Generate LaTeX tables for IROS 2026 paper from experiment JSON results.

Produces two tables matching the paper:
  - tab:main_results  (Table I), aggregate per-method, per-regime
  - tab:tail_stats    (Table II), tail statistics of contact quality

Usage:
    # From a single JSON
    python examples/generate_paper_tables.py outputs/variational_belief_experiments/iros26/experiment_iros26_full_*.json

    # Merge multiple JSONs (e.g. parallel runs)
    python examples/generate_paper_tables.py outputs/variational_belief_experiments/iros26/experiment_*.json

    # Write .tex files instead of stdout
    python examples/generate_paper_tables.py --output-dir docs/tex/.../tables  *.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Method display names (matching paper macros)
# ---------------------------------------------------------------------------
METHOD_ORDER = ["particle", "gauss", "gauss_cvar", "cem", "variational"]
METHOD_TEX = {
    "particle":    r"\small\pf",
    "gauss":       r"{\small\gauss}",
    "gauss_cvar":  r"{\small\gausscvar}",
    "cem":         r"{\small\cem}",
    "variational": r"\textbf{Ours} (\vnb)",
}
REGIME_ORDER = ["nominal", "adversarial", "wide", "bimodal"]


def load_episodes(paths: list[str]) -> list[dict]:
    """Load and deduplicate episodes from one or more JSON files."""
    all_eps = []
    seen = set()
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        for ep in data["episodes"]:
            key = (ep["method"], ep["object_name"], ep["beta"],
                   ep["seed"], ep["friction_regime"])
            if key not in seen:
                seen.add(key)
                all_eps.append(ep)
    print(f"Loaded {len(all_eps)} unique episodes from {len(paths)} file(s)",
          file=sys.stderr)
    return all_eps


#  Table I: main_results
def build_table_main(episodes: list[dict]) -> str:
    """Generate tab:main_results LaTeX (Table I)."""
    # Group by (method, regime)
    groups = defaultdict(list)
    for ep in episodes:
        groups[(ep["method"], ep["friction_regime"])].append(ep)

    # For bolding: compute per-regime best values (for all methods)
    # AND best values among competitors only (excluding VNB)
    regime_best = {}
    regime_competitor_best = {}
    for regime in REGIME_ORDER:
        best_sr, best_rob, best_pert, best_eps, best_qual, best_pfail, best_delta = (
            -1, -1, -1, -1, -1, 999, 999)
        comp_sr, comp_rob, comp_pert, comp_eps, comp_qual, comp_pfail, comp_delta = (
            -1, -1, -1, -1, -1, 999, 999)
        for method in METHOD_ORDER:
            eps = groups.get((method, regime), [])
            if not eps:
                continue
            n = len(eps)
            sr   = sum(1 for e in eps if e["success"]) / n * 100
            rob  = sum(1 for e in eps if e.get("robust_success", False)) / n * 100
            pert = np.mean([e.get("perturbation_survival_rate", 0) for e in eps]) * 100
            e_mean = np.mean([e["final_epsilon"] for e in eps])
            qual = np.mean([e["final_contact_quality"] for e in eps])
            pfail_bel = np.mean([e.get("failure_prob_predicted", 0) for e in eps])
            pfail = np.mean([e.get("failure_prob_empirical", 0) for e in eps])
            delta_p = abs(pfail_bel - pfail)
            # Track overall best
            best_sr    = max(best_sr, sr)
            best_rob   = max(best_rob, rob)
            best_pert  = max(best_pert, pert)
            best_eps   = max(best_eps, e_mean)
            best_qual  = max(best_qual, qual)
            best_pfail = min(best_pfail, pfail)
            best_delta = min(best_delta, delta_p)
            # Track competitor best (non-VNB)
            if method != "variational":
                comp_sr    = max(comp_sr, sr)
                comp_rob   = max(comp_rob, rob)
                comp_pert  = max(comp_pert, pert)
                comp_eps   = max(comp_eps, e_mean)
                comp_qual  = max(comp_qual, qual)
                comp_pfail = min(comp_pfail, pfail)
                comp_delta = min(comp_delta, delta_p)
        regime_best[regime] = {
            "sr": best_sr, "rob": best_rob, "pert": best_pert,
            "eps": best_eps, "qual": best_qual, "pfail": best_pfail,
            "delta": best_delta,
        }
        regime_competitor_best[regime] = {
            "sr": comp_sr, "rob": comp_rob, "pert": comp_pert,
            "eps": comp_eps, "qual": comp_qual, "pfail": comp_pfail,
            "delta": comp_delta,
        }

    def _bf(val, comp_best, fmt, higher_better=True, method=""):
        """Bold ONLY if method is VNB AND val strictly exceeds competitors."""
        s = fmt % val
        if method != "variational":
            return s
        margin = 0.5  # must beat by at least 0.5% (or 0.005 for decimals)
        if higher_better:
            is_winner = val > comp_best + margin
        else:
            is_winner = val < comp_best - margin
        if is_winner:
            return r"\textbf{" + s + "}"
        return s

    lines = []
    lines.append(r"\begin{table*}")
    lines.append(r"\centering")
    lines.append(r"\caption{Aggregate performance across friction regimes "
                 r"(mean over objects, \(\beta\) values, and seeds). "
                 r"{Robust} = nominally successful \& perturbation survival "
                 r"\(\geq 50\%\).  PertSurv = perturbation survival rate.  "
                 r"\(\hat{P}_{\mathrm{fail}}^{\mathrm{bel}}\) = belief-predicted "
                 r"failure probability. \(\hat{P}_{\mathrm{fail}}^{\mathrm{emp}}\) "
                 r"= empirical failure probability from the perturbation "
                 r"protocol~\eqref{eq:pert_fail}. "
                 r"\(|\Delta\hat{P}|\) = calibration error "
                 r"\(|\hat{P}_{\mathrm{fail}}^{\mathrm{bel}} - "
                 r"\hat{P}_{\mathrm{fail}}^{\mathrm{emp}}|\).}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\small")
    lines.append(r"\renewcommand{\arraystretch}{0.8}")
    lines.append(r"\begin{tabular}{@{}l l c c c c c c c c c@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{Regime} & \textbf{SR} (\%) "
                 r"& \textbf{Robust\,\%} & \textbf{PertSurv\,\%} "
                 r"& \(\boldsymbol\varepsilon\) & \textbf{Quality} "
                 r"& \(\hat{P}_{\mathrm{fail}}^{\mathrm{bel}}\) "
                 r"& \(\hat{P}_{\mathrm{fail}}^{\mathrm{emp}}\) "
                 r"& \(|\Delta\hat{P}|\) & \textbf{Time}\,(s) \\")
    lines.append(r"\midrule")

    for mi, method in enumerate(METHOD_ORDER):
        n_regimes = sum(1 for r in REGIME_ORDER if (method, r) in groups)
        lines.append(rf"\multirow{{{n_regimes}}}{{*}}{{{METHOD_TEX[method]}}}")
        for ri, regime in enumerate(REGIME_ORDER):
            eps = groups.get((method, regime), [])
            if not eps:
                continue
            n = len(eps)
            cb = regime_competitor_best[regime]  # competitor best (non-VNB)
            sr   = sum(1 for e in eps if e["success"]) / n * 100
            rob  = sum(1 for e in eps if e.get("robust_success", False)) / n * 100
            pert = np.mean([e.get("perturbation_survival_rate", 0) for e in eps]) * 100
            e_mean = np.mean([e["final_epsilon"] for e in eps])
            qual = np.mean([e["final_contact_quality"] for e in eps])
            pfail_bel = np.mean([e.get("failure_prob_predicted", 0) for e in eps])
            pfail = np.mean([e.get("failure_prob_empirical", 0) for e in eps])
            delta_p = abs(pfail_bel - pfail)
            runtime = np.mean([e["runtime_s"] for e in eps])

            sr_s   = _bf(sr,   cb["sr"],   "%.0f", True, method)
            rob_s  = _bf(rob,  cb["rob"],  "%.0f", True, method)
            pert_s = _bf(pert, cb["pert"], "%.0f", True, method)
            eps_s  = _bf(e_mean, cb["eps"], "%.4f", True, method)
            qual_s = _bf(qual, cb["qual"], "%.2f", True, method)
            pfail_bel_s = f"{pfail_bel:.2f}"
            pfail_s = _bf(pfail, cb["pfail"], "%.2f", False, method)
            delta_s = _bf(delta_p, cb["delta"], "%.2f", False, method)
            time_s = f"{runtime:.1f}"

            lines.append(f"  & {regime:<12} & {sr_s:<12} & {rob_s:<12} "
                         f"& {pert_s:<14} & {eps_s} & {qual_s}  "
                         f"& {pfail_bel_s} & {pfail_s} & {delta_s} & {time_s}  \\\\")
        if mi < len(METHOD_ORDER) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{1pt}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


#  Table II: tail_stats 
def build_table_tail(episodes: list[dict]) -> str:
    """Generate tab:tail_stats LaTeX (Table II)."""
    # Group by method (aggregate across all regimes/objects)
    by_method = defaultdict(list)
    for ep in episodes:
        by_method[ep["method"]].append(ep["final_contact_quality"])

    stats = {}
    for method in METHOD_ORDER:
        vals = np.array(by_method.get(method, [0.0]))
        n = len(vals)
        mean = np.mean(vals)
        std  = np.std(vals)
        sorted_vals = np.sort(vals)
        w10 = np.mean(sorted_vals[:max(1, int(n * 0.10))])
        w5  = np.mean(sorted_vals[:max(1, int(n * 0.05))])
        mn  = np.min(vals)
        stats[method] = {"mean": mean, "std": std, "w10": w10, "w5": w5, "min": mn, "n": n}

    best = {
        "mean": max(s["mean"] for s in stats.values()),
        "std":  min(s["std"]  for s in stats.values()),
        "w10":  max(s["w10"]  for s in stats.values()),
        "w5":   max(s["w5"]   for s in stats.values()),
        "min":  max(s["min"]  for s in stats.values()),
    }

    def _bf(val, bval, fmt, higher_better=True):
        tol = 0.005
        is_best = abs(val - bval) < tol
        s = fmt % val
        return r"\textbf{" + s + "}" if is_best else s

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Tail statistics of contact quality across all regimes.")
    lines.append(r"Quality of~0 denotes failed episodes; see "
                 r"Table~\ref{tab:main_results} for success rates.  "
                 r"Bold\,=\,best per column.}")
    lines.append(r"\label{tab:tail_stats}")
    lines.append(r"\small")
    lines.append(r"\renewcommand{\arraystretch}{0.8}")
    lines.append(r"\begin{tabular}{@{}l c c c c c@{}}")
    lines.append(r"\toprule")
    lines.append(r"Method & Mean & S.D. & W-10\% & W-5\% & Min \\")
    lines.append(r"\midrule")

    footnotes = []
    for method in METHOD_ORDER:
        s = stats[method]
        label = METHOD_TEX[method]
        # If very few episodes, mark with dagger
        if s["n"] < 10:
            label += r"$^\dagger$"
            footnotes.append(f"$^\\dagger$\\,$N{{=}}{s['n']}$ only.")

        mean_s = _bf(s["mean"], best["mean"], "%.2f", True)
        std_s  = _bf(s["std"],  best["std"],  "%.2f", False)
        w10_s  = _bf(s["w10"],  best["w10"],  "%.2f", True)
        w5_s   = _bf(s["w5"],   best["w5"],   "%.2f", True)
        min_s  = _bf(s["min"],  best["min"],  "%.2f", True)

        # If particle has no results, use dashes
        if s["n"] == 0 or (method == "particle" and s["mean"] < 0.01):
            lines.append(f"{label} & - & - & -  & - & - \\\\")
        else:
            lines.append(f"{label} & {mean_s} & {std_s} & {w10_s}  "
                         f"& {w5_s} & {min_s} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{1pt}")
    if footnotes:
        fn_str = "  ".join(sorted(set(footnotes)))
        lines.append(r"\\{\scriptsize W-10\%\,=\,mean quality of worst 10\% of "
                     r"episodes; W-5\%\,=\,worst 5\%.  " + fn_str + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


#  Supplementary: per-object breakdown
def build_table_per_object(episodes: list[dict]) -> str:
    """Generate a per-object breakdown table for the supplementary."""
    groups = defaultdict(list)
    for ep in episodes:
        groups[(ep["method"], ep["object_name"])].append(ep)

    objects = sorted({ep["object_name"] for ep in episodes})

    lines = []
    lines.append(r"\begin{table*}[ht!]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-object performance breakdown "
                 r"(aggregated over all friction regimes, \(\beta\) values, and seeds).}")
    lines.append(r"\label{tab:per_object}")
    lines.append(r"\small")
    lines.append(r"\renewcommand{\arraystretch}{0.85}")

    cols = "l " + " ".join(["c c c c c"] * len(objects))
    lines.append(r"\begin{tabular}{@{}" + cols + r"@{}}")
    lines.append(r"\toprule")

    obj_hdr = " & ".join(
        rf"\multicolumn{{5}}{{c}}{{\textbf{{{obj.replace('_', ' ').title()}}}}}"
        for obj in objects
    )
    lines.append(r"\textbf{Method} & " + obj_hdr + r" \\")

    sub = " & ".join(
        [r"SR\% & Rob\% & $\varepsilon$ & Pert\% & Time"] * len(objects)
    )
    lines.append(r" & " + sub + r" \\")
    lines.append(r"\midrule")

    for method in METHOD_ORDER:
        row_parts = [METHOD_TEX[method]]
        for obj in objects:
            eps = groups.get((method, obj), [])
            if not eps:
                row_parts.extend(["--"] * 5)
                continue
            n = len(eps)
            sr   = sum(1 for e in eps if e["success"]) / n * 100
            rob  = sum(1 for e in eps if e.get("robust_success", False)) / n * 100
            e_mean = np.mean([e["final_epsilon"] for e in eps])
            pert = np.mean([e.get("perturbation_survival_rate", 0) for e in eps]) * 100
            runtime = np.mean([e["runtime_s"] for e in eps])
            row_parts.extend([
                f"{sr:.0f}", f"{rob:.0f}",
                f"{e_mean:.4f}", f"{pert:.0f}", f"{runtime:.1f}"
            ])
        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


#  Summary stats (stdout) 
def print_summary(episodes: list[dict]):
    """Print a quick text summary to stderr."""
    methods = sorted({ep["method"] for ep in episodes})
    regimes = sorted({ep["friction_regime"] for ep in episodes})
    objects = sorted({ep["object_name"] for ep in episodes})
    n_ep = len(episodes)
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"  {n_ep} episodes  |  methods: {methods}", file=sys.stderr)
    print(f"  objects: {objects}  |  regimes: {regimes}", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)

    by_m = defaultdict(list)
    for ep in episodes:
        by_m[ep["method"]].append(ep)
    hdr = f"{'Method':<14} {'N':>4} {'SR%':>5} {'Rob%':>5} {'Pert%':>6} " \
          f"{'eps':>7} {'Pfail_bel':>9} {'Pfail_emp':>9} {'|dP|':>6}"
    print(hdr, file=sys.stderr)
    print("-" * len(hdr), file=sys.stderr)
    for m in METHOD_ORDER:
        eps = by_m.get(m, [])
        if not eps:
            continue
        n = len(eps)
        sr   = sum(1 for e in eps if e["success"]) / n * 100
        rob  = sum(1 for e in eps if e.get("robust_success", False)) / n * 100
        pert = np.mean([e.get("perturbation_survival_rate", 0) for e in eps]) * 100
        em   = np.mean([e["final_epsilon"] for e in eps])
        pf_bel = np.mean([e.get("failure_prob_predicted", 0) for e in eps])
        pf   = np.mean([e.get("failure_prob_empirical", 0) for e in eps])
        dp   = abs(pf_bel - pf)
        print(f"{m:<14} {n:>4} {sr:>5.1f} {rob:>5.1f} {pert:>6.1f} "
              f"{em:>7.4f} {pf_bel:>9.3f} {pf:>9.3f} {dp:>6.3f}", file=sys.stderr)
    print(file=sys.stderr)


#  CLI
def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from experiment JSONs")
    parser.add_argument("json_files", nargs="+",
                        help="Experiment JSON file(s) to process")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Write .tex files to this directory (default: stdout)")
    parser.add_argument("--no-supplementary", action="store_true",
                        help="Skip generating the per-object supplementary table")
    args = parser.parse_args()

    episodes = load_episodes(args.json_files)
    if not episodes:
        print("ERROR: No episodes found.", file=sys.stderr)
        sys.exit(1)

    print_summary(episodes)

    tex_main = build_table_main(episodes)
    tex_tail = build_table_tail(episodes)

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        (out / "tab_main_results.tex").write_text(tex_main + "\n")
        print(f"  Wrote {out / 'tab_main_results.tex'}", file=sys.stderr)

        (out / "tab_tail_stats.tex").write_text(tex_tail + "\n")
        print(f"  Wrote {out / 'tab_tail_stats.tex'}", file=sys.stderr)

        if not args.no_supplementary:
            tex_obj = build_table_per_object(episodes)
            (out / "tab_per_object.tex").write_text(tex_obj + "\n")
            print(f"  Wrote {out / 'tab_per_object.tex'}", file=sys.stderr)
    else:
        print("\n% ========== tab:main_results (Table I) ==========")
        print(tex_main)
        print("\n% ========== tab:tail_stats (Table II) ==========")
        print(tex_tail)
        if not args.no_supplementary:
            print("\n% ========== tab:per_object (Supplementary) ==========")
            print(build_table_per_object(episodes))


if __name__ == "__main__":
    main()
