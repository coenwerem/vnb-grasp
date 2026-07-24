#!/usr/bin/env python3
"""
Compute statistics for paper Tables I & II from experiment summary output.
"""

import numpy as np
from collections import defaultdict

# Raw data from experiment output (parsed from the summary table)
# Format: (method, object, regime, beta, sr%, rob%, pert%, epsilon, quality, p_fail, time)
raw_data = [
    # cem - cube
    ("cem", "cube", "adversarial", 0.50, 33, 33, 30, 0.0023, 0.470, 0.702, 4.1),
    ("cem", "cube", "adversarial", 0.90, 0, 0, 0, 0.0000, 0.250, 1.000, 5.0),
    ("cem", "cube", "adversarial", 0.95, 0, 0, 0, 0.0000, 0.250, 1.000, 5.0),
    ("cem", "cube", "adversarial", 0.99, 0, 0, 0, 0.0000, 0.250, 1.000, 5.0),
    ("cem", "cube", "bimodal", 0.50, 33, 33, 30, 0.0042, 0.471, 0.702, 4.0),
    ("cem", "cube", "bimodal", 0.90, 67, 67, 60, 0.0084, 0.693, 0.405, 3.3),
    ("cem", "cube", "bimodal", 0.95, 33, 33, 30, 0.0038, 0.473, 0.702, 4.0),
    ("cem", "cube", "bimodal", 0.99, 100, 100, 89, 0.0111, 0.923, 0.107, 2.2),
    ("cem", "cube", "nominal", 0.50, 100, 100, 89, 0.0128, 0.918, 0.107, 2.6),
    ("cem", "cube", "nominal", 0.90, 100, 100, 89, 0.0151, 0.925, 0.107, 2.8),
    ("cem", "cube", "nominal", 0.95, 100, 100, 89, 0.0144, 0.928, 0.107, 2.5),
    ("cem", "cube", "nominal", 0.99, 100, 100, 89, 0.0094, 0.917, 0.107, 2.5),
    ("cem", "cube", "wide", 0.50, 67, 67, 60, 0.0076, 0.687, 0.405, 3.2),
    ("cem", "cube", "wide", 0.90, 100, 100, 89, 0.0092, 0.924, 0.107, 2.5),
    ("cem", "cube", "wide", 0.95, 67, 67, 60, 0.0109, 0.699, 0.405, 3.2),
    ("cem", "cube", "wide", 0.99, 100, 100, 89, 0.0134, 0.925, 0.107, 2.4),
    # cem - graspit_box
    ("cem", "graspit_box", "adversarial", 0.50, 0, 0, 0, 0.0000, 0.000, 1.000, 3.6),
    ("cem", "graspit_box", "adversarial", 0.90, 0, 0, 0, 0.0000, 0.000, 1.000, 3.5),
    ("cem", "graspit_box", "adversarial", 0.95, 0, 0, 0, 0.0000, 0.000, 1.000, 3.6),
    ("cem", "graspit_box", "adversarial", 0.99, 0, 0, 0, 0.0000, 0.000, 1.000, 3.7),
    ("cem", "graspit_box", "bimodal", 0.50, 67, 67, 60, 0.0106, 0.571, 0.405, 2.9),
    ("cem", "graspit_box", "bimodal", 0.90, 67, 67, 60, 0.0112, 0.814, 0.405, 2.2),
    ("cem", "graspit_box", "bimodal", 0.95, 67, 0, 1, 0.0051, 0.578, 0.988, 2.8),
    ("cem", "graspit_box", "bimodal", 0.99, 0, 0, 0, 0.0000, 0.000, 1.000, 3.5),
    ("cem", "graspit_box", "nominal", 0.50, 100, 100, 88, 0.0118, 0.854, 0.119, 2.3),
    ("cem", "graspit_box", "nominal", 0.90, 100, 67, 51, 0.0093, 0.868, 0.488, 2.3),
    ("cem", "graspit_box", "nominal", 0.95, 100, 33, 31, 0.0084, 0.869, 0.690, 2.2),
    ("cem", "graspit_box", "nominal", 0.99, 100, 67, 60, 0.0084, 0.864, 0.405, 2.4),
    ("cem", "graspit_box", "wide", 0.50, 33, 33, 30, 0.0022, 0.676, 0.702, 3.0),
    ("cem", "graspit_box", "wide", 0.90, 33, 0, 0, 0.0024, 0.556, 1.000, 2.8),
    ("cem", "graspit_box", "wide", 0.95, 67, 67, 60, 0.0109, 0.769, 0.405, 3.2),
    ("cem", "graspit_box", "wide", 0.99, 67, 33, 30, 0.0072, 0.739, 0.702, 2.8),
    # gauss - cube
    ("gauss", "cube", "adversarial", 0.50, 0, 0, 0, 0.0000, 0.170, 1.000, 5.5),
    ("gauss", "cube", "adversarial", 0.90, 33, 33, 30, 0.0007, 0.458, 0.702, 4.7),
    ("gauss", "cube", "adversarial", 0.95, 0, 0, 0, 0.0000, 0.250, 1.000, 5.6),
    ("gauss", "cube", "adversarial", 0.99, 0, 0, 0, 0.0000, 0.250, 1.000, 5.4),
    ("gauss", "cube", "bimodal", 0.50, 100, 100, 89, 0.0072, 0.915, 0.107, 2.9),
    ("gauss", "cube", "bimodal", 0.90, 67, 67, 60, 0.0104, 0.616, 0.405, 3.9),
    ("gauss", "cube", "bimodal", 0.95, 33, 33, 30, 0.0041, 0.469, 0.702, 4.7),
    ("gauss", "cube", "bimodal", 0.99, 100, 100, 89, 0.0156, 0.925, 0.107, 2.9),
    ("gauss", "cube", "nominal", 0.50, 67, 67, 60, 0.0110, 0.620, 0.405, 3.8),
    ("gauss", "cube", "nominal", 0.90, 100, 100, 89, 0.0148, 0.918, 0.107, 3.4),
    ("gauss", "cube", "nominal", 0.95, 100, 100, 89, 0.0157, 0.926, 0.107, 3.3),
    ("gauss", "cube", "nominal", 0.99, 100, 100, 89, 0.0117, 0.928, 0.107, 2.7),
    ("gauss", "cube", "wide", 0.50, 67, 67, 60, 0.0093, 0.702, 0.405, 3.8),
    ("gauss", "cube", "wide", 0.90, 100, 100, 89, 0.0114, 0.922, 0.107, 2.8),
    ("gauss", "cube", "wide", 0.95, 100, 100, 89, 0.0098, 0.922, 0.107, 3.1),
    ("gauss", "cube", "wide", 0.99, 100, 100, 89, 0.0135, 0.917, 0.107, 2.8),
    # gauss - graspit_box
    ("gauss", "graspit_box", "adversarial", 0.50, 0, 0, 0, 0.0000, 0.000, 1.000, 4.3),
    ("gauss", "graspit_box", "adversarial", 0.90, 0, 0, 0, 0.0000, 0.000, 1.000, 4.2),
    ("gauss", "graspit_box", "adversarial", 0.95, 0, 0, 0, 0.0000, 0.000, 1.000, 4.3),
    ("gauss", "graspit_box", "adversarial", 0.99, 0, 0, 0, 0.0000, 0.000, 1.000, 4.4),
    ("gauss", "graspit_box", "bimodal", 0.50, 67, 67, 60, 0.0053, 0.587, 0.405, 3.2),
    ("gauss", "graspit_box", "bimodal", 0.90, 100, 100, 89, 0.0100, 0.871, 0.107, 2.6),
    ("gauss", "graspit_box", "bimodal", 0.95, 67, 67, 60, 0.0054, 0.590, 0.405, 3.3),
    ("gauss", "graspit_box", "bimodal", 0.99, 0, 0, 0, 0.0000, 0.000, 1.000, 4.4),
    ("gauss", "graspit_box", "nominal", 0.50, 100, 67, 60, 0.0095, 0.881, 0.405, 2.6),
    ("gauss", "graspit_box", "nominal", 0.90, 100, 67, 60, 0.0067, 0.880, 0.405, 2.4),
    ("gauss", "graspit_box", "nominal", 0.95, 100, 67, 58, 0.0092, 0.862, 0.417, 2.4),
    ("gauss", "graspit_box", "nominal", 0.99, 100, 67, 60, 0.0060, 0.872, 0.405, 2.6),
    ("gauss", "graspit_box", "wide", 0.50, 33, 33, 30, 0.0022, 0.451, 0.702, 2.5),
    ("gauss", "graspit_box", "wide", 0.90, 33, 33, 30, 0.0026, 0.559, 0.702, 2.4),
    ("gauss", "graspit_box", "wide", 0.95, 100, 33, 35, 0.0066, 0.874, 0.655, 2.8),
    ("gauss", "graspit_box", "wide", 0.99, 100, 33, 30, 0.0079, 0.885, 0.702, 2.7),
    # gauss_cvar - cube
    ("gauss_cvar", "cube", "adversarial", 0.50, 33, 33, 30, 0.0024, 0.486, 0.702, 4.8),
    ("gauss_cvar", "cube", "adversarial", 0.90, 0, 0, 0, 0.0000, 0.250, 1.000, 5.6),
    ("gauss_cvar", "cube", "adversarial", 0.95, 0, 0, 0, 0.0000, 0.250, 1.000, 5.5),
    ("gauss_cvar", "cube", "adversarial", 0.99, 0, 0, 0, 0.0000, 0.250, 1.000, 5.7),
    ("gauss_cvar", "cube", "bimodal", 0.50, 33, 33, 30, 0.0044, 0.471, 0.702, 4.6),
    ("gauss_cvar", "cube", "bimodal", 0.90, 67, 67, 60, 0.0093, 0.691, 0.405, 3.8),
    ("gauss_cvar", "cube", "bimodal", 0.95, 33, 33, 30, 0.0041, 0.467, 0.702, 4.6),
    ("gauss_cvar", "cube", "bimodal", 0.99, 100, 100, 89, 0.0167, 0.931, 0.107, 3.0),
    ("gauss_cvar", "cube", "nominal", 0.50, 100, 100, 89, 0.0127, 0.925, 0.107, 3.1),
    ("gauss_cvar", "cube", "nominal", 0.90, 100, 100, 89, 0.0165, 0.926, 0.107, 3.3),
    ("gauss_cvar", "cube", "nominal", 0.95, 100, 100, 89, 0.0159, 0.928, 0.107, 3.1),
    ("gauss_cvar", "cube", "nominal", 0.99, 100, 100, 89, 0.0098, 0.916, 0.107, 2.6),
    ("gauss_cvar", "cube", "wide", 0.50, 67, 67, 60, 0.0067, 0.681, 0.405, 3.9),
    ("gauss_cvar", "cube", "wide", 0.90, 100, 100, 89, 0.0114, 0.921, 0.107, 2.7),
    ("gauss_cvar", "cube", "wide", 0.95, 67, 67, 60, 0.0105, 0.702, 0.405, 3.8),
    ("gauss_cvar", "cube", "wide", 0.99, 100, 100, 89, 0.0152, 0.924, 0.107, 2.6),
    # gauss_cvar - graspit_box
    ("gauss_cvar", "graspit_box", "adversarial", 0.50, 0, 0, 0, 0.0000, 0.000, 1.000, 4.2),
    ("gauss_cvar", "graspit_box", "adversarial", 0.90, 0, 0, 0, 0.0000, 0.000, 1.000, 4.1),
    ("gauss_cvar", "graspit_box", "adversarial", 0.95, 0, 0, 0, 0.0000, 0.000, 1.000, 4.4),
    ("gauss_cvar", "graspit_box", "adversarial", 0.99, 0, 0, 0, 0.0000, 0.000, 1.000, 4.5),
    ("gauss_cvar", "graspit_box", "bimodal", 0.50, 67, 67, 60, 0.0068, 0.576, 0.405, 3.0),
    ("gauss_cvar", "graspit_box", "bimodal", 0.90, 100, 67, 63, 0.0085, 0.880, 0.369, 2.6),
    ("gauss_cvar", "graspit_box", "bimodal", 0.95, 67, 33, 18, 0.0052, 0.575, 0.821, 3.1),
    ("gauss_cvar", "graspit_box", "bimodal", 0.99, 0, 0, 0, 0.0000, 0.000, 1.000, 4.2),
    ("gauss_cvar", "graspit_box", "nominal", 0.50, 100, 67, 60, 0.0073, 0.894, 0.405, 2.6),
    ("gauss_cvar", "graspit_box", "nominal", 0.90, 100, 100, 87, 0.0086, 0.875, 0.131, 2.6),
    ("gauss_cvar", "graspit_box", "nominal", 0.95, 100, 0, 8, 0.0059, 0.870, 0.917, 2.6),
    ("gauss_cvar", "graspit_box", "nominal", 0.99, 100, 67, 57, 0.0064, 0.870, 0.429, 2.6),
    ("gauss_cvar", "graspit_box", "wide", 0.50, 33, 0, 0, 0.0020, 0.497, 1.000, 2.4),
    ("gauss_cvar", "graspit_box", "wide", 0.90, 33, 33, 30, 0.0020, 0.571, 0.702, 3.0),
    ("gauss_cvar", "graspit_box", "wide", 0.95, 67, 67, 60, 0.0042, 0.763, 0.405, 2.4),
    ("gauss_cvar", "graspit_box", "wide", 0.99, 100, 67, 60, 0.0109, 0.874, 0.405, 2.4),
    # variational (Ours) - cube
    ("variational", "cube", "adversarial", 0.50, 0, 0, 0, 0.0000, 0.085, 1.000, 6.2),
    ("variational", "cube", "adversarial", 0.90, 0, 0, 0, 0.0000, 0.000, 1.000, 6.0),
    ("variational", "cube", "adversarial", 0.95, 0, 0, 0, 0.0000, 0.000, 1.000, 6.8),
    ("variational", "cube", "adversarial", 0.99, 0, 0, 0, 0.0000, 0.000, 1.000, 6.1),
    ("variational", "cube", "bimodal", 0.50, 33, 33, 30, 0.0044, 0.300, 0.702, 6.3),
    ("variational", "cube", "bimodal", 0.90, 67, 67, 60, 0.0084, 0.599, 0.405, 4.7),
    ("variational", "cube", "bimodal", 0.95, 33, 33, 30, 0.0044, 0.299, 0.702, 5.7),
    ("variational", "cube", "bimodal", 0.99, 100, 100, 89, 0.0130, 0.906, 0.107, 3.7),
    ("variational", "cube", "nominal", 0.50, 67, 67, 60, 0.0058, 0.596, 0.405, 5.1),
    ("variational", "cube", "nominal", 0.90, 100, 100, 89, 0.0145, 0.918, 0.107, 4.1),
    ("variational", "cube", "nominal", 0.95, 100, 100, 89, 0.0139, 0.912, 0.107, 5.7),
    ("variational", "cube", "nominal", 0.99, 67, 67, 60, 0.0074, 0.613, 0.405, 6.5),
    ("variational", "cube", "wide", 0.50, 67, 67, 60, 0.0063, 0.597, 0.405, 4.4),
    ("variational", "cube", "wide", 0.90, 67, 67, 60, 0.0071, 0.601, 0.405, 4.4),
    ("variational", "cube", "wide", 0.95, 67, 67, 60, 0.0072, 0.618, 0.405, 4.1),
    ("variational", "cube", "wide", 0.99, 100, 100, 89, 0.0129, 0.909, 0.107, 4.4),
    # variational (Ours) - graspit_box
    ("variational", "graspit_box", "adversarial", 0.50, 0, 0, 0, 0.0000, 0.000, 1.000, 7.3),
    ("variational", "graspit_box", "adversarial", 0.90, 0, 0, 0, 0.0000, 0.000, 1.000, 5.4),
    ("variational", "graspit_box", "adversarial", 0.95, 0, 0, 0, 0.0000, 0.000, 1.000, 5.2),
    ("variational", "graspit_box", "adversarial", 0.99, 0, 0, 0, 0.0000, 0.000, 1.000, 6.2),
    ("variational", "graspit_box", "bimodal", 0.50, 67, 0, 0, 0.0055, 0.571, 1.000, 5.4),
    ("variational", "graspit_box", "bimodal", 0.90, 67, 33, 26, 0.0045, 0.817, 0.738, 5.1),
    ("variational", "graspit_box", "bimodal", 0.95, 67, 33, 30, 0.0083, 0.571, 0.702, 5.1),
    ("variational", "graspit_box", "bimodal", 0.99, 0, 0, 0, 0.0000, 0.000, 1.000, 5.3),
    ("variational", "graspit_box", "nominal", 0.50, 100, 100, 79, 0.0115, 0.863, 0.214, 3.5),
    ("variational", "graspit_box", "nominal", 0.90, 100, 33, 45, 0.0077, 0.872, 0.548, 3.2),
    ("variational", "graspit_box", "nominal", 0.95, 100, 67, 60, 0.0082, 0.876, 0.405, 3.3),
    ("variational", "graspit_box", "nominal", 0.99, 100, 33, 30, 0.0068, 0.879, 0.702, 3.2),
    ("variational", "graspit_box", "wide", 0.50, 33, 0, 0, 0.0016, 0.475, 1.000, 3.5),
    ("variational", "graspit_box", "wide", 0.90, 33, 0, 0, 0.0043, 0.296, 1.000, 6.4),
    ("variational", "graspit_box", "wide", 0.95, 67, 67, 60, 0.0107, 0.743, 0.405, 3.8),
    ("variational", "graspit_box", "wide", 0.99, 100, 67, 60, 0.0099, 0.862, 0.405, 4.3),
]

def compute_table_stats():
    """Compute statistics for Table I: aggregate by method and regime."""
    
    method_display = {
        "gauss": "Gauss",
        "gauss_cvar": "Gauss-CVaR",
        "cem": "CEM",
        "variational": "Ours (VNB)"
    }

    # Group by method and regime
    stats = defaultdict(lambda: defaultdict(list))
    
    for row in raw_data:
        method, obj, regime, beta, sr, rob, pert, eps, qual, pfail, time = row
        key = (method, regime)
        stats[key]["sr"].append(sr)
        stats[key]["rob"].append(rob)
        stats[key]["pert"].append(pert)
        stats[key]["eps"].append(eps)
        stats[key]["qual"].append(qual)
        stats[key]["pfail"].append(pfail)
        stats[key]["time"].append(time)
    
    print("=" * 100)
    print("TABLE I: Aggregate performance across friction regimes")
    print("(mean over objects, beta values, and seeds)")
    print("=" * 100)
    print(f"{'Method':<15} {'Regime':<12} {'SR%':>6} {'Rob%':>6} {'Pert%':>7} {'eps':>8} {'Qual':>6} {'P_fail':>7} {'Time':>6}")
    print("-" * 100)
    
    regimes_order = ["nominal", "adversarial", "wide", "bimodal"]
    methods_order = ["gauss", "gauss_cvar", "cem", "variational"]
    
    for method in methods_order:
        for regime in regimes_order:
            key = (method, regime)
            if key in stats:
                s = stats[key]
                sr_mean = np.mean(s["sr"])
                rob_mean = np.mean(s["rob"])
                pert_mean = np.mean(s["pert"])
                eps_mean = np.mean(s["eps"])
                qual_mean = np.mean(s["qual"])
                pfail_mean = np.mean(s["pfail"])
                time_mean = np.mean(s["time"])
                
                print(f"{method_display[method]:<15} {regime:<12} {sr_mean:>5.0f}% {rob_mean:>5.0f}% {pert_mean:>6.0f}% {eps_mean:>8.4f} {qual_mean:>6.2f} {pfail_mean:>7.2f} {time_mean:>5.1f}s")
        print()
    
    print()

    print("=" * 100)
    print("TABLE I - LaTeX Format:")
    print("=" * 100)
    for method in methods_order:
        for i, regime in enumerate(regimes_order):
            key = (method, regime)
            if key in stats:
                s = stats[key]
                sr = int(round(np.mean(s["sr"])))
                rob = int(round(np.mean(s["rob"])))
                pert = int(round(np.mean(s["pert"])))
                eps = np.mean(s["eps"])
                qual = np.mean(s["qual"])
                pfail = np.mean(s["pfail"])
                time = np.mean(s["time"])
                
                prefix = f"\\multirow{{4}}{{*}}{{\\small\\{method if method != 'variational' else 'textbf{Ours}'}}}" if i == 0 else "  "
                print(f"{prefix}")
                print(f"  & {regime:<12} & {sr:<10} & {rob:<10} & {pert:<14} & {eps:.4f} & {qual:.2f}  & {pfail:.2f} & {time:.1f}   \\\\")
        print("\\midrule")

def compute_tail_stats():
    """Compute statistics for Table II: tail statistics."""
    
    method_display = {
        "gauss": "Gauss",
        "gauss_cvar": "Gauss-CVaR",
        "cem": "CEM", 
        "variational": "Ours (VNB)"
    }
    
    qual_by_method = defaultdict(list)
    
    for row in raw_data:
        method, obj, regime, beta, sr, rob, pert, eps, qual, pfail, time = row
        qual_by_method[method].append(qual)
    
    print("=" * 80)
    print("TABLE II: Tail statistics of contact quality across all regimes")
    print("=" * 80)
    print(f"{'Method':<15} {'Mean':>8} {'S.D.':>8} {'W-10%':>8} {'W-5%':>8} {'Min':>8}")
    print("-" * 80)
    
    methods_order = ["gauss", "gauss_cvar", "cem", "variational"]
    
    for method in methods_order:
        quals = np.array(qual_by_method[method])
        n = len(quals)
        
        # Sort to get worst cases
        sorted_quals = np.sort(quals)
        
        mean_q = np.mean(quals)
        std_q = np.std(quals)
        
        n_10 = max(1, int(np.ceil(n * 0.10)))
        w10 = np.mean(sorted_quals[:n_10])

        n_5 = max(1, int(np.ceil(n * 0.05)))
        w5 = np.mean(sorted_quals[:n_5])
        
        min_q = np.min(quals)
        
        print(f"{method_display[method]:<15} {mean_q:>8.2f} {std_q:>8.2f} {w10:>8.2f} {w5:>8.2f} {min_q:>8.2f}")
    
    print()

    print("=" * 80)
    print("TABLE II - LaTeX Format:")
    print("=" * 80)
    for method in methods_order:
        quals = np.array(qual_by_method[method])
        n = len(quals)
        sorted_quals = np.sort(quals)
        
        mean_q = np.mean(quals)
        std_q = np.std(quals)
        n_10 = max(1, int(np.ceil(n * 0.10)))
        w10 = np.mean(sorted_quals[:n_10])
        n_5 = max(1, int(np.ceil(n * 0.05)))
        w5 = np.mean(sorted_quals[:n_5])
        min_q = np.min(quals)
        
        disp = method_display[method]
        if method == "variational":
            disp = "\\textbf{Ours}"
        else:
            disp = f"{{\\small\\{method}}}" if method != "gauss_cvar" else "{\\small\\gausscvar}"
        
        print(f"{disp:<20} & {mean_q:.2f} & {std_q:.2f} & {w10:.2f}  & {w5:.2f} & {min_q:.2f} \\\\")


def compute_aggregate_stats():
    """Compute overall aggregate statistics per method."""
    
    method_display = {
        "gauss": "Gauss",
        "gauss_cvar": "Gauss-CVaR",
        "cem": "CEM",
        "variational": "Ours (VNB)"
    }
    
    stats = defaultdict(lambda: {"sr": [], "rob": [], "pert": [], "eps": [], "pfail": [], "n": 0})
    
    for row in raw_data:
        method, obj, regime, beta, sr, rob, pert, eps, qual, pfail, time = row
        stats[method]["sr"].append(sr)
        stats[method]["rob"].append(rob)
        stats[method]["pert"].append(pert)
        stats[method]["eps"].append(eps)
        stats[method]["pfail"].append(pfail)
        stats[method]["n"] += 1
    
    print("=" * 80)
    print("OVERALL AGGREGATE STATISTICS (Per Method)")
    print("=" * 80)
    print(f"{'Method':<15} {'SR%':>8} {'Rob%':>8} {'Pert%':>8} {'eps':>10} {'P_fail':>8} {'N':>6}")
    print("-" * 80)
    
    methods_order = ["gauss", "gauss_cvar", "cem", "variational"]
    
    for method in methods_order:
        s = stats[method]
        print(f"{method_display[method]:<15} {np.mean(s['sr']):>7.0f}% {np.mean(s['rob']):>7.0f}% {np.mean(s['pert']):>7.0f}% {np.mean(s['eps']):>10.4f} {np.mean(s['pfail']):>8.3f} {s['n']:>6}")


def compute_success_counts():
    """Compute success and robustness counts for cross-checking."""
    
    methods_order = ["gauss", "gauss_cvar", "cem", "variational"]
    regimes_order = ["nominal", "adversarial", "wide", "bimodal"]
    
    print("=" * 100)
    print("SUCCESS/ROBUSTNESS COUNTS (for verification)")
    print("=" * 100)
    
    for method in methods_order:
        print(f"\n{method.upper()}")
        for regime in regimes_order:
            successes = 0
            robust = 0
            total = 0
            for row in raw_data:
                if row[0] == method and row[2] == regime:
                    total += 1
                    if row[4] > 50:  # SR > 50%
                        successes += 1
                    if row[5] > 50:  # Rob > 50%
                        robust += 1
            print(f"  {regime}: {successes}/{total} successful, {robust}/{total} robust")


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("PAPER STATISTICS COMPUTATION")
    print("=" * 100 + "\n")
    
    compute_table_stats()
    print("\n")
    compute_tail_stats()
    print("\n")
    compute_aggregate_stats()
    print("\n")
    compute_success_counts()
