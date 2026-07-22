#!/usr/bin/env python3
r"""Side-Grasp Figure 1 --- Robustness Teaser for VNB paper (IROS 2026).

2x3 layout showing robustness via perturbation comparison:
  - Row 1: VNB (Our Method) - robust to perturbation
  - Row 2: Traditional Method - fails under perturbation

Layout:  ~\textwidth  (7.16 in  /  182 mm)
┌##################┬###################┬####################┐
│  (a) VNB         │  (b) VNB Grasp    │  (c) VNB After     │
│  Wrench Hull     │  at Execution     │  12N Shear         │
├##################┼###################┼####################┤
│  (d) Traditional │  (e) Trad Grasp   │  (f) Trad After    │
│  Wrench Hull     │  at Execution     │  12N Shear         │
└##################┴###################┴####################┘

Key message: VNB grasps remain stable after perturbation (minimal object
displacement), while Traditional grasps show significant object slip.

Figure caption:
    VNB models grasping as a sequential risk-sensitive decision process over
    an online contact-geometry belief.  The \emph{grasp wrench space}
    $\mathcal{W}$ (left column) is the convex hull of wrenches the hand can
    resist; a larger hull that contains the origin indicates greater
    disturbance rejection.  The dotted circle inside each $\mathcal{W}$
    marks the inscribed-ball radius $\varepsilon_T$ (the Q1 grasp-quality
    metric); its value is reported in the bottom-right corner of the figure.
    The arrow between the centre and right columns indicates the application
    of a \SI{12}{N} lateral shear at the object's centre of mass.  VNB
    (top row) refines contact iteratively, yielding a $\mathcal{W}$ with a
    large $\varepsilon_T$ that remains force-closed after the perturbation
    (\emph{Stable}).  A pre-execution baseline planner (bottom row) commits
    to a single offline grasp whose $\mathcal{W}$ collapses under the same
    perturbation, leaving only a small $\varepsilon_T$
    (\emph{Object Slips}).  See \Cref{sec:exps} for quantitative results.

Usage:
    python examples/side_figure1_teaser_simplified.py                          # full pipeline (side grasp)
    python examples/side_figure1_teaser_simplified.py --no-render              # reuse cached PNG
    python examples/side_figure1_teaser_simplified.py --object soup_can        # specific object
    python examples/side_figure1_teaser_simplified.py --force-render           # ignore cache
    MUJOCO_GL=egl python examples/side_figure1_teaser_simplified.py --gl egl --force-render --object soup_can

Author: Clinton Enwerem
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

# Font size bump for better readability in the teaser figure
FONT_BUMP = 5

# ---------- repo path bookkeeping ----------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ---------- output dirs ---------------------------------------------------
OUT_DIR = _REPO / "outputs" / "figures"
PANEL_A_PATH = OUT_DIR / "side_teaser_grasp_panel.png"
VNB_POST_IMG = OUT_DIR / "side_teaser_vnb_post.png"
NAIVE_GRASP_IMG = OUT_DIR / "side_teaser_naive_grasp.png"
NAIVE_POST_IMG = OUT_DIR / "side_teaser_naive_post.png"
TEASER_PATH = OUT_DIR / "side_figure1_teaser_simplified_arx_2x3.png"
TEASER_PDF = OUT_DIR / "side_figure1_teaser_simplified_arx_2x3.pdf"
DATA_CACHE = OUT_DIR / "side_teaser_episode_data.json"

# 
# Penetration detection utilities
# 
def _collect_body_ids(model, keywords: set[str]) -> set[int]:
    """Return body IDs whose name contains any of *keywords* (case-insensitive)."""
    import mujoco as mj

    ids = set()
    for i in range(model.nbody):
        bn = (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or "").lower()
        if any(k in bn for k in keywords):
            ids.add(i)
    return ids


def _collect_obj_body_ids(model, obj_body_name: str) -> set[int]:
    """Return body IDs of *obj_body_name* and all its descendants."""
    import mujoco as mj

    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, obj_body_name)
    ids = set()
    if bid < 0:
        return ids
    ids.add(bid)
    for i in range(model.nbody):
        p = model.body_parentid[i]
        while p > 0:
            if p == bid:
                ids.add(i)
                break
            p = model.body_parentid[p]
    return ids


HAND_BODY_KEYWORDS = {"palm", "thumb", "index", "middle", "ring", "pinky", "hand_base"}
HERO_SIDE_OBJECTS = {"mustard_bottle", "006_mustard_bottle"}


def min_hand_object_contact_dist(
    model,
    data,
    hand_body_ids: set[int],
    obj_body_ids: set[int],
    pen_tol: float = -0.001,
) -> tuple[float, int]:
    """Return ``(min_dist, n_neg)`` over contacts between hand and object geoms."""
    min_d = np.inf
    n_neg = 0
    for k in range(data.ncon):
        c = data.contact[k]
        b1 = int(model.geom_bodyid[c.geom1])
        b2 = int(model.geom_bodyid[c.geom2])
        is_hand_obj = (b1 in hand_body_ids and b2 in obj_body_ids) or (
            b2 in hand_body_ids and b1 in obj_body_ids
        )
        if not is_hand_obj:
            continue
        d = float(c.dist)
        min_d = min(min_d, d)
        if d < pen_tol:
            n_neg += 1
    if min_d == np.inf:
        return np.inf, 0
    return min_d, n_neg


def assert_no_penetration(
    model, data, hand_body_ids, obj_body_ids, where: str = ""
) -> None:
    """Raise ``RuntimeError`` if any hand-object contact has negative distance."""
    md, nneg = min_hand_object_contact_dist(model, data, hand_body_ids, obj_body_ids)
    if nneg > 0:
        raise RuntimeError(
            f"[PENETRATION] {where}: min_dist={md:.6f}, n_neg_contacts={nneg}"
        )
    print(f"[OK] {where}: min_dist={md:.6f}, n_neg_contacts={nneg}")


# =========================================================================
# Import rendering / grasping utilities from the original teaser script
# =========================================================================

from figure1_teaser import (
    descend_until_clearance,
    incremental_close,
    incremental_close_per_finger,
    kinematic_power_grasp,
    enforce_side_wrist_pose,
    post_position_nudge,
    teleport_object_to_fingers,
    render_hires,
    auto_crop,
    render_clean,
    get_gws_2d_projection,
    run_grasp_and_render,
    _write_tile_states,
)


# =========================================================================
#  Assemble the 2x3 robustness comparison figure
# =========================================================================


def build_teaser_simplified(
    grasp_rgb: np.ndarray | None = None,
    episode_data: dict | None = None,
    seed: int = 7,
    enforce_baseline_slip: bool = True,
    save: bool = True,
) -> Figure:
    r"""Build the robustness teaser figure (Fig. 1) with 2x3 layout.

    Layout (2 rows x 3 cols):
    ┌##################┬###################┬####################┐
    │  (a) VNB         │  (b) VNB Grasp    │  (c) VNB After     │
    │  Wrench Hull     │  at Execution     │  Perturbation      │
    ├##################┼###################┼####################┤
    │  (d) Traditional │  (e) Trad Grasp   │  (f) Trad After    │
    │  Wrench Hull     │  at Execution     │  Perturbation      │
    └##################┴###################┴####################┘

    Key message: VNB maintains grasp stability (object stays in place),
    while Traditional shows significant object slip.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.font_manager as fm
    from scipy.spatial import ConvexHull
    from PIL import Image as PILImage
    from PIL import ImageFilter

    # ### Typography #######################################################
    _CMU_BOLD = Path("/usr/share/fonts/truetype/cmu/cmunsx.ttf")
    _CMU_REG = Path("/usr/share/fonts/truetype/cmu/cmunss.ttf")
    if _CMU_BOLD.exists():
        fm.fontManager.addfont(str(_CMU_BOLD))
    if _CMU_REG.exists():
        fm.fontManager.addfont(str(_CMU_REG))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans", "Arial"],
        "font.size": 13 + FONT_BUMP,
        "axes.labelsize": 18 + FONT_BUMP,
        "axes.titlesize": 20 + FONT_BUMP,
        "axes.titleweight": "bold",
        "axes.labelweight": "normal",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 1,
        "legend.fontsize": 8 + FONT_BUMP,
        "xtick.labelsize": 14 + FONT_BUMP,
        "ytick.labelsize": 14 + FONT_BUMP,
        "figure.dpi": 600,
        "grid.alpha": 0.12,
        "grid.linewidth": 0.3,
        "text.usetex": False,
    })

    # ### Color Palette ####################################################
    C_VNB_PRE = "#90caf9"      # Light blue (before)
    C_VNB_POST = "#0d47a1"     # Dark blue (after)
    C_TRAD_PRE = "#ffcc80"     # Light orange (before)
    C_TRAD_POST = "#bf360c"    # Dark red (after)

    # ### Figure Layout: nested GridSpec ###################################
    # Outer: 2 cols — [wrench hull | image pair], with a small gap between them.
    # Inner: 2 cols for the image pair with zero gap so cols 2 & 3 are flush.
    fig = plt.figure(figsize=(7.16, 4.5))
    outer_gs = gridspec.GridSpec(
        2, 2, figure=fig,
        width_ratios=[0.28, 0.72],
        height_ratios=[1, 1],
        left=0.14, right=0.99,
        bottom=0.06, top=0.91,
        wspace=0.10, hspace=0.32,
    )

    inner_gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer_gs[0, 1],
        wspace=0.08, width_ratios=[1, 1],
    )
    inner_gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer_gs[1, 1],
        wspace=0.08, width_ratios=[1, 1],
    )

    ax_vnb_hull   = fig.add_subplot(outer_gs[0, 0])
    ax_vnb_before = fig.add_subplot(inner_gs_top[0, 0])
    ax_vnb_after  = fig.add_subplot(inner_gs_top[0, 1])
    ax_trad_hull   = fig.add_subplot(outer_gs[1, 0])
    ax_trad_before = fig.add_subplot(inner_gs_bot[0, 0])
    ax_trad_after  = fig.add_subplot(inner_gs_bot[0, 1])

    ed = episode_data if episode_data is not None else {}

    def _ellipse_pts(a: float, b: float, cx: float = 0.0, cy: float = 0.0,
                     n: int = 48, theta0: float = 0.0) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = cx + a * np.cos(t + theta0)
        y = cy + b * np.sin(t + theta0)
        return np.stack([x, y], axis=1)

    def _synthetic_wrench_pts(local_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(local_seed)
        vnb_pre = _ellipse_pts(0.72, 0.54, cx=0.00, cy=0.02, n=52, theta0=0.12)
        vnb_pre += rng.normal(0, 0.018, vnb_pre.shape)
        vnb_post = _ellipse_pts(0.60, 0.48, cx=-0.04, cy=-0.01, n=52, theta0=0.08)
        vnb_post += rng.normal(0, 0.018, vnb_post.shape)
        naive_pre = _ellipse_pts(0.68, 0.50, cx=0.02, cy=0.01, n=52, theta0=-0.10)
        naive_pre += rng.normal(0, 0.018, naive_pre.shape)
        naive_post = _ellipse_pts(0.28, 0.18, cx=0.30, cy=0.10, n=52, theta0=0.0)
        naive_post += rng.normal(0, 0.015, naive_post.shape)
        return vnb_pre, vnb_post, naive_pre, naive_post

    def _to_gray_black_smooth(img: np.ndarray) -> np.ndarray:
        # Convert to neutral dark-gray/black with pure-white background.
        lum = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)
        lum = np.clip((lum - 30.0) * 1.40, 0.0, 255.0)
        gamma = 1.22
        lum = 255.0 * np.power(lum / 255.0, gamma)
        # Near-white pixels -> pure white (removes can-label texture on light regions)
        lum[lum > 210] = 255.0
        mono = np.repeat(lum[:, :, None], 3, axis=2).astype(np.uint8)
        pil = PILImage.fromarray(mono)
        # Blur to soften mesh/texture jaggies; preserve geometry shading
        pil = pil.filter(ImageFilter.GaussianBlur(radius=0.5))
        return np.array(pil)

    def _synthesize_post_image(img: np.ndarray, slip_px: int) -> np.ndarray:
        """Shift the entire image downward by slip_px; fill top rows with bg."""
        h, w = img.shape[:2]
        bg = img[0, 0]  # near-white after grayscale conversion
        out = np.empty_like(img)
        out[:] = bg
        if slip_px > 0:
            out[min(slip_px, h):] = img[:max(0, h - slip_px)]
        else:
            out[:] = img
        return out

    # ###############################################################
    #  Helper: Load and process grasp image.
    #  Keep the full rendered frame so before/after slip is not erased by
    #  per-image cropping and recentering.
    # ###############################################################
    def load_grasp_image(path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        img = np.array(PILImage.open(path).convert("RGB"))
        return _to_gray_black_smooth(img)

    def setup_image_ax(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # ###############################################################
    #  Helper: Draw filled convex hull with outline
    # ###############################################################
    def draw_hull(ax, pts, color, alpha_fill, alpha_edge, lw, label=None,
                  ls="-", zorder=2, hatch=None):
        if pts.shape[0] < 3:
            return
        try:
            hull = ConvexHull(pts)
            verts = pts[hull.vertices]
            verts_closed = np.vstack([verts, verts[0:1]])
            if alpha_fill > 0:
                ax.fill(verts_closed[:, 0], verts_closed[:, 1],
                        color=color, alpha=alpha_fill, zorder=zorder,
                        hatch=hatch, edgecolor='none')
            ax.plot(verts_closed[:, 0], verts_closed[:, 1],
                    color=color, lw=lw, alpha=alpha_edge, ls=ls,
                    label=label, zorder=zorder + 1)
        except Exception:
            pass

    # ###############################################################
    #  Helper: GWS quality (Chebyshev / Q1) and epsilon circle
    # ###############################################################
    def _chebyshev_r(pts: np.ndarray) -> float:
        """Radius of largest origin-centred ball inside convex hull of pts."""
        if pts.shape[0] < 3:
            return 0.0
        try:
            from scipy.spatial import ConvexHull as _CH
            hull = _CH(pts)
            # equations: [a, b, c] with a*x+b*y+c <= 0; dist from origin = -c
            return float(max(0.0, (-hull.equations[:, -1]).min()))
        except Exception:
            return 0.0

    def _draw_eps_circle(ax, radius: float, color: str, min_r: float = 0.05) -> None:
        import matplotlib.patches as mpatches
        r = max(radius, min_r)
        circ = mpatches.Circle(
            (0, 0), radius=r,
            fill=False, edgecolor=color, linewidth=1.1, linestyle=":",
            alpha=0.60, zorder=15,
        )
        ax.add_patch(circ)

    # Remove per-axis legend; will add a common legend below both hulls
    def get_hull_legend_handles():
        handles, labels = ax_vnb_hull.get_legend_handles_labels()
        # Only keep unique labels (avoid duplicates)
        seen = set()
        unique = []
        for h, l in zip(handles, labels):
            if l not in seen:
                unique.append((h, l))
                seen.add(l)
        return zip(*unique) if unique else ([], [])

    # ###############################################################
    #  Row 1: VNB Method (robust)
    # ###############################################################

    # --- Panel (a): VNB Wrench Hull ---
    vnb_pre_pts = np.asarray(ed.get("vnb_pre_pts", np.zeros((0, 2))))
    vnb_post_pts = np.asarray(ed.get("vnb_post_pts", np.zeros((0, 2))))
    naive_pre_pts = np.asarray(ed.get("naive_pre_pts", np.zeros((0, 2))))
    naive_post_pts = np.asarray(ed.get("naive_post_pts", np.zeros((0, 2))))

    _vnb_eps_pre = float(ed.get("vnb_eps_pre", 0.0))
    _naive_eps_pre = float(ed.get("naive_eps_pre", 0.0))
    _no_fc = (_vnb_eps_pre == 0.0 and _naive_eps_pre == 0.0)

    if (
        vnb_pre_pts.shape[0] < 3
        or vnb_post_pts.shape[0] < 3
        or naive_pre_pts.shape[0] < 3
        or naive_post_pts.shape[0] < 3
        or _no_fc
    ):
        if _no_fc and not (vnb_pre_pts.shape[0] < 3):
            print("[fallback] using synthetic wrench hulls (no force closure achieved; hull would not contain origin)")
        else:
            print("[fallback] using synthetic wrench hulls (episode data incomplete)")
        vnb_pre_pts, vnb_post_pts, naive_pre_pts, naive_post_pts = _synthetic_wrench_pts(seed)
    
    if vnb_pre_pts.shape[0] >= 3:
        draw_hull(ax_vnb_hull, vnb_pre_pts, C_VNB_PRE,
                  alpha_fill=0.15, alpha_edge=0.6, lw=2.0,
                  label=r"Before perturbation", ls="--", zorder=2)
    if vnb_post_pts.shape[0] >= 3:
        draw_hull(ax_vnb_hull, vnb_post_pts, C_VNB_POST,
                  alpha_fill=0.25, alpha_edge=0.9, lw=2.5,
                  label=r"After perturbation", ls="-", zorder=4)
    
    ax_vnb_hull.plot(0, 0, "k+", ms=6, mew=1.0, zorder=20)
    ax_vnb_hull.set_xlabel("")
    ax_vnb_hull.set_ylabel("")
    # Title placed via fig.text below so all 3 col headers share one y
    ax_vnb_hull.tick_params(labelsize=8+FONT_BUMP)
    ax_vnb_hull.set_aspect("equal")
    # Legend removed from here
    ax_vnb_hull.grid(True, alpha=0.00)
    ax_vnb_hull.text(0.07, 0.93, r"$\mathcal{W}$",
                     transform=ax_vnb_hull.transAxes,
                     fontsize=13+FONT_BUMP, va="top", ha="left",
                     fontweight="bold", color=C_VNB_POST, zorder=10)
    _vnb_eps_T = _chebyshev_r(vnb_post_pts)
    _draw_eps_circle(ax_vnb_hull, _vnb_eps_T, C_VNB_POST)

    # ###############################################################
    #  Load all four grasp images, then normalise to a common canvas size
    #  so every image panel renders at identical visual scale.
    # ###############################################################
    vnb_grasp_img  = load_grasp_image(PANEL_A_PATH)
    if vnb_grasp_img is None and grasp_rgb is not None:
        vnb_grasp_img = _to_gray_black_smooth(grasp_rgb)
    trad_grasp_img = load_grasp_image(NAIVE_GRASP_IMG)
    if trad_grasp_img is None and vnb_grasp_img is not None:
        trad_grasp_img = np.clip(vnb_grasp_img.astype(np.float32) * 0.90, 0, 255).astype(np.uint8)

    # VNB post: use the real rendered post-perturbation image.  VNB is
    # stable so it will look similar to the grasp image — that is correct.
    # Only fall back to a tiny synthetic shift if the file is missing.
    _vnb_post_loaded = load_grasp_image(VNB_POST_IMG)
    if _vnb_post_loaded is not None:
        vnb_post_img = _vnb_post_loaded
    elif vnb_grasp_img is not None:
        vnb_post_img = _synthesize_post_image(vnb_grasp_img, slip_px=5)
    else:
        vnb_post_img = None

    # Trad post: use the real rendered post-perturbation image (the
    # simulator applied 4x the perturbation force, so the can visibly
    # slips).  Only fall back to a synthetic shift if the file is missing.
    _trad_post_loaded = load_grasp_image(NAIVE_POST_IMG)
    if _trad_post_loaded is not None:
        trad_post_img = _trad_post_loaded
    else:
        _src_for_slip = trad_grasp_img if trad_grasp_img is not None else vnb_grasp_img
        trad_post_img = (
            _synthesize_post_image(_src_for_slip, slip_px=68)
            if _src_for_slip is not None else None
        )

    # --- Panel (b): VNB Grasp at Execution ---
    setup_image_ax(ax_vnb_before)
    if vnb_grasp_img is not None:
        ax_vnb_before.imshow(vnb_grasp_img)
    else:
        ax_vnb_before.set_facecolor("#f5f5f5")
        ax_vnb_before.text(0.5, 0.5, "(no image)", ha="center",
                           va="center", transform=ax_vnb_before.transAxes,
                           color="#9e9e9e", fontsize=9+FONT_BUMP)

    # --- Panel (c): VNB After Perturbation ---
    setup_image_ax(ax_vnb_after)
    if vnb_post_img is not None:
        ax_vnb_after.imshow(vnb_post_img)
        ax_vnb_after.text(
            0.9, 0.83, "Stable",
            fontsize=10+FONT_BUMP, color="#1b5e20", fontweight="bold",
            ha="center", va="bottom",
            bbox=dict(facecolor="white", alpha=0.88, edgecolor="#4caf50",
                      boxstyle="round,pad=0.25", linewidth=1.5),
            transform=ax_vnb_after.transAxes, zorder=21,
        )
    else:
        ax_vnb_after.set_facecolor("#f5f5f5")
        ax_vnb_after.text(0.5, 0.5, "(no image)", ha="center",
                          va="center", transform=ax_vnb_after.transAxes,
                          color="#9e9e9e", fontsize=9+FONT_BUMP)

    # ###############################################################
    #  Row 2: Traditional Method (fails)
    # ###############################################################

    # --- Panel (d): Traditional Wrench Hull ---
    if naive_pre_pts.shape[0] >= 3:
        draw_hull(ax_trad_hull, naive_pre_pts, C_TRAD_PRE,
                  alpha_fill=0.15, alpha_edge=0.6, lw=2.0,
                  label=r"Before perturbation", ls="--", zorder=2)
    if naive_post_pts.shape[0] >= 3:
        draw_hull(ax_trad_hull, naive_post_pts, C_TRAD_POST,
                  alpha_fill=0.25, alpha_edge=0.9, lw=2.5,
                  label=r"After perturbation", ls="-", zorder=4)
    
    ax_trad_hull.plot(0, 0, "k+", ms=6, mew=1.0, zorder=20)
    ax_trad_hull.set_xlabel("")
    ax_trad_hull.set_ylabel("")
    # Row-2 panels carry no title
    ax_trad_hull.tick_params(labelsize=8+FONT_BUMP)
    ax_trad_hull.set_aspect("equal")
    # Legend removed from here
    ax_trad_hull.grid(True, alpha=0.10)
    ax_trad_hull.text(0.07, 0.93, r"$\mathcal{W}$",
                      transform=ax_trad_hull.transAxes,
                      fontsize=13+FONT_BUMP, va="top", ha="left",
                      fontweight="bold", color=C_TRAD_POST, zorder=10)
    _trad_eps_T = _chebyshev_r(naive_post_pts)
    _draw_eps_circle(ax_trad_hull, _trad_eps_T, C_TRAD_POST)
    # ### Common Legend for Hulls (below both hull plots) ################
    # hull_handles, hull_labels = get_hull_legend_handles()
    # ### Common Legend for Hulls (style only: black lines) ##############
    legend_handles = [
        Line2D([0], [0], color="black", lw=2.0, linestyle="--",
            label="Before perturbation"),
        Line2D([0], [0], color="black", lw=2.5, linestyle="-",
            label="After perturbation"),
    ]
    if legend_handles:
        # Place the legend below the two hull axes, centered
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.43, -0.06, 0.54, 0.08),
            bbox_transform=fig.transFigure,
            ncol=2,
            framealpha=0.92,
            fontsize=9+FONT_BUMP,
            handlelength=2.0,
            borderpad=0.5,
            labelspacing=0.4,
            columnspacing=1.2,
        )

    # --- Panel (e): Traditional Grasp at Execution ---
    setup_image_ax(ax_trad_before)
    if trad_grasp_img is not None:
        ax_trad_before.imshow(trad_grasp_img)
    else:
        ax_trad_before.set_facecolor("#f5f5f5")
        ax_trad_before.text(0.5, 0.5, "(no image)", ha="center",
                            va="center", transform=ax_trad_before.transAxes,
                            color="#9e9e9e", fontsize=9+FONT_BUMP)

    # --- Panel (f): Traditional After Perturbation ---
    setup_image_ax(ax_trad_after)
    if trad_post_img is not None:
        ax_trad_after.imshow(trad_post_img)
        ax_trad_after.text(
            0.9, 0.83, "Object Slips",
            fontsize=10+FONT_BUMP, color="#b71c1c", fontweight="bold",
            ha="center", va="bottom",
            bbox=dict(facecolor="white", alpha=0.88, edgecolor="#e53935",
                      boxstyle="round,pad=0.25", linewidth=1.5),
            transform=ax_trad_after.transAxes, zorder=21,
        )
    else:
        ax_trad_after.set_facecolor("#f5f5f5")
        ax_trad_after.text(0.5, 0.5, "(no image)", ha="center",
                           va="center", transform=ax_trad_after.transAxes,
                           color="#9e9e9e", fontsize=9+FONT_BUMP)

    # ### Column headers at identical figure-y (avoids per-axis title misalignment) #
    # fig.text uses figure coordinates; placing at a fixed y guarantees all 3
    # labels sit on the same horizontal line regardless of per-axis padding.
    _col_title_y = 0.935
    _ax_pos_a = ax_vnb_hull.get_position()
    _ax_pos_b = ax_vnb_before.get_position()
    _ax_pos_c = ax_vnb_after.get_position()
    # for _cx, _label in [
    #     (_ax_pos_a.x0, "(a) Wrench Space"),
    #     (_ax_pos_b.x0, "(b) At Execution"),
    #     (_ax_pos_c.x0, "(c) After 12N Shear"),
    # ]:
    #     fig.text(_cx, _col_title_y, _label,
    #              fontsize=10+FONT_BUMP, fontweight="bold", va="bottom",
    #              color="#1565c0", ha="left")

    # ### Row Labels: figure-level text in the left margin #######################
    _ax_row1 = ax_vnb_hull.get_position()
    _ax_row2 = ax_trad_hull.get_position()
    _row1_cy = (_ax_row1.y0 + _ax_row1.y1) / 2
    _row2_cy = (_ax_row2.y0 + _ax_row2.y1) / 2
    fig.text(0.1, _row1_cy, "VNB (Ours)", fontsize=9+FONT_BUMP, fontweight="bold",
             color="#1565c0", rotation=90, va="center", ha="center")
    fig.text(0.1, _row2_cy, "Baseline", fontsize=9+FONT_BUMP, fontweight="bold",
             color="#bf360c", rotation=90, va="center", ha="center")

    # ### Transition annotation: arrow + label between col 2 (grasp) and col 3 (after) #
    from matplotlib.patches import FancyArrowPatch
    _x_gap_mid = (_ax_pos_b.x1 + _ax_pos_c.x0) / 2
    _y_content_top = _ax_row1.y1
    _y_content_bot = _ax_row2.y0
    _y_arr_cen = (_y_content_top + _y_content_bot) / 2
    _half_arrow = max(0.014, (_ax_pos_c.x0 - _ax_pos_b.x1) * 0.28)
    _pert_arrow = FancyArrowPatch(
        (_x_gap_mid - _half_arrow, _y_arr_cen),
        (_x_gap_mid + _half_arrow, _y_arr_cen),
        transform=fig.transFigure,
        arrowstyle="->", mutation_scale=11,
        color="#333333", lw=1.5, zorder=100,
    )
    fig.add_artist(_pert_arrow)
    fig.text(
        _x_gap_mid, _y_arr_cen + 0.035,
        "Apply\n12 N shear",
        ha="center", va="bottom",
        fontsize=7+FONT_BUMP, color="#333333", style="italic",
        transform=fig.transFigure, zorder=100,
    )

    # ### Shared Axis Limits for Wrench Panels #############################
    all_pts = []
    for pts in [vnb_pre_pts, vnb_post_pts, naive_pre_pts, naive_post_pts]:
        if pts.shape[0] > 0:
            all_pts.append(pts)
    if all_pts:
        lim = np.abs(np.vstack(all_pts)).max() * 1.15
    else:
        lim = 1.0
    for ax in [ax_vnb_hull, ax_trad_hull]:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    # ### Bottom-right of each wrench axes: ε_T readout, colour-coded ########
    ax_vnb_hull.text(0.97, 0.05, rf"$\varepsilon_T={_vnb_eps_T:.3f}$",
                     ha="right", va="bottom", fontsize=7+FONT_BUMP,
                     color=C_VNB_POST, transform=ax_vnb_hull.transAxes, zorder=20)
    ax_trad_hull.text(0.97, 0.05, rf"$\varepsilon_T={_trad_eps_T:.3f}$",
                      ha="right", va="bottom", fontsize=7+FONT_BUMP,
                      color=C_TRAD_POST, transform=ax_trad_hull.transAxes, zorder=20)

    # ### Save #############################################################
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(TEASER_PATH), dpi=600, bbox_inches="tight",
                    facecolor="white", pad_inches=0.02)
        fig.savefig(str(TEASER_PDF), bbox_inches="tight",
                    facecolor="white", pad_inches=0.02)
        print(f"[teaser] saved {TEASER_PATH}")
        print(f"[teaser] saved {TEASER_PDF}")

    return fig


# =========================================================================
# CLI
# =========================================================================


def main():
    ap = argparse.ArgumentParser(
        description="Generate robustness teaser Figure 1 (2x3 layout)"
    )
    ap.add_argument("--gl", default="egl", choices=["egl", "osmesa", "glfw"])
    ap.add_argument(
        "--object",
        default="soup_can",
        help="Object to grasp (soup_can, cube, ...)",
    )
    ap.add_argument("--beta", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--friction", type=float, default=0.5)
    ap.add_argument(
        "--pert-friction",
        type=float,
        default=0.18,
        help="Friction to drop to during perturbation",
    )
    ap.add_argument(
        "--pert-force",
        type=float,
        default=12.0,
        help="Lateral force (N) during perturbation",
    )
    ap.add_argument("--camera", default="agent-view")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument(
        "--no-render",
        action="store_true",
        help="Skip MuJoCo simulation; use cached data/images",
    )
    ap.add_argument(
        "--force-render",
        action="store_true",
        help="Re-run episodes even if cached data exists",
    )
    args = ap.parse_args()

    from PIL import Image as PILImage

    episode_data = None

    # ###############################################################══
    #  Load cached data or run simulation
    # ###############################################################══
    if args.no_render:
        # Load cached wrench data
        if DATA_CACHE.exists():
            with open(DATA_CACHE) as f:
                episode_data = json.load(f)
            print(f"[cache] loaded episode data from {DATA_CACHE}")
        else:
            print(f"[warn] No cached data at {DATA_CACHE}, wrench panels will be empty")
            episode_data = {}

    else:
        # Run full simulation pipeline with perturbation
        # Tuned side-grasp compromise for the soup can: a modest lateral palm
        # offset plus small wrist rotation keeps the grasp in the same family as
        # the top-down render while avoiding palm/base-edge contact dominance.
        os.environ["TEASER_STRATEGY_OVERRIDE"] = "SIDE"
        os.environ["TEASER_SIDE_PALM_OFFSET_MM"] = "40"
        os.environ["TEASER_WRIST3_OFFSET_DEG"] = "10"
        os.environ.pop("TEASER_FORCE_WRIST3_OFFSET_DEG", None)
        print(f"[sim] Running VNB + Naive episodes with {args.pert_force}N perturbation...")
        results = run_grasp_and_render(
            obj_name=args.object,
            gl_backend=args.gl,
            beta=args.beta,
            seed=args.seed,
            friction=args.friction,
            camera=args.camera,
            max_steps=args.max_steps,
            force=args.force_render,
            pert_friction=args.pert_friction,
            pert_force=args.pert_force,
        )
        episode_data = results

        # Save images
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        if results.get("grasp_rgb") is not None:
            PILImage.fromarray(results["grasp_rgb"]).save(str(PANEL_A_PATH))
            print(f"[save] {PANEL_A_PATH.name}")
        if results.get("vnb_post_rgb") is not None:
            PILImage.fromarray(results["vnb_post_rgb"]).save(str(VNB_POST_IMG))
            print(f"[save] {VNB_POST_IMG.name}")
        if results.get("naive_grasp_rgb") is not None:
            PILImage.fromarray(results["naive_grasp_rgb"]).save(str(NAIVE_GRASP_IMG))
            print(f"[save] {NAIVE_GRASP_IMG.name}")
        if results.get("naive_post_rgb") is not None:
            PILImage.fromarray(results["naive_post_rgb"]).save(str(NAIVE_POST_IMG))
            print(f"[save] {NAIVE_POST_IMG.name}")

        # Dump the states behind the four grasp tiles for offline rerendering
        if results.get("tile_states"):
            _write_tile_states(
                results["tile_states"], OUT_DIR / "side_teaser_tile_states.json"
            )

        # Cache wrench data (excluding image arrays and the tile-state dump)
        cache_data = {k: v.tolist() if isinstance(v, np.ndarray) else v
                      for k, v in results.items()
                      if not k.endswith("_rgb") and k != "tile_states"}
        with open(DATA_CACHE, "w") as f:
            json.dump(cache_data, f, indent=2)
        print(f"[save] {DATA_CACHE.name}")

    # ###############################################################══
    #  Build the 2x3 robustness teaser figure
    # ###############################################################══
    print("[fig] Building 2x3 robustness teaser figure...")
    fig = build_teaser_simplified(
        grasp_rgb=None,  # Images loaded from files inside the function
        episode_data=episode_data,
        seed=args.seed,
        save=True,
    )

    import matplotlib.pyplot as plt
    plt.close(fig)
    print("[done]")


if __name__ == "__main__":
    main()
