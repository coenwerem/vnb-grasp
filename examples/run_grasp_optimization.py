#!/usr/bin/env python3
"""Grasp optimization: solve, save JSON results, render visualizations.

Supports multiple optimization methods:
- sqp: SQP-based optimizer (default)
- precision: Precision-grip optimizer (standoff-based, zero proximal penetration)
Outputs
-------
outputs/sampled_grasps/<run_id>_<object>_<method>_grasps.json
    Full run record: config, per-grasp metrics, qpos, fingertip positions.
    grasp_01_<view>.png  ...  (one multi-view panel per top-k grasp)
    top_k_panel.png           (all top-k grasps in iso view, side-by-side)
    summary_card.png          (quality bar chart + best-grasp render)
Usage
-----
    # Default: SQP method, 32 multi-start, top-5, cube object
    python examples/run_grasp_optimization.py
    # Precision grip (standoff-based, zero proximal penetration)
    python examples/run_grasp_optimization.py --method precision --starts 192 --top-k 20

    # Custom SQP
    python examples/run_grasp_optimization.py --method sqp --starts 64 --top-k 8 --friction 1.0
    # Specify object
    python examples/run_grasp_optimization.py --object cube
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import mujoco
except ImportError:
    print("mujoco is required: pip install mujoco")
    sys.exit(1)

from vnb_grasp.grasping.object_surface import ObjectSurface, GeomKind
from vnb_grasp.grasping.grasp_optimizer import GraspOptimizer, OptimizerConfig
from vnb_grasp.grasping.grasp_sampler import SampledGrasp


#
# JSON serialisation helpers
#


class _NumpyEncoder(json.JSONEncoder):
    """Serialise numpy arrays / scalars to native Python types"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def _vec3(v) -> List[float]:
    return [float(x) for x in v]


def _grasp_to_dict(
    grasp: SampledGrasp,
    rank: int,
    viz_path: Optional[str] = None,
    mesh_pen_mm: float = 0.0,
) -> dict:
    """Serialise a SampledGrasp to a plain dict"""
    d: dict = {
        "rank": rank,
        "seed_source": grasp.seed_source,
        # GWS quality
        "epsilon": round(float(grasp.gws.epsilon), 6),
        "volume": round(float(grasp.gws.volume), 6),
        "min_singular": round(float(grasp.gws.min_singular), 6),
        "is_force_closure": bool(grasp.gws.is_force_closure),
        "n_contacts": int(grasp.gws.n_contacts),
        # IK quality
        "residual_m2": round(float(grasp.residual), 8),
        "max_tip_penetration_mm": round(float(grasp.max_penetration * 1000), 3),
        "max_mesh_penetration_mm": round(float(mesh_pen_mm), 3),
        # Configuration
        "hand_qpos": grasp.hand_qpos.tolist(),
        "finger_qpos": {fn: q.tolist() for fn, q in grasp.finger_qpos.items()},
        "fingertip_positions": {
            fn: _vec3(p) for fn, p in grasp.fingertip_positions.items()
        },
        "target_contacts": {fn: _vec3(p) for fn, p in grasp.target_contacts.items()},
        "target_normals": {fn: _vec3(n) for fn, n in grasp.target_normals.items()},
    }
    if viz_path is not None:
        d["viz_path"] = viz_path
    return d


def _measure_mesh_pen(
    model,
    data,
    hand_geom_ids: set,
    obj_geom_ids: set,
    grasp: SampledGrasp,
) -> float:
    """Return worst mesh-type penetration depth in mm for a grasp qpos"""
    data.qpos[:] = grasp.hand_qpos
    mujoco.mj_forward(model, data)
    worst = 0.0
    for ci in range(data.ncon):
        c = data.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        is_ho = (g1 in hand_geom_ids and g2 in obj_geom_ids) or (
            g1 in obj_geom_ids and g2 in hand_geom_ids
        )
        if is_ho and c.dist < 0:
            worst = max(worst, -c.dist)
    return worst * 1000.0


#
# Visualization helpers
#


def _render_grasp_views(
    viz,
    grasp: SampledGrasp,
    out_dir: Path,
    rank: int,
    friction: float,
    views=("iso", "front", "side", "top"),
    w: int = 800,
    h: int = 600,
) -> Path:
    """Render a 2x2 multi-view panel for a single grasp. Returns saved path"""
    try:
        from PIL import Image, ImageDraw

        HAS_PIL = True
    except ImportError:
        HAS_PIL = False

    viz.set_grasp(grasp).with_friction_coef(friction)

    imgs = []
    for view in views:
        viz.set_camera(preset=view)
        img = viz.render(w, h)
        if HAS_PIL:
            pil = Image.fromarray(img)
            draw = ImageDraw.Draw(pil)
            draw.text((10, 10), view.upper(), fill=(220, 220, 220, 255))
            img = np.array(pil)
        imgs.append(img)

    # Arrange 2x2
    row0 = np.hstack(imgs[:2])
    row1 = np.hstack(imgs[2:])
    panel = np.vstack([row0, row1])

    path = out_dir / f"grasp_{rank:02d}_views.png"
    _save_img(panel, path)
    return path


def _render_top_k_panel(
    viz,
    grasps: List[SampledGrasp],
    out_dir: Path,
    friction: float,
    view: str = "iso",
    w: int = 640,
    h: int = 480,
) -> Path:
    """Render a side-by-side panel of all top-k grasps in a single view"""
    try:
        from PIL import Image, ImageDraw

        HAS_PIL = True
    except ImportError:
        HAS_PIL = False

    imgs = []
    viz.with_friction_coef(friction)
    for rank, g in enumerate(grasps, start=1):
        viz.set_grasp(g).set_camera(preset=view)
        img = viz.render(w, h)
        if HAS_PIL:
            pil = Image.fromarray(img)
            draw = ImageDraw.Draw(pil)
            fc_str = "FC" if g.gws.is_force_closure else "--"
            label = f"#{rank}  ε={g.gws.epsilon:.4f}  {fc_str}  {g.gws.n_contacts}ctc"
            draw.rectangle([0, h - 30, w, h], fill=(20, 20, 20, 200))
            draw.text((8, h - 22), label, fill=(240, 240, 100, 255))
            src_col = (100, 200, 100, 255)
            draw.text((w - 90, h - 22), g.seed_source, fill=src_col)
            img = np.array(pil)
        imgs.append(img)

    panel = np.hstack(imgs)
    path = out_dir / "top_k_panel.png"
    _save_img(panel, path)
    return path


def _render_summary_card(
    viz,
    grasps: List[SampledGrasp],
    out_dir: Path,
    friction: float,
    n_starts: int,
    method_label: str = "SQP",
) -> Path:
    """Render a summary card: best grasp iso view + quality bars"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from PIL import Image

        HAS_BOTH = True
    except ImportError:
        HAS_BOTH = False

    path = out_dir / "summary_card.png"

    if not HAS_BOTH:
        if grasps:
            viz.set_grasp(grasps[0]).set_camera(preset="iso")
            img = viz.render(1024, 768)
            _save_img(img, path)
        return path

    best = grasps[0]
    viz.set_grasp(best).set_camera(preset="three_quarter").with_friction_coef(friction)
    render_img = viz.render(900, 700)
    render_pil = Image.fromarray(render_img)

    epsilons = [g.gws.epsilon for g in grasps]
    fc_flags = [g.gws.is_force_closure for g in grasps]
    ranks = [f"#{i + 1}" for i in range(len(grasps))]
    bar_cols = ["#4CAF50" if fc else "#E57373" for fc in fc_flags]

    fig = plt.figure(figsize=(14, 6), facecolor="#1a1a1a")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1], wspace=0.04)

    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(render_pil)
    ax_img.axis("off")
    ax_img.set_title(
        f"Best grasp  ε={best.gws.epsilon:.4f}  "
        f"{'FC ✓' if best.gws.is_force_closure else '✗'}  "
        f"{best.gws.n_contacts} contacts  "
        f"pen={best.max_penetration * 1000:.1f}mm",
        color="white",
        fontsize=13,
        pad=8,
    )

    ax_bar = fig.add_subplot(gs[1])
    ax_bar.set_facecolor("#242424")
    bars = ax_bar.barh(
        ranks[::-1], epsilons[::-1], color=bar_cols[::-1], edgecolor="none", height=0.6
    )
    for bar, eps, fc in zip(bars, epsilons[::-1], fc_flags[::-1]):
        ax_bar.text(
            bar.get_width() + max(epsilons) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{eps:.4f}",
            va="center",
            ha="left",
            color="white",
            fontsize=9,
        )
    ax_bar.set_xlabel("Ferrari-Canny ε", color="white", fontsize=11)
    ax_bar.set_title(
        f"Top-{len(grasps)} {method_label} grasps  ({n_starts} starts)\n"
        f"FC: {sum(fc_flags)}/{len(grasps)}",
        color="white",
        fontsize=11,
        pad=8,
    )
    ax_bar.tick_params(colors="white")
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#555")
    ax_bar.xaxis.label.set_color("white")
    from matplotlib.patches import Patch

    legend_els = [
        Patch(facecolor="#4CAF50", label="Force closure"),
        Patch(facecolor="#E57373", label="Not FC"),
    ]
    ax_bar.legend(
        handles=legend_els,
        facecolor="#333",
        labelcolor="white",
        fontsize=9,
        loc="lower right",
    )

    fig.suptitle(
        f"VNB-Grasp: {method_label} Grasp Optimization  |  friction μ={friction}",
        color="white",
        fontsize=14,
        y=1.01,
    )

    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _save_img(arr: np.ndarray, path: Path) -> None:
    """Save numpy HxWx3 image to path using PIL or imageio"""
    try:
        from PIL import Image

        Image.fromarray(arr).save(str(path))
    except ImportError:
        import imageio

        imageio.imwrite(str(path), arr)


#
# Main
#


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grasp optimizer: run, save JSON + render visualizations."
    )
    parser.add_argument("--arena", default="hand_object_testbed")
    parser.add_argument("--object", default="cube", help="Object body name")
    parser.add_argument(
        "--starts", type=int, default=32, help="Number of multi-start initialisations"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Keep top-k grasps")
    parser.add_argument(
        "--friction", type=float, default=1.0, help="Friction coefficient"
    )
    parser.add_argument(
        "--max-iters", type=int, default=200, help="Max SLSQP iterations per start"
    )
    parser.add_argument(
        "--n-fingers", type=int, default=5, help="Number of fingers to use"
    )
    parser.add_argument(
        "--scheme",
        default="paper",
        choices=["default", "paper", "dark", "minimal", "natural"],
    )
    parser.add_argument(
        "--no-render", action="store_true", help="Skip visualization (JSON only)"
    )
    parser.add_argument(
        "--run-id", default=None, help="Override run ID (default: timestamp)"
    )
    parser.add_argument(
        "--method",
        default="sqp",
        choices=["sqp", "precision", "contact-first", "arm-ik"],
        help="Optimization method: sqp (default), precision (standoff-based), contact-first, or arm-ik (full arm+hand)",
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Scene XML filename within arena dir (default: scene.xml). E.g. scene_mustard.xml",
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_iso = datetime.datetime.now().isoformat(timespec="seconds")
    method = args.method
    method_label = {"sqp": "SQP", "precision": "Precision-Grip", "contact-first": "Contact-First", "arm-ik": "Arm-IK"}[method]

    print(f"\n{'=' * 62}")
    print(f"  VNB-Grasp {method_label} Grasp Optimization  [{run_id}]")
    print(
        f"  object={args.object}  method={method}  starts={args.starts}  "
        f"top_k={args.top_k}  μ={args.friction}"
    )
    print(f"{'=' * 62}\n")

    #  Output directories
    viz_dir = REPO_ROOT / "outputs" / "grasp_viz" / run_id
    json_dir = REPO_ROOT / "outputs" / "sampled_grasps"
    json_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_render:
        viz_dir.mkdir(parents=True, exist_ok=True)

    #  Load scene
    if args.scene:
        scene_xml = REPO_ROOT / "arenas" / args.arena / args.scene
    else:
        scene_xml = REPO_ROOT / "arenas" / args.arena / "scene.xml"
        if not scene_xml.exists():
            scene_xml = REPO_ROOT / "arenas" / args.arena / f"{args.arena}.xml"
    if not scene_xml.exists():
        print(f"Scene not found: {scene_xml}")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    #  Build object surface
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.object)
    if bid >= 0:
        surface = ObjectSurface.from_model(model, body_name=args.object)
        # Update surface pose from simulation state
        surface.position = data.xpos[bid].copy()
        surface.rotation = data.xmat[bid].reshape(3, 3).copy()
        print(f"  Surface: {args.object} ({surface.kind.name})")
        print(f"  Object position: {surface.position}")
    else:
        surface = ObjectSurface.from_primitive(GeomKind.BOX, [0.025, 0.025, 0.025])
        print(f"  Surface: fallback primitive box")

    #  Configure optimizer
    if method == "precision":
        from vnb_grasp.grasping.precision_grip_optimizer import (
            PrecisionGripOptimizer,
            PrecisionGripConfig,
        )
        # For precision method, --starts controls n_monte_carlo_seeds (FK samples).
        # If user didn't specify --starts (still at default 32), use config default.
        mc_seeds = args.starts if args.starts != 32 else None
        cfg = PrecisionGripConfig(
            top_k=args.top_k,
            friction_coef=args.friction,
            **(dict(n_monte_carlo_seeds=mc_seeds, n_starts=mc_seeds) if mc_seeds else {}),
        )
        optimizer = PrecisionGripOptimizer(
            model,
            data,
            surface,
            config=cfg,
            object_body_name=args.object if bid >= 0 else None,
        )
        hand_geom_ids = optimizer._hand_geom_ids
        obj_geom_ids = optimizer._obj_geom_ids
    elif method == "contact-first":
        from vnb_grasp.grasping.contact_first_optimizer import (
            ContactFirstOptimizer,
            OptimizerConfig as ContactFirstConfig,
        )

        cfg = ContactFirstConfig(
            n_starts=args.starts,
            top_k=args.top_k,
            friction_coef=args.friction,
            active_fingers=["thumb", "index", "middle"],
        )
        optimizer = ContactFirstOptimizer(
            model,
            data,
            surface,
            config=cfg,
            object_body_name=args.object if bid >= 0 else None,
        )
        hand_geom_ids = optimizer._hand_geom_ids
        obj_geom_ids = optimizer._obj_geom_ids
    elif method == "arm-ik":
        from vnb_grasp.grasping.arm_grasp_optimizer import (
            ArmGraspOptimizer,
            ArmGraspConfig,
        )

        cfg = ArmGraspConfig(
            n_starts=args.starts,
            top_k=args.top_k,
            friction_coef=args.friction,
            n_fingers=args.n_fingers,
        )
        optimizer = ArmGraspOptimizer(
            model,
            data,
            surface,
            config=cfg,
            object_body_name=args.object if bid >= 0 else None,
        )
        hand_geom_ids = optimizer._hand_geom_ids
        obj_geom_ids = optimizer._obj_geom_ids
    else:
        cfg = OptimizerConfig(
            n_starts=args.starts,
            top_k=args.top_k,
            friction_coef=args.friction,
            max_sqp_iters=args.max_iters,
            n_fingers=args.n_fingers,
        )
        optimizer = GraspOptimizer(
            model,
            data,
            surface,
            config=cfg,
            object_body_name=args.object if bid >= 0 else None,
        )
        hand_geom_ids = optimizer._hand_geom_ids
        obj_geom_ids = optimizer._object_geom_ids

    #  Solve
    print(f"\n  Running {args.starts} multi-start {method_label} optimizations ...")
    t0 = datetime.datetime.now()
    grasps = optimizer.solve()
    elapsed = (datetime.datetime.now() - t0).total_seconds()
    print(f"  Solver done in {elapsed:.1f}s  ({len(grasps)} grasps found)")

    # Measure actual collision penetration
    mesh_pens = [
        _measure_mesh_pen(model, data, hand_geom_ids, obj_geom_ids, g) for g in grasps
    ]
    # Restore qpos to best grasp
    if grasps:
        data.qpos[:] = grasps[0].hand_qpos
        mujoco.mj_forward(model, data)

    #  Print summary
    fc_grasps = [g for g in grasps if g.gws.is_force_closure]
    valid = [g for g in grasps if g.gws.epsilon > 0]
    best_eps = grasps[0].gws.epsilon if grasps else 0.0

    print(f"\n{'#' * 72}")
    print(
        f"  {'Rank':<5} {'Source':<8} {'ε':>8} {'FC':>3} {'ctcs':>5} "
        f"{'tip-pen':>8} {'mesh-pen':>9} {'residual':>10}"
    )
    print(f"{'#' * 72}")
    for i, (g, mp) in enumerate(zip(grasps, mesh_pens)):
        print(
            f"  {i + 1:<5} {g.seed_source:<8} {g.gws.epsilon:>8.4f} "
            f"{'Y' if g.gws.is_force_closure else 'N':>3} "
            f"{g.gws.n_contacts:>5}  "
            f"{g.max_penetration * 1000:>6.1f}mm "
            f"{mp:>7.1f}mm "
            f"{g.residual:>10.6f}"
        )
    print(f"{'#' * 72}")
    print(
        f"  FC: {len(fc_grasps)}/{len(grasps)}   valid (ε>0): {len(valid)}/{len(grasps)}   "
        f"best ε={best_eps:.4f}   {elapsed:.1f}s\n"
    )

    if not grasps:
        print(
            "\n  ⚠  No valid grasps found. Try increasing --starts or relaxing constraints."
        )
        return

    #  Save JSON
    cfg_dict = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}

    grasp_records = []
    for i, (g, mp) in enumerate(zip(grasps, mesh_pens)):
        viz_rel = None
        if not args.no_render:
            viz_rel = str(
                viz_dir.relative_to(REPO_ROOT) / f"grasp_{i + 1:02d}_views.png"
            )
        grasp_records.append(
            _grasp_to_dict(g, rank=i + 1, viz_path=viz_rel, mesh_pen_mm=mp)
        )

    run_record = {
        "run_id": run_id,
        "timestamp": ts_iso,
        "method": method,
        "object": args.object,
        "arena": args.arena,
        "n_starts": args.starts,
        "elapsed_seconds": round(elapsed, 2),
        "config": cfg_dict,
        "summary": {
            "n_total": len(grasps),
            "n_force_closure": len(fc_grasps),
            "n_valid_epsilon": len(valid),
            "best_epsilon": round(best_eps, 6),
            "mean_epsilon_fc": round(
                float(np.mean([g.gws.epsilon for g in fc_grasps])), 6
            )
            if fc_grasps
            else 0.0,
            "worst_penetration_mm": round(max(mesh_pens) if mesh_pens else 0.0, 3),
        },
        "grasps": grasp_records,
    }

    json_path = json_dir / f"{run_id}_{args.object}_{method}_grasps.json"
    with open(json_path, "w") as f:
        json.dump(run_record, f, indent=2, cls=_NumpyEncoder)
    print(f"  JSON --> {json_path.relative_to(REPO_ROOT)}")

    #  Visualize
    if args.no_render:
        return

    try:
        from vnb_grasp.visualization.grasp_viz import GraspVisualizer
    except Exception as e:
        print(f"\n  [WARN] Cannot import GraspVisualizer: {e}")
        return

    print(f"\n  Rendering to {viz_dir.relative_to(REPO_ROOT)}/")
    viz = (
        GraspVisualizer(model, data, scheme=args.scheme)
        .with_friction_cones(True)
        .with_normals(True)
    )

    # Per-grasp multi-view panels
    for i, g in enumerate(grasps):
        _render_grasp_views(
            viz,
            g,
            viz_dir,
            rank=i + 1,
            friction=args.friction,
            views=("iso", "front", "side", "top"),
            w=800,
            h=600,
        )
        print(
            f"    grasp_{i + 1:02d}_views.png  ε={g.gws.epsilon:.4f}  "
            f"{'FC' if g.gws.is_force_closure else '--'}  "
            f"pen={mesh_pens[i]:.1f}mm"
        )

    # Top-k panel
    _render_top_k_panel(
        viz,
        grasps,
        viz_dir,
        friction=args.friction,
        view="iso",
        w=640,
        h=480,
    )
    print(f"    top_k_panel.png")

    # Summary card
    _render_summary_card(
        viz,
        grasps,
        viz_dir,
        friction=args.friction,
        n_starts=args.starts,
        method_label=method_label,
    )
    print(f"    summary_card.png")

    # Also save a simple metrics JSON alongside the viz
    metrics = {
        "run_id": run_id,
        "method": method,
        "object": args.object,
        "n_starts": args.starts,
        "n_grasps": len(grasps),
        "n_force_closure": len(fc_grasps),
        "best_epsilon": round(best_eps, 6),
        "worst_penetration_mm": round(max(mesh_pens) if mesh_pens else 0.0, 3),
        "grasps": [
            {
                "rank": i + 1,
                "epsilon": round(float(g.gws.epsilon), 6),
                "is_force_closure": bool(g.gws.is_force_closure),
                "n_contacts": int(g.gws.n_contacts),
                "max_penetration_mm": round(float(g.max_penetration * 1000), 3),
                "mesh_penetration_mm": round(float(mp), 3),
            }
            for i, (g, mp) in enumerate(zip(grasps, mesh_pens))
        ],
    }
    metrics_path = viz_dir / "grasp_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"    grasp_metrics.json")

    print(f"\n  Done.  All outputs under outputs/")


if __name__ == "__main__":
    main()
