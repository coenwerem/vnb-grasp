"""SQP-based multi-finger grasp optimizer.

This module provides a constrained optimization alternative to the sampling+IK
pipeline for generating high-quality, low-penetration grasps. The optimizer
uses SLSQP over hand base pose and finger joints, enforces surface contact
constraints, and scores candidate grasps with Ferrari-Canny GWS metrics.

Key design decisions
--------------------
- **No tip overshoot compensation.** The tip sites sit at 40 mm along the
  distal-body Z axis, which is *inside* (or level with) the distal collision
  geometry (cylinder top approx. 40-55 mm).  Subtracting a fake "overshoot" from
  the site position pushes the target contact deep into the palm and causes
  the massive penetrations seen in the first version.
- Instead we add a small *forward* offset per finger so the SDF-zero
  constraint lands the actual collision surface on the object, not the
  internal site marker.
- Standoff is computed from actual fingertip reach in the default pose so
  that the hand starts at a distance where fingertips are near the object
  surface (not where the palm rams the object).
- A minimum Z bound prevents the hand from intersecting the table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

try:
    import mujoco as _mujoco
except ImportError:
    _mujoco = None

mujoco = cast(Any, _mujoco)

from .object_surface import ObjectSurface, SurfaceSample, GeomKind
from .gws_quality import GWSResult, analyze_gws
from ..belief.mujoco_rollout import ContactInfo
from .grasp_sampler import DEFAULT_BASE_JOINT, DEFAULT_FINGER_MAP, SampledGrasp

logger = logging.getLogger(__name__)


# Helpers


def _is_descendant(model: Any, body_id: int, root_id: int) -> bool:
    bid = int(body_id)
    while bid != 0:
        if bid == root_id:
            return True
        bid = int(model.body_parentid[bid])
    return root_id == 0


def _collect_geoms_in_subtree(model: Any, root_body_id: int) -> set[int]:
    geom_ids: set[int] = set()
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        if _is_descendant(model, body_id, root_body_id):
            geom_ids.add(geom_id)
    return geom_ids


def _compute_tip_forward_offset(model: Any, site_id: int) -> float:
    """Return the distance from the tip site to the end of the collision
    geometry along the distal body's +Z axis.

    A positive value means the collision geom extends *past* the site
    (site is inside the mesh).  We will push the SDF query point forward
    by this amount so that ``SDF = 0`` corresponds to the physical
    fingertip surface touching the object.
    """
    body_id = int(model.site_bodyid[site_id])
    site_z = float(model.site_pos[site_id][2])
    max_geom_z = 0.0
    for gi in range(model.ngeom):
        if int(model.geom_bodyid[gi]) != body_id:
            continue
        gtype = int(model.geom_type[gi])
        gpos_z = float(model.geom_pos[gi][2])
        gsize = model.geom_size[gi]
        if gtype == 5:  # cylinder
            top = gpos_z + float(gsize[1])
        elif gtype == 7:  # mesh (bounding box approximation)
            top = gpos_z + float(gsize[2])
        elif gtype == 3:  # capsule
            top = gpos_z + float(gsize[1]) + float(gsize[0])
        elif gtype == 2:  # sphere
            top = gpos_z + float(gsize[0])
        else:
            continue
        max_geom_z = max(max_geom_z, top)
    return max(0.0, max_geom_z - site_z)


# Config


@dataclass
class OptimizerConfig:
    n_starts: int = 32
    top_k: int = 5
    friction_coef: float = 1.0
    max_sqp_iters: int = 200
    sdf_tol: float = 0.003  # 3 mm surface tolerance for validation
    collision_margin: float = 0.001  # 1 mm clearance for non-distal geoms
    min_finger_sep: float = 0.010  # 10 mm min fingertip separation
    n_fingers: int = 5
    fingertip_only: bool = True
    max_penetration: float = 0.002  # 2 mm max penetration for validation
    table_z: float = 0.825  # table top Z (for collision avoidance)


# Main optimizer


class GraspOptimizer:
    """Constrained SQP grasp optimizer.

    Parameters
    ----------
    model, data : mujoco.MjModel, mujoco.MjData
        MuJoCo model and *scratch* simulation data (will be mutated).
    surface : ObjectSurface
        Object SDF / surface representation (world frame).
    config : OptimizerConfig, optional
    object_body_name : str, optional
    finger_map : dict, optional
    base_joint : str, optional
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        surface: ObjectSurface,
        *,
        config: OptimizerConfig | None = None,
        object_body_name: str | None = None,
        finger_map: dict[str, tuple[list[str], str]] | None = None,
        base_joint: str | None = None,
    ):
        if mujoco is None:
            raise ImportError("mujoco is required")

        self.model = model
        self.data = data
        self.surface = surface
        self.cfg = config or OptimizerConfig()
        self._rng = np.random.default_rng()

        self._finger_map = finger_map or DEFAULT_FINGER_MAP
        self._base_joint_name = base_joint or DEFAULT_BASE_JOINT
        mujoco.mj_forward(self.model, self.data)

        #  resolve fingers 
        self._resolved_fingers: list[tuple[str, list[str], int]] = []
        for finger_name, (joint_names, site_name) in self._finger_map.items():
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id < 0:
                logger.warning("Missing tip site '%s' for '%s'", site_name, finger_name)
                continue
            self._resolved_fingers.append(
                (finger_name, list(joint_names), int(site_id))
            )
        self._resolved_fingers = self._resolved_fingers[: self.cfg.n_fingers]
        if not self._resolved_fingers:
            raise ValueError("No valid fingertip sites found for optimizer")

        self._finger_names: list[str] = [f for f, _, _ in self._resolved_fingers]
        self._tip_site_ids: list[int] = [sid for _, _, sid in self._resolved_fingers]

        #  base joint 
        self._base_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, self._base_joint_name
        )
        if self._base_jid < 0:
            raise ValueError(f"Base joint '{self._base_joint_name}' not found")
        if int(self.model.jnt_type[self._base_jid]) != 0:
            raise ValueError(f"Base joint '{self._base_joint_name}' must be freejoint")

        self._base_qadr = int(self.model.jnt_qposadr[self._base_jid])
        self._base_body_id = int(self.model.jnt_bodyid[self._base_jid])

        #  finger joints 
        self._finger_joint_ids: list[int] = []
        self._finger_joint_names: list[str] = []
        self._finger_joint_qadr: list[int] = []
        self._finger_joint_bounds: list[tuple[float, float]] = []
        self._finger_joint_by_finger: dict[str, list[int]] = {}

        for finger_name, joint_names, _ in self._resolved_fingers:
            per_finger: list[int] = []
            for jn in joint_names:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid < 0:
                    logger.warning("Missing joint '%s' for '%s'", jn, finger_name)
                    continue
                jtype = int(self.model.jnt_type[jid])
                if jtype not in (2, 3):
                    continue
                qadr = int(self.model.jnt_qposadr[jid])
                if bool(self.model.jnt_limited[jid]):
                    lo = float(self.model.jnt_range[jid, 0])
                    hi = float(self.model.jnt_range[jid, 1])
                else:
                    lo, hi = -np.pi, np.pi
                per_finger.append(len(self._finger_joint_ids))
                self._finger_joint_ids.append(int(jid))
                self._finger_joint_names.append(jn)
                self._finger_joint_qadr.append(qadr)
                self._finger_joint_bounds.append((lo, hi))
            self._finger_joint_by_finger[finger_name] = per_finger

        self._n_params = 6 + len(self._finger_joint_ids)

        #  geom id sets 
        self._obj_body_id = self._resolve_object_body_id(object_body_name)
        if self._obj_body_id is not None:
            self._object_geom_ids = {
                gi
                for gi in range(self.model.ngeom)
                if int(self.model.geom_bodyid[gi]) == self._obj_body_id
            }
        else:
            self._object_geom_ids = set()

        self._hand_geom_ids = _collect_geoms_in_subtree(self.model, self._base_body_id)
        self._distal_body_ids = {
            int(self.model.site_bodyid[sid]) for sid in self._tip_site_ids
        }
        self._distal_geom_ids = {
            gi
            for gi in self._hand_geom_ids
            if int(self.model.geom_bodyid[gi]) in self._distal_body_ids
        }
        self._non_distal_geom_ids = self._hand_geom_ids - self._distal_geom_ids
        self._non_distal_geom_ids_sorted = sorted(self._non_distal_geom_ids)

        #  representative non-distal geoms for inequality constraints 
        # Instead of checking all 21 non-distal geoms (which overwhelms SLSQP),
        # pick a representative subset: palm + hand_base + proximal bodies.
        self._repr_non_distal_geom_ids = self._pick_representative_geoms()

        #  per-finger forward offset (site --> physical fingertip surface) 
        self._tip_forward: dict[str, float] = {}
        for fname, sid in zip(self._finger_names, self._tip_site_ids):
            self._tip_forward[fname] = _compute_tip_forward_offset(self.model, sid)
            logger.debug(
                "tip forward offset %s = %.1f mm",
                fname,
                self._tip_forward[fname] * 1000,
            )

        #  compute fingertip reach from model default pose 
        self._fingertip_reach = self._compute_fingertip_reach()
        logger.debug("fingertip reach = %.1f mm", self._fingertip_reach * 1000)

    def _pick_representative_geoms(self) -> list[int]:
        """Pick a small set of non-distal geoms for inequality constraints.

        Checking all non-distal geoms gives SLSQP 21+ inequality constraints
        just for clearance, which makes convergence very hard.  Instead we
        pick one geom per non-distal body (the largest), keeping the count
        manageable.
        """
        body_best: dict[int, tuple[int, float]] = {}
        for gid in self._non_distal_geom_ids:
            bid = int(self.model.geom_bodyid[gid])
            size_val = float(np.max(self.model.geom_size[gid]))
            if bid not in body_best or size_val > body_best[bid][1]:
                body_best[bid] = (gid, size_val)
        return sorted(gid for gid, _ in body_best.values())

    def _compute_fingertip_reach(self) -> float:
        """Average distance from hand base to fingertip sites in default pose"""
        base_pos = self.data.xpos[self._base_body_id].copy()
        dists = []
        for sid in self._tip_site_ids:
            tip_pos = self.data.site_xpos[sid].copy()
            dists.append(float(np.linalg.norm(tip_pos - base_pos)))
        return float(np.mean(dists)) if dists else 0.15

    # Public API

    def solve(self) -> list[SampledGrasp]:
        seeds = self._generate_initial_configs()
        grasps: list[SampledGrasp] = []
        for q_init in seeds:
            grasp = self._optimize_single(q_init)
            if grasp is not None:
                grasps.append(grasp)
        if not grasps:
            return []
        grasps.sort(key=lambda g: float(g.gws.epsilon), reverse=True)
        return grasps[: self.cfg.top_k]

    # Initialization

    def _generate_initial_configs(self) -> list[NDArray[np.float64]]:
        obj_center = self._get_object_center()
        obj_radius = self._estimate_object_radius()
        q_inits: list[NDArray[np.float64]] = []
        # Standoff strategy: place the hand base far enough that the
        # *entire hand body* clears the object.  fingertip_reach (~167mm)
        # is the distance from base to tip sites in the default (straight)
        # pose.  We use the full reach + obj_radius so that the tips are
        # roughly at the object surface with straight fingers.  The optimizer
        # will curl fingers inward and slide the base closer.
        base_standoff = self._fingertip_reach + obj_radius

        # Minimum Z to keep the hand above the table.  The hand body
        # extends ~31mm below the base position, so we need clearance.
        min_z = self.cfg.table_z + 0.06  # 60mm above table top
        # Build diverse approach directions: avoid pure downward approaches
        approach_dirs: list[NDArray[np.float64]] = []
        side_dirs = [
            np.array([1.0, 0.0, 0.3]),
            np.array([-1.0, 0.0, 0.3]),
            np.array([0.0, 1.0, 0.3]),
            np.array([0.0, -1.0, 0.3]),
        ]
        for d in side_dirs:
            approach_dirs.append(d / np.linalg.norm(d))
        # Top-down (good for power grasps)
        approach_dirs.append(np.array([0.0, 0.0, 1.0]))
        oblique_dirs = [
            np.array([1.0, 0.0, 1.0]),
            np.array([-1.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 1.0]),
            np.array([0.0, -1.0, 1.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0, -1.0, 1.0]),
            np.array([-1.0, 1.0, 1.0]),
            np.array([-1.0, -1.0, 1.0]),
        ]
        for d in oblique_dirs:
            approach_dirs.append(d / np.linalg.norm(d))
        # Random upper-hemisphere directions to fill remaining starts
        while len(approach_dirs) < self.cfg.n_starts:
            v = self._rng.normal(size=3)
            v[2] = abs(v[2]) * 0.5 + 0.3  # bias upward, avoid below table
            nv = np.linalg.norm(v)
            if nv > 1e-8:
                approach_dirs.append(v / nv)
        # Build one initial configuration per approach direction
        for i in range(self.cfg.n_starts):
            q = self.data.qpos.copy()
            d = approach_dirs[i % len(approach_dirs)]
            # Randomize standoff slightly (mostly positive to err on safe side)
            standoff = base_standoff + float(self._rng.uniform(-0.005, 0.02))
            base_pos = obj_center + d * standoff
            base_pos[2] = max(base_pos[2], min_z)
            finger_dir = obj_center - base_pos
            finger_dir_norm = np.linalg.norm(finger_dir)
            if finger_dir_norm < 1e-6:
                finger_dir = np.array([0.0, 0.0, -1.0])
            else:
                finger_dir = finger_dir / finger_dir_norm
            up = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(up, finger_dir)) > 0.95:
                up = np.array([0.0, 1.0, 0.0])
            x_axis = np.cross(up, finger_dir)
            x_axis /= np.linalg.norm(x_axis) + 1e-12
            y_axis = np.cross(finger_dir, x_axis)
            y_axis /= np.linalg.norm(y_axis) + 1e-12
            R = np.column_stack([x_axis, y_axis, finger_dir])
            quat = self._matrix_to_quat(R)
            rv_noise = self._rng.normal(scale=0.06, size=3)
            quat_noise = self._rotvec_to_quat(rv_noise)
            quat = self._quat_multiply(quat_noise, quat)
            quat /= np.linalg.norm(quat) + 1e-12
            # Write base position AND orientation to qpos
            q[self._base_qadr : self._base_qadr + 3] = base_pos
            q[self._base_qadr + 3 : self._base_qadr + 7] = quat
            # Finger curl: 15-35% of range (nearly straight).
            # Start with open fingers so the hand is definitely clear
            # of the object.  The optimizer will curl them inward.
            for j, qadr in enumerate(self._finger_joint_qadr):
                lo, hi = self._finger_joint_bounds[j]
                frac = float(self._rng.uniform(0.15, 0.35))
                jitter = self._rng.normal(scale=0.03 * max(hi - lo, 1e-3))
                q[qadr] = np.clip(lo + frac * (hi - lo) + jitter, lo, hi)
            q_inits.append(q)
        return q_inits

    def _estimate_object_radius(self) -> float:
        """Rough radius of the object for standoff computation.

        For a BOX, uses the max half-extent (not half-diagonal) so the
        standoff isn't inflated by the diagonal.
        """
        if self.surface.kind == GeomKind.BOX:
            return float(np.max(self.surface.size[:3]))
        elif self.surface.kind == GeomKind.SPHERE:
            return float(self.surface.size[0])
        elif self.surface.kind in (GeomKind.CYLINDER, GeomKind.CAPSULE):
            r, h = float(self.surface.size[0]), float(self.surface.size[1])
            return max(r, h)
        elif self.surface.kind == GeomKind.ELLIPSOID:
            return float(np.max(self.surface.size[:3]))
        else:
            return 0.05  # default 50 mm

    # Single-start optimization

    def _optimize_single(self, q_init: NDArray[np.float64]) -> SampledGrasp | None:
        x0 = self._qpos_to_x(q_init)
        center = self._get_object_center()
        # Bounds (shared by both phases)
        lb = np.empty(self._n_params, dtype=np.float64)
        ub = np.empty(self._n_params, dtype=np.float64)
        lb[0:3] = center - 0.30
        ub[0:3] = center + 0.30
        # Enforce minimum Z to stay above table
        lb[2] = max(lb[2], self.cfg.table_z + 0.03)
        lb[3:6] = -np.pi
        ub[3:6] = np.pi
        for i, (lo, hi) in enumerate(self._finger_joint_bounds):
            lb[6 + i] = lo
            ub[6 + i] = hi
        bounds = [(float(lb[i]), float(ub[i])) for i in range(self._n_params)]
        # === Phase 1: Approach (unconstrained, pull fingertips toward surface) ===
        # Far from the object, SDF=0 equality constraints have zero gradient
        # signal.  Use L-BFGS-B to minimize sum(sdf^2) and get close first.
        try:
            p1_result = minimize(
                fun=self._phase1_objective,
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": 150,
                    "ftol": 1e-10,
                    "gtol": 1e-7,
                },
            )
            x1 = np.asarray(p1_result.x, dtype=np.float64)
        except Exception as exc:
            logger.debug("Phase 1 (L-BFGS-B) failed: %s", exc)
            x1 = x0  # fall back to original

        # Check how close Phase 1 got us
        _q1, tips1 = self._forward_and_tips(x1)
        tip_arr1 = np.array([tips1[n] for n in self._finger_names])
        sdf1 = self.surface.signed_distance(tip_arr1)
        max_sdf1 = float(np.max(np.abs(sdf1)))
        logger.debug(
            "Phase 1 done: max|sdf|=%.1fmm, pen=%.1fmm",
            max_sdf1 * 1000,
            self._measure_penetration_from_data() * 1000,
        )

        # === Phase 2: Constrained refinement (SLSQP) ===
        constraints = [
            {"type": "eq", "fun": self._eq_constraints},
            {"type": "ineq", "fun": self._ineq_constraints},
        ]

        try:
            result = minimize(
                fun=self._objective,
                x0=x1,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={
                    "maxiter": self.cfg.max_sqp_iters,
                    "ftol": 1e-8,
                    "disp": False,
                },
            )
        except Exception as exc:
            logger.debug("Phase 2 (SLSQP) failed: %s", exc)
            return None
        x_final = np.asarray(result.x, dtype=np.float64)
        q_final = self._x_to_qpos(x_final)
        return self._validate_and_score(q_final)

    def _phase1_objective(self, x: NDArray[np.float64]) -> float:
        """Phase 1 cost: pull fingertips toward the surface while avoiding penetration.

        No equality constraints: just a smooth cost that has gradient everywhere.
        The SDF^2 term pulls tips to the surface; the penetration term keeps the
        palm and knuckles clear.  A small force-closure proxy term encourages
        good finger spread even during approach.
        """
        _qpos, tips = self._forward_and_tips(x)
        tip_arr = np.array([tips[n] for n in self._finger_names])
        sdf = self.surface.signed_distance(tip_arr)
        pen = self._measure_penetration_from_data()

        # Primary: bring fingertips to surface
        sdf_cost = float(np.sum(sdf ** 2))
        # Secondary: avoid penetration
        pen_cost = pen ** 2
        # Tertiary: encourage finger spread (helps Phase 2)
        tip_normals = self.surface.outward_direction(tip_arr)
        proxy = self._compute_force_closure_proxy(tip_arr, tip_normals)

        return 500.0 * sdf_cost + 50000.0 * pen_cost - 0.1 * proxy

    # Objective & constraints

    def _forward_and_tips(self, x: NDArray[np.float64]):
        """Set qpos, run mj_forward, return (qpos, tip_contact_points).

        tip_contact_points are the physical fingertip surface positions
        (site + forward offset along distal Z).
        """
        qpos = self._x_to_qpos(x)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

        tips: dict[str, NDArray[np.float64]] = {}
        for fname, sid in zip(self._finger_names, self._tip_site_ids):
            site_pos = self.data.site_xpos[sid].copy()
            body_id = int(self.model.site_bodyid[sid])
            distal_z = self.data.xmat[body_id].reshape(3, 3)[:, 2].copy()
            distal_z /= np.linalg.norm(distal_z) + 1e-12
            fwd = self._tip_forward.get(fname, 0.0)
            tips[fname] = site_pos + distal_z * fwd
        return qpos, tips

    def _objective(self, x: NDArray[np.float64]) -> float:
        qpos, tips = self._forward_and_tips(x)
        tip_arr = np.array([tips[n] for n in self._finger_names])
        tip_normals = self.surface.outward_direction(tip_arr)

        proxy = self._compute_force_closure_proxy(tip_arr, tip_normals)
        sdf = self.surface.signed_distance(tip_arr)

        pen = self._measure_penetration_from_data()
        cost = -proxy + 200.0 * float(np.sum(sdf**2)) + 1e6 * pen**2
        return float(cost)

    def _eq_constraints(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        _qpos, tips = self._forward_and_tips(x)
        tip_arr = np.array([tips[n] for n in self._finger_names])
        sdf_vals = self.surface.signed_distance(tip_arr)
        return sdf_vals.astype(np.float64)

    def _ineq_constraints(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # NOTE: _forward_and_tips already called mj_forward so
        # self.data.geom_xpos is up to date.
        _qpos, tips = self._forward_and_tips(x)
        tip_arr = np.array([tips[n] for n in self._finger_names])

        values: list[float] = []

        # Fingertip separation
        for i, j in combinations(range(len(tip_arr)), 2):
            sep = float(np.linalg.norm(tip_arr[i] - tip_arr[j]))
            values.append(sep - self.cfg.min_finger_sep)

        # Representative non-distal geom clearance (SDF > margin)
        for gid in self._repr_non_distal_geom_ids:
            gpos = self.data.geom_xpos[gid].copy()
            sdf_val = float(self.surface.signed_distance(gpos[None, :])[0])
            values.append(sdf_val - self.cfg.collision_margin)


        # Contact-based penetration constraint: pen <= 0
        pen = self._measure_penetration_from_data()
        values.append(-pen)  # ineq: -pen >= 0 => pen <= 0
        return np.asarray(values, dtype=np.float64)

    # Validation

    def _validate_and_score(self, qpos: NDArray[np.float64]) -> SampledGrasp | None:
        q = qpos.copy()
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)

        # Compute contact points
        tips: dict[str, NDArray[np.float64]] = {}
        for fname, sid in zip(self._finger_names, self._tip_site_ids):
            site_pos = self.data.site_xpos[sid].copy()
            body_id = int(self.model.site_bodyid[sid])
            distal_z = self.data.xmat[body_id].reshape(3, 3)[:, 2].copy()
            distal_z /= np.linalg.norm(distal_z) + 1e-12
            fwd = self._tip_forward.get(fname, 0.0)
            tips[fname] = site_pos + distal_z * fwd

        tip_arr = np.array([tips[n] for n in self._finger_names])

        # SDF check (relaxed tolerance)
        sdf_vals = self.surface.signed_distance(tip_arr)
        if np.any(np.abs(sdf_vals) > self.cfg.sdf_tol):
            logger.debug("SDF check failed: max |sdf| = %.4f", np.max(np.abs(sdf_vals)))
            return None

        # Penetration check (ALL hand geoms, not just non-distal)
        pen = self._measure_all_penetration(q)
        if pen > self.cfg.max_penetration:
            logger.debug("Penetration check failed: %.2f mm", pen * 1000)
            return None

        # Fingertip separation
        for i, j in combinations(range(len(tip_arr)), 2):
            if np.linalg.norm(tip_arr[i] - tip_arr[j]) < self.cfg.min_finger_sep * 0.8:
                return None

        # GWS analysis
        tip_normals = self._get_tip_normals(tips)
        obj_center = self._get_object_center()
        contacts = self._build_contact_infos(tips, tip_normals)
        gws = analyze_gws(
            contacts, object_center=obj_center, friction_coef=self.cfg.friction_coef
        )
        if gws.epsilon <= 0.0:
            return None

        # Build output
        finger_qpos: dict[str, NDArray[np.float64]] = {}
        for fname in self._finger_names:
            idxs = self._finger_joint_by_finger.get(fname, [])
            finger_qpos[fname] = np.array(
                [q[self._finger_joint_qadr[k]] for k in idxs], dtype=np.float64
            )

        return SampledGrasp(
            hand_qpos=q.copy(),
            finger_qpos=finger_qpos,
            fingertip_positions={n: tips[n].copy() for n in self._finger_names},
            target_contacts={n: tips[n].copy() for n in self._finger_names},
            target_normals={n: tip_normals[n].copy() for n in self._finger_names},
            residual=float(np.mean(sdf_vals**2)),
            max_penetration=pen,
            gws=gws,
            seed_source="sqp",
        )

    # Penetration measurement

    def _measure_penetration_from_data(self) -> float:
        """Measure worst penetration from *current* self.data (no mj_forward)"""
        worst = 0.0
        if not self._object_geom_ids:
            return 0.0
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            is_hand_obj = (
                g1 in self._hand_geom_ids and g2 in self._object_geom_ids
            ) or (g2 in self._hand_geom_ids and g1 in self._object_geom_ids)
            if is_hand_obj and c.dist < 0.0:
                worst = max(worst, -c.dist)
        return worst

    def _measure_all_penetration(self, qpos: NDArray[np.float64]) -> float:
        """Set qpos, forward, and measure worst hand-object penetration"""
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        return self._measure_penetration_from_data()

    # For backward compat with run script
    def _measure_penetration(self, qpos: NDArray[np.float64]) -> float:
        return self._measure_all_penetration(qpos)

    # Tip normals / force closure proxy

    def _get_tip_normals(
        self, tip_positions: dict[str, NDArray[np.float64]]
    ) -> dict[str, NDArray[np.float64]]:
        pts = np.array([tip_positions[n] for n in self._finger_names])
        nrms = self.surface.outward_direction(pts)
        return {n: nrms[i].copy() for i, n in enumerate(self._finger_names)}

    def _compute_force_closure_proxy(
        self,
        tip_positions: NDArray[np.float64],
        tip_normals: NDArray[np.float64],
    ) -> float:
        obj_center = self._get_object_center()
        cols: list[NDArray[np.float64]] = []
        for pos, normal in zip(tip_positions, tip_normals):
            nrm = normal / (np.linalg.norm(normal) + 1e-12)
            r = pos - obj_center
            wrench = np.concatenate([nrm, np.cross(r, nrm)])
            cols.append(wrench)
        if len(cols) < 2:
            return 0.0
        G = np.column_stack(cols)
        svs = np.linalg.svd(G, compute_uv=False)
        return float(svs[-1]) if len(svs) > 0 else 0.0

    # Contact info builder

    def _build_contact_infos(
        self,
        tip_positions: dict[str, NDArray[np.float64]],
        tip_normals: dict[str, NDArray[np.float64]],
    ) -> list[ContactInfo]:
        contacts: list[ContactInfo] = []
        for name in self._finger_names:
            pos = tip_positions[name]
            nrm = tip_normals[name]
            nrm = nrm / (np.linalg.norm(nrm) + 1e-12)
            t1, t2 = self._orthonormal_tangent_basis(nrm)
            frame = np.vstack([nrm, t1, t2])
            contacts.append(
                ContactInfo(
                    geom1=-1,
                    geom2=-1,
                    pos=pos.copy(),
                    frame=frame,
                    dist=0.0,
                    force=np.array([1.0, 0.0, 0.0], dtype=np.float64),
                )
            )
        return contacts

    @staticmethod
    def _orthonormal_tangent_basis(
        normal: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if abs(normal[2]) < 0.9:
            t1 = np.cross(normal, np.array([0.0, 0.0, 1.0]))
        else:
            t1 = np.cross(normal, np.array([1.0, 0.0, 0.0]))
        t1 /= np.linalg.norm(t1) + 1e-12
        t2 = np.cross(normal, t1)
        t2 /= np.linalg.norm(t2) + 1e-12
        return t1, t2

    # Coordinate conversions

    def _x_to_qpos(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        qpos = self.data.qpos.copy()
        qpos[self._base_qadr : self._base_qadr + 3] = x[0:3]
        quat = self._rotvec_to_quat(np.asarray(x[3:6], dtype=np.float64))
        quat /= np.linalg.norm(quat) + 1e-12
        qpos[self._base_qadr + 3 : self._base_qadr + 7] = quat
        for i, qadr in enumerate(self._finger_joint_qadr):
            qpos[qadr] = x[6 + i]
        return qpos

    def _qpos_to_x(self, qpos: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.zeros(self._n_params, dtype=np.float64)
        x[0:3] = qpos[self._base_qadr : self._base_qadr + 3]
        x[3:6] = self._quat_to_rotvec(qpos[self._base_qadr + 3 : self._base_qadr + 7])
        for i, qadr in enumerate(self._finger_joint_qadr):
            x[6 + i] = qpos[qadr]
        return x

    # Object helpers

    def _get_object_center(self) -> NDArray[np.float64]:
        if self._obj_body_id is not None:
            return self.data.xpos[self._obj_body_id].copy()
        return np.asarray(self.surface.position, dtype=np.float64).copy()

    def _resolve_object_body_id(self, name: str | None) -> int | None:
        if name is not None:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            return int(bid) if bid >= 0 else None
        for guess in ("cube", "mustard_bottle", "mustard", "object"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, guess)
            if bid >= 0:
                return int(bid)
        # Fallback: closest body to surface position
        target = np.asarray(self.surface.position, dtype=np.float64)
        best_bid, best_dist = None, np.inf
        for bid in range(1, self.model.nbody):
            if _is_descendant(self.model, bid, self._base_body_id):
                continue
            d = float(np.linalg.norm(self.data.xpos[bid] - target))
            if d < best_dist:
                best_dist, best_bid = d, bid
        return int(best_bid) if best_bid is not None else None

    # Quaternion / rotation utilities

    @staticmethod
    def _rotvec_to_quat(rv: NDArray[np.float64]) -> NDArray[np.float64]:
        angle = float(np.linalg.norm(rv))
        if angle < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        axis = rv / angle
        w = np.cos(angle / 2.0)
        xyz = axis * np.sin(angle / 2.0)
        return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float64)

    @staticmethod
    def _quat_to_rotvec(q: NDArray[np.float64]) -> NDArray[np.float64]:
        qn = np.asarray(q, dtype=np.float64)
        qn = qn / (np.linalg.norm(qn) + 1e-12)
        w = float(np.clip(qn[0], -1.0, 1.0))
        xyz = qn[1:4]
        sin_half = float(np.linalg.norm(xyz))
        if sin_half < 1e-10:
            return np.zeros(3, dtype=np.float64)
        axis = xyz / sin_half
        angle = 2.0 * np.arctan2(sin_half, w)
        if angle > np.pi:
            angle -= 2.0 * np.pi
        return axis * angle

    @staticmethod
    def _quat_multiply(
        q1: NDArray[np.float64], q2: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _matrix_to_quat(R: NDArray[np.float64]) -> NDArray[np.float64]:
        tr = float(R[0, 0] + R[1, 1] + R[2, 2])
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        q = np.array([qw, qx, qy, qz], dtype=np.float64)
        q /= np.linalg.norm(q) + 1e-12
        return q
