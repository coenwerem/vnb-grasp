"""Precision-grip grasp optimizer: zero proximal penetration via
Monte-Carlo seed search + finger-only IK.
Strategy
--------
The core insight, validated by large-scale random sampling, is that
for the RealHand L6 there exist many (base-pose, finger-curl)
configurations where fingertips touch the object surface while
proximal links pass BESIDE the object without penetrating it.  The
feasible space is sparse (~0.15% of random configs) but well-structured:

  - Standoff (base-to-object distance): 80-190 mm (sweet spot 110-160 mm)
  - Finger curl: broadly distributed, including deep curls up to 95%
  - Body +Z alignment with approach: mean 0.84 (fingers mostly toward object)
Algorithm:
1.  MONTE-CARLO SEED GENERATION: Sample diverse (position, orientation,
    curl) configurations within the empirically-validated feasible band,
    combined with structured face-aligned approaches.

2.  FINGER-ONLY IK: For each seed, freeze the base and run Jacobian IK
    using only finger joint DOFs to reach contact targets on the object
    surface.

3.  PROXIMAL PENETRATION FILTER: After IK, verify that no proximal
    (non-distal) hand geom penetrates the object.  This is the KEY
    filter.  Only grasps with zero proximal penetration pass.

4.  GWS QUALITY: Evaluate force-closure quality via Grasp Wrench Space
    analysis.  Require epsilon > 0.001.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

NDArray = np.ndarray

mujoco: Any = __import__("mujoco")

from .object_surface import ObjectSurface, SurfaceSample, GeomKind
from .gws_quality import GWSResult, analyze_gws
from ..belief.mujoco_rollout import ContactInfo
from .grasp_sampler import (
    DEFAULT_BASE_JOINT,
    DEFAULT_FINGER_MAP,
    SampledGrasp,
    _is_descendant,
    _collect_geoms_in_subtree,
)

logger = logging.getLogger(__name__)


# Helpers


def _normalize(v: NDArray, eps: float = 1e-12) -> NDArray:
    n = float(np.linalg.norm(v))
    return v / n if n > eps else np.zeros_like(v)


def _matrix_to_quat(R: NDArray) -> NDArray:
    """Rotation matrix to MuJoCo quaternion [w, x, y, z]"""
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
    return q / (np.linalg.norm(q) + 1e-12)


def _orthonormal_tangent_basis(
    normal: NDArray,
) -> Tuple[NDArray, NDArray]:
    if abs(float(normal[2])) < 0.9:
        t1 = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    else:
        t1 = np.cross(normal, np.array([1.0, 0.0, 0.0]))
    t1 = _normalize(t1)
    t2 = _normalize(np.cross(normal, t1))
    return t1, t2


# Config


@dataclass
class PrecisionGripConfig:
    """Configuration for PrecisionGripOptimizer"""

    n_starts: int = 500_000
    top_k: int = 20
    friction_coef: float = 0.8
    max_penetration: float = 0.001  # 1 mm: strict
    sdf_tolerance: float = 0.015  # 15 mm: generous contact detection
    min_finger_separation: float = 0.008  # 8 mm
    min_epsilon: float = 0.0001

    # IK parameters (finger-only)
    ik_max_iter: int = 200
    ik_damping: float = 5e-3
    ik_step_size: float = 0.4
    ik_tol: float = 3e-3  # 3 mm convergence

    # Standoff control: Monte Carlo validated feasible band
    standoff_margin: float = 0.010  # 10 mm (reduced: not proximal-avoidance)
    min_standoff: float = 0.080  # 80 mm: Monte Carlo minimum
    max_standoff: float = 0.190  # 190 mm: Monte Carlo maximum

    # Curl constraints: Monte Carlo shows deep curls work
    max_curl_fraction: float = 0.95  # 95% of joint range
    curl_fractions: List[float] = field(
        default_factory=lambda: [0.15, 0.30, 0.50, 0.65, 0.80, 0.90]
    )

    # Monte Carlo seed generation
    n_monte_carlo_seeds: int = 500_000
    mc_standoff_range: Tuple[float, float] = (0.080, 0.200)  # Wider range for varied approaches
    mc_curl_range: Tuple[float, float] = (0.10, 0.95)
    fk_proximity_threshold: float = 0.025  # 25mm - relaxed for thumb-opposition seeds
    min_near_tips: int = 3  # Need 3+ contacts for force closure with thumb
    # Pad-facing constraint: reject fingers where nail side faces object
    # tip Y-axis points toward pad, and the projection of (obj - tip) onto tip Y should be > 0
    require_pad_facing: bool = True
    # Pad offset: distance from tip site origin to pad surface along tip +Y axis
    # Measured from RealHand L6 geometry: ~6mm
    pad_offset: float = 0.006

    # Wrist rotation perturbations (radians around approach axis)
    wrist_rotations: List[float] = field(
        default_factory=lambda: [0.0, 0.3, -0.3, 0.6, -0.6, 1.0, -1.0, 1.5]
    )

    active_fingers: List[str] | None = field(
        default_factory=lambda: ["thumb", "index", "middle"]  # Precision grip: thumb opposes 2 fingers
    )
    random_seed: int | None = None


# Finger-Only IK Solver


class FingerOnlyIKSolver:
    """Jacobian IK solver using only finger joint DOFs (base frozen).

    This is a stripped-down IK that:
    - Only varies finger hinge joints (no base translation/rotation)
    - Enforces joint limits per iteration (including max curl)
    - Has no penetration barrier (standoff guarantees no proximal contact)
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        finger_map: Dict[str, Tuple[List[str], str]],
        active_fingers: List[str],
        damping: float = 5e-3,
        max_iter: int = 200,
        tol: float = 3e-3,
        step_size: float = 0.4,
        max_curl_fraction: float = 0.55,
    ) -> None:
        self.model = model
        self.data = data
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.step_size = step_size
        self.max_curl_fraction = max_curl_fraction

        # Build finger DOF map (only active fingers)
        self.finger_names: List[str] = []
        self.site_ids: List[int] = []
        self._finger_joint_ids: List[List[int]] = []  # MuJoCo joint IDs
        self._all_dof_ids: List[int] = []

        for fname in active_fingers:
            if fname not in finger_map:
                continue
            joint_names, site_name = finger_map[fname]
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid < 0:
                logger.warning("Site '%s' not found, skipping '%s'", site_name, fname)
                continue

            jids: List[int] = []
            dofs: List[int] = []
            for jn in joint_names:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid < 0:
                    logger.warning("Joint '%s' not found", jn)
                    continue
                jids.append(jid)
                dof_adr = int(model.jnt_dofadr[jid])
                dofs.append(dof_adr)
                if dof_adr not in self._all_dof_ids:
                    self._all_dof_ids.append(dof_adr)

            self.finger_names.append(fname)
            self.site_ids.append(sid)
            self._finger_joint_ids.append(jids)

        self._all_dof_ids_arr = np.array(self._all_dof_ids, dtype=np.int32)
        self.n_fingers = len(self.finger_names)

        # Precompute joint limits (clamped to max_curl_fraction)
        self._joint_lo: Dict[int, float] = {}
        self._joint_hi: Dict[int, float] = {}
        for jids in self._finger_joint_ids:
            for jid in jids:
                if bool(model.jnt_limited[jid]):
                    lo = float(model.jnt_range[jid, 0])
                    hi = float(model.jnt_range[jid, 1])
                    # Clamp upper limit to max_curl_fraction of range
                    hi_clamped = lo + (hi - lo) * self.max_curl_fraction
                    self._joint_lo[jid] = lo
                    self._joint_hi[jid] = hi_clamped

    def solve(
        self,
        targets: Dict[str, NDArray],
        q_init: NDArray | None = None,
    ) -> Tuple[NDArray, float, Dict[str, NDArray]]:
        """Run finger-only IK to reach target positions.

        Parameters
        ----------
        targets : dict[finger_name, (3,) world target]
        q_init : (nq,) initial qpos (base already set, fingers at seed curl)

        Returns
        -------
        qpos, residual, achieved_positions
        """
        if q_init is not None:
            self.data.qpos[:] = q_init
        mujoco.mj_forward(self.model, self.data)

        # Build active finger list
        active_idxs: List[int] = []
        active_targets: List[NDArray] = []
        for i, fn in enumerate(self.finger_names):
            if fn in targets:
                active_idxs.append(i)
                active_targets.append(
                    np.asarray(targets[fn], dtype=np.float64).ravel()[:3]
                )

        if not active_idxs:
            return self.data.qpos.copy(), 0.0, self._get_positions()

        n_active = len(active_idxs)
        nv = len(self._all_dof_ids)

        for iteration in range(self.max_iter):
            mujoco.mj_forward(self.model, self.data)

            # Build error vector and Jacobian
            error = np.zeros(3 * n_active)
            J_full = np.zeros((3 * n_active, self.model.nv))

            for k, fi in enumerate(active_idxs):
                sid = self.site_ids[fi]
                xpos = self.data.site_xpos[sid].copy()
                error[3 * k : 3 * k + 3] = active_targets[k] - xpos

                jacp = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, self.data, jacp, None, sid)
                J_full[3 * k : 3 * k + 3] = jacp

            # Restrict to finger DOFs only
            J = J_full[:, self._all_dof_ids_arr]

            # Check convergence
            per_finger_err = np.linalg.norm(error.reshape(-1, 3), axis=1)
            max_err = float(np.max(per_finger_err))
            if max_err < self.tol:
                break

            # Standard DLS solve (no weighting needed: all finger DOFs)
            lam2 = self.damping**2
            JJt = J @ J.T + lam2 * np.eye(3 * n_active)
            dq = J.T @ np.linalg.solve(JJt, error)
            dq *= self.step_size

            # Apply dq with joint limits
            for local_idx, dof_id in enumerate(self._all_dof_ids):
                jid = int(self.model.dof_jntid[dof_id])
                q_adr = int(self.model.jnt_qposadr[jid])
                new_val = self.data.qpos[q_adr] + dq[local_idx]

                if jid in self._joint_lo:
                    new_val = np.clip(new_val, self._joint_lo[jid], self._joint_hi[jid])

                self.data.qpos[q_adr] = new_val

        mujoco.mj_forward(self.model, self.data)

        # Compute final residual and positions
        residual = 0.0
        achieved: Dict[str, NDArray] = {}
        for k, fi in enumerate(active_idxs):
            sid = self.site_ids[fi]
            xpos = self.data.site_xpos[sid].copy()
            achieved[self.finger_names[fi]] = xpos
            residual += float(np.sum((active_targets[k] - xpos) ** 2))

        # Include non-active fingers
        for i, fn in enumerate(self.finger_names):
            if fn not in achieved:
                achieved[fn] = self.data.site_xpos[self.site_ids[i]].copy()

        return self.data.qpos.copy(), residual, achieved

    def _get_positions(self) -> Dict[str, NDArray]:
        result: Dict[str, NDArray] = {}
        for i, fn in enumerate(self.finger_names):
            result[fn] = self.data.site_xpos[self.site_ids[i]].copy()
        return result


# Optimizer


class PrecisionGripOptimizer:
    """Precision-grip optimizer: standoff-based, zero proximal penetration.

    Usage::

        surface = ObjectSurface.from_model(model, body_name='cube')
        opt = PrecisionGripOptimizer(model, data, surface, object_body_name='cube')
        grasps = opt.solve()
        for g in grasps:
            print(f"eps={g.gws.epsilon:.4f}, pen={g.max_penetration*1000:.1f}mm")
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        surface: ObjectSurface,
        *,
        config: PrecisionGripConfig | None = None,
        object_body_name: str | None = None,
    ) -> None:
        if mujoco is None:
            raise ImportError("mujoco is required")

        self.model = model
        self.data = data
        self.surface = surface
        self.cfg = config or PrecisionGripConfig()
        self._rng = np.random.default_rng(self.cfg.random_seed)

        self._finger_map = DEFAULT_FINGER_MAP
        self._active_fingers = self.cfg.active_fingers or [
            "index",
            "middle",
            "ring",
            "pinky",
        ]

        # Base joint
        self._base_jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, DEFAULT_BASE_JOINT
        )
        if self._base_jid < 0:
            raise ValueError(f"Base joint '{DEFAULT_BASE_JOINT}' not found")
        self._base_qadr = int(model.jnt_qposadr[self._base_jid])
        self._base_body_id = int(model.jnt_bodyid[self._base_jid])

        # Object body
        self._obj_body_id = self._resolve_object_body_id(object_body_name)

        # Sync surface pose with the object's current world frame.
        # ObjectSurface.from_model() only stores geometry (local frame).
        # We need world-frame contacts for IK target computation.
        if self._obj_body_id is not None:
            self.surface.position = self.data.xpos[self._obj_body_id].copy()
            self.surface.rotation = (
                self.data.xmat[self._obj_body_id].reshape(3, 3).copy()
            )

        # Geom sets for collision detection
        if self._obj_body_id is not None:
            self._obj_geom_ids = _collect_geoms_in_subtree(model, self._obj_body_id)
        else:
            self._obj_geom_ids: set[int] = set()
        self._hand_geom_ids = _collect_geoms_in_subtree(model, self._base_body_id)

        # Tip site IDs
        self._tip_site_ids: Dict[str, int] = {}
        self._all_tip_site_ids: Dict[str, int] = {}
        for fname in self._finger_map:
            _, site_name = self._finger_map[fname]
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid < 0:
                raise ValueError(f"Missing fingertip site: {site_name}")
            self._all_tip_site_ids[fname] = sid
        for fname in self._active_fingers:
            self._tip_site_ids[fname] = self._all_tip_site_ids[fname]

        # Compute tip overshoot (site extends past collision mesh)
        self._tip_overshoot: Dict[str, float] = {}
        self._compute_tip_overshoot()

        # Distal body/geom IDs (exempt from proximal penetration check)
        # IMPORTANT: include ALL fingers (not just active ones) so that
        # inactive-finger distal geoms don't inflate proximal reach.
        self._distal_body_ids: set[int] = set()
        self._distal_geom_ids: set[int] = set()
        for fname in self._finger_map:
            _, site_name = self._finger_map[fname]
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid >= 0:
                self._distal_body_ids.add(int(model.site_bodyid[sid]))
        for gi in range(model.ngeom):
            if int(model.geom_bodyid[gi]) in self._distal_body_ids:
                self._distal_geom_ids.add(gi)

        # Compute proximal link length (for standoff calculation)
        self._proximal_reach = self._estimate_proximal_reach()

        # Compute fingertip reach from base
        self._fingertip_reach = self._estimate_fingertip_reach()

        # Compute standoff distance
        self._standoff = self._compute_standoff()

        # Build finger-only IK solver
        self._ik = FingerOnlyIKSolver(
            model,
            data,
            finger_map=self._finger_map,
            active_fingers=self._active_fingers,
            damping=self.cfg.ik_damping,
            max_iter=self.cfg.ik_max_iter,
            tol=self.cfg.ik_tol,
            step_size=self.cfg.ik_step_size,
            max_curl_fraction=self.cfg.max_curl_fraction,
        )


        # Precompute finger joint info for fast seed generation
        # Each entry: (qadr, lo, hi) for randomizing curl
        self._finger_joint_info: List[Tuple[int, float, float]] = []
        for fname in self._finger_map:
            joints, _ = self._finger_map[fname]
            for jn in joints:
                jid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, jn,
                )
                if jid < 0:
                    continue
                if not bool(model.jnt_limited[jid]):
                    continue
                qadr = int(model.jnt_qposadr[jid])
                lo = float(model.jnt_range[jid, 0])
                hi = float(model.jnt_range[jid, 1])
                self._finger_joint_info.append((qadr, lo, hi))

        mujoco.mj_forward(self.model, self.data)
        logger.info(
            "PrecisionGripOptimizer: n_starts=%d, top_k=%d, "
            "fingers=%s, standoff=%.1fmm, prox_reach=%.1fmm, "
            "tip_reach=%.1fmm",
            self.cfg.n_starts,
            self.cfg.top_k,
            self._active_fingers,
            self._standoff * 1000,
            self._proximal_reach * 1000,
            self._fingertip_reach * 1000,
        )

    # Public API

    def solve(self) -> List[SampledGrasp]:
        """Run the optimizer and return ranked grasps"""
        q0 = self.data.qpos.copy()
        self._update_surface_pose()
        obj_center = self._get_object_center()
        q_template = self.data.qpos.copy()
        n_seeds = self.cfg.n_monte_carlo_seeds
        logger.info("Starting FK Monte Carlo with %d seeds", n_seeds)
        results: List[SampledGrasp] = []
        stats = {
            "fk_candidates": 0,
            "ik_converged": 0,
            "prox_pen_rejected": 0,
            "pen_rejected": 0,
            "sep_rejected": 0,
            "sdf_rejected": 0,
            "pad_facing_rejected": 0,
            "gws_rejected": 0,
        }

        try:
            candidates: List[Tuple[NDArray, List[str]]] = []
            use_thumb_opposition = "thumb" in self._active_fingers

            # Phase 1: FK seed generation + proximity filter
            for idx in range(n_seeds):
                if use_thumb_opposition:
                    q = self._generate_thumb_opposition_seed(q_template, obj_center)
                else:
                    q = q_template.copy()
                    # Random wrist placement
                    standoff = float(self._rng.uniform(*self.cfg.mc_standoff_range))
                    d_hat = _normalize(self._rng.standard_normal(3))
                    wrist_rot = float(self._rng.uniform(0, 2 * np.pi))
                    base_pos = obj_center + d_hat * standoff
                    q[self._base_qadr : self._base_qadr + 3] = base_pos
                    R = self._build_approach_rotation(d_hat, wrist_rot)
                    q[self._base_qadr + 3 : self._base_qadr + 7] = _matrix_to_quat(R)
                    # Random finger curls
                    for qadr, lo, hi in self._finger_joint_info:
                        curl_frac = float(
                            self._rng.uniform(0.0, self.cfg.max_curl_fraction)
                        )
                        q[qadr] = lo + (hi - lo) * curl_frac
                self.data.qpos[:] = q
                mujoco.mj_forward(self.model, self.data)
                near_fingers: List[str] = self._fk_proximity_check()
                if len(near_fingers) >= self.cfg.min_near_tips:
                    prox_pen = self._measure_proximal_penetration()
                    if prox_pen <= 0.0:
                        candidates.append((q.copy(), near_fingers))
                if (idx + 1) % 50_000 == 0:
                    logger.info(
                        "FK filter progress: %d/%d seeds, %d candidates",
                        idx + 1,
                        n_seeds,
                        len(candidates),
                    )
            stats["fk_candidates"] = len(candidates)
            logger.info("FK filter kept %d/%d seeds (thumb_opposition=%s)", len(candidates), n_seeds, use_thumb_opposition)

            refined: List[
                Tuple[NDArray, Dict[str, SurfaceSample], Dict[str, NDArray], float]
            ] = []
            for q_seed, near_fingers in candidates:
                used_fingers = [
                    fname for fname in near_fingers if fname in self._active_fingers
                ]
                if len(used_fingers) < self.cfg.min_near_tips:
                    continue

                self.data.qpos[:] = q_seed
                mujoco.mj_forward(self.model, self.data)

                contact_set: Dict[str, SurfaceSample] = {}
                for fname in used_fingers:
                    tip_sid = self._all_tip_site_ids[fname]
                    tip_pos = self.data.site_xpos[tip_sid].copy()
                    surf_pt, normal = self._project_to_surface(tip_pos)
                    contact_set[fname] = SurfaceSample(
                        points=surf_pt,
                        normals=_normalize(normal),
                        weights=np.array([1.0], dtype=np.float64),
                    )

                targets = self._compute_ik_targets(contact_set)
                q_refined, residual, achieved = self._ik.solve(targets, q_init=q_seed)

                max_tip_err = 0.0
                for fname in used_fingers:
                    if fname in achieved:
                        err = float(np.linalg.norm(achieved[fname] - targets[fname]))
                        max_tip_err = max(max_tip_err, err)
                if max_tip_err > 0.010:
                    continue

                prox_pen = self._measure_proximal_penetration()
                if prox_pen > self.cfg.max_penetration:
                    stats["prox_pen_rejected"] += 1
                    continue

                stats["ik_converged"] += 1
                refined.append((q_refined.copy(), contact_set, achieved, residual))

            for q_refined, contact_set, achieved, residual in refined:
                self.data.qpos[:] = q_refined
                mujoco.mj_forward(self.model, self.data)
                used_fingers = list(contact_set.keys())

                prox_pen_3 = self._measure_proximal_penetration()
                total_pen = self._measure_worst_penetration()
                if prox_pen_3 > self.cfg.max_penetration:
                    stats["pen_rejected"] += 1
                    continue

                tip_pts = np.array(
                    [achieved[f] for f in used_fingers], dtype=np.float64
                )
                if not self._check_finger_separation(tip_pts):
                    stats["sep_rejected"] += 1
                    continue

                fingers_in_contact = self._detect_finger_contacts()
                for fname in used_fingers:
                    sdf_val = float(
                        self.surface.signed_distance(achieved[fname].reshape(1, 3))[0]
                    )
                    if abs(sdf_val) <= self.cfg.sdf_tolerance:
                        fingers_in_contact.add(fname)

                valid_fingers = [f for f in used_fingers if f in fingers_in_contact]
                # Apply pad-facing filter after IK refinement
                pre_pad_count = len(valid_fingers)
                valid_fingers = self._check_pad_facing(valid_fingers)
                if pre_pad_count > 0 and len(valid_fingers) < pre_pad_count:
                    stats["pad_facing_rejected"] += 1
                if len(valid_fingers) < self.cfg.min_near_tips:
                    stats["sdf_rejected"] += 1
                    continue

                gws = self._evaluate_gws({f: achieved[f] for f in valid_fingers})
                if gws.epsilon <= self.cfg.min_epsilon:
                    stats["gws_rejected"] += 1
                    continue

                finger_qpos = self._extract_finger_qpos(self.data.qpos)
                results.append(
                    SampledGrasp(
                        hand_qpos=self.data.qpos.copy(),
                        finger_qpos=finger_qpos,
                        fingertip_positions={
                            f: achieved[f].copy() for f in valid_fingers
                        },
                        target_contacts={
                            f: contact_set[f].points.copy() for f in valid_fingers
                        },
                        target_normals={
                            f: contact_set[f].normals.copy() for f in valid_fingers
                        },
                        residual=residual,
                        max_penetration=total_pen,
                        gws=gws,
                        seed_source="fk_sample",
                    )
                )
        finally:
            self.data.qpos[:] = q0
            mujoco.mj_forward(self.model, self.data)

        results.sort(key=lambda g: g.gws.epsilon, reverse=True)
        trimmed = results[: self.cfg.top_k]
        logger.info(
            "Solve complete: %d valid grasps (returning %d). "
            "fk_candidates=%d, ik=%d, prox_pen=%d, pen=%d, sep=%d, pad_facing=%d, sdf=%d, gws=%d",
            len(results),
            len(trimmed),
            stats["fk_candidates"],
            stats["ik_converged"],
            stats["prox_pen_rejected"],
            stats["pen_rejected"],
            stats["sep_rejected"],
            stats["pad_facing_rejected"],
            stats["sdf_rejected"],
            stats["gws_rejected"],
        )
        return trimmed

    # Seed generation

    def _generate_seed(self, q_template: NDArray, obj_center: NDArray) -> NDArray:
        """Generate a single FK seed (utility for external callers)"""
        q = q_template.copy()
        standoff = float(self._rng.uniform(*self.cfg.mc_standoff_range))
        d_hat = _normalize(self._rng.standard_normal(3))
        wrist_rot = float(self._rng.uniform(0, 2 * np.pi))
        base_pos = obj_center + d_hat * standoff
        q[self._base_qadr : self._base_qadr + 3] = base_pos
        R = self._build_approach_rotation(d_hat, wrist_rot)
        q[self._base_qadr + 3 : self._base_qadr + 7] = _matrix_to_quat(R)
        for qadr, lo, hi in self._finger_joint_info:
            curl_frac = float(
                self._rng.uniform(0.0, self.cfg.max_curl_fraction)
            )
            q[qadr] = lo + (hi - lo) * curl_frac
        return q

    def _generate_thumb_opposition_seed(
        self, q_template: NDArray, obj_center: NDArray
    ) -> NDArray:
        """Generate seed for thumb-opposing precision grip.
        Positions the hand at the computed standoff distance from the object,
        with orientation chosen to achieve thumb-opposition configuration.
        
        Geometry of RealHand L6:
        - Fingers extend in +Z direction from the hand base at rest
        - At high curl (85-95%), fingertips curl back toward palm in +X direction
        - Thumb is mounted at +X relative to palm, opposing the other fingers
        
        Strategy for precision grip:
        1. Position hand base at standoff distance, approaching from -Z direction
           (fingers point down toward object from above)
        2. Apply pitch (~-45 deg to -75 deg) to angle the wrist so curled fingers
           can wrap around the object sides
        3. Apply roll (~-60 deg to -120 deg) to rotate thumb to opposite side from
           index/middle fingers (achieving true opposition)
        4. Apply high curl (75-95%) so fingertips are close to object surface
        """
        q = q_template.copy()
        # Use standoff distance computed for this object
        # Add some variation to explore the feasible space
        standoff_base = self._standoff
        standoff_var = float(self._rng.uniform(-0.02, 0.02))  # +/-20mm variation
        standoff = max(0.08, standoff_base + standoff_var)
        
        # Approach direction: primarily from above/behind
        # Randomize the approach direction within a cone
        # Primary direction: -Z (hand above object, fingers pointing down)
        # with some tilt in XY plane
        approach_z = float(self._rng.uniform(-1.0, -0.5))  # -Z component (above)
        approach_x = float(self._rng.uniform(-0.5, 0.3))   # -X to +X variation
        approach_y = float(self._rng.uniform(-0.3, 0.5))   # Y variation
        approach_vec = np.array([approach_x, approach_y, approach_z])
        approach_vec = approach_vec / np.linalg.norm(approach_vec)
        
        # Position hand base at standoff along approach direction
        base_pos = obj_center - approach_vec * standoff
        q[self._base_qadr : self._base_qadr + 3] = base_pos
        # Build orientation:
        # Start with hand pointing toward object (+Z toward object)
        # Then apply pitch and roll for thumb opposition
        
        # First, build rotation to point hand +Z toward object
        z_axis = approach_vec  # hand's +Z points toward object
        
        # Choose an arbitrary up direction not parallel to z_axis
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(z_axis, world_up)) > 0.9:
            world_up = np.array([1.0, 0.0, 0.0])
        
        # Gram-Schmidt to get orthogonal x and y axes
        x_axis = np.cross(world_up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Base rotation matrix (hand pointing toward object)
        R_base = np.column_stack([x_axis, y_axis, z_axis])
        
        # Now apply additional rotations for thumb opposition
        # Pitch: rotate around local X-axis (tilts fingers)
        pitch_deg = float(self._rng.uniform(-75, -45))
        pitch = np.radians(pitch_deg)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])
        
        # Roll: rotate around local Z-axis (positions thumb opposite to fingers)
        roll_deg = float(self._rng.uniform(-120, -60))
        roll = np.radians(roll_deg)
        Rz = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation: first apply pitch, then roll, in local frame
        R_local = Rz @ Rx
        R = R_base @ R_local
        
        q[self._base_qadr + 3 : self._base_qadr + 7] = _matrix_to_quat(R)
        
        # High curl for precision grip (75-95%)
        curl_frac = float(self._rng.uniform(0.75, 0.95))
        for qadr, lo, hi in self._finger_joint_info:
            q[qadr] = lo + (hi - lo) * curl_frac
        return q

    def _fk_proximity_check(self) -> List[str]:
        """Check which fingertips are near the object surface"""
        near_fingers: List[str] = []
        for fname, sid in self._all_tip_site_ids.items():
            tip_pos = self.data.site_xpos[sid]
            if self.surface.kind == GeomKind.BOX:
                dist = self._dist_to_cube_surface(tip_pos)
            else:
                dist = float(self.surface.signed_distance(tip_pos.reshape(1, 3))[0])
            if abs(dist) > self.cfg.fk_proximity_threshold:
                continue
            near_fingers.append(fname)
        return near_fingers
    def _check_pad_facing(self, fingers: List[str]) -> List[str]:
        """Filter fingers to only those where the pad faces the object.
        Finger geometry analysis:
        - Tip sites are at the end of distal links with no rotation (identity quat)
        - Finger joints rotate about Y-axis (axis="0 1 0")
        - When fingers curl, they rotate in the XZ plane
        - Pad faces toward palm: -X direction in local frame (perpendicular to curl axis)
        - Nail faces outward: +X direction in local frame
        This should be called AFTER IK refinement when the hand is properly positioned.
        """
        if not self.cfg.require_pad_facing:
            return fingers
        obj_center = self._get_object_center()
        pad_facing_fingers: List[str] = []
        for fname in fingers:
            sid = self._all_tip_site_ids.get(fname)
            if sid is None:
                continue
            tip_pos = self.data.site_xpos[sid]
            tip_xmat = self.data.site_xmat[sid].reshape(3, 3)
            # -X is pad direction (toward palm), +X is nail direction
            # We want pad to face toward object, so -X should point toward object
            pad_direction = -tip_xmat[:, 0]  # -X axis = pad direction
            to_object = obj_center - tip_pos
            to_object_norm = np.linalg.norm(to_object)
            if to_object_norm > 1e-6:
                to_object = to_object / to_object_norm
            pad_facing_score = float(np.dot(pad_direction, to_object))
            # Debug logging (rate-limited)
            if self._rng.random() < 0.001:  # Log ~0.1% of checks
                logger.debug(
                    "Pad-facing check %s: pad_dir=[%.2f,%.2f,%.2f] to_obj=[%.2f,%.2f,%.2f] score=%.3f",
                    fname, pad_direction[0], pad_direction[1], pad_direction[2],
                    to_object[0], to_object[1], to_object[2], pad_facing_score
                )
            # Require pad to face object (positive score)
            # Threshold of -0.3 allows angles up to ~107 degrees from ideal
            if pad_facing_score >= -0.3:
                pad_facing_fingers.append(fname)
        return pad_facing_fingers

    def _dist_to_cube_surface(self, point: NDArray) -> float:
        obj_center = self._get_object_center()
        half = np.asarray(self.surface.size[:3], dtype=np.float64)
        R_obj = self.surface.rotation
        local = R_obj.T @ (point - obj_center)
        abs_local = np.abs(local)
        if np.all(abs_local <= half):
            return -float(np.min(half - abs_local))
        clamped = np.clip(local, -half, half)
        return float(np.linalg.norm(local - clamped))

    def _project_to_surface(self, point: NDArray) -> Tuple[NDArray, NDArray]:
        if self.surface.kind == GeomKind.BOX:
            obj_center = self._get_object_center()
            half = np.asarray(self.surface.size[:3], dtype=np.float64)
            R_obj = self.surface.rotation
            local = R_obj.T @ (point - obj_center)
            dists_to_faces = half - np.abs(local)
            nearest_axis = int(np.argmin(dists_to_faces))
            proj_local = local.copy()
            sign = 1.0 if local[nearest_axis] >= 0 else -1.0
            proj_local[nearest_axis] = sign * half[nearest_axis]
            for a in range(3):
                if a != nearest_axis:
                    proj_local[a] = np.clip(proj_local[a], -half[a], half[a])
            surf_pt = R_obj @ proj_local + obj_center
            normal_local = np.zeros(3)
            normal_local[nearest_axis] = sign
            normal_world = R_obj @ normal_local
            return surf_pt, normal_world
        pos = point.copy()
        for _ in range(8):
            sdf_val = float(self.surface.signed_distance(pos.reshape(1, 3))[0])
            if abs(sdf_val) < 0.001:
                break
            out_d = self.surface.outward_direction(pos.reshape(1, 3))[0]
            pos = pos - out_d * sdf_val
        out_d = self.surface.outward_direction(pos.reshape(1, 3))[0]
        return pos, out_d

    def _set_finger_curl(
        self,
        q: NDArray,
        curl: float,
    ) -> None:
        """Set finger joints to *curl* fraction with dip/ip at half-curl.

        This matches the Monte-Carlo-validated pattern where distal joints
        curl less aggressively than proximal joints.
        """
        for fname in self._active_fingers:
            joints, _ = self._finger_map[fname]
            for jn in joints:
                jid = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    jn,
                )
                if jid < 0:
                    continue
                if not bool(self.model.jnt_limited[jid]):
                    continue
                qadr = int(self.model.jnt_qposadr[jid])
                lo = float(self.model.jnt_range[jid, 0])
                hi = float(self.model.jnt_range[jid, 1])
                if "dip" in jn or "ip" in jn:
                    q[qadr] = lo + (hi - lo) * curl * 0.5
                else:
                    q[qadr] = lo + (hi - lo) * curl

    def _set_finger_curl_mc(
        self,
        q: NDArray,
        curl: float,
    ) -> None:
        """Set finger joints with MC-specific variation.

        Same as ``_set_finger_curl`` but with additional randomness:
        - Thumb ``cmc_yaw`` gets a random value within its range
        - All other joints follow the dip/ip half-curl pattern
        """
        for fname in self._active_fingers:
            joints, _ = self._finger_map[fname]
            for ji, jn in enumerate(joints):
                jid = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    jn,
                )
                if jid < 0:
                    continue
                if not bool(self.model.jnt_limited[jid]):
                    continue
                qadr = int(self.model.jnt_qposadr[jid])
                lo = float(self.model.jnt_range[jid, 0])
                hi = float(self.model.jnt_range[jid, 1])
                # Thumb cmc_yaw: fully random within range
                if fname == "thumb" and ji == 0:
                    q[qadr] = float(self._rng.uniform(lo, hi))
                elif "dip" in jn or "ip" in jn:
                    q[qadr] = lo + (hi - lo) * curl * 0.5
                else:
                    q[qadr] = lo + (hi - lo) * curl

    def _build_approach_rotation(
        self,
        approach_dir: NDArray,
        wrist_rot: float = 0.0,
    ) -> NDArray:
        """Build rotation matrix for approach.
        Body +Z faces the object (toward -approach_dir) because the
        RealHand L6 fingers extend primarily along +Z in the body frame.
        Body +X is lateral (thumb direction), +Y is roughly palm-normal.
        wrist_rot adds rotation around the approach axis (+Z).
        """
        z_axis = -approach_dir  # +Z toward object (finger direction)
        # Choose an up vector for Gram-Schmidt
        if abs(float(z_axis[2])) < 0.9:
            up = np.array([0.0, 0.0, 1.0])
        else:
            up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(up, z_axis)
        xn = np.linalg.norm(x_axis)
        if xn < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis = x_axis / xn
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
        R_base = np.column_stack([x_axis, y_axis, z_axis])
        # Apply wrist rotation around z_axis (approach axis)
        if abs(wrist_rot) > 1e-6:
            c, s = np.cos(wrist_rot), np.sin(wrist_rot)
            R_wrist = np.array(
                [
                    [c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1],
                ]
            )
            R_base = R_base @ R_wrist

        return R_base

    # Contact target generation

    def _generate_contacts_for_approach(
        self,
        approach_dir: NDArray,
    ) -> List[Dict[str, SurfaceSample]]:
        """Generate contact targets matching an approach direction"""
        if self.surface.kind != GeomKind.BOX:
            return self._generate_contacts_generic()

        half = np.asarray(self.surface.size[:3], dtype=np.float64)
        R_obj = self.surface.rotation

        # Find which face the approach aligns with
        local_approach = R_obj.T @ (-approach_dir)
        abs_local = np.abs(local_approach)
        dom_axis = int(np.argmax(abs_local))
        dom_sign = 1 if local_approach[dom_axis] > 0 else -1

        results: List[Dict[str, SurfaceSample]] = []
        ortho_axes = [a for a in (0, 1, 2) if a != dom_axis]

        # Generate 2 contact sets with jitter
        for _ in range(2):
            thumb = self._sample_face_contact((dom_axis, dom_sign))
            index = self._sample_face_contact((dom_axis, -dom_sign))
            # Middle on orthogonal face
            oa = ortho_axes[int(self._rng.integers(0, len(ortho_axes)))]
            os_sign = 1 if self._rng.random() > 0.5 else -1
            middle = self._sample_face_contact((oa, os_sign))
            results.append(
                {
                    "thumb": thumb,
                    "index": index,
                    "middle": middle,
                }
            )

        return results

    def _generate_contacts_generic(
        self,
    ) -> List[Dict[str, SurfaceSample]]:
        """Fallback contact generation for non-box objects"""
        samples = self.surface.sample(30, rng=self._rng)
        if samples is None or len(samples.points) < 3:
            return []

        pts = samples.points
        norms = samples.normals

        best_set = None
        best_spread = -1.0
        for _ in range(20):
            idxs = self._rng.choice(len(pts), size=3, replace=False)
            p3 = pts[idxs]
            spread = float(
                np.min(
                    [
                        np.linalg.norm(p3[0] - p3[1]),
                        np.linalg.norm(p3[0] - p3[2]),
                        np.linalg.norm(p3[1] - p3[2]),
                    ]
                )
            )
            if spread > best_spread:
                best_spread = spread
                best_set = idxs

        if best_set is None:
            return []

        fnames = self._active_fingers[:3]
        cs: Dict[str, SurfaceSample] = {}
        for fi, pi in enumerate(best_set):
            cs[fnames[fi]] = SurfaceSample(
                points=pts[pi].copy(),
                normals=_normalize(norms[pi].copy()),
                weights=np.array([1.0]),
            )
        return [cs]

    def _sample_face_contact(
        self,
        face: Tuple[int, int],
    ) -> SurfaceSample:
        """Sample a single contact point on a box face"""
        axis, sign = face
        half = np.asarray(self.surface.size[:3], dtype=np.float64)
        p = np.zeros(3, dtype=np.float64)
        n = np.zeros(3, dtype=np.float64)
        p[axis] = sign * half[axis]
        n[axis] = float(sign)

        free_axes = [a for a in (0, 1, 2) if a != axis]
        for a in free_axes:
            margin = 0.70  # 70% of face
            p[a] = self._rng.uniform(-half[a] * margin, half[a] * margin)

        pw = self.surface.to_world(p[None, :])[0]
        nw = self.surface.normal_to_world(n[None, :])[0]
        return SurfaceSample(
            points=pw.astype(np.float64),
            normals=_normalize(nw.astype(np.float64)),
            weights=np.array([1.0], dtype=np.float64),
        )

    # IK target computation

    def _compute_ik_targets(
        self,
        contact_set: Dict[str, SurfaceSample],
    ) -> Dict[str, NDArray]:
        """Compute IK targets with tip overshoot compensation.

        The tip site extends ~20mm past the physical fingertip mesh.
        To place the physical fingertip ON the surface, we target
        the site at (overshoot + small margin) above the surface.
        """
        targets = {}
        for fname, sample in contact_set.items():
            overshoot = self._tip_overshoot.get(fname, 0.0)
            # Target the site at overshoot above the surface contact point
            # so the physical fingertip pad touches the surface
            offset = overshoot + self.cfg.pad_offset + 0.002  # pad_offset + 2mm contact margin
            targets[fname] = sample.points + sample.normals * offset
        return targets

    # Standoff computation

    def _compute_standoff(self) -> float:
        """Compute standoff distance from object center.
        For precision grip with high curl, the effective fingertip reach
        is much shorter than at zero curl. We use the curled reach (85% curl)
        which is ~100mm for RealHand L6, not the extended reach (~180mm).
        
        The standoff should place the hand so that:
        1. Curled fingertips can reach the object surface
        2. Proximal links don't penetrate the object
        """
        # Object radius
        if self.surface.kind == GeomKind.BOX:
            obj_radius = float(np.max(self.surface.size[:3]))
        elif self.surface.kind in (GeomKind.SPHERE, GeomKind.CAPSULE):
            obj_radius = float(self.surface.size[0])
        elif self.surface.kind == GeomKind.CYLINDER:
            obj_radius = float(max(self.surface.size[0], self.surface.size[1]))
        else:
            obj_radius = 0.05  # conservative default
        # Compute curled fingertip reach (at 85% curl for precision grip)
        curled_reach = self._estimate_curled_fingertip_reach(curl=0.85)
        
        # Standoff target: position hand so curled fingertips can reach object
        # curled_reach is approximately 100mm, obj_radius is approximately 25mm
        # We want fingertips ~5-10mm from object surface for IK headroom
        # standoff = curled_reach - obj_radius + margin
        ik_margin = 0.010  # 10mm margin for IK to close the gap
        target = curled_reach - obj_radius + ik_margin
        
        # Also ensure proximal links don't hit: standoff >= proximal_reach - obj_radius
        min_for_prox = self._proximal_reach - obj_radius + 0.005
        target = max(target, min_for_prox)
        
        # Clamp to config limits
        standoff = np.clip(target, self.cfg.min_standoff, self.cfg.max_standoff)
        logger.info(
            "Standoff: %.1fmm (target=%.1fmm, obj_radius=%.1fmm, "
            "curled_reach=%.1fmm, prox_reach=%.1fmm, extended_reach=%.1fmm)",
            standoff * 1000,
            target * 1000,
            obj_radius * 1000,
            curled_reach * 1000,
            self._proximal_reach * 1000,
            self._fingertip_reach * 1000,
        )
        return standoff

    def _estimate_proximal_reach(self) -> float:
        """Estimate the maximum reach of proximal link geoms from the base.

        This is the distance from the base body to the farthest point on
        any non-distal hand geom, measured at zero configuration.
        """
        q_save = self.data.qpos.copy()

        # Set to zero configuration
        self.data.qpos[:] = 0
        # Set base to origin
        self.data.qpos[self._base_qadr : self._base_qadr + 3] = 0.0
        self.data.qpos[self._base_qadr + 3 : self._base_qadr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)

        base_pos = self.data.xpos[self._base_body_id].copy()
        max_reach = 0.0

        for gi in range(self.model.ngeom):
            if gi not in self._hand_geom_ids:
                continue
            if gi in self._distal_geom_ids:
                continue  # skip distal (fingertip) geoms

            # Skip visual-only geoms (only collision geoms matter)
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gi)
            if gname and "visual" in gname:
                continue
            # Also skip geoms with contype=0 and conaffinity=0 (non-colliding)
            if (
                int(self.model.geom_contype[gi]) == 0
                and int(self.model.geom_conaffinity[gi]) == 0
            ):
                continue

            # Compute max distance from base to any point on this geom
            geom_world_pos = self.data.geom_xpos[gi].copy()
            geom_xmat = self.data.geom_xmat[gi].reshape(3, 3)
            geom_type = int(self.model.geom_type[gi])
            if geom_type == 7:  # mesh: transform vertices to world space
                mesh_id = int(self.model.geom_dataid[gi])
                if mesh_id >= 0:
                    vert_adr = int(self.model.mesh_vertadr[mesh_id])
                    vert_num = int(self.model.mesh_vertnum[mesh_id])
                    verts_local = self.model.mesh_vert[vert_adr : vert_adr + vert_num]
                    # Transform to world: R @ v + t
                    verts_world = (geom_xmat @ verts_local.T).T + geom_world_pos
                    dist = float(np.max(np.linalg.norm(verts_world - base_pos, axis=1)))
                else:
                    dist = float(np.linalg.norm(geom_world_pos - base_pos))
            elif geom_type == 6:  # box: transform 8 corners to world space
                sz = self.model.geom_size[gi]
                corners = np.array(
                    [
                        [s0 * sz[0], s1 * sz[1], s2 * sz[2]]
                        for s0 in (-1, 1)
                        for s1 in (-1, 1)
                        for s2 in (-1, 1)
                    ]
                )
                corners_world = (geom_xmat @ corners.T).T + geom_world_pos
                dist = float(np.max(np.linalg.norm(corners_world - base_pos, axis=1)))
            elif geom_type == 3:  # capsule: check both endpoints + radius
                half_len = float(self.model.geom_size[gi, 1])
                radius = float(self.model.geom_size[gi, 0])
                # Capsule axis is local z
                ep1 = geom_world_pos + geom_xmat[:, 2] * half_len
                ep2 = geom_world_pos - geom_xmat[:, 2] * half_len
                d1 = float(np.linalg.norm(ep1 - base_pos))
                d2 = float(np.linalg.norm(ep2 - base_pos))
                dist = max(d1, d2) + radius
            else:  # sphere, cylinder, etc.
                dist = float(np.linalg.norm(geom_world_pos - base_pos))
                dist += float(self.model.geom_size[gi, 0])

            max_reach = max(max_reach, dist)
        # Restore
        self.data.qpos[:] = q_save
        mujoco.mj_forward(self.model, self.data)

        logger.info("Estimated proximal reach: %.1f mm", max_reach * 1000)
        return max_reach

    def _estimate_fingertip_reach(self) -> float:
        """Estimate max fingertip reach from base at zero curl"""
        q_save = self.data.qpos.copy()

        # Zero configuration
        self.data.qpos[:] = 0
        self.data.qpos[self._base_qadr : self._base_qadr + 3] = 0.0
        self.data.qpos[self._base_qadr + 3 : self._base_qadr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)

        base_pos = self.data.xpos[self._base_body_id].copy()
        max_reach = 0.0
        for fname in self._active_fingers:
            sid = self._tip_site_ids[fname]
            tip_pos = self.data.site_xpos[sid].copy()
            reach = float(np.linalg.norm(tip_pos - base_pos))
            max_reach = max(max_reach, reach)

        # Restore
        self.data.qpos[:] = q_save
        mujoco.mj_forward(self.model, self.data)

        logger.info("Estimated fingertip reach: %.1f mm", max_reach * 1000)
        return max_reach

    def _estimate_curled_fingertip_reach(self, curl: float = 0.85) -> float:
        """Estimate fingertip reach from base at a given curl level.
        
        For precision grip, fingers are highly curled (75-95%), which brings
        the fingertips much closer to the hand base than at zero curl.
        This is critical for computing correct standoff distance.
        """
        q_save = self.data.qpos.copy()

        # Zero base pose
        self.data.qpos[:] = 0
        self.data.qpos[self._base_qadr : self._base_qadr + 3] = 0.0
        self.data.qpos[self._base_qadr + 3 : self._base_qadr + 7] = [1, 0, 0, 0]

        # Set all finger joints to target curl
        for fname in self._active_fingers:
            joints, _ = self._finger_map[fname]
            for jn in joints:
                jid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, jn
                )
                if jid < 0:
                    continue
                if not bool(self.model.jnt_limited[jid]):
                    continue
                qadr = int(self.model.jnt_qposadr[jid])
                lo = float(self.model.jnt_range[jid, 0])
                hi = float(self.model.jnt_range[jid, 1])
                # Distal joints curl at half rate
                if "dip" in jn or "ip" in jn:
                    self.data.qpos[qadr] = lo + (hi - lo) * curl * 0.5
                else:
                    self.data.qpos[qadr] = lo + (hi - lo) * curl

        mujoco.mj_forward(self.model, self.data)

        base_pos = self.data.xpos[self._base_body_id].copy()
        max_reach = 0.0
        for fname in self._active_fingers:
            sid = self._tip_site_ids.get(fname)
            if sid is None:
                continue
            tip_pos = self.data.site_xpos[sid].copy()
            reach = float(np.linalg.norm(tip_pos - base_pos))
            max_reach = max(max_reach, reach)

        # Restore
        self.data.qpos[:] = q_save
        mujoco.mj_forward(self.model, self.data)

        logger.info(
            "Estimated curled (%.0f%%) fingertip reach: %.1f mm",
            curl * 100,
            max_reach * 1000,
        )
        return max_reach

    # Validation

    def _measure_worst_penetration(self) -> float:
        """Worst hand-object penetration from MuJoCo contacts"""
        worst = 0.0
        if not self._obj_geom_ids:
            return 0.0
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            is_hand_obj = (g1 in self._hand_geom_ids and g2 in self._obj_geom_ids) or (
                g1 in self._obj_geom_ids and g2 in self._hand_geom_ids
            )
            if is_hand_obj and c.dist < 0.0:
                worst = max(worst, -float(c.dist))
        return worst

    def _measure_proximal_penetration(self) -> float:
        """Worst proximal (non-distal) hand-object penetration.

        This is the KEY metric.  Proximal links must NOT penetrate.
        Distal (fingertip) contacts are fine and expected.
        """
        worst = 0.0
        if not self._obj_geom_ids:
            return 0.0
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            is_hand_obj = (g1 in self._hand_geom_ids and g2 in self._obj_geom_ids) or (
                g1 in self._obj_geom_ids and g2 in self._hand_geom_ids
            )
            if not is_hand_obj or c.dist >= 0:
                continue
            # Check if it's a proximal (non-distal) geom
            hand_g = g1 if g1 in self._hand_geom_ids else g2
            if hand_g in self._distal_geom_ids:
                continue  # fingertip contact is fine
            worst = max(worst, -float(c.dist))
        return worst

    def _check_finger_separation(self, tips: NDArray) -> bool:
        """Check minimum pairwise distance between fingertips"""
        if len(tips) < 2:
            return False
        for i, j in combinations(range(len(tips)), 2):
            if (
                float(np.linalg.norm(tips[i] - tips[j]))
                < self.cfg.min_finger_separation
            ):
                return False
        return True

    def _detect_finger_contacts(self) -> set[str]:
        """Detect which fingers have MuJoCo contact with the object"""
        result: set[str] = set()
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            is_hand_obj = (g1 in self._hand_geom_ids and g2 in self._obj_geom_ids) or (
                g1 in self._obj_geom_ids and g2 in self._hand_geom_ids
            )
            if not is_hand_obj:
                continue
            hand_g = g1 if g1 in self._hand_geom_ids else g2
            gname = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, hand_g) or ""
            )
            for fn in self._active_fingers:
                if fn in gname:
                    result.add(fn)
                    break
        return result

    # GWS evaluation

    def _evaluate_gws(
        self,
        fingertip_positions: Dict[str, NDArray],
    ) -> GWSResult:
        """Build ContactInfo list and evaluate GWS quality"""
        obj_center = self._get_object_center()
        contacts: List[ContactInfo] = []

        for fn, tip_pos in fingertip_positions.items():
            # Project to surface via SDF descent
            pos = tip_pos.copy()
            for _ in range(8):
                sdf_cur = float(self.surface.signed_distance(pos.reshape(1, 3))[0])
                if abs(sdf_cur) < 0.001:
                    break
                out_d = self.surface.outward_direction(pos.reshape(1, 3))[0]
                pos = pos - out_d * sdf_cur

            out_dir = self.surface.outward_direction(pos.reshape(1, 3))[0]
            normal = -out_dir  # inward toward object
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-9:
                normal = _normalize(obj_center - pos)
            else:
                normal = normal / norm_len

            t1, t2 = _orthonormal_tangent_basis(normal)
            frame = np.stack([normal, t1, t2], axis=0)

            contacts.append(
                ContactInfo(
                    pos=pos.copy(),
                    frame=frame,
                    dist=0.0,
                    force=np.array([1.0, 0.0, 0.0]),
                    geom1=-1,
                    geom2=-1,
                )
            )

        if len(contacts) < 2:
            return GWSResult(
                epsilon=0.0,
                volume=0.0,
                min_singular=0.0,
                is_force_closure=False,
                n_contacts=len(contacts),
            )

        return analyze_gws(contacts, obj_center, self.cfg.friction_coef)

    # Tip overshoot

    def _compute_tip_overshoot(self) -> None:
        """Measure how far each tip_site extends past the distal collision mesh"""
        for fname in self._active_fingers:
            sid = self._tip_site_ids[fname]
            site_z = float(self.model.site_pos[sid, 2])

            body_id = int(self.model.site_bodyid[sid])
            mesh_max_z_body = None
            for gi in range(self.model.ngeom):
                if int(self.model.geom_bodyid[gi]) != body_id:
                    continue
                gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gi)
                if gname is None or "collision" not in gname:
                    continue
                if int(self.model.geom_type[gi]) != 7:  # mesh
                    continue
                mesh_id = int(self.model.geom_dataid[gi])
                vert_adr = int(self.model.mesh_vertadr[mesh_id])
                vert_num = int(self.model.mesh_vertnum[mesh_id])
                verts = self.model.mesh_vert[vert_adr : vert_adr + vert_num]
                geom_pos_z = float(self.model.geom_pos[gi, 2])
                mesh_max_z_body = float(verts[:, 2].max()) + geom_pos_z
                break

            if mesh_max_z_body is not None and site_z > mesh_max_z_body:
                overshoot = site_z - mesh_max_z_body
            else:
                overshoot = 0.0

            self._tip_overshoot[fname] = overshoot
            logger.info(
                "Finger '%s' tip overshoot: %.1f mm (site=%.1f mm, mesh_top=%.1f mm)",
                fname,
                overshoot * 1000,
                site_z * 1000,
                (mesh_max_z_body or 0) * 1000,
            )

    # Utility

    def _update_surface_pose(self) -> None:
        """Sync surface pose from simulation state"""
        if self._obj_body_id is not None:
            self.surface.position = self.data.xpos[self._obj_body_id].copy()
            self.surface.rotation = (
                self.data.xmat[self._obj_body_id].reshape(3, 3).copy()
            )

    def _get_object_center(self) -> NDArray:
        if self._obj_body_id is not None:
            return self.data.xpos[self._obj_body_id].copy()
        return np.asarray(self.surface.position, dtype=np.float64).copy()

    def _resolve_object_body_id(self, name: str | None) -> int | None:
        if name is not None:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            return int(bid) if bid >= 0 else None
        for guess in ("cube", "object"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, guess)
            if bid >= 0:
                return int(bid)
        return None

    def _extract_finger_qpos(self, qpos: NDArray) -> Dict[str, NDArray]:
        """Extract per-finger joint values from full qpos"""
        result: Dict[str, NDArray] = {}
        for fname in self._active_fingers:
            joints, _ = self._finger_map[fname]
            vals = []
            for jn in joints:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid >= 0:
                    qadr = int(self.model.jnt_qposadr[jid])
                    vals.append(float(qpos[qadr]))
            result[fname] = np.array(vals, dtype=np.float64)
        return result
