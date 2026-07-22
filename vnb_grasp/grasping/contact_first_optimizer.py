"""Contact-first grasp optimizer using approach-based initial poses + Jacobian IK.

Algorithm
---------
1.  Generate approach-based initial hand poses: orient the hand so that
    body +X faces the object (opposition pose), position at a standoff,
    and set fingers to partial curl.  This ensures the curl arc sweeps
    toward the object, enabling force-closure opposition.

2.  For each approach pose, pick contact targets on opposing cube faces
    that match the approach geometry (thumb on the face nearest the palm,
    index/middle on opposite/orthogonal faces).

3.  Run ContactIKSolver (MuJoCo Jacobian-based DLS IK) to drive
    fingertips to their targets.

4.  Retreat from any remaining penetration by nudging the base away.

5.  Validate: penetration, finger separation, surface contacts, GWS.

Author: Clinton Enwerem
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import mujoco as _mujoco
except ImportError:
    _mujoco = None

mujoco = _mujoco

from .object_surface import ObjectSurface, SurfaceSample, GeomKind
from .gws_quality import GWSResult, analyze_gws
from ..belief.mujoco_rollout import ContactInfo
from .grasp_sampler import (
    DEFAULT_BASE_JOINT,
    DEFAULT_FINGER_MAP,
    SampledGrasp,
    ContactIKSolver,
    _is_descendant,
    _collect_geoms_in_subtree,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class OptimizerConfig:
    """Configuration for ContactFirstOptimizer"""

    n_starts: int = 96
    top_k: int = 20
    friction_coef: float = 0.8
    max_penetration: float = 0.002  # 2 mm
    sdf_tolerance: float = 0.015  # 15 mm: generous contact detection
    min_finger_separation: float = 0.008  # 8 mm
    min_epsilon: float = 0.001
    contact_margin: float = 0.002  # 2 mm standoff for IK targets
    ik_max_iter: int = 800
    ik_damping: float = 1e-2
    ik_step_size: float = 0.30
    ik_tol: float = 3e-3  # 3 mm convergence
    penetration_weight: float = 5.0
    retreat_max_steps: int = 100
    retreat_step_size: float = 0.001  # 1 mm per retreat step
    standoff: float = 0.055  # 55 mm from object center
    curl_fractions: List[float] = field(default_factory=lambda: [0.40, 0.50, 0.60])
    active_fingers: List[str] | None = None
    random_seed: int | None = None


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class ContactFirstOptimizer:
    """Contact-first grasp optimizer using approach-based seeds + Jacobian IK.

    Typical usage::

        surface = ObjectSurface.from_model(model, body_name='cube')
        opt = ContactFirstOptimizer(model, data, surface, object_body_name='cube')
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
        config: OptimizerConfig | None = None,
        object_body_name: str | None = None,
    ) -> None:
        if mujoco is None:
            raise ImportError("mujoco is required")

        self.model = model
        self.data = data
        self.surface = surface
        self.cfg = config or OptimizerConfig()
        self._rng = np.random.default_rng(self.cfg.random_seed)

        self._finger_map = DEFAULT_FINGER_MAP
        self._active_fingers = self.cfg.active_fingers or [
            "thumb",
            "index",
            "middle",
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

        # Geom sets for collision detection
        if self._obj_body_id is not None:
            self._obj_geom_ids = _collect_geoms_in_subtree(model, self._obj_body_id)
        else:
            self._obj_geom_ids: set = set()
        self._hand_geom_ids = _collect_geoms_in_subtree(model, self._base_body_id)

        # Build the IK solver (reuses the proven ContactIKSolver)
        self._ik = ContactIKSolver(
            model,
            data,
            finger_map=self._finger_map,
            base_joint=DEFAULT_BASE_JOINT,
            damping=self.cfg.ik_damping,
            max_iter=self.cfg.ik_max_iter,
            tol=self.cfg.ik_tol,
            step_size=self.cfg.ik_step_size,
            surface=surface,
            penetration_weight=self.cfg.penetration_weight,
            contact_margin=self.cfg.contact_margin,
        )
        self._ik.set_collision_geoms(self._hand_geom_ids, self._obj_geom_ids)

        # Tip site IDs for the active fingers
        self._tip_site_ids: Dict[str, int] = {}
        for fname in self._active_fingers:
            _, site_name = self._finger_map[fname]
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid < 0:
                raise ValueError(f"Missing fingertip site: {site_name}")
            self._tip_site_ids[fname] = sid

        # Compute tip overshoot
        self._tip_overshoot: Dict[str, float] = {}
        self._compute_tip_overshoot()

        # Distal body/geom IDs (exempt from proximal penetration check)
        self._distal_body_ids: set = set()
        self._distal_geom_ids: set = set()
        for fname in self._active_fingers:
            sid = self._tip_site_ids[fname]
            bid = int(model.site_bodyid[sid])
            self._distal_body_ids.add(bid)
        for gi in range(model.ngeom):
            if int(model.geom_bodyid[gi]) in self._distal_body_ids:
                self._distal_geom_ids.add(gi)

        mujoco.mj_forward(self.model, self.data)
        logger.info(
            "ContactFirstOptimizer initialized: "
            "n_starts=%d, top_k=%d, fingers=%s, object_body=%s",
            self.cfg.n_starts,
            self.cfg.top_k,
            self._active_fingers,
            str(self._obj_body_id),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self) -> List[SampledGrasp]:
        """Run the optimizer and return ranked grasps"""
        q0 = self.data.qpos.copy()

        # Update surface pose from sim
        self._update_surface_pose()
        obj_center = self._get_object_center()

        # Generate approach seeds paired with contact targets
        seeds = self._generate_approach_seeds()
        logger.info("Generated %d approach seeds", len(seeds))

        # Trim to n_starts
        if len(seeds) > self.cfg.n_starts:
            idxs = self._rng.permutation(len(seeds))[: self.cfg.n_starts]
            seeds = [seeds[int(i)] for i in idxs]

        results: List[SampledGrasp] = []
        stats = {
            "ik_converged": 0,
            "pen_rejected": 0,
            "sep_rejected": 0,
            "sdf_rejected": 0,
            "gws_rejected": 0,
        }

        for idx, (q_init, contact_set) in enumerate(seeds):
            # Phase 1: Compute IK targets with overshoot compensation
            targets = self._compute_ik_targets(contact_set)

            # Phase 2: Solve IK
            qpos, residual, achieved = self._ik.solve(targets, q_init=q_init)

            # Check IK convergence
            max_tip_err = 0.0
            for fname in targets:
                if fname in achieved:
                    err = float(np.linalg.norm(achieved[fname] - targets[fname]))
                    max_tip_err = max(max_tip_err, err)
            if max_tip_err > 0.025:  # 25mm: didn't converge
                continue
            stats["ik_converged"] += 1

            # Phase 3: Retreat from penetration
            pen = self._retreat_from_penetration()

            # Re-read achieved positions after retreat
            mujoco.mj_forward(self.model, self.data)
            achieved = {
                fname: self.data.site_xpos[self._tip_site_ids[fname]].copy()
                for fname in self._active_fingers
            }

            # Phase 4: Validate
            used_fingers = list(contact_set.keys())

            # 4a: Penetration check (ALL hand geoms, not just proximal)
            total_pen = self._measure_worst_penetration()
            if total_pen > self.cfg.max_penetration:
                stats["pen_rejected"] += 1
                continue

            # 4b: Finger separation
            tip_pts = np.array(
                [achieved[f] for f in used_fingers if f in achieved],
                dtype=np.float64,
            )
            if not self._check_finger_separation(tip_pts):
                stats["sep_rejected"] += 1
                continue

            # 4c: Contact validity
            fingers_in_contact = self._detect_finger_contacts()
            for fname in used_fingers:
                if fname in achieved:
                    sdf_val = float(
                        self.surface.signed_distance(achieved[fname].reshape(1, 3))[0]
                    )
                    if abs(sdf_val) <= self.cfg.sdf_tolerance:
                        fingers_in_contact.add(fname)

            valid_fingers = [f for f in used_fingers if f in fingers_in_contact]
            if len(valid_fingers) < 2:
                stats["sdf_rejected"] += 1
                continue

            # 4d: GWS quality
            gws = self._evaluate_gws({f: achieved[f] for f in valid_fingers})
            if gws.epsilon <= self.cfg.min_epsilon:
                stats["gws_rejected"] += 1
                continue

            # Build result
            finger_qpos = self._extract_finger_qpos(self.data.qpos)
            results.append(
                SampledGrasp(
                    hand_qpos=self.data.qpos.copy(),
                    finger_qpos=finger_qpos,
                    fingertip_positions={
                        f: achieved[f].copy() for f in used_fingers if f in achieved
                    },
                    target_contacts={
                        f: contact_set[f].points.copy() for f in used_fingers
                    },
                    target_normals={
                        f: contact_set[f].normals.copy() for f in used_fingers
                    },
                    residual=residual,
                    max_penetration=total_pen,
                    gws=gws,
                    seed_source="contact_first",
                )
            )

            if (idx + 1) % 16 == 0:
                logger.info(
                    "Processed %d/%d -> %d valid (IK: %d, pen: %d, "
                    "sep: %d, sdf: %d, gws: %d)",
                    idx + 1,
                    len(seeds),
                    len(results),
                    stats["ik_converged"],
                    stats["pen_rejected"],
                    stats["sep_rejected"],
                    stats["sdf_rejected"],
                    stats["gws_rejected"],
                )

        # Restore original state
        self.data.qpos[:] = q0
        mujoco.mj_forward(self.model, self.data)

        # Sort by quality
        results.sort(key=lambda g: g.gws.epsilon, reverse=True)
        trimmed = results[: self.cfg.top_k]
        logger.info(
            "Solve complete: %d valid grasps (returning %d). "
            "IK: %d, pen rej: %d, sep rej: %d, "
            "sdf rej: %d, gws rej: %d",
            len(results),
            len(trimmed),
            stats["ik_converged"],
            stats["pen_rejected"],
            stats["sep_rejected"],
            stats["sdf_rejected"],
            stats["gws_rejected"],
        )
        return trimmed

    # ------------------------------------------------------------------
    # Approach seed generation
    # ------------------------------------------------------------------

    def _generate_approach_seeds(
        self,
    ) -> List[Tuple[NDArray, Dict[str, SurfaceSample]]]:
        """Generate (q_init, contact_set) pairs using approach-based poses.

        For each approach direction, orient the hand in "opposition mode"
        (body +X faces the object, so finger curl sweeps toward it),
        position at standoff distance, and generate matching contact
        targets on opposing faces.

        The hand anatomy:
        - At zero joints, fingers extend along body +Z (~175mm)
        - When pitch joints curl, tips sweep from +Z toward +X
        - Thumb is at body (+X, +Y, +Z) relative to base
        - So body +X should face the object for fingertips to reach it
        """
        obj_center = self._get_object_center()
        q_template = self.data.qpos.copy()
        seeds: List[Tuple[NDArray, Dict[str, SurfaceSample]]] = []

        # Approach directions: lateral and tilted
        # Each direction is where the hand approaches FROM (base --> object)
        approach_dirs = [
            np.array([+1, 0, 0]),
            np.array([-1, 0, 0]),
            np.array([0, +1, 0]),
            np.array([0, -1, 0]),
            np.array([+1, +1, 0]),
            np.array([+1, -1, 0]),
            np.array([-1, +1, 0]),
            np.array([-1, -1, 0]),
            np.array([+1, 0, +1]),
            np.array([-1, 0, +1]),
            np.array([0, +1, +1]),
            np.array([0, -1, +1]),
            # Near-top approaches (tilted)
            np.array([+1, 0, +2]),
            np.array([-1, 0, +2]),
            np.array([0, +1, +2]),
            np.array([0, -1, +2]),
        ]

        standoffs = [self.cfg.standoff, self.cfg.standoff + 0.015]

        for d in approach_dirs:
            d_hat = d / np.linalg.norm(d)

            for standoff in standoffs:
                for curl in self.cfg.curl_fractions:
                    q = q_template.copy()

                    # Position: object_center + approach_dir * standoff
                    base_pos = obj_center + d_hat * standoff
                    q[self._base_qadr : self._base_qadr + 3] = base_pos

                    # Orientation: body +X faces object (toward -d_hat)
                    # This is the "opposition" orientation from GraspSampler
                    x_axis = -d_hat  # +X toward object
                    # Choose up vector
                    if abs(float(x_axis[2])) < 0.9:
                        up = np.array([0.0, 0.0, 1.0])
                    else:
                        up = np.array([0.0, 1.0, 0.0])
                    z_axis = np.cross(x_axis, up)
                    zn = np.linalg.norm(z_axis)
                    if zn < 1e-6:
                        z_axis = np.array([0.0, 0.0, 1.0])
                    else:
                        z_axis = z_axis / zn
                    y_axis = np.cross(z_axis, x_axis)
                    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
                    R = np.column_stack([x_axis, y_axis, z_axis])
                    q[self._base_qadr + 3 : self._base_qadr + 7] = _matrix_to_quat(R)

                    # Set finger joints to curl fraction
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
                            if bool(self.model.jnt_limited[jid]):
                                qadr = int(self.model.jnt_qposadr[jid])
                                lo = float(self.model.jnt_range[jid, 0])
                                hi = float(self.model.jnt_range[jid, 1])
                                q[qadr] = lo + (hi - lo) * curl

                    # Generate contact targets matching this approach
                    contact_sets = self._generate_contacts_for_approach(d_hat)
                    for cs in contact_sets:
                        seeds.append((q.copy(), cs))

        logger.info(
            "Generated %d approach seeds from %d directions x %d standoffs x %d curls",
            len(seeds),
            len(approach_dirs),
            len(standoffs),
            len(self.cfg.curl_fractions),
        )
        return seeds

    def _generate_contacts_for_approach(
        self,
        approach_dir: NDArray,
    ) -> List[Dict[str, SurfaceSample]]:
        """Generate contact target sets matching an approach direction.

        For a box, assigns thumb to the face that the palm faces (nearest
        to the approach direction), index on the opposite face, and middle
        on an orthogonal face for tripod stability.

        Returns multiple contact sets (with random jitter) for diversity.
        """
        if self.surface.kind != GeomKind.BOX:
            return self._generate_contacts_generic()

        half = np.asarray(self.surface.size[:3], dtype=np.float64)
        R_obj = self.surface.rotation  # world-to-local

        # Find which box face the approach direction most aligns with
        # approach_dir points FROM base TOWARD object
        # Palm faces +X in body frame, which is -approach_dir in world
        # The thumb should contact the face that -approach_dir points to
        local_approach = R_obj.T @ (-approach_dir)  # in object local frame

        # Find dominant axis
        abs_local = np.abs(local_approach)
        dom_axis = int(np.argmax(abs_local))
        dom_sign = 1 if local_approach[dom_axis] > 0 else -1

        # Thumb face = (dom_axis, dom_sign)
        # Index face = (dom_axis, -dom_sign) (opposing)
        # Middle face = orthogonal
        results: List[Dict[str, SurfaceSample]] = []

        # Generate 2 contact sets per approach (for diversity)
        ortho_axes = [a for a in (0, 1, 2) if a != dom_axis]
        for _ in range(2):
            thumb = self._sample_face_contact((dom_axis, dom_sign))
            index = self._sample_face_contact((dom_axis, -dom_sign))
            # Middle on random orthogonal face
            oa = ortho_axes[int(self._rng.integers(0, len(ortho_axes)))]
            os = 1 if self._rng.random() > 0.5 else -1
            middle = self._sample_face_contact((oa, os))
            results.append({"thumb": thumb, "index": index, "middle": middle})

        return results

    def _generate_contacts_generic(
        self,
    ) -> List[Dict[str, SurfaceSample]]:
        """Fallback contact generation for non-box objects"""
        samples = self.surface.sample(20)
        if samples is None or len(samples.points) < 3:
            return []

        # Pick 3 well-spread points
        pts = samples.points
        norms = samples.normals

        best_set = None
        best_spread = -1.0
        for _ in range(10):
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
            margin = 0.80  # 80% of face
            p[a] = self._rng.uniform(-half[a] * margin, half[a] * margin)

        pw = self.surface.to_world(p[None, :])[0]
        nw = self.surface.normal_to_world(n[None, :])[0]
        return SurfaceSample(
            points=pw.astype(np.float64),
            normals=_normalize(nw.astype(np.float64)),
            weights=np.array([1.0], dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # IK target computation
    # ------------------------------------------------------------------

    def _compute_ik_targets(
        self,
        contact_set: Dict[str, SurfaceSample],
    ) -> Dict[str, NDArray]:
        """Compute IK targets with tip overshoot compensation"""
        targets = {}
        for fname, sample in contact_set.items():
            overshoot = self._tip_overshoot.get(fname, 0.0)
            offset = overshoot + self.cfg.contact_margin
            targets[fname] = sample.points + sample.normals * offset
        return targets

    # ------------------------------------------------------------------
    # Penetration retreat
    # ------------------------------------------------------------------

    def _retreat_from_penetration(self) -> float:
        """Nudge base away from object to resolve remaining penetration.

        Returns the final penetration depth.
        """
        mujoco.mj_forward(self.model, self.data)

        for step in range(self.cfg.retreat_max_steps):
            pen = self._measure_worst_penetration()
            if pen <= self.cfg.max_penetration:
                return pen

            # Find the deepest penetrating contact and push along its normal
            worst_normal = None
            worst_pen = 0.0
            for ci in range(self.data.ncon):
                c = self.data.contact[ci]
                g1, g2 = int(c.geom1), int(c.geom2)
                is_hand_obj = (
                    g1 in self._hand_geom_ids and g2 in self._obj_geom_ids
                ) or (g1 in self._obj_geom_ids and g2 in self._hand_geom_ids)
                if not is_hand_obj or c.dist >= 0:
                    continue
                depth = -float(c.dist)
                if depth > worst_pen:
                    worst_pen = depth
                    # Contact normal: frame[:3] points from g1 to g2
                    normal = c.frame[:3].copy()
                    # We want to push hand away from object
                    if g1 in self._hand_geom_ids:
                        # g1=hand, g2=obj; normal points hand->obj
                        # Push hand opposite: -normal
                        worst_normal = -normal
                    else:
                        # g1=obj, g2=hand; normal points obj->hand
                        # Push hand along normal
                        worst_normal = normal

            if worst_normal is None:
                break

            # Push base along contact normal
            self.data.qpos[self._base_qadr : self._base_qadr + 3] += (
                worst_normal * self.cfg.retreat_step_size
            )
            mujoco.mj_forward(self.model, self.data)

        return self._measure_worst_penetration()

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

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

    def _detect_finger_contacts(self) -> set:
        """Detect which fingers have MuJoCo contact with the object"""
        result: set = set()
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

    # ------------------------------------------------------------------
    # GWS evaluation
    # ------------------------------------------------------------------

    def _evaluate_gws(
        self,
        fingertip_positions: Dict[str, NDArray],
    ) -> GWSResult:
        """Build ContactInfo and evaluate GWS quality.

        Projects tip sites to the object surface via SDF descent for
        accurate contact locations.
        """
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
            normal = -out_dir  # inward
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                normal = _normalize(obj_center - pos)
            else:
                normal = normal / norm

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

    # ------------------------------------------------------------------
    # Tip overshoot computation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

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
