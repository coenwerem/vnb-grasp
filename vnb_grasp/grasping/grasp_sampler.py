"""Sampling-based multi-finger grasp solver with GraspIt! warm-start.

Algorithm
---------
1.  Surface sampling --- draw candidate contact points on the object
    surface, either uniformly or weighted by a user-supplied energy
    functional (e.g. curvature, reachability, normals aligned with palm).
2.  Finger assignment --- assign each fingertip to its nearest
    (or best-scoring) surface point, respecting finger kinematics.
3.  Contact IK --- solve for hand joint angles that minimise the sum of
    squared distances from each fingertip site to its assigned surface
    point, using damped-least-squares (DLS) IK over the hand's DOFs.
4.  GWS evaluation --- score the resulting contact set with the exact
    Ferrari-Canny epsilon metric from gws_quality.py.
5.  Warm-start --- when GraspIt! database grasps exist for the
    (object, hand) pair, use their joint angles as seeds for the IK solve
    instead of (or in addition to) random initialisation, giving the
    solver a head start near known good configurations.

The solver is purely kinematic / geometric --- it does not simulate
dynamics or require MJX.  It operates on a single mujoco.MjModel and
a scratch mujoco.MjData.

Public API
----------
GraspSampler
    Main entry point.  Construct with a model + hand / object metadata,
    then call solve() to get ranked SampledGrasp results.

SampledGrasp
    Result dataclass carrying joint angles, fingertip positions, contact
    targets, GWS quality, and distance residual.

ContactIKSolver
    Low-level multi-site DLS IK that drives fingertips to target positions.

FingerAssigner
    Assigns target surface points to fingertips based on distance and
    normal alignment.

Author: Clinton Enwerem
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Callable

import numpy as np
from numpy.typing import NDArray

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None

from .object_surface import ObjectSurface, SurfaceSample, GeomKind
from .gws_quality import GWSResult, analyze_gws
from ..belief.mujoco_rollout import ContactInfo

logger = logging.getLogger(__name__)


def _is_descendant(model, body_id: int, root_id: int) -> bool:
    """Check if body_id is root_id or a descendant of root_id"""
    bid = body_id
    while bid != 0:
        if bid == root_id:
            return True
        bid = int(model.body_parentid[bid])
    return bid == root_id  # root_id might be 0 (worldbody)


def _collect_geoms_in_subtree(model, root_body_id: int) -> set:
    """Collect all geom IDs whose body is root_body_id or a descendant"""
    ids = set()
    for gi in range(model.ngeom):
        bid = int(model.geom_bodyid[gi])
        if _is_descendant(model, bid, root_body_id):
            ids.add(gi)
    return ids


# ------
# Data types
# ------

# Default hand configuration: (finger_name --> (joint_names, tip_site_name))
DEFAULT_FINGER_MAP: Dict[str, Tuple[List[str], str]] = {
    "thumb": (
        ["thumb_cmc_yaw", "thumb_cmc_pitch", "thumb_ip"],
        "thumb_tip_site",
    ),
    "index": (
        ["index_mcp_pitch", "index_dip"],
        "index_tip_site",
    ),
    "middle": (
        ["middle_mcp_pitch", "middle_dip"],
        "middle_tip_site",
    ),
    "ring": (
        ["ring_mcp_pitch", "ring_dip"],
        "ring_tip_site",
    ),
    "pinky": (
        ["pinky_mcp_pitch", "pinky_dip"],
        "pinky_tip_site",
    ),
}

# The freejoint controlling the floating hand base
DEFAULT_BASE_JOINT = "hand_free"


@dataclass
class SampledGrasp:
    """Result of one grasp-sampling trial.

    Attributes
    ----------
    hand_qpos : (nq,) float
        Full qpos vector for the hand model (base + fingers).
    finger_qpos : dict[str, NDArray]
        Per-finger joint positions in radians.
    fingertip_positions : dict[str, NDArray]
        Achieved fingertip world positions.
    target_contacts : dict[str, NDArray]
        Target surface points (world frame) that were assigned to each finger.
    target_normals : dict[str, NDArray]
        Surface normals at target contacts (world frame).
    residual : float
        Sum of squared fingertip-to-target distances after IK.
    max_penetration : float
        Maximum penetration depth (metres) of any fingertip into the object.
        Zero means no penetration.
    gws : GWSResult
        Grasp Wrench Space quality evaluation.
    seed_source : str
        "graspit" if warm-started from the database, "random"
        otherwise.
    """

    hand_qpos: NDArray
    finger_qpos: Dict[str, NDArray] = field(default_factory=dict)
    fingertip_positions: Dict[str, NDArray] = field(default_factory=dict)
    target_contacts: Dict[str, NDArray] = field(default_factory=dict)
    target_normals: Dict[str, NDArray] = field(default_factory=dict)
    residual: float = float("inf")
    max_penetration: float = 0.0
    gws: GWSResult = field(
        default_factory=lambda: GWSResult(
            epsilon=0.0,
            volume=0.0,
            min_singular=0.0,
            is_force_closure=False,
            n_contacts=0,
        )
    )
    seed_source: str = "random"


# ------
# Contact IK solver
# ------

class ContactIKSolver:
    """Multi-site damped-least-squares IK for fingertip contact targets.
-
    Given K (fingertip_site, target_position) pairs and a set of DOFs,
    solve for joint angles that minimise ∑ᵢ ‖ xᵢ(q) - pᵢ ‖².

    When an ObjectSurface is provided the solver also enforces a
    penetration barrier: each iteration, any fingertip that has moved
    inside the object receives an outward push proportional to its
    penetration depth, preventing the final configuration from having
    fingers inside the object mesh.
    """

    def __init__(
        self,
        model,
        data,
        *,
        finger_map: Optional[Dict[str, Tuple[List[str], str]]] = None,
        base_joint: Optional[str] = None,
        damping: float = 1e-2,
        max_iter: int = 80,
        tol: float = 3e-3,       # metres
        step_size: float = 0.3,
        surface: Optional["ObjectSurface"] = None,
        penetration_weight: float = 5.0,
        contact_margin: float = 0.003,  # 3 mm standoff
        sdf_refine_iters: int = 30,
        sdf_refine_tol: float = 0.0005,
    ) -> None:
        if mujoco is None:
            raise ImportError("mujoco is required")

        self.model = model
        self.data = data
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.step_size = step_size
        self.surface = surface
        self.penetration_weight = penetration_weight
        self.contact_margin = contact_margin
        self._sdf_refine_iters = sdf_refine_iters
        self._sdf_refine_tol = sdf_refine_tol

        fmap = finger_map or DEFAULT_FINGER_MAP
        base_jnt = base_joint or DEFAULT_BASE_JOINT

        # Resolve joint/site ids
        self.finger_names: List[str] = []
        self.site_ids: List[int] = []
        self.finger_dof_slices: List[List[int]] = []  # per-finger DOF indices into full dof list
        self._all_dof_ids: List[int] = []

        # Base joint DOFs (6 for freejoint)
        # Include base DOFs in IK but apply HIGHER regularisation to
        # them via a diagonal weight matrix. This way the DLS solver
        # preferentially uses finger joints and only translates /
        # rotates the base as a last resort, preventing the solver
        # from wildly shoving the whole hand through the object.
        base_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, base_jnt)
        if base_jid >= 0:
            base_dofs = self._joint_dofs(base_jid)
            self._base_dof_ids = base_dofs
            for d in base_dofs:
                if d not in self._all_dof_ids:
                    self._all_dof_ids.append(d)
        else:
            self._base_dof_ids = []

        for fname, (joint_names, site_name) in fmap.items():
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid < 0:
                logger.warning("Site '%s' not found --- skipping finger '%s'", site_name, fname)
                continue
            dofs: List[int] = []
            for jn in joint_names:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid < 0:
                    logger.warning("Joint '%s' not found", jn)
                    continue
                dofs.extend(self._joint_dofs(jid))
            self.finger_names.append(fname)
            self.site_ids.append(sid)
            self.finger_dof_slices.append(dofs)
            # Add to global DOF list (avoiding duplicates)
            for d in dofs:
                if d not in self._all_dof_ids:
                    self._all_dof_ids.append(d)

        self._all_dof_ids_arr = np.array(self._all_dof_ids, dtype=np.int32)
        self.n_fingers = len(self.finger_names)

        # Weighted DLS: base DOFs get higher regularisation so the IK
        # preferentially uses finger joints over base translation.
        # Split trans/rot: translation is lightly penalised so the palm
        # can slide to put fingers in reach; rotation is heavily penalised
        # to preserve the approach orientation from the warm-start.
        base_set = set(self._base_dof_ids)
        base_trans_set = set(self._base_dof_ids[:3])  # first 3 = translation
        base_rot_set = set(self._base_dof_ids[3:])    # remaining = rotation
        nv_ik = len(self._all_dof_ids)
        self._dof_weight = np.ones(nv_ik)
        BASE_TRANS_REG = 1.2   # moderate: allow small base shifts
        BASE_ROT_REG = 3.0     # high: keep approach orientation stable
        for local_idx, dof_id in enumerate(self._all_dof_ids):
            if dof_id in base_trans_set:
                self._dof_weight[local_idx] = BASE_TRANS_REG
            elif dof_id in base_rot_set:
                self._dof_weight[local_idx] = BASE_ROT_REG
        self._W_diag = self._dof_weight          # shape (nv_ik,)
        self._W_inv_diag = 1.0 / self._dof_weight  # shape (nv_ik,)

        # Build a map from per-finger DOFs --> index into _all_dof_ids
        self._finger_dof_local: List[List[int]] = []
        for dofs in self.finger_dof_slices:
            local = [self._all_dof_ids.index(d) for d in dofs]
            self._finger_dof_local.append(local)

        # Hand / object geom ID sets for contact-based collision
        # repulsion during IK.  Populated by the GraspSampler after
        # construction via set_collision_geoms().
        self._hand_geom_ids: set = set()
        self._obj_geom_ids: set = set()

        # Distal body IDs: the bodies carrying each finger's tip site.
        # These are EXEMPTED from link repulsion because they are supposed
        # to make contact with the object.  Without this exemption the
        # repulsion pushes the WHOLE hand away when any distal mesh touches
        # the cube, preventing all fingers from converging.
        self._distal_body_ids: set = set()
        for fname, (joint_names, site_name) in fmap.items():
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid >= 0:
                self._distal_body_ids.add(int(model.site_bodyid[sid]))

        # Geoms belonging to distal bodies (for fast lookup in repulsion)
        self._distal_geom_ids: set = set()
        for gi in range(model.ngeom):
            if int(model.geom_bodyid[gi]) in self._distal_body_ids:
                self._distal_geom_ids.add(gi)

    def set_collision_geoms(
        self,
        hand_geom_ids: set,
        obj_geom_ids: set,
    ) -> None:
        """Provide hand and object geom ID sets for contact-based repulsion"""
        self._hand_geom_ids = hand_geom_ids
        self._obj_geom_ids = obj_geom_ids

    def _joint_dofs(self, jid: int) -> List[int]:
        dof_adr = int(self.model.jnt_dofadr[jid])
        jnt_type = int(self.model.jnt_type[jid])
        if jnt_type == 0:   # free
            return list(range(dof_adr, dof_adr + 6))
        elif jnt_type == 1: # ball
            return list(range(dof_adr, dof_adr + 3))
        else:               # slide or hinge
            return [dof_adr]

    def _joint_qpos_adr(self, jid: int) -> List[int]:
        q_adr = int(self.model.jnt_qposadr[jid])
        jnt_type = int(self.model.jnt_type[jid])
        if jnt_type == 0:   # free --> 7 qpos (pos + quat)
            return list(range(q_adr, q_adr + 7))
        elif jnt_type == 1: # ball --> 4 qpos (quat)
            return list(range(q_adr, q_adr + 4))
        else:
            return [q_adr]

    # 

    def solve(
        self,
        targets: Dict[str, NDArray],
        q_init: Optional[NDArray] = None,
    ) -> Tuple[NDArray, float, Dict[str, NDArray]]:
        """Run IK to move fingertips to target positions.

        Parameters
        ----------
        targets : dict[finger_name, (3,) target position in world frame]
            Only fingers present in both targets and the solver's
            finger map are driven; others are left unchanged.
        q_init : (nq,) array, optional
            Initial qpos.  When None, the current data.qpos is used.

        Returns
        -------
        qpos : (nq,)
            Joint positions after solve.
        residual : float
            Sum of squared position errors (metres²).
        achieved : dict[finger_name, (3,)]
            Achieved fingertip world positions.
        """
        if q_init is not None:
            self.data.qpos[:] = q_init
        mujoco.mj_forward(self.model, self.data)

        # Build active finger list for this solve
        active_idxs: List[int] = []   # index into self.finger_names
        active_targets: List[NDArray] = []
        for i, fn in enumerate(self.finger_names):
            if fn in targets:
                active_idxs.append(i)
                active_targets.append(np.asarray(targets[fn], dtype=np.float64).ravel()[:3])

        if not active_idxs:
            achieved = self._get_fingertip_positions()
            return self.data.qpos.copy(), 0.0, achieved

        n_active = len(active_idxs)
        nv = len(self._all_dof_ids)

        for iteration in range(self.max_iter):
            mujoco.mj_forward(self.model, self.data)

            # Build stacked error vector e ∈ R^{3K} and Jacobian J ∈ R^{3K x nv}
            error = np.zeros(3 * n_active)
            J_full = np.zeros((3 * n_active, self.model.nv))

            for k, fi in enumerate(active_idxs):
                sid = self.site_ids[fi]
                xpos = self.data.site_xpos[sid].copy()
                error[3 * k: 3 * k + 3] = active_targets[k] - xpos

                jacp = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, self.data, jacp, None, sid)
                J_full[3 * k: 3 * k + 3] = jacp

            # Restrict to our DOFs
            J = J_full[:, self._all_dof_ids_arr]

            # Penetration barrier 
            # When a fingertip is inside the object, REPLACE the target
            # error with a pure outward push so the solver ejects it
            # instead of the competing target-pull cancelling the push.
            if self.surface is not None:
                tip_positions = np.array([
                    self.data.site_xpos[self.site_ids[fi]].copy()
                    for fi in active_idxs
                ])
                sdf_vals = self.surface.signed_distance(tip_positions)
                for k in range(n_active):
                    pen = -sdf_vals[k]  # positive when inside
                    if pen > 1e-4:  # inside by > 0.1 mm
                        out_dir = self.surface.outward_direction(
                            tip_positions[k:k+1]
                        )[0]
                        # REPLACE error: eject to surface + margin
                        eject = out_dir * (pen + self.contact_margin) * self.penetration_weight
                        error[3 * k: 3 * k + 3] = eject

            # Check convergence
            max_err = np.max(np.linalg.norm(error.reshape(-1, 3), axis=1))
            if max_err < self.tol:
                break

            # Weighted DLS solve to penalise base motion:
            #   dq = W⁻¹ Jᵀ (J W⁻¹ Jᵀ + λ²I)⁻¹ e
            # W = diag(w) where base DOFs have w=4 and finger DOFs w=1.
            lam2 = self.damping ** 2
            J_winv = J * self._W_inv_diag[np.newaxis, :]   # J W⁻¹
            JWJt = J_winv @ J.T + lam2 * np.eye(3 * n_active)
            dq = self._W_inv_diag * (J.T @ np.linalg.solve(JWJt, error))
            dq *= self.step_size

            #  Base approach guard 
            # Prevent the base from translating TOWARD the object.
            # The DLS often tries to shrink tip-to-target error by
            # sliding the whole hand into the object.  We allow lateral
            # and outward base motion but zero out the inward component.
            if self._base_dof_ids and self.surface is not None:
                # Base translational DOFs are the first 3 of the freejoint
                base_local = [self._all_dof_ids.index(d) for d in self._base_dof_ids[:3]]
                base_trans = np.array([dq[li] for li in base_local])
                # Current base position
                base_jid = self.model.dof_jntid[self._base_dof_ids[0]]
                base_qadr = int(self.model.jnt_qposadr[base_jid])
                base_pos = self.data.qpos[base_qadr:base_qadr + 3].copy()
                # Direction from base toward object (inward)
                obj_pos = self.surface.position
                toward_obj = obj_pos - base_pos
                toward_norm = np.linalg.norm(toward_obj)
                if toward_norm > 1e-6:
                    toward_hat = toward_obj / toward_norm
                    # Component of base translation toward object
                    inward_component = np.dot(base_trans, toward_hat)
                    if inward_component > 0:
                        # Damp inward component (keep 70%);
                        # Allow moderate inward base motion so fingers
                        # can reach the surface, while preventing runaway
                        # sliding that buries the palm in the object.
                        base_trans -= inward_component * toward_hat * 0.30
                        for li, val in zip(base_local, base_trans):
                            dq[li] = val

            #  Z-floor constraint 
            # Prevent the hand base from dropping below the object's
            # bottom face.  Without this, the IK freely translates
            # the base downward (into the table) to reduce finger
            # errors, producing grasps where the palm is under the table.
            if self._base_dof_ids and self.surface is not None:
                base_jid = self.model.dof_jntid[self._base_dof_ids[0]]
                base_qadr = int(self.model.jnt_qposadr[base_jid])
                base_z = self.data.qpos[base_qadr + 2]
                # Floor = bottom of object in world frame - 15mm clearance.
                # For a box: accounting for rotation via projected half-extents
                if self.surface.kind == GeomKind.BOX:
                    z_half = float(np.sum(
                        np.abs(self.surface.rotation[2, :]) * self.surface.size[:3]
                    ))
                else:
                    z_half = float(self.surface.size[0])  # radius for sphere/cylinder
                obj_bottom_z = self.surface.position[2] - z_half
                z_floor = obj_bottom_z - 0.015
                z_local = self._all_dof_ids.index(self._base_dof_ids[2])
                if base_z + dq[z_local] < z_floor:
                    dq[z_local] = z_floor - base_z

            # Joint limits enforcement: clip dq so qpos stays in bounds
            self._apply_dq(dq)

            #  Link-body contact repulsion 
            # The DLS step drives fingertip SITES toward targets but cannot
            # prevent PROXIMAL LINK BODIES from penetrating the object.
            # We must call mj_forward FIRST to refresh kinematics/contacts
            # for the NEW qpos created by _apply_dq; using stale contacts
            # from the top of this iteration causes Jacobians to be computed
            # at the wrong configuration, making the IK diverge.
            #
            # IMPORTANT: Distal link geoms (the fingertip meshes) are
            # EXEMPTED from repulsion.  They are supposed to make contact
            # with the object: repulsing them pushes the entire hand away
            # and prevents any finger from touching the cube.
            if self._hand_geom_ids and self._obj_geom_ids:
                mujoco.mj_forward(self.model, self.data)  # ← MUST precede mj_jac
                any_link_pen = False
                for ci in range(self.data.ncon):
                    c = self.data.contact[ci]
                    g1, g2 = int(c.geom1), int(c.geom2)
                    is_hand_obj = (
                        (g1 in self._hand_geom_ids and g2 in self._obj_geom_ids) or
                        (g1 in self._obj_geom_ids and g2 in self._hand_geom_ids)
                    )
                    if not is_hand_obj or c.dist >= 0:
                        continue
                    # Skip distal finger geoms: they are supposed to contact
                    hand_g = g1 if g1 in self._hand_geom_ids else g2
                    if hand_g in self._distal_geom_ids:
                        continue
                    pen = max(-float(c.dist), 0.0)
                    if pen < 1e-3:  # ignore < 1 mm micro-contacts
                        continue
                    any_link_pen = True
                    hand_body = int(self.model.geom_bodyid[hand_g])
                    # MuJoCo contact normal: frame col-0 points from g1-->g2
                    normal = c.frame[:3].copy()
                    if g2 in self._hand_geom_ids:
                        normal = -normal  # flip: object --> hand (push out)
                    # Jacobian at contact point: kinematics are fresh
                    jacp_link = np.zeros((3, self.model.nv))
                    mujoco.mj_jac(self.model, self.data, jacp_link, None, c.pos, hand_body)
                    J_link = jacp_link[:, self._all_dof_ids_arr]
                    # Push firmly: with base DOFs unfrozen, the repulsion
                    # must be strong enough to keep links from pushing
                    # through the object.  Weight 20x and 30% step-size
                    # are aggressive enough to eject links while the
                    # weighted DLS still converges fingertips.
                    push = normal * (pen + 0.003) * 20.0
                    lam2_rep = (self.damping * 2.0) ** 2
                    JJt_rep = J_link @ J_link.T + lam2_rep * np.eye(3)
                    dq_rep = J_link.T @ np.linalg.solve(JJt_rep, push)
                    dq_rep *= self.step_size * 0.3
                    self._apply_dq(dq_rep)

        # NOTE: SDF surface-projection refinement was removed because
        # it targeted SDF=0 (tip site ON surface), which undoes the
        # overshoot compensation from the main IK loop.  With the
        # tip site 20mm past the physical fingertip, placing the site
        # on the surface means the physical finger is 20mm inside the
        # object.  The main IK targets at (overshoot + margin) above
        # the surface are correct: when converged, the physical
        # fingertip lands at ~margin (2mm) from the surface.

        mujoco.mj_forward(self.model, self.data)

        # Compute final residual and achieved positions
        residual = 0.0
        achieved = {}
        for k, fi in enumerate(active_idxs):
            sid = self.site_ids[fi]
            xpos = self.data.site_xpos[sid].copy()
            achieved[self.finger_names[fi]] = xpos
            residual += float(np.sum((active_targets[k] - xpos) ** 2))

        # Add non-active fingers to achieved
        for i, fn in enumerate(self.finger_names):
            if fn not in achieved:
                achieved[fn] = self.data.site_xpos[self.site_ids[i]].copy()

        return self.data.qpos.copy(), residual, achieved

    def _apply_dq(self, dq: NDArray) -> None:
        """Integrate dq into qpos while respecting joint limits.

        For freejoints, uses mj_integratePos which correctly
        handles the translation + quaternion representation
        (exponential map for rotation).  This replaces the old
        broken direct Euler update on quaternion elements which
        corrupted the base pose.

        dq is assumed to already be scaled by step_size (the caller
        does dq = self.step_size`` before calling this method).
        """
        # Collect freejoint updates and apply them via mj_integratePos
        # in a single call (handles all joints at once).
        dq_full = np.zeros(self.model.nv)
        has_freejoint = False

        for local_idx, dof_id in enumerate(self._all_dof_ids):
            jid = int(self.model.dof_jntid[dof_id])
            jnt_type = int(self.model.jnt_type[jid])

            if jnt_type == 0:
                # Freejoint; accumulate into dq_full for mj_integratePos
                dq_full[dof_id] = dq[local_idx]
                has_freejoint = True
            else:
                # Hinge or slide; direct update with limits
                q_adr = int(self.model.jnt_qposadr[jid])
                new_val = self.data.qpos[q_adr] + dq[local_idx]

                limited = bool(self.model.jnt_limited[jid])
                if limited:
                    lo = float(self.model.jnt_range[jid, 0])
                    hi = float(self.model.jnt_range[jid, 1])
                    new_val = np.clip(new_val, lo, hi)

                self.data.qpos[q_adr] = new_val

        if has_freejoint:
            # mj_integratePos: qpos += dt * qvel, with proper
            # quaternion exponential map for rotation DOFs.
            mujoco.mj_integratePos(self.model, self.data.qpos, dq_full, 1.0)

    def _get_fingertip_positions(self) -> Dict[str, NDArray]:
        """Return current fingertip world positions"""
        result: Dict[str, NDArray] = {}
        for i, fn in enumerate(self.finger_names):
            result[fn] = self.data.site_xpos[self.site_ids[i]].copy()
        return result


# ------
# Finger-to-contact assignment
# ------

class FingerAssigner:
    """Assign surface contact points to fingertips.

    Given N candidate surface points and K fingertips, compute a
    one-to-one assignment that minimises a cost combining Euclidean
    distance and (optionally) normal alignment with the fingertip's
    approach direction.

    The assignment is greedy-optimal: at each step the (finger, point)
    pair with lowest cost is committed.

    Reachability filter --- A finger can only be assigned to a
    surface point where the outward normal points toward the finger's
    current position (i.e.  the finger approaches the surface from the
    outside, not from inside the object).  Assignments where the normal
    faces away are masked out with infinite cost.
    """

    def __init__(
        self,
        model,
        data,
        finger_map: Optional[Dict[str, Tuple[List[str], str]]] = None,
        w_distance: float = 1.0,
        w_normal: float = 0.1,       # reduced from 0.3; less bias toward approach-aligned face
        w_opposition: float = 0.4,   # diversity bonus for opposing faces
        min_approach_cos: float = 0.0,  # 0 = only block wrong-side; raise to filter oblique
    ) -> None:
        if mujoco is None:
            raise ImportError("mujoco is required")
        self.model = model
        self.data = data
        self.w_distance = w_distance
        self.w_normal = w_normal
        self.w_opposition = w_opposition
        self.min_approach_cos = min_approach_cos

        fmap = finger_map or DEFAULT_FINGER_MAP
        self.finger_names: List[str] = []
        self.site_ids: List[int] = []
        for fname, (joint_names, site_name) in fmap.items():
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid >= 0:
                self.finger_names.append(fname)
                self.site_ids.append(sid)

    def assign(
        self,
        points_world: NDArray,
        normals_world: NDArray,
        weights: Optional[NDArray] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Tuple[NDArray, NDArray]]:
        """Assign surface points to fingertips.

        Parameters
        ----------
        points_world : (N, 3) surface candidate points in world frame.
        normals_world : (N, 3) outward normals in world frame.
        weights : (N,), optional
            Sampling weights (from energy functional).  Higher --> preferred.
        k : int, optional
            Number of fingers to assign (default: all available).

        Returns
        -------
        assignment : dict[finger_name --> (point, normal)]
            One surface point per assigned finger.
        """
        mujoco.mj_forward(self.model, self.data)

        n_fingers = min(k or len(self.finger_names), len(self.finger_names))
        n_pts = len(points_world)
        if n_pts == 0 or n_fingers == 0:
            return {}

        # Fingertip current positions and approach directions (-Z of site frame)
        tip_pos = np.zeros((len(self.finger_names), 3))
        tip_approach = np.zeros((len(self.finger_names), 3))
        for i, sid in enumerate(self.site_ids):
            tip_pos[i] = self.data.site_xpos[sid]
            # Site frame rotation matrix (3x3, row-major in xmat)
            R = self.data.site_xmat[sid].reshape(3, 3)
            # Approach direction = negative Z of site frame (into the object)
            tip_approach[i] = -R[:, 2]

        # Cost matrix: (n_fingers x N)
        # cost = w_d  distance - w_n  (normal · approach) - bonus * weight
        diffs = tip_pos[:, None, :] - points_world[None, :, :]  # (F, N, 3)
        dists = np.linalg.norm(diffs, axis=2)                    # (F, N)

        # Normal alignment: dot(outward_normal, approach_dir) --- want positive
        # (approach pointing into surface)
        alignment = np.einsum("ij,kj->ki", normals_world, tip_approach)  # (F, N)

        cost = self.w_distance * dists - self.w_normal * alignment

        # Reachability --- soft penalty instead of hard block 
        # For force closure, the thumb MUST reach the opposing face.
        # Hard-blocking wrong-side assignments prevents opposition, so
        # we use a continuous penalty:  large cost increase for
        # wrong-side approaches, mild penalty for oblique angles,
        # no penalty for face-on approaches.
        approach_dot = np.einsum("ijk,jk->ij", diffs, normals_world)  # (F, N)
        normalized_cos = approach_dot / (dists + 1e-9)               # (F, N)
        # Penalty: 0 for cos >= 0.3, ramps linearly to 2.0 for cos = -1
        reachability_penalty = np.clip(0.3 - normalized_cos, 0, None) * 1.5
        cost += reachability_penalty

        # Bias towards higher-weight points
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            # Subtract weight bonus (so lower cost = better)
            cost -= w[None, :] * 0.5

        # Greedy one-to-one assignment with opposition diversity
        assigned_fingers: List[int] = []
        assigned_points: List[int] = []
        assigned_normals: List[NDArray] = []  # normals of assigned points
        cost_work = cost.copy()

        for step in range(n_fingers):
            # Mask already assigned
            if assigned_fingers:
                cost_work[assigned_fingers, :] = np.inf
                cost_work[:, assigned_points] = np.inf

            # Opposition diversity bonus 
            # After the first assignment, give a cost bonus (reduction)
            # to surface points whose normal OPPOSES already-assigned
            # normals.  This encourages wrapping: thumb on one face,
            # fingers on the opposing face.
            if step > 0 and self.w_opposition > 0:
                for prev_nrm in assigned_normals:
                    # dot(candidate_normal, previous_normal) < 0 --> opposing
                    opp = normals_world @ prev_nrm  # (N,)
                    # Bonus proportional to opposition (-1 = max opposition)
                    # bonus = w_opp * max(0, -dot)
                    bonus = self.w_opposition * np.maximum(-opp, 0.0)
                    cost_work -= bonus[None, :]  # apply to all remaining fingers

            min_cost = np.min(cost_work)
            if not np.isfinite(min_cost):
                break  # No reachable assignments left

            fi, pi = np.unravel_index(np.argmin(cost_work), cost_work.shape)
            assigned_fingers.append(int(fi))
            assigned_points.append(int(pi))
            assigned_normals.append(normals_world[int(pi)].copy())

        result: Dict[str, Tuple[NDArray, NDArray]] = {}
        for fi, pi in zip(assigned_fingers, assigned_points):
            result[self.finger_names[fi]] = (
                points_world[pi].copy(),
                normals_world[pi].copy(),
            )
        return result


# ------
# Grasp sampler (main entry point)
# ------

@dataclass
class SamplerConfig:
    """Configuration for the grasp sampler.

    Attributes
    ----------
    n_surface_points : int
        Number of candidate points sampled on the object surface per trial.
    n_random_trials : int
        Number of random-seed IK trials (in addition to warm-start seeds).
    n_fingers : int or None
        How many fingers to engage (None = all 5).
    ik_max_iter : int
        Max DLS iterations per IK solve.
    ik_tol : float
        IK convergence tolerance in metres.
    ik_damping : float
        DLS damping factor.
    ik_step_size : float
        IK integration step size.
    friction_coef : float
        Assumed friction coefficient for GWS analysis.
    top_k : int
        Return the top-k grasps by quality.
    base_joint : str or None
        Name of the floating base joint (None = hand_free).
    resample_surface : bool
        If True, draw fresh surface points for each trial.  If False,
        sample once and reuse (faster, but less diverse).
    penetration_weight : float
        Strength of the IK penetration barrier.  Higher values make the
        outward push stronger when a fingertip enters the object.
    contact_margin : float
        Standoff distance (metres) added to IK targets along the surface
        normal so that fingertips approach from outside.
    max_penetration : float
        Maximum allowable penetration depth (metres) in the final grasp.
        Grasps with deeper penetration have their epsilon zeroed.
    contact_tolerance : float
        Maximum distance (metres) from the surface for a fingertip to be
        considered a valid contact.  Fingers farther away are excluded
        from GWS evaluation.
    min_finger_separation : float
        Minimum pairwise distance (metres) between any two fingertips.
        Grasps violating this are rejected (finger-finger collision).
    min_valid_contacts : int
        Minimum number of validated surface contacts for a grasp to be
        scored.  Grasps with fewer get epsilon = 0.
    """

    n_surface_points: int = 500
    n_random_trials: int = 60
    n_fingers: Optional[int] = None
    ik_max_iter: int = 800          # more iters for better convergence
    ik_tol: float = 2e-3
    ik_damping: float = 1e-2
    ik_step_size: float = 0.30      # slightly larger step for faster convergence
    friction_coef: float = 0.8
    top_k: int = 10
    base_joint: Optional[str] = None
    resample_surface: bool = True
    penetration_weight: float = 0.0    # disabled: distal contacts are expected; link repulsion handles proximal
    contact_margin: float = 0.002     # 2 mm standoff: tip sites target 2mm above surface
    max_penetration: float = 0.015    # 15mm: tip site penetration limit
    max_mesh_penetration: float = 0.020  # 20 mm: proximal-only mesh collision limit (distal exempt)
    contact_tolerance: float = 0.012  # 12 mm: SDF-based fallback for marginal contacts
    min_finger_separation: float = 0.005
    min_valid_contacts: int = 2       # lowered: hard with 2-DOF fingers
    sdf_refine_iters: int = 30        # SDF projection refinement iterations
    sdf_refine_tol: float = 0.0005    # 0.5 mm convergence threshold
    min_approach_cos: float = 0.0     # 0 = only block wrong-side (approach filter in assigner)


class GraspSampler:
    """Sampling-based multi-finger grasp solver.

    Typical usage::

        import mujoco
        model = mujoco.MjModel.from_xml_path("scene.xml")
        data  = mujoco.MjData(model)

        surface = ObjectSurface.from_model(model, geom_name="cube")
        sampler = GraspSampler(model, data, surface)

        # Optionally load GraspIt! seeds
        from vnb_grasp.grasping import load_grasps
        db = load_grasps("cube")
        sampler.add_warm_starts(db)

        grasps = sampler.solve()
        best = grasps[0]
        print(f"Best epsilon quality: {best.gws.epsilon:.4f}")
    """

    def __init__(
        self,
        model,
        data,
        surface: ObjectSurface,
        *,
        config: Optional[SamplerConfig] = None,
        finger_map: Optional[Dict[str, Tuple[List[str], str]]] = None,
        object_body_name: Optional[str] = None,
        energy_fn: Optional[Callable[[NDArray, NDArray], NDArray]] = None,
    ) -> None:
        """
        Parameters
        ----------
        model, data : mujoco.MjModel, mujoco.MjData
        surface : ObjectSurface
            Surface representation for the target object.
        config : SamplerConfig, optional
        finger_map : dict, optional
            Override the default finger --> (joints, site) mapping.
        object_body_name : str, optional
            Name of the MuJoCo body for the object (used to read its current
            pose for GWS evaluation).  When None the surface's stored
            position is used as the object center.
        energy_fn : callable, optional
            Energy functional for biased surface sampling.
        """
        if mujoco is None:
            raise ImportError("mujoco is required")

        self.model = model
        self.data = data
        self.surface = surface
        self.cfg = config or SamplerConfig()
        self.energy_fn = energy_fn
        self._finger_map = finger_map or DEFAULT_FINGER_MAP
        self._rng = np.random.default_rng()

        # Object body for pose lookup
        self._obj_body_id: Optional[int] = None
        if object_body_name is not None:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)
            if bid >= 0:
                self._obj_body_id = bid

        # Object / hand geom IDs for collision detection: subtree-based
        # (robust: catches all geoms in body subtree, not just name-matched)
        if self._obj_body_id is not None:
            self._obj_geom_ids = _collect_geoms_in_subtree(model, self._obj_body_id)
        else:
            self._obj_geom_ids: set = set()

        # Find hand root body ID (the body carrying the freejoint)
        base_jnt_name = self.cfg.base_joint or DEFAULT_BASE_JOINT
        _bjid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, base_jnt_name)
        if _bjid >= 0:
            self._hand_root_body_id = int(model.jnt_bodyid[_bjid])
        else:
            # Fallback: try body named "hand"
            self._hand_root_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "hand"
            )
        if self._hand_root_body_id >= 0:
            self._hand_geom_ids = _collect_geoms_in_subtree(model, self._hand_root_body_id)
        else:
            self._hand_geom_ids: set = set()

        # Sub-components
        self._ik = ContactIKSolver(
            model, data,
            finger_map=self._finger_map,
            base_joint=self.cfg.base_joint,
            damping=self.cfg.ik_damping,
            max_iter=self.cfg.ik_max_iter,
            tol=self.cfg.ik_tol,
            step_size=self.cfg.ik_step_size,
            surface=surface,
            penetration_weight=self.cfg.penetration_weight,
            contact_margin=self.cfg.contact_margin,
            sdf_refine_iters=self.cfg.sdf_refine_iters,
            sdf_refine_tol=self.cfg.sdf_refine_tol,
        )
        # Provide collision geom sets so IK uses contact-based repulsion
        self._ik.set_collision_geoms(self._hand_geom_ids, self._obj_geom_ids)

        #  Per-finger tip-site overshoot 
        # The tip_site sits at pos="0 0 0.04" in the distal body frame,
        # but the distal collision mesh only extends ~20 mm along +Z.
        # The tip site therefore floats ~20 mm PAST the physical fingertip.
        # IK targets must be offset by this amount so the actual finger
        # pad (not the phantom site) lands on the object surface.
        self._tip_overshoot: Dict[str, float] = {}  # finger_name --> metres
        self._compute_tip_overshoot()

        self._assigner = FingerAssigner(
            model, data,
            finger_map=self._finger_map,
            min_approach_cos=self.cfg.min_approach_cos,
        )

        # Warm start seeds: list of (qpos, source_label) tuples
        self._warm_starts: List[Tuple[NDArray, str]] = []

        # Auto-generate canonical approach seeds
        self._add_canonical_approach_seeds()

    def _compute_tip_overshoot(self) -> None:
        """Measure how far each tip_site extends past the distal collision mesh.

        For each finger, reads the distal collision geom's mesh vertices,
        finds the maximum Z coordinate in the body frame (accounting
        for the geom position offset that MuJoCo applies when it centres
        mesh vertices around their centroid), and compares it to the
        tip_site's Z offset.  The difference is the "overshoot": the
        distance by which the IK target must be pushed outward along
        the surface normal so the physical finger pad lands on the surface.

        IMPORTANT: model.mesh_vert stores vertices relative to the
        mesh centroid, not the body origin.  model.geom_pos gives the
        centroid position in the body frame.  The true vertex Z in the
        body frame is mesh_vert_z + geom_pos_z.
        """
        fmap = self._finger_map
        for fname, (joint_names, site_name) in fmap.items():
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid < 0:
                continue
            # Site local Z offset in its parent body
            site_z = float(self.model.site_pos[sid, 2])

            # Find the collision mesh geom on the same body
            body_id = int(self.model.site_bodyid[sid])
            mesh_max_z_body = None
            for gi in range(self.model.ngeom):
                if int(self.model.geom_bodyid[gi]) != body_id:
                    continue
                gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gi)
                if gname is None or 'collision' not in gname:
                    continue
                if int(self.model.geom_type[gi]) != 7:  # 7 = mesh
                    continue
                mesh_id = int(self.model.geom_dataid[gi])
                vert_adr = int(self.model.mesh_vertadr[mesh_id])
                vert_num = int(self.model.mesh_vertnum[mesh_id])
                verts = self.model.mesh_vert[vert_adr:vert_adr + vert_num]
                # mesh_vert is centroid-relative; add geom_pos to get body frame
                geom_pos_z = float(self.model.geom_pos[gi, 2])
                mesh_max_z_body = float(verts[:, 2].max()) + geom_pos_z
                break

            if mesh_max_z_body is not None and site_z > mesh_max_z_body:
                overshoot = site_z - mesh_max_z_body
            else:
                overshoot = 0.0

            self._tip_overshoot[fname] = overshoot
            logger.info("Finger '%s' tip overshoot: %.1f mm "
                        "(site=%.1f mm, mesh_top_body=%.1f mm)",
                        fname, overshoot * 1000, site_z * 1000,
                        (mesh_max_z_body or 0) * 1000)

    def _add_canonical_approach_seeds(self) -> None:
        r"""Generate approach seeds from canonical directions.

        Creates initial hand poses approaching the object from lateral
        cardinal directions (+-X, +-Y) plus diagonal and tilted approaches,
        with diverse finger pre-curl settings.  These give the IK solver a
        much better starting point than random orientations.
        """
        obj_center = self._get_object_center()
        base_jnt = self.cfg.base_joint or DEFAULT_BASE_JOINT
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, base_jnt)
        if jid < 0 or int(self.model.jnt_type[jid]) != 0:
            return

        q_adr = int(self.model.jnt_qposadr[jid])

        # Palm approach directions: cardinal (+X, -X, +Y, -Y, +Z, -Z) + diagonals
        # NOTE: +Z (top-down) is excluded because the hand approaching from above
        # always wraps finger links through the cube top face, causing large mesh
        # penetration. Side approaches allow fingers to wrap around the cube
        # without the proximal links piercing adjacent faces.
        # Only lateral + diagonal approaches are included (no pure top-down).
        approach_dirs = [
            np.array([+1, 0, 0]),  # +X
            np.array([-1, 0, 0]),  # -X
            np.array([0, +1, 0]),  # +Y
            np.array([0, -1, 0]),  # -Y
            np.array([+1, +1, 0]),  # XY diagonals (lateral only)
            np.array([+1, -1, 0]),
            np.array([-1, +1, 0]),
            np.array([-1, -1, 0]),
            np.array([+1, 0, +1]),  # XZ tilted
            np.array([-1, 0, +1]),
            np.array([0, +1, +1]),  # YZ tilted
            np.array([0, -1, +1]),
        ]

        # Stand-off: far enough that palm/proximal start outside the
        # object, but close enough that the IK can approach the surface.
        # 60 mm from centre ≈ 35 mm from a 50 mm cube face.
        standoff = 0.06  # 60 mm from object center

        for d in approach_dirs:
            d_norm = d / np.linalg.norm(d)
            q = self.data.qpos.copy()

            # Position
            q[q_adr: q_adr + 3] = obj_center + d_norm * standoff

            # Orient palm toward object
            z_axis = d_norm  # hand Z away from object
            up = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x_axis = np.cross(up, z_axis)
            xn = np.linalg.norm(x_axis)
            if xn < 1e-6:
                x_axis = np.array([1.0, 0.0, 0.0])
            else:
                x_axis /= xn
            y_axis = np.cross(z_axis, x_axis)
            R = np.column_stack([x_axis, y_axis, z_axis])

            # R to quat
            tr = R[0, 0] + R[1, 1] + R[2, 2]
            if tr > 0:
                s_ = np.sqrt(tr + 1.0) * 2
                qw, qx, qy, qz = 0.25*s_, (R[2,1]-R[1,2])/s_, (R[0,2]-R[2,0])/s_, (R[1,0]-R[0,1])/s_
            elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s_ = np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2]) * 2
                qw, qx, qy, qz = (R[2,1]-R[1,2])/s_, 0.25*s_, (R[0,1]+R[1,0])/s_, (R[0,2]+R[2,0])/s_
            elif R[1,1] > R[2,2]:
                s_ = np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2]) * 2
                qw, qx, qy, qz = (R[0,2]-R[2,0])/s_, (R[0,1]+R[1,0])/s_, 0.25*s_, (R[1,2]+R[2,1])/s_
            else:
                s_ = np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1]) * 2
                qw, qx, qy, qz = (R[1,0]-R[0,1])/s_, (R[0,2]+R[2,0])/s_, (R[1,2]+R[2,1])/s_, 0.25*s_
            quat = np.array([qw, qx, qy, qz])
            quat /= np.linalg.norm(quat)
            q[q_adr + 3: q_adr + 7] = quat

            # Set finger joints to mid-range (partially curled)
            for dofs in self._ik.finger_dof_slices:
                for dof_id in dofs:
                    djid = int(self.model.dof_jntid[dof_id])
                    jtype = int(self.model.jnt_type[djid])
                    if jtype in (2, 3) and self.model.jnt_limited[djid]:
                        qa = int(self.model.jnt_qposadr[djid])
                        lo = float(self.model.jnt_range[djid, 0])
                        hi = float(self.model.jnt_range[djid, 1])
                        q[qa] = (lo + hi) * 0.45  # 45% curl: natural pre-grasp

            # Add two curl variants for each approach direction:
            # 1) 45% pre-curl (default, already set above)
            self._warm_starts.append((q.copy(), "canonical"))

            # 2) 60% pre-curl: more aggressive wrap for power grasp
            q2 = q.copy()
            for dofs in self._ik.finger_dof_slices:
                for dof_id in dofs:
                    djid = int(self.model.dof_jntid[dof_id])
                    jtype = int(self.model.jnt_type[djid])
                    if jtype in (2, 3) and self.model.jnt_limited[djid]:
                        qa = int(self.model.jnt_qposadr[djid])
                        lo = float(self.model.jnt_range[djid, 0])
                        hi = float(self.model.jnt_range[djid, 1])
                        q2[qa] = (lo + hi) * 0.60
            self._warm_starts.append((q2.copy(), "canonical"))

        #  Opposition warm-starts 
        # The canonical seeds orient the hand's body-Z away from the
        # object (z_axis = d_norm).  In that orientation the fingers
        # extend along +Z (away from the object) and curl toward +X,
        # which is perpendicular to the approach: all contacts land on
        # the SAME side, preventing force closure.
        #
        # These "opposition" seeds instead orient the hand so that the
        # body +X axis points TOWARD the object.  Finger curl (+Z --> +X)
        # then sweeps TOWARD the cube, and the thumb's cmc_yaw allows
        # it to wrap the opposing face, producing true opposition.
        # Cardinal + diagonal approach directions for opposition grasps
        _s2 = 1.0 / np.sqrt(2.0)
        opposition_specs = [
            # (approach_dir, standoff, curl_fraction)
            # Cardinal directions at two standoffs and curl levels.
            # Moderate standoffs with low BASE_TRANS_REG let IK approach.
            (np.array([+1, 0, 0]), 0.055, 0.45),
            (np.array([+1, 0, 0]), 0.070, 0.60),
            (np.array([-1, 0, 0]), 0.055, 0.45),
            (np.array([-1, 0, 0]), 0.070, 0.60),
            (np.array([0, +1, 0]), 0.055, 0.45),
            (np.array([0, +1, 0]), 0.070, 0.60),
            (np.array([0, -1, 0]), 0.055, 0.45),
            (np.array([0, -1, 0]), 0.070, 0.60),
            # Top / bottom approaches
            (np.array([0, 0, +1]), 0.060, 0.50),
            (np.array([0, 0, -1]), 0.060, 0.50),
            # Diagonal approaches in XY plane
            (np.array([_s2, _s2, 0]), 0.060, 0.50),
            (np.array([_s2, -_s2, 0]), 0.060, 0.50),
            (np.array([-_s2, _s2, 0]), 0.060, 0.50),
            (np.array([-_s2, -_s2, 0]), 0.060, 0.50),
        ]

        for d_raw, opp_standoff, curl_frac in opposition_specs:
            d_norm = d_raw / np.linalg.norm(d_raw)
            q = self.data.qpos.copy()

            # Position hand
            q[q_adr: q_adr + 3] = obj_center + d_norm * opp_standoff

            # Orient so body X points TOWARD the object (-d_norm)
            x_body = -d_norm  # body +X toward cube
            # Choose body Z perpendicular to x_body (fingers extend this way)
            if abs(x_body[2]) < 0.9:
                z_body = np.array([0.0, 0.0, 1.0])
            else:
                z_body = np.array([0.0, 1.0, 0.0])
            # Gram-Schmidt: make z_body perpendicular to x_body
            z_body = z_body - np.dot(z_body, x_body) * x_body
            z_body /= np.linalg.norm(z_body)
            y_body = np.cross(z_body, x_body)
            R = np.column_stack([x_body, y_body, z_body])

            # R to quat
            tr = R[0, 0] + R[1, 1] + R[2, 2]
            if tr > 0:
                s_ = np.sqrt(tr + 1.0) * 2
                qw, qx, qy, qz = 0.25 * s_, (R[2, 1] - R[1, 2]) / s_, (R[0, 2] - R[2, 0]) / s_, (R[1, 0] - R[0, 1]) / s_
            elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s_ = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw, qx, qy, qz = (R[2, 1] - R[1, 2]) / s_, 0.25 * s_, (R[0, 1] + R[1, 0]) / s_, (R[0, 2] + R[2, 0]) / s_
            elif R[1, 1] > R[2, 2]:
                s_ = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw, qx, qy, qz = (R[0, 2] - R[2, 0]) / s_, (R[0, 1] + R[1, 0]) / s_, 0.25 * s_, (R[1, 2] + R[2, 1]) / s_
            else:
                s_ = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw, qx, qy, qz = (R[1, 0] - R[0, 1]) / s_, (R[0, 2] + R[2, 0]) / s_, (R[1, 2] + R[2, 1]) / s_, 0.25 * s_
            quat = np.array([qw, qx, qy, qz])
            quat /= np.linalg.norm(quat)
            q[q_adr + 3: q_adr + 7] = quat

            # Set fingers to specified curl fraction
            for dofs in self._ik.finger_dof_slices:
                for dof_id in dofs:
                    djid = int(self.model.dof_jntid[dof_id])
                    jtype = int(self.model.jnt_type[djid])
                    if jtype in (2, 3) and self.model.jnt_limited[djid]:
                        qa = int(self.model.jnt_qposadr[djid])
                        lo = float(self.model.jnt_range[djid, 0])
                        hi = float(self.model.jnt_range[djid, 1])
                        q[qa] = (lo + hi) * curl_frac

            self._warm_starts.append((q.copy(), "opposition"))

    # 
    # Warm-start interface
    # 

    def add_warm_starts(
        self,
        grasp_db,     # GraspDatabase from graspit_loader
        *,
        top_k: int = 10,
        object_position: Optional[NDArray] = None,
        object_orientation: Optional[NDArray] = None,
    ) -> int:
        """Seed the solver from a GraspIt! database.

        The best top_k grasps by epsilon quality are converted from
        GraspIt! DOF ordering into qpos vectors suitable for the
        MuJoCo model.

        Parameters
        ----------
        grasp_db : GraspDatabase
        top_k : int
        object_position : (3,) optional --- current object world position
        object_orientation : (4,) optional --- current object quaternion [w,x,y,z]

        Returns
        -------
        int --- number of seeds added.
        """
        from .graspit_loader import REALHAND_L6_DOF_NAMES

        seeds_added = 0
        for grasp in grasp_db.top_k_grasps(top_k):
            try:
                q_init = self._graspit_to_qpos(grasp)
                self._warm_starts.append((q_init, "graspit"))
                seeds_added += 1
            except Exception as exc:
                logger.debug("Skipped GraspIt! grasp %d: %s", grasp.grasp_id, exc)

        logger.info("Added %d GraspIt! warm-start seeds", seeds_added)
        return seeds_added

    def add_qpos_seeds(
        self,
        seeds: Sequence[NDArray],
        label: str = "custom",
    ) -> None:
        """Add arbitrary qpos vectors as warm-start seeds"""
        for q in seeds:
            self._warm_starts.append((np.asarray(q, dtype=np.float64).copy(), label))

    # 
    # Main solve
    # 

    def solve(
        self,
        *,
        n_trials: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> List[SampledGrasp]:
        """Run the sampling-based solver and return ranked grasps.

        Parameters
        ----------
        n_trials : int, optional
            Override config.n_random_trials.
        top_k : int, optional
            Override config.top_k.

        Returns
        -------
        list[SampledGrasp]
            Grasps sorted by descending GWS epsilon quality (best first).
        """
        n_rand = n_trials if n_trials is not None else self.cfg.n_random_trials
        k = top_k if top_k is not None else self.cfg.top_k

        # --- shared surface sample (if not resampling per trial) ---
        shared_sample: Optional[SurfaceSample] = None
        if not self.cfg.resample_surface:
            shared_sample = self.surface.sample(
                self.cfg.n_surface_points,
                rng=self._rng,
                energy_fn=self.energy_fn,
            )

        results: List[SampledGrasp] = []

        # --- warm-start trials ---
        for q_seed, source in self._warm_starts:
            sample = shared_sample or self.surface.sample(
                self.cfg.n_surface_points,
                rng=self._rng,
                energy_fn=self.energy_fn,
            )
            g = self._single_trial(sample, q_init=q_seed, source=source)
            results.append(g)

        # --- random trials ---
        for _ in range(n_rand):
            sample = shared_sample or self.surface.sample(
                self.cfg.n_surface_points,
                rng=self._rng,
                energy_fn=self.energy_fn,
            )
            q_rand = self._random_qpos()
            g = self._single_trial(sample, q_init=q_rand, source="random")
            results.append(g)

        # --- rank by epsilon quality ---
        results.sort(
            key=lambda g: (g.gws.epsilon, -g.residual),
            reverse=True,
        )
        return results[:k]

    # 
    # Internal helpers
    # 

    def _single_trial(
        self,
        surface_sample: SurfaceSample,
        *,
        q_init: NDArray,
        source: str,
    ) -> SampledGrasp:
        """Run one trial: assign --> IK --> validate --> GWS.

        Post-IK validation
        ------------------
        1. Penetration: reject grasps where any fingertip is deeper
           than max_penetration inside the object.
        2. Contact validity: only fingertips within contact_tolerance
           of the surface count as real contacts for GWS.  Others are
           hovering or missed.
        3. Finger separation: every pair of contact fingertips must
           be at least min_finger_separation apart to rule out
           finger-to-finger occlusion.
        4. Minimum contacts: at least min_valid_contacts valid
           surface contacts are required for a non-zero epsilon.
        """
        _bad = SampledGrasp(hand_qpos=q_init.copy(), seed_source=source)

        # Update object surface pose from simulation
        self._update_surface_pose()

        # Transform sampled points to world frame
        pts_world = self.surface.to_world(surface_sample.points)
        nrm_world = self.surface.normal_to_world(surface_sample.normals)

        # Set initial qpos and forward to get fingertip positions
        self.data.qpos[:] = q_init
        mujoco.mj_forward(self.model, self.data)

        # Assign contact targets to fingers
        assignment = self._assigner.assign(
            pts_world, nrm_world,
            weights=surface_sample.weights,
            k=self.cfg.n_fingers,
        )

        if not assignment:
            return _bad

        # Target positions: surface point + (overshoot + margin) along normal.
        # The tip_site may extend past the physical fingertip mesh.
        # Overshoot compensation ensures the IK drives the physical
        # finger pad to the surface rather than the phantom site.
        # After the XML fix (tip_site moved to mesh boundary) overshoot
        # is ~0 and this reduces to  pt + nrm * margin : harmless but
        # still correct for any future hand model with non-zero overshoot.
        margin = self.cfg.contact_margin
        targets = {}
        for fn, (pt, nrm) in assignment.items():
            overshoot = self._tip_overshoot.get(fn, 0.0)
            targets[fn] = pt + nrm * (overshoot + margin)

        # Solve IK (barrier inside the loop ejects penetrating tips)
        qpos, residual, achieved = self._ik.solve(targets, q_init=q_init)

        # 
        #  POST-IK VALIDATION
        # 
        finger_names = list(achieved.keys())
        tip_pts = np.array([achieved[fn] for fn in finger_names])
        sdf_vals = self.surface.signed_distance(tip_pts)  # neg = inside

        # (1) Tip site penetration check.
        # With no overshoot compensation, tip sites should be near the
        # surface (SDF ≈ margin ≈ 2mm).  A tip more than max_penetration
        # INSIDE the surface means the IK overshot badly.
        max_pen = float(np.max(-sdf_vals))  # positive = inside
        max_pen = max(max_pen, 0.0)

        if max_pen > self.cfg.max_penetration:
            logger.debug("Trial (%s) REJECTED: tip SDF penetration %.1f mm",
                         source, max_pen * 1000)
            _bad.max_penetration = max_pen
            return _bad

        # (1b) Mesh collision check for NON-DISTAL links only.
        # Distal links are expected to make contact; their penetration
        # is acceptable.  Proximal/palm/base links should NOT penetrate.
        mesh_pen = self._check_mesh_collision(skip_distal=True)
        if mesh_pen > self.cfg.max_mesh_penetration:
            logger.debug("Trial (%s) REJECTED: proximal mesh collision %.1f mm",
                         source, mesh_pen * 1000)
            _bad.max_penetration = mesh_pen
            return _bad
        max_pen = max(max_pen, mesh_pen)

        # (2) Contact validity using MuJoCo contacts.
        # Check which finger distal links actually touch the cube via
        # MuJoCo's collision detection.  This is the ground truth for
        # "finger touches the object": no SDF approximation needed.
        mujoco.mj_forward(self.model, self.data)
        fingers_in_contact = self._detect_finger_contacts()

        # Also accept fingers whose tip site is within tolerance of the
        # surface (SDF-based fallback for marginal contacts).
        for i, fn in enumerate(finger_names):
            if abs(sdf_vals[i]) <= self.cfg.contact_tolerance:
                fingers_in_contact.add(fn)

        valid_fingers = [fn for fn in finger_names if fn in fingers_in_contact]
        n_valid = len(valid_fingers)

        if n_valid < self.cfg.min_valid_contacts:
            logger.debug("Trial (%s) REJECTED: only %d/%d contacts on surface",
                         source, n_valid, len(finger_names))
            _bad.max_penetration = max_pen
            return _bad

        # (3) Finger-finger separation; no two contact tips too close
        is_contact_mask = np.array([fn in fingers_in_contact for fn in finger_names])
        contact_pts = tip_pts[is_contact_mask]
        if len(contact_pts) >= 2:
            from itertools import combinations
            for i, j in combinations(range(len(contact_pts)), 2):
                sep = float(np.linalg.norm(contact_pts[i] - contact_pts[j]))
                if sep < self.cfg.min_finger_separation:
                    logger.debug(
                        "Trial (%s) REJECTED: fingers %.0fmm apart (min %dmm)",
                        source, sep * 1000,
                        int(self.cfg.min_finger_separation * 1000),
                    )
                    _bad.max_penetration = max_pen
                    return _bad

        # (4) GWS using ONLY validated contacts
        valid_positions = {fn: achieved[fn] for fn in valid_fingers}
        gws_result = self._evaluate_gws(valid_positions)

        # Log successful trial details
        sdf_summary = ", ".join(
            f"{fn}:sdf={sdf_vals[i]*1000:.1f}mm"
            for i, fn in enumerate(finger_names) if fn in fingers_in_contact
        )
        logger.debug(
            "Trial (%s) ACCEPTED: eps=%.4f, %d contacts, pen=%.1fmm, sdf=[%s]",
            source, gws_result.epsilon, n_valid, max_pen * 1000, sdf_summary,
        )

        # Build per-finger qpos dict
        finger_qpos = self._extract_finger_qpos(qpos)

        target_contacts = {fn: pt.copy() for fn, (pt, _) in assignment.items()}
        target_normals = {fn: nrm.copy() for fn, (_, nrm) in assignment.items()}

        return SampledGrasp(
            hand_qpos=qpos.copy(),
            finger_qpos=finger_qpos,
            fingertip_positions=achieved,
            target_contacts=target_contacts,
            target_normals=target_normals,
            residual=residual,
            max_penetration=max_pen,
            gws=gws_result,
            seed_source=source,
        )

    def _update_surface_pose(self) -> None:
        """Sync object surface pose from simulation state"""
        if self._obj_body_id is not None:
            self.surface.position = self.data.xpos[self._obj_body_id].copy()
            self.surface.rotation = self.data.xmat[self._obj_body_id].reshape(3, 3).copy()

    def _check_mesh_collision(self, skip_distal: bool = False) -> float:
        """Check actual geometry interpenetration via MuJoCo contacts.

        When skip_distal is True, distal finger link geoms are excluded
        (they are expected to make contact with the object).

        Returns the worst penetration depth in metres (positive = deeper).
        """
        mujoco.mj_forward(self.model, self.data)
        worst_pen = 0.0
        distal_geoms = self._ik._distal_geom_ids if skip_distal else set()
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            is_hand_obj = (
                (g1 in self._hand_geom_ids and g2 in self._obj_geom_ids) or
                (g1 in self._obj_geom_ids and g2 in self._hand_geom_ids)
            )
            if not is_hand_obj or c.dist >= 0:
                continue
            # Skip distal finger geoms if requested
            if skip_distal:
                hand_g = g1 if g1 in self._hand_geom_ids else g2
                if hand_g in distal_geoms:
                    continue
            pen = -c.dist
            worst_pen = max(worst_pen, pen)
        return worst_pen

    def _detect_finger_contacts(self) -> set:
        """Detect which fingers have actual MuJoCo contact with the object.

        Returns a set of finger names whose distal link geoms overlap
        with the object geoms (contact.dist < 0).
        """
        result: set = set()
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            is_hand_obj = (
                (g1 in self._hand_geom_ids and g2 in self._obj_geom_ids) or
                (g1 in self._obj_geom_ids and g2 in self._hand_geom_ids)
            )
            if not is_hand_obj:
                continue
            hand_g = g1 if g1 in self._hand_geom_ids else g2
            hand_body = int(self.model.geom_bodyid[hand_g])
            # Map body to finger name
            gname = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, hand_g
            ) or ""
            for fn in self._ik.finger_names:
                if fn in gname:
                    result.add(fn)
                    break
        return result

    def _evaluate_gws(self, fingertip_positions: Dict[str, NDArray]) -> GWSResult:
        """Build ContactInfo from validated fingertip positions and run GWS.

        Tip site positions are projected to the nearest surface point via
        iterative SDF descent so the GWS contacts lie on the actual
        object boundary.
        """
        obj_center = self._get_object_center()

        # Use the surface SDF to get true outward normals at each contact
        contacts: List[ContactInfo] = []
        for fn, tip_pos in fingertip_positions.items():
            # Project tip site position to the object surface using
            # iterative SDF descent.  The tip site sits ~overshoot mm
            # above the surface; we step along the SDF gradient until
            # SDF ≈ 0 (on the surface).  This is more robust than a
            # single overshoot-magnitude step, especially near cube
            # edges/corners where the gradient direction changes.
            pos = tip_pos.copy()
            for _proj_it in range(8):
                sdf_cur = float(self.surface.signed_distance(pos.reshape(1, 3))[0])
                if abs(sdf_cur) < 0.001:  # within 1mm of surface
                    break
                out_d = self.surface.outward_direction(pos.reshape(1, 3))[0]
                pos = pos - out_d * sdf_cur  # step exactly to surface

            out_dir = self.surface.outward_direction(pos.reshape(1, 3))[0]

            # Contact normal points inward (toward object interior)
            normal = -out_dir
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                # Fallback: point toward center
                normal = obj_center - pos
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    normal = normal / norm
                else:
                    normal = np.array([0.0, 0.0, 1.0])

            # Build a 3x3 contact frame (row 0 = normal, rows 1-2 = tangents)
            if abs(normal[2]) < 0.9:
                t1 = np.cross(normal, np.array([0.0, 0.0, 1.0]))
            else:
                t1 = np.cross(normal, np.array([1.0, 0.0, 0.0]))
            t1 /= np.linalg.norm(t1)
            t2 = np.cross(normal, t1)
            frame = np.stack([normal, t1, t2], axis=0)  # (3, 3)

            # Distance-weighted force (GraspIt!-inspired virtual contacts)
            # Cosine decay: w = (cos(π·d/d_max) + 1) / 2
            # = 1.0 at surface, 0.5 at d_max/2, 0.0 at d_max
            d_max = 0.050  # 50 mm max influence range
            sdf_val = abs(float(self.surface.signed_distance(pos.reshape(1, 3))[0]))
            w_contact = max(0.0, (np.cos(np.pi * min(sdf_val, d_max) / d_max) + 1.0) / 2.0)

            contacts.append(ContactInfo(
                pos=pos.copy(),
                frame=frame,
                dist=0.0,
                force=np.array([w_contact, 0.0, 0.0]),  # distance-weighted normal force
                geom1=-1,
                geom2=-1,
            ))

        if len(contacts) < 2:
            return GWSResult(
                epsilon=0.0, volume=0.0, min_singular=0.0,
                is_force_closure=False, n_contacts=len(contacts),
            )

        return analyze_gws(contacts, obj_center, self.cfg.friction_coef)

    def _get_object_center(self) -> NDArray:
        """Get object center position"""
        if self._obj_body_id is not None:
            return self.data.xpos[self._obj_body_id].copy()
        return self.surface.position.copy()

    def _random_qpos(self) -> NDArray:
        """Generate a random initial qpos within joint limits.

        The base (freejoint) is placed at a random approach direction
        around the object (not just from above) so that the thumb can
        oppose the other four fingers from diverse angles.  Approach
        direction is sampled uniformly on the upper hemisphere.
        """
        q = self.data.qpos.copy()

        for dof_id in self._ik._all_dof_ids:
            jid = int(self.model.dof_jntid[dof_id])
            jnt_type = int(self.model.jnt_type[jid])

            if jnt_type == 0:
                obj_center = self._get_object_center()
                q_adr = int(self.model.jnt_qposadr[jid])

                # --- Sample approach direction on the sphere ---
                # Hand must start OUTSIDE the object so IK can curl inward.
                # 40-70 mm from centre gives ~15-45 mm clearance; low
                # BASE_TRANS_REG lets the IK close the gap as needed.
                approach_r = self._rng.uniform(0.04, 0.070)  # 40-70 mm
                # spherical coords: θ ∈ [0, π/2] (upper hemi), φ ∈ [0, 2π)
                theta = self._rng.uniform(0, np.pi * 0.5)  # polar (0=above)
                phi = self._rng.uniform(0, 2 * np.pi)      # azimuthal

                # Bias toward side approaches (theta near π/2)
                # Use sin weighting: sample u uniform then theta = arccos(u)
                u = self._rng.uniform(0.0, 1.0)
                theta = np.arccos(u)  # concentrates near pi/2 (horizon)

                # Direction FROM object TO hand
                dx = np.sin(theta) * np.cos(phi)
                dy = np.sin(theta) * np.sin(phi)
                dz = np.cos(theta)
                approach_dir = np.array([dx, dy, dz])

                # Position: object center + approach offset
                hand_pos = obj_center + approach_dir * approach_r
                q[q_adr: q_adr + 3] = hand_pos

                # --- Orientation: palm faces the object ---
                # We want the hand's -Z axis (palm normal) to point toward
                # the object.  Build a rotation that aligns -Z with
                # -approach_dir (i.e. Z points away from object, -Z toward).
                z_axis = approach_dir  # hand Z = away from object
                # Choose an "up" reference to avoid degenerate cross products
                if abs(z_axis[2]) < 0.9:
                    up = np.array([0.0, 0.0, 1.0])
                else:
                    up = np.array([0.0, 1.0, 0.0])
                x_axis = np.cross(up, z_axis)
                x_axis /= np.linalg.norm(x_axis)
                y_axis = np.cross(z_axis, x_axis)
                R = np.column_stack([x_axis, y_axis, z_axis])  # 3x3

                # Add small random rotation perturbation
                angle = self._rng.uniform(-0.3, 0.3)
                axis_rand = self._rng.standard_normal(3)
                axis_rand /= np.linalg.norm(axis_rand)
                c, s = np.cos(angle), np.sin(angle)
                K = np.array([
                    [0, -axis_rand[2], axis_rand[1]],
                    [axis_rand[2], 0, -axis_rand[0]],
                    [-axis_rand[1], axis_rand[0], 0],
                ])
                R_pert = np.eye(3) + s * K + (1 - c) * (K @ K)
                R = R @ R_pert

                # Convert rotation to quaternion (w, x, y, z)
                tr = R[0, 0] + R[1, 1] + R[2, 2]
                if tr > 0:
                    s_ = np.sqrt(tr + 1.0) * 2
                    qw = 0.25 * s_
                    qx = (R[2, 1] - R[1, 2]) / s_
                    qy = (R[0, 2] - R[2, 0]) / s_
                    qz = (R[1, 0] - R[0, 1]) / s_
                elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                    s_ = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                    qw = (R[2, 1] - R[1, 2]) / s_
                    qx = 0.25 * s_
                    qy = (R[0, 1] + R[1, 0]) / s_
                    qz = (R[0, 2] + R[2, 0]) / s_
                elif R[1, 1] > R[2, 2]:
                    s_ = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                    qw = (R[0, 2] - R[2, 0]) / s_
                    qx = (R[0, 1] + R[1, 0]) / s_
                    qy = 0.25 * s_
                    qz = (R[1, 2] + R[2, 1]) / s_
                else:
                    s_ = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                    qw = (R[1, 0] - R[0, 1]) / s_
                    qx = (R[0, 2] + R[2, 0]) / s_
                    qy = (R[1, 2] + R[2, 1]) / s_
                    qz = 0.25 * s_

                quat = np.array([qw, qx, qy, qz])
                quat /= np.linalg.norm(quat)
                q[q_adr + 3: q_adr + 7] = quat
                break  # Only set freejoint once
            # (finger joints handled below)

        # Finger joints: partially curled with noise
        # Bias toward 40% of joint range (pre-shaped for wrapping)
        for dofs in self._ik.finger_dof_slices:
            for dof_id in dofs:
                jid = int(self.model.dof_jntid[dof_id])
                jnt_type = int(self.model.jnt_type[jid])
                if jnt_type in (2, 3):  # slide or hinge
                    q_adr = int(self.model.jnt_qposadr[jid])
                    limited = bool(self.model.jnt_limited[jid])
                    if limited:
                        lo = float(self.model.jnt_range[jid, 0])
                        hi = float(self.model.jnt_range[jid, 1])
                        # 40% curl \pm  15% noise (concentrated, not uniform)
                        mid = lo + (hi - lo) * 0.4
                        noise = (hi - lo) * 0.15 * self._rng.standard_normal()
                        q[q_adr] = np.clip(mid + noise, lo, hi)
                    else:
                        q[q_adr] = 0.3 + self._rng.uniform(-0.2, 0.2)

        return q

    def _graspit_to_qpos(self, grasp) -> NDArray:
        """Convert a GraspItGrasp into a full qpos vector.

        Maps the 11 GraspIt! DOFs to the corresponding MuJoCo joints and
        sets the base freejoint to place the hand near the object.
        """
        q = self.data.qpos.copy()

        # -- finger DOFs --
        # GraspIt! DOF order (from graspit_loader):
        #  0: thumb_cmc_yaw, 1: thumb_cmc_pitch, 2: thumb_ip,
        #  3: index_mcp_pitch, 4: index_dip, 5: (index_dip duplicate),
        #  6: middle_mcp_pitch, 7: middle_dip,
        #  8: ring_mcp_pitch, 9: ring_dip,
        # 10: pinky_mcp_pitch
        GRASPIT_TO_JOINT = [
            ("thumb_cmc_yaw",    0),
            ("thumb_cmc_pitch",  1),
            ("thumb_ip",         2),
            ("index_mcp_pitch",  3),
            ("index_dip",        4),
            # index_q3 (5) --> also maps to index_dip; skip or average
            ("middle_mcp_pitch", 6),
            ("middle_dip",       7),
            ("ring_mcp_pitch",   8),
            ("ring_dip",         9),
            ("pinky_mcp_pitch", 10),
        ]

        dof_vals = grasp.hand_dof_values
        for joint_name, gi in GRASPIT_TO_JOINT:
            if gi >= len(dof_vals):
                continue
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid < 0:
                continue
            q_adr = int(self.model.jnt_qposadr[jid])
            val = float(dof_vals[gi])
            # Clamp to limits
            if self.model.jnt_limited[jid]:
                lo = float(self.model.jnt_range[jid, 0])
                hi = float(self.model.jnt_range[jid, 1])
                val = np.clip(val, lo, hi)
            q[q_adr] = val

        # -- base freejoint: position hand relative to object --
        base_jnt = self.cfg.base_joint or DEFAULT_BASE_JOINT
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, base_jnt)
        if jid >= 0 and int(self.model.jnt_type[jid]) == 0:
            q_adr = int(self.model.jnt_qposadr[jid])
            obj_center = self._get_object_center()

            # Use contact centroid direction if contacts are available,
            # otherwise default to a side approach
            if hasattr(grasp, 'contact_points') and len(grasp.contact_points) > 0:
                ctc_centroid = grasp.contact_points.mean(axis=0) + obj_center
                approach = ctc_centroid - obj_center
                norm = np.linalg.norm(approach)
                if norm > 1e-6:
                    approach = approach / norm
                else:
                    approach = np.array([0.0, -1.0, 0.5])
                    approach /= np.linalg.norm(approach)
            else:
                approach = np.array([0.0, -1.0, 0.5])
                approach /= np.linalg.norm(approach)

            # GraspIt approach direction: start 60 mm from object centre.
            # The weighted DLS IK will adjust base pose while link
            # repulsion prevents penetration.
            q[q_adr: q_adr + 3] = obj_center + approach * 0.06

            # Orient palm toward object
            z_axis = approach
            up = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x_axis = np.cross(up, z_axis)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            R = np.column_stack([x_axis, y_axis, z_axis])

            # Convert R to quaternion
            tr = R[0, 0] + R[1, 1] + R[2, 2]
            if tr > 0:
                s_ = np.sqrt(tr + 1.0) * 2
                qw, qx, qy, qz = 0.25*s_, (R[2,1]-R[1,2])/s_, (R[0,2]-R[2,0])/s_, (R[1,0]-R[0,1])/s_
            elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s_ = np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2]) * 2
                qw, qx, qy, qz = (R[2,1]-R[1,2])/s_, 0.25*s_, (R[0,1]+R[1,0])/s_, (R[0,2]+R[2,0])/s_
            elif R[1,1] > R[2,2]:
                s_ = np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2]) * 2
                qw, qx, qy, qz = (R[0,2]-R[2,0])/s_, (R[0,1]+R[1,0])/s_, 0.25*s_, (R[1,2]+R[2,1])/s_
            else:
                s_ = np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1]) * 2
                qw, qx, qy, qz = (R[1,0]-R[0,1])/s_, (R[0,2]+R[2,0])/s_, (R[1,2]+R[2,1])/s_, 0.25*s_
            quat = np.array([qw, qx, qy, qz])
            quat /= np.linalg.norm(quat)
            q[q_adr + 3: q_adr + 7] = quat

        return q

    def _extract_finger_qpos(self, qpos: NDArray) -> Dict[str, NDArray]:
        """Extract per-finger joint values from a full qpos vector"""
        result: Dict[str, NDArray] = {}
        for i, fn in enumerate(self._ik.finger_names):
            dofs = self._ik.finger_dof_slices[i]
            vals = []
            for dof_id in dofs:
                jid = int(self.model.dof_jntid[dof_id])
                q_adr = int(self.model.jnt_qposadr[jid])
                vals.append(float(qpos[q_adr]))
            result[fn] = np.array(vals)
        return result
