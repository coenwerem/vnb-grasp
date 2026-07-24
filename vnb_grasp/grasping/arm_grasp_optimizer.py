"""Arm-aware grasp optimizer for full arm+hand systems.

This optimizer works with the full arm+hand arena (zarm_realhand_l6_right_arena)
where the hand base is rigidly attached to the arm wrist. Instead of optimizing
a freejoint base pose, it:

1. Samples approach directions relative to the object
2. Uses inverse kinematics to solve for arm joint angles
3. Optimizes hand joint angles to achieve finger-pad contacts

Key insight: The arm constrains approach directions via forward kinematics,
making grasp geometry more meaningful than the floating hand testbed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

try:
    import mujoco as _mujoco
except ImportError:
    _mujoco = None

mujoco = cast(Any, _mujoco)

try:
    from mink import (
        Configuration as MinkConfiguration,
        FrameTask as MinkFrameTask,
        SE3 as MinkSE3,
        SO3 as MinkSO3,
        solve_ik as mink_solve_ik,
    )

    MINK_AVAILABLE = True
except ImportError:
    MINK_AVAILABLE = False

from .object_surface import ObjectSurface, GeomKind
from .gws_quality import GWSResult, analyze_gws
from .grasp_sampler import DEFAULT_FINGER_MAP, SampledGrasp

logger = logging.getLogger(__name__)


# Default arm joint names for ZArm
DEFAULT_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


@dataclass
class ArmGraspConfig:
    """Configuration for arm-aware grasp optimizer"""

    n_starts: int = 64
    top_k: int = 10
    friction_coef: float = 1.0
    max_ik_iters: int = 100
    max_hand_iters: int = 100
    sdf_tol: float = 0.003  # 3mm surface tolerance
    ik_tolerance: float = 1e-3  # IK position tolerance
    min_finger_sep: float = 0.010  # 10mm min fingertip separation
    n_fingers: int = 3  # Default to precision grip (thumb, index, middle)
    active_fingers: List[str] = None  # If None, uses first n_fingers

    # Approach geometry
    pregrasp_distance: float = 0.08  # 8cm pregrasp standoff
    grasp_standoff: float = 0.0  # Position TCP at object surface for finger reach

    # Table collision
    table_z: float = 0.775
    min_tcp_z: float = 0.78  # Minimum TCP height (slightly above table)

    # Finger curl parameters for precision grip
    thumb_curl_range: Tuple[float, float] = (0.4, 0.8)
    finger_curl_range: Tuple[float, float] = (0.5, 0.9)

    def __post_init__(self):
        if self.active_fingers is None:
            self.active_fingers = ["thumb", "index", "middle"][: self.n_fingers]


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


class ArmGraspOptimizer:
    """Arm-aware grasp optimizer using IK for arm positioning.

    This optimizer works with full arm+hand systems where the hand base
    is attached to the arm wrist. It uses inverse kinematics to position
    the arm and optimizes hand joints for precision grip.

    Parameters
    ----------
    model, data : mujoco.MjModel, mujoco.MjData
        MuJoCo model and scratch simulation data.
    surface : ObjectSurface
        Object SDF / surface representation.
    config : ArmGraspConfig, optional
    object_body_name : str, optional
    tcp_site_name : str
        Tool center point site name for IK targeting.
    arm_joints : list[str], optional
        Arm joint names (defaults to ZArm joints).
    finger_map : dict, optional
        Finger configuration map.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        surface: ObjectSurface,
        *,
        config: ArmGraspConfig | None = None,
        object_body_name: str | None = None,
        tcp_site_name: str = "tcp_site",
        arm_joints: list[str] | None = None,
        finger_map: dict[str, tuple[list[str], str]] | None = None,
    ):
        if mujoco is None:
            raise ImportError("mujoco is required")
        if not MINK_AVAILABLE:
            raise ImportError("mink is required for arm IK")

        self.model = model
        self.data = data
        self.surface = surface
        self.cfg = config or ArmGraspConfig()
        self._rng = np.random.default_rng()

        self._finger_map = finger_map or DEFAULT_FINGER_MAP
        self._arm_joint_names = arm_joints or DEFAULT_ARM_JOINTS

        mujoco.mj_forward(self.model, self.data)

        # Resolve TCP site
        self._tcp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, tcp_site_name
        )
        if self._tcp_site_id < 0:
            raise ValueError(f"TCP site '{tcp_site_name}' not found")

        # Resolve arm joints
        self._arm_joint_ids: list[int] = []
        self._arm_joint_qadr: list[int] = []
        self._arm_joint_bounds: list[tuple[float, float]] = []

        for jn in self._arm_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                logger.warning(f"Arm joint '{jn}' not found")
                continue
            qadr = int(self.model.jnt_qposadr[jid])
            if bool(self.model.jnt_limited[jid]):
                lo = float(self.model.jnt_range[jid, 0])
                hi = float(self.model.jnt_range[jid, 1])
            else:
                lo, hi = -np.pi, np.pi
            self._arm_joint_ids.append(int(jid))
            self._arm_joint_qadr.append(qadr)
            self._arm_joint_bounds.append((lo, hi))

        if len(self._arm_joint_ids) < 6:
            raise ValueError(f"Expected 6 arm joints, found {len(self._arm_joint_ids)}")

        # Resolve finger joints and tip sites
        self._resolved_fingers: list[tuple[str, list[str], int]] = []
        for finger_name, (joint_names, site_name) in self._finger_map.items():
            if finger_name not in self.cfg.active_fingers:
                continue
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id < 0:
                logger.warning(f"Missing tip site '{site_name}' for '{finger_name}'")
                continue
            self._resolved_fingers.append(
                (finger_name, list(joint_names), int(site_id))
            )

        if not self._resolved_fingers:
            raise ValueError("No valid fingertip sites found")

        self._finger_names = [f for f, _, _ in self._resolved_fingers]
        self._tip_site_ids = [sid for _, _, sid in self._resolved_fingers]

        # Resolve finger joints
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
                    continue
                jtype = int(self.model.jnt_type[jid])
                if jtype not in (2, 3):  # hinge or slide
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

        # Object and hand geom sets
        self._obj_body_id = self._resolve_object_body_id(object_body_name)
        if self._obj_body_id is not None:
            self._obj_geom_ids = {
                gi
                for gi in range(self.model.ngeom)
                if int(self.model.geom_bodyid[gi]) == self._obj_body_id
            }
        else:
            self._obj_geom_ids = set()

        # Find hand base body (parent of first finger's tip site)
        first_tip_body = int(self.model.site_bodyid[self._tip_site_ids[0]])
        hand_base_body = first_tip_body
        while hand_base_body > 0:
            parent = int(self.model.body_parentid[hand_base_body])
            parent_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, parent
            )
            if parent_name and "hand" in parent_name.lower():
                hand_base_body = parent
            elif parent_name and any(
                x in parent_name.lower() for x in ["wrist", "tool"]
            ):
                break
            else:
                break

        self._hand_base_body_id = hand_base_body
        self._hand_geom_ids = _collect_geoms_in_subtree(
            self.model, self._hand_base_body_id
        )

        # Distal body geom IDs (fingertips)
        self._distal_body_ids = {
            int(self.model.site_bodyid[sid]) for sid in self._tip_site_ids
        }
        self._distal_geom_ids = {
            gi
            for gi in self._hand_geom_ids
            if int(self.model.geom_bodyid[gi]) in self._distal_body_ids
        }

        # Initialize Mink IK
        self._mink_config = MinkConfiguration(self.model)
        self._mink_frame_task = MinkFrameTask(
            frame_name=tcp_site_name,
            frame_type="site",
            position_cost=np.array([1.0, 1.0, 1.0]),
            orientation_cost=np.array([0.5, 0.5, 0.5]),
            lm_damping=1e-3,
        )

        logger.info(
            f"ArmGraspOptimizer initialized: {len(self._arm_joint_ids)} arm joints, "
            f"{len(self._finger_joint_ids)} finger joints, {len(self._finger_names)} fingers"
        )

    def _resolve_object_body_id(self, name: str | None) -> int | None:
        if name is None:
            return None
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return int(bid) if bid >= 0 else None

    def _get_object_center(self) -> NDArray[np.float64]:
        """Get object center position"""
        if self._obj_body_id is not None:
            return self.data.xpos[self._obj_body_id].copy()
        return self.surface.position.copy()

    def _estimate_object_radius(self) -> float:
        """Estimate object bounding radius for approach standoff.
        
        For precision grasping, we want the TCP positioned close to the object
        surface, not at the bounding sphere. Returns max half-extent for boxes,
        not the diagonal.
        """
        if self.surface.kind == GeomKind.BOX:
            # BOX: size = half-extents (sx, sy, sz)
            # Use max half-extent for approach (not diagonal)
            return float(np.max(self.surface.size[:3]))
        elif self.surface.kind == GeomKind.SPHERE:
            # SPHERE: size[0] = radius
            return float(self.surface.size[0])
        elif self.surface.kind == GeomKind.CYLINDER:
            # CYLINDER: size[0] = radius, size[1] = half-height
            # Use max of radius and half-height
            return float(max(self.surface.size[0], self.surface.size[1]))
        elif self.surface.kind == GeomKind.CAPSULE:
            # CAPSULE: size[0] = radius, size[1] = half-height
            return float(self.surface.size[0] + self.surface.size[1])
        elif self.surface.kind == GeomKind.ELLIPSOID:
            # ELLIPSOID: size = semi-axes (a, b, c)
            return float(np.max(self.surface.size[:3]))
        return 0.025  # Default 2.5cm

    def _compute_approach_poses(self) -> list[tuple[NDArray, NDArray]]:
        """Generate approach poses (position, quaternion) for the TCP.
        Returns list of (position, quaternion) tuples for the TCP to target.
        
        Key insight from hand geometry analysis:
        - Open fingers extend along TCP +Z axis (~135mm)
        - Curled fingers move toward TCP +X axis (at 100% curl: X=45mm, Z=25mm)
        - For precision grip, TCP +X should point toward object center
        - TCP positioned above/behind object so curled fingers wrap around it
        
        Best strategy: Top-down approaches with fingers pointing down toward object.
        """
        obj_center = self._get_object_center()
        obj_radius = self._estimate_object_radius()
        # Height above object for top-down grasps
        # At 90% curl, fingertips are ~35mm from TCP in X direction
        # So TCP should be ~(obj_radius + 35mm) above object for fingertips to reach surface
        grasp_height = obj_radius + 0.05  # 50mm above object center

        poses: list[tuple[NDArray, NDArray]] = []

        # Strategy 1: Top-down approaches (most reliable for this arm)
        # TCP directly above object, fingers pointing down
        # Vary the yaw angle around the vertical axis
        for yaw_deg in np.linspace(0, 360, 12, endpoint=False):
            yaw_rad = np.radians(yaw_deg)
            
            # Position: above object
            pos = obj_center.copy()
            pos[2] += grasp_height
            
            # Small XY offset based on yaw for variety
            offset = 0.01  # 10mm offset
            pos[0] += offset * np.cos(yaw_rad)
            pos[1] += offset * np.sin(yaw_rad)
            
            # Orientation: X-axis pointing down toward object (for curled fingers)
            # Z-axis pointing forward (+Y world direction, matching home orientation)
            # This means: X=[0,0,-1], Y=[-1,0,0], Z=[0,1,0] rotated by yaw
            x_axis = np.array([0.0, 0.0, -1.0])  # Down
            z_axis = np.array([np.sin(yaw_rad), np.cos(yaw_rad), 0.0])  # Horizontal, rotated
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis) + 1e-12
            
            R = np.column_stack([x_axis, y_axis, z_axis])
            
            # Add small random noise
            rv_noise = self._rng.normal(scale=0.05, size=3)
            R_noise = Rotation.from_rotvec(rv_noise).as_matrix()
            R = R @ R_noise
            quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
            quat_mj = np.array([quat[3], quat[0], quat[1], quat[2]])
            
            pos[2] = max(pos[2], self.cfg.min_tcp_z)
            poses.append((pos.copy(), quat_mj.copy()))

        # Strategy 2: Oblique approaches (tilted top-down)
        # TCP above and slightly offset, fingers angled toward object
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            for pitch_deg in [15, 30]:  # Tilt angles from vertical
                pitch_rad = np.radians(pitch_deg)
                
                # Position: above and offset from object
                offset_dist = grasp_height * np.sin(pitch_rad)
                height = grasp_height * np.cos(pitch_rad)
                
                pos = obj_center.copy()
                pos[0] += offset_dist * np.cos(angle)
                pos[1] += offset_dist * np.sin(angle)
                pos[2] += height
                
                # Direction from TCP to object
                to_obj = obj_center - pos
                to_obj_norm = np.linalg.norm(to_obj)
                if to_obj_norm > 1e-6:
                    to_obj = to_obj / to_obj_norm
                else:
                    to_obj = np.array([0.0, 0.0, -1.0])
                
                # X-axis points toward object (for curled fingers)
                x_axis = to_obj
                
                # Z-axis: roughly horizontal, perpendicular to approach
                up = np.array([0.0, 0.0, 1.0])
                z_axis = np.cross(x_axis, up)
                z_norm = np.linalg.norm(z_axis)
                if z_norm < 0.1:
                    # Near vertical approach, use Y-based up
                    up = np.array([0.0, 1.0, 0.0])
                    z_axis = np.cross(x_axis, up)
                z_axis /= np.linalg.norm(z_axis) + 1e-12
                
                # Y-axis completes the frame
                y_axis = np.cross(z_axis, x_axis)
                y_axis /= np.linalg.norm(y_axis) + 1e-12
                
                R = np.column_stack([x_axis, y_axis, z_axis])
                
                # Add random noise
                rv_noise = self._rng.normal(scale=0.05, size=3)
                R_noise = Rotation.from_rotvec(rv_noise).as_matrix()
                R = R @ R_noise
                
                quat = Rotation.from_matrix(R).as_quat()
                quat_mj = np.array([quat[3], quat[0], quat[1], quat[2]])
                
                pos[2] = max(pos[2], self.cfg.min_tcp_z)
                poses.append((pos.copy(), quat_mj.copy()))

        # Fill remaining with random top-down variations
        while len(poses) < self.cfg.n_starts:
            yaw = self._rng.uniform(0, 2 * np.pi)
            pitch = self._rng.uniform(0, np.radians(45))
            
            offset_dist = grasp_height * np.sin(pitch)
            height = grasp_height * np.cos(pitch)
            
            pos = obj_center.copy()
            pos[0] += offset_dist * np.cos(yaw)
            pos[1] += offset_dist * np.sin(yaw)  
            pos[2] += height
            
            # X toward object
            to_obj = obj_center - pos
            to_obj /= np.linalg.norm(to_obj) + 1e-12
            x_axis = to_obj
            up = np.array([0.0, 0.0, 1.0])
            z_axis = np.cross(x_axis, up)
            if np.linalg.norm(z_axis) < 0.1:
                up = np.array([0.0, 1.0, 0.0])
                z_axis = np.cross(x_axis, up)
            z_axis /= np.linalg.norm(z_axis) + 1e-12
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis) + 1e-12
            
            R = np.column_stack([x_axis, y_axis, z_axis])
            rv_noise = self._rng.normal(scale=0.08, size=3)
            R = R @ Rotation.from_rotvec(rv_noise).as_matrix()
            
            quat = Rotation.from_matrix(R).as_quat()
            quat_mj = np.array([quat[3], quat[0], quat[1], quat[2]])
            
            pos[2] = max(pos[2], self.cfg.min_tcp_z)
            poses.append((pos.copy(), quat_mj.copy()))

        return poses[:self.cfg.n_starts]

    def _solve_arm_ik(self, target_pos: NDArray, target_quat: NDArray) -> bool:
        """Solve 6-DOF IK for arm joints using damped least squares.

        Controls BOTH position AND orientation to ensure the hand faces the object.
        Uses native MuJoCo Jacobian for stable IK with arm joints only.
        Returns True if IK succeeded, False otherwise.
        Modifies self.data.qpos in place.
        """
        try:
            damping = 0.01  # Damping factor for regularization
            pos_weight = 1.0
            ori_weight = 0.5  # Orientation error weight (less important than position)
            for _ in range(self.cfg.max_ik_iters):
                # Compute site Jacobians for position and rotation
                jacp = np.zeros((3, self.model.nv))
                jacr = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._tcp_site_id)

                # Extract arm-only Jacobians (first 6 DoFs)
                Jp = jacp[:, :6]
                Jr = jacr[:, :6]
                tcp_pos = self.data.site_xpos[self._tcp_site_id]
                pos_err = target_pos - tcp_pos
                pos_error_norm = np.linalg.norm(pos_err)

                # Compute orientation error using quaternion difference
                # Current orientation: extract from site_xmat (3x3 rotation matrix)
                current_xmat = self.data.site_xmat[self._tcp_site_id].reshape(3, 3)
                current_quat_scipy = Rotation.from_matrix(current_xmat).as_quat()  # [x,y,z,w]
                # Convert target_quat from MuJoCo [w,x,y,z] to scipy [x,y,z,w]
                target_quat_scipy = np.array([target_quat[1], target_quat[2], target_quat[3], target_quat[0]])
                
                # Compute rotation error as rotation vector (axis-angle)
                # R_err = R_target @ R_current.T -> rotation from current to target
                R_current = Rotation.from_quat(current_quat_scipy)
                R_target = Rotation.from_quat(target_quat_scipy)
                R_err = R_target * R_current.inv()
                ori_err = R_err.as_rotvec()  # 3D rotation vector
                ori_error_norm = np.linalg.norm(ori_err)

                # Check convergence
                if pos_error_norm < self.cfg.ik_tolerance and ori_error_norm < 0.1:  # 0.1 rad ~ 6 deg
                    return True
                # Stack weighted Jacobians and errors
                J = np.vstack([pos_weight * Jp, ori_weight * Jr])  # 6x6
                err = np.concatenate([pos_weight * pos_err, ori_weight * ori_err])  # 6D
                # Damped least squares: dq = (J^T J + lambda*I)^-1 J^T e
                JTJ = J.T @ J + damping * np.eye(6)
                dq = np.linalg.solve(JTJ, J.T @ err)
                for i, qadr in enumerate(self._arm_joint_qadr):
                    jnt_id = self._arm_joint_ids[i]
                    if bool(self.model.jnt_limited[jnt_id]):
                        lo, hi = self._arm_joint_bounds[i]
                        self.data.qpos[qadr] = np.clip(
                            self.data.qpos[qadr] + dq[i], lo, hi
                        )
                    else:
                        # Unlimited joint, no clamping
                        self.data.qpos[qadr] += dq[i]
                mujoco.mj_forward(self.model, self.data)

            # Check final errors
            tcp_pos = self.data.site_xpos[self._tcp_site_id]
            pos_error = np.linalg.norm(tcp_pos - target_pos)
            # Recompute orientation error
            current_xmat = self.data.site_xmat[self._tcp_site_id].reshape(3, 3)
            current_quat_scipy = Rotation.from_matrix(current_xmat).as_quat()
            target_quat_scipy = np.array([target_quat[1], target_quat[2], target_quat[3], target_quat[0]])
            R_current = Rotation.from_quat(current_quat_scipy)
            R_target = Rotation.from_quat(target_quat_scipy)
            ori_error = np.linalg.norm((R_target * R_current.inv()).as_rotvec())
            
            # Accept if position is close enough (orientation less critical)
            return pos_error < self.cfg.ik_tolerance * 10 and ori_error < 0.5
        except Exception as e:
            logger.debug(f"IK solve failed: {e}")
            return False

    def _optimize_hand_joints(self, target_contacts: dict[str, NDArray]) -> tuple[float, dict]:
        """Optimize hand joint angles for precision grip.

        Returns (best_residual, finger_qpos_dict).
        """
        # target_contacts passed from _optimize_single (precision grip contact points)

        # Build optimization parameters (finger joint angles)
        n_params = len(self._finger_joint_ids)
        bounds = self._finger_joint_bounds

        def objective(x):
            # Set finger joint angles
            for j, qadr in enumerate(self._finger_joint_qadr):
                lo, hi = bounds[j]
                self.data.qpos[qadr] = np.clip(x[j], lo, hi)

            mujoco.mj_forward(self.model, self.data)

            # Compute fingertip distance to target contacts
            total_dist = 0.0
            for i, (finger_name, _, site_id) in enumerate(self._resolved_fingers):
                tip_pos = self.data.site_xpos[site_id]
                target = target_contacts[finger_name]
                dist = np.linalg.norm(tip_pos - target)
                total_dist += dist**2

            return total_dist

        # Initial guess: curl fingers to precision grip position
        x0 = np.zeros(n_params)
        for j, qadr in enumerate(self._finger_joint_qadr):
            lo, hi = bounds[j]
            # Determine if this is thumb or other finger
            is_thumb = any(
                j in self._finger_joint_by_finger.get("thumb", []) for _ in [1]
            )
            if is_thumb:
                frac = self._rng.uniform(*self.cfg.thumb_curl_range)
            else:
                frac = self._rng.uniform(*self.cfg.finger_curl_range)
            x0[j] = lo + frac * (hi - lo)

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.cfg.max_hand_iters},
        )

        # Apply best solution
        for j, qadr in enumerate(self._finger_joint_qadr):
            lo, hi = bounds[j]
            self.data.qpos[qadr] = np.clip(result.x[j], lo, hi)
        mujoco.mj_forward(self.model, self.data)

        # Build finger qpos dict
        finger_qpos = {}
        for finger_name, joint_indices in self._finger_joint_by_finger.items():
            finger_qpos[finger_name] = np.array(
                [self.data.qpos[self._finger_joint_qadr[j]] for j in joint_indices]
            )

        return result.fun, finger_qpos

    def _evaluate_grasp(
        self, target_contacts: dict, target_normals: dict
    ) -> SampledGrasp:
        """Evaluate current configuration as a grasp"""
        # Get fingertip positions
        fingertip_positions = {}
        for finger_name, _, site_id in self._resolved_fingers:
            fingertip_positions[finger_name] = self.data.site_xpos[site_id].copy()

        # Build contact info for GWS analysis
        contacts = []
        obj_center = self._get_object_center()
        for finger_name in self._finger_names:
            tip_pos = fingertip_positions[finger_name]
            target = target_contacts.get(finger_name, tip_pos)
            normal = target_normals.get(finger_name, np.array([0, 0, 1]))
            dist = np.linalg.norm(tip_pos - target)
            if dist < 0.03:  # 30mm tolerance for contact registration
                from ..belief.mujoco_rollout import ContactInfo
                # Build contact frame: [normal, tangent1, tangent2]
                if abs(normal[2]) < 0.9:
                    tangent1 = np.cross(normal, np.array([0, 0, 1]))
                else:
                    tangent1 = np.cross(normal, np.array([1, 0, 0]))
                tangent1 /= np.linalg.norm(tangent1) + 1e-12
                tangent2 = np.cross(normal, tangent1)
                frame = np.row_stack([normal, tangent1, tangent2])
                contacts.append(
                    ContactInfo(
                        geom1=-1,  # dummy
                        geom2=-1,
                        pos=tip_pos.copy(),
                        frame=frame,
                        dist=-dist,  # negative = penetrating/in contact
                        force=np.array([1.0, 0.0, 0.0]),  # unit normal force
                    )
                )

        # Analyze GWS, needs object_center
        if contacts:
            gws = analyze_gws(contacts, object_center=obj_center, friction_coef=self.cfg.friction_coef)
        else:
            gws = GWSResult(
                epsilon=0.0,
                volume=0.0,
                min_singular=0.0,
                is_force_closure=False,
                n_contacts=0,
            )

        # Compute residual (sum of squared distances)
        residual = 0.0
        max_pen = 0.0
        for finger_name in self._finger_names:
            tip_pos = fingertip_positions[finger_name]
            target = target_contacts.get(finger_name, tip_pos)
            residual += np.linalg.norm(tip_pos - target) ** 2

        # Build finger qpos
        finger_qpos = {}
        for finger_name, joint_indices in self._finger_joint_by_finger.items():
            finger_qpos[finger_name] = np.array(
                [self.data.qpos[self._finger_joint_qadr[j]] for j in joint_indices]
            )

        return SampledGrasp(
            hand_qpos=self.data.qpos.copy(),
            finger_qpos=finger_qpos,
            fingertip_positions=fingertip_positions,
            target_contacts=target_contacts,
            target_normals=target_normals,
            gws=gws,
            residual=float(residual),
            max_penetration=max_pen,
            seed_source="arm_ik",
        )

    def _optimize_single(
        self, target_pos: NDArray, target_quat: NDArray
    ) -> SampledGrasp | None:
        """Optimize a single grasp from a target approach pose.
        
        Uses fingertip-projection approach:
        1. Position arm via IK
        2. Pre-curl fingers partway toward grasp position
        3. Project each fingertip toward object to get realistic target contacts
        4. Optimize finger joints to reach those targets
        """
        q_init = self.data.qpos.copy()
        try:
            # Step 1: Solve arm IK to reach target pose
            if not self._solve_arm_ik(target_pos, target_quat):
                logger.debug("IK failed for target pose")
                self.data.qpos[:] = q_init
                return None

            # Step 2: Pre-curl fingers to get reasonable starting positions
            # This gives us fingertip positions that can reach the object
            for j, qadr in enumerate(self._finger_joint_qadr):
                lo, hi = self._finger_joint_bounds[j]
                # Start at 50% curl, gives fingertips closer to palm
                is_thumb = any(
                    j in self._finger_joint_by_finger.get("thumb", []) for _ in [1]
                )
                if is_thumb:
                    frac = 0.5  # Thumb at 50%
                else:
                    frac = 0.6  # Other fingers at 60%
                self.data.qpos[qadr] = lo + frac * (hi - lo)
            mujoco.mj_forward(self.model, self.data)

            # Step 3: Compute OPPOSING target contacts for precision grip
            # Key insight: thumb must oppose index/middle for force closure
            obj_center = self._get_object_center()
            obj_radius = self._estimate_object_radius()
            if self.surface.kind == GeomKind.BOX:
                contact_radius = float(np.max(self.surface.size[:3]))
            else:
                contact_radius = obj_radius
            # Get TCP orientation to determine grasp axis
            tcp_xmat = self.data.site_xmat[self._tcp_site_id].reshape(3, 3)
            # X-axis of TCP points toward object (per our approach strategy)
            # This will be the grasp axis, with thumb on the +X side and fingers on the -X side
            grasp_axis = tcp_xmat[:, 0]  # TCP X-axis
            grasp_axis /= np.linalg.norm(grasp_axis) + 1e-12
            
            # Y-axis gives spread direction for index/middle separation
            spread_axis = tcp_xmat[:, 1]  # TCP Y-axis
            spread_axis /= np.linalg.norm(spread_axis) + 1e-12
            target_contacts = {}
            target_normals = {}
            for finger_name, _, site_id in self._resolved_fingers:
                if finger_name == 'thumb':
                    # Thumb contacts on the POSITIVE grasp axis side
                    # (toward where TCP X-axis points)
                    contact_dir = grasp_axis
                    normal = -grasp_axis  # Normal points back toward fingers
                elif finger_name == 'index':
                    # Index contacts on NEGATIVE grasp axis side, slightly spread in +Y
                    contact_dir = -grasp_axis + 0.3 * spread_axis
                    contact_dir /= np.linalg.norm(contact_dir) + 1e-12
                    normal = -contact_dir  # Normal points toward thumb
                else:  # middle
                    # Middle contacts on NEGATIVE grasp axis side, slightly spread in -Y
                    contact_dir = -grasp_axis - 0.3 * spread_axis
                    contact_dir /= np.linalg.norm(contact_dir) + 1e-12
                    normal = -contact_dir  # Normal points toward thumb
                
                # Target contact is on object surface
                contact_pt = obj_center + contact_dir * contact_radius
                target_contacts[finger_name] = contact_pt
                target_normals[finger_name] = normal

            # Step 4: Optimize hand joints to reach target contacts
            residual, finger_qpos = self._optimize_hand_joints(target_contacts)
            # Step 5: Evaluate final grasp
            grasp = self._evaluate_grasp(target_contacts, target_normals)
            return grasp
        except Exception as e:
            logger.debug(f"Optimization failed: {e}")
            self.data.qpos[:] = q_init
            return None

    def _get_object_freejoint_slice(self) -> tuple[int, int] | None:
        """Get the qpos slice for the object's freejoint if it has one"""
        if self._obj_body_id is None:
            return None
        
        # Find joint attached to object body
        for jid in range(self.model.njnt):
            if int(self.model.jnt_bodyid[jid]) == self._obj_body_id:
                jtype = int(self.model.jnt_type[jid])
                if jtype == 0:  # freejoint
                    qadr = int(self.model.jnt_qposadr[jid])
                    return (qadr, qadr + 7)  # freejoint has 7 DOF
        return None

    def _reset_to_home(self, object_qpos: NDArray | None = None) -> None:
        """Reset arm and hand joints to home position, preserving object positions.
        Uses the model's first keyframe for arm joint values if available,
        otherwise resets to neutral positions that work for reaching.
        Parameters
        ----------
        object_qpos : array, optional
            The object freejoint qpos to restore. If None, object is left as-is.
        """
        # Try to use keyframe 'home' if available
        if self.model.nkey > 0:
            # Get arm joint values from first keyframe
            key_qpos = self.model.key_qpos[0]
            for qadr in self._arm_joint_qadr:
                self.data.qpos[qadr] = key_qpos[qadr]
        else:
            # Fallback: use neutral arm configuration that enables reaching
            # These values approximate a ready-to-reach pose
            arm_home = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]  # shoulder_lift and wrist_1 at -90 deg
            for i, qadr in enumerate(self._arm_joint_qadr):
                self.data.qpos[qadr] = arm_home[i]
        
        # Reset finger joints to zero (open hand)
        for qadr in self._finger_joint_qadr:
            self.data.qpos[qadr] = 0.0
        # Restore object freejoint if provided
        obj_slice = self._get_object_freejoint_slice()
        if obj_slice is not None and object_qpos is not None:
            start, end = obj_slice
            self.data.qpos[start:end] = object_qpos
        
        # Forward kinematics to update body positions
        mujoco.mj_forward(self.model, self.data)

    def solve(self) -> list[SampledGrasp]:
        """Run multi-start optimization and return top-k grasps"""
        # CRITICAL: Save the original object freejoint qpos before any optimization
        # This prevents object position corruption during IK iterations
        obj_slice = self._get_object_freejoint_slice()
        if obj_slice is not None:
            start, end = obj_slice
            original_object_qpos = self.data.qpos[start:end].copy()
            logger.info(f"Object freejoint qpos: {original_object_qpos[:3]} (position)")
        else:
            original_object_qpos = None
            logger.info("No object freejoint found - using surface.position")
        
        # Save complete initial state
        q_home = self.data.qpos.copy()
        
        # Generate approach poses (uses _get_object_center which reads from data.xpos)
        # Make sure object is at correct position first
        mujoco.mj_forward(self.model, self.data)
        approach_poses = self._compute_approach_poses()
        grasps: list[SampledGrasp] = []
        for i, (target_pos, target_quat) in enumerate(approach_poses):
            # CRITICAL: Reset to home before each optimization attempt
            # This ensures object position is correct and arm starts from known state
            self._reset_to_home(original_object_qpos)
            
            # Verify object position is correct
            obj_center = self._get_object_center()
            expected_center = self.surface.position
            center_error = np.linalg.norm(obj_center - expected_center)
            if center_error > 0.01:  # 1cm tolerance
                logger.warning(
                    f"Object center drift detected: {center_error*1000:.1f}mm. "
                    f"Got {obj_center}, expected {expected_center}"
                )
            grasp = self._optimize_single(target_pos, target_quat)
            if grasp is not None and grasp.gws.n_contacts > 0:
                grasps.append(grasp)
            if (i + 1) % 10 == 0:
                logger.info(
                    f"Processed {i + 1}/{len(approach_poses)} approaches, "
                    f"{len(grasps)} valid grasps"
                )
        # Restore original state
        self.data.qpos[:] = q_home
        mujoco.mj_forward(self.model, self.data)
        if not grasps:
            logger.warning("No valid grasps found")
            return []
        grasps.sort(key=lambda g: float(g.gws.epsilon), reverse=True)
        logger.info(f"Found {len(grasps)} valid grasps, returning top {self.cfg.top_k}")
        return grasps[: self.cfg.top_k]