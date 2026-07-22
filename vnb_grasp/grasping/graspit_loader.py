"""
Load grasps from vnb_grasp grasp database (aggregated GraspIt! exports).

The grasp database contains JSON files organized by object, with each grasp
containing:
- hand_dof_values: 11 DOF values for RealHand L6
- object_position_mm: Object position in millimeters
- epsilon_quality/volume_quality: GraspIt! quality metrics
- contact_points_mm: Contact locations on object surface

Coordinate conventions:
- GraspIt!: Z-up, millimeters
- MuJoCo: Z-up, meters

The main transformation is unit conversion (mm -> m).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from scipy.spatial.transform import Rotation

# Package-relative path to grasp database
GRASP_DB_PATH = Path(__file__).parent.parent.parent / "grasp_db"


@dataclass
class GraspItGrasp:
    """A grasp loaded from the vnb_grasp grasp database"""

    # Hand joint configuration ; 11 DOFs for RealHand L6
    hand_dof_values: np.ndarray  # ; 11, radians

    # Object pose the grasp was computed for
    object_position: np.ndarray  # ; 3, meters
    object_orientation: np.ndarray  # ; 4, quaternion [w, x, y, z]

    # Quality metrics from GraspIt!
    epsilon_quality: float = 0.0
    volume_quality: float = 0.0
    n_contacts: int = 0

    # Contact geometry ; in object frame, meters
    contact_points: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3))
    )  # ; n, 3
    contact_normals: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3))
    )  # ; n, 3

    # Metadata
    object_name: str = ""
    grasp_id: int = 0
    source_file: str = ""

    def to_mujoco_ctrl(self) -> np.ndarray:
        """
        Return hand DOF values for MuJoCo control.

        The 11 DOFs for RealHand L6 are:
        0: thumb_q1 (thumb base rotation)
        1: thumb_q2 (thumb proximal)
        2: thumb_q3 (thumb distal)
        3: index_q1 (index base)
        4: index_q2 (index proximal)
        5: index_q3 (index distal)  
        6: middle_q1 (middle base)
        7: middle_q2 (middle proximal)
        8: ring_q1 (ring base) - coupled with ring_q2
        9: ring_q2 (ring proximal) - same as ring_q1
        10: pinky_q1 (pinky base)

        Returns:
            np.ndarray of shape (11,) with joint positions in radians
        """
        return self.hand_dof_values.copy()


@dataclass
class GraspDatabase:
    """A collection of grasps for an object"""

    object_name: str
    grasps: List[GraspItGrasp]

    def filter_by_quality(
        self, min_epsilon: float = 0.0, min_contacts: int = 0
    ) -> "GraspDatabase":
        """Return a new database with only grasps above quality thresholds"""
        filtered = [
            g
            for g in self.grasps
            if g.epsilon_quality >= min_epsilon and g.n_contacts >= min_contacts
        ]
        return GraspDatabase(object_name=self.object_name, grasps=filtered)

    def best_grasp(self) -> Optional[GraspItGrasp]:
        """Return the highest quality grasp"""
        if not self.grasps:
            return None
        return max(self.grasps, key=lambda g: g.epsilon_quality)

    def top_k_grasps(self, k: int = 5) -> List[GraspItGrasp]:
        """Return top k grasps by epsilon quality"""
        sorted_grasps = sorted(
            self.grasps, key=lambda g: g.epsilon_quality, reverse=True
        )
        return sorted_grasps[:k]

    def random_grasp(self, rng: Optional[np.random.Generator] = None) -> GraspItGrasp:
        """Return a random grasp from the database"""
        if not self.grasps:
            raise ValueError(f"No grasps available for {self.object_name}")
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.integers(len(self.grasps))
        return self.grasps[idx]

    def __len__(self) -> int:
        return len(self.grasps)

    def __iter__(self):
        return iter(self.grasps)


class GraspLoader:
    """Load grasps from the vnb_grasp grasp database"""

    # GraspIt! uses millimeters, MuJoCo uses meters
    UNIT_SCALE = 0.001

    def __init__(self, grasp_db_path: Optional[Path] = None):
        """
        Initialize the loader.

        Args:
            grasp_db_path: Path to grasp database directory.
                          Defaults to vnb_grasp/grasp_db/
        """
        self.grasp_db_path = Path(grasp_db_path) if grasp_db_path else GRASP_DB_PATH

    def available_objects(self) -> List[str]:
        """List all objects with available grasps"""
        if not self.grasp_db_path.exists():
            return []
        return [p.stem for p in self.grasp_db_path.glob("*.json")]

    def load(self, object_name: str) -> GraspDatabase:
        """
        Load all grasps for an object.

        Args:
            object_name: Name of object (e.g., "cube", "ycb_mustard_bottle")

        Returns:
            GraspDatabase containing all grasps for the object

        Raises:
            FileNotFoundError: If no grasps exist for the object
        """
        grasp_file = self.grasp_db_path / f"{object_name}.json"
        if not grasp_file.exists():
            available = self.available_objects()
            raise FileNotFoundError(
                f"No grasps found for '{object_name}'. "
                f"Available objects: {available}"
            )

        with open(grasp_file) as f:
            data = json.load(f)

        grasps = []
        for i, g in enumerate(data.get("grasps", [])):
            grasp = self._parse_grasp(g, i)
            grasp.object_name = object_name
            grasps.append(grasp)

        return GraspDatabase(object_name=object_name, grasps=grasps)

    def load_all(self) -> Dict[str, GraspDatabase]:
        """Load grasps for all available objects"""
        result = {}
        for obj_name in self.available_objects():
            result[obj_name] = self.load(obj_name)
        return result

    def _parse_grasp(self, data: dict, grasp_id: int = 0) -> GraspItGrasp:
        """Parse a single grasp from JSON dict"""
        # Extract hand DOF values
        hand_dof = np.array(data.get("hand_dof_values", []), dtype=np.float64)

        # Extract object pose ; convert mm to m
        obj_pos_mm = data.get("object_position_mm", [0, 0, 0])
        obj_pos = np.array(obj_pos_mm, dtype=np.float64) * self.UNIT_SCALE

        obj_quat = np.array(
            data.get("object_orientation", [1, 0, 0, 0]), dtype=np.float64
        )

        # Extract quality metrics
        epsilon = data.get("epsilon_quality", 0.0)
        volume = data.get("volume_quality", 0.0)
        n_contacts = data.get("n_contacts", 0)

        # Extract contact geometry ; convert mm to m
        contact_pts_mm = data.get("contact_points_mm", [])
        contact_pts = np.array(contact_pts_mm, dtype=np.float64) * self.UNIT_SCALE
        if contact_pts.ndim == 1:
            contact_pts = contact_pts.reshape(-1, 3) if len(contact_pts) > 0 else np.zeros((0, 3))

        contact_normals = np.array(
            data.get("contact_normals", []), dtype=np.float64
        )
        if contact_normals.ndim == 1:
            contact_normals = contact_normals.reshape(-1, 3) if len(contact_normals) > 0 else np.zeros((0, 3))

        return GraspItGrasp(
            hand_dof_values=hand_dof,
            object_position=obj_pos,
            object_orientation=obj_quat,
            epsilon_quality=epsilon,
            volume_quality=volume,
            n_contacts=n_contacts,
            contact_points=contact_pts,
            contact_normals=contact_normals,
            grasp_id=grasp_id,
            source_file=data.get("source_file", ""),
        )


def transform_grasp_to_current_pose(
    grasp: GraspItGrasp,
    current_object_position: np.ndarray,
    current_object_orientation: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute the hand pose adjustment needed for a new object pose.

    When the object is at a different pose than what the grasp was computed for,
    we need to transform the grasp accordingly.

    Args:
        grasp: Original grasp computed for grasp.object_position/orientation
        current_object_position: Current object position in world frame (meters)
        current_object_orientation: Current object orientation [w,x,y,z]

    Returns:
        Dict with:
            - "object_delta_position": Position delta to apply
            - "object_delta_rotation": Rotation delta (as rotation matrix)
            - "contact_points_world": Contact points in world frame
    """
    # Original object pose
    orig_pos = grasp.object_position
    orig_quat = grasp.object_orientation

    # Current object pose
    curr_pos = current_object_position
    curr_quat = current_object_orientation

    # Compute position delta
    delta_pos = curr_pos - orig_pos

    # Compute rotation delta: R_delta = R_curr * R_orig^T
    R_orig = Rotation.from_quat(
        [orig_quat[1], orig_quat[2], orig_quat[3], orig_quat[0]]
    )  # scipy: [x,y,z,w]
    R_curr = Rotation.from_quat(
        [curr_quat[1], curr_quat[2], curr_quat[3], curr_quat[0]]
    )
    R_delta = R_curr * R_orig.inv()

    # Transform contact points to world frame
    if len(grasp.contact_points) > 0:
        contact_pts_world = R_curr.apply(grasp.contact_points) + curr_pos
    else:
        contact_pts_world = np.zeros((0, 3))

    return {
        "object_delta_position": delta_pos,
        "object_delta_rotation": R_delta.as_matrix(),
        "contact_points_world": contact_pts_world,
    }


# Convenience functions
def load_grasps(
    object_name: str,
    grasp_db_path: Optional[Path] = None,
) -> GraspDatabase:
    """
    Load grasps for an object from the grasp database.

    Args:
        object_name: Name of object (e.g., "cube", "ycb_mustard_bottle")
        grasp_db_path: Optional custom path to grasp database

    Returns:
        GraspDatabase containing all grasps for the object
    """
    loader = GraspLoader(grasp_db_path)
    return loader.load(object_name)


def list_available_objects(grasp_db_path: Optional[Path] = None) -> List[str]:
    """List all objects with available grasps in the database"""
    loader = GraspLoader(grasp_db_path)
    return loader.available_objects()


# RealHand L6 DOF names in order as stored in hand_dof_values
# (matches GraspIt internal order from zarm_realhand_l6_right/eigen/eigen.xml)
REALHAND_L6_DOF_NAMES = [
    "index_mcp_pitch",   # d0:  index MCP flexion         [0, 1.57]
    "index_dip",         # d1:  index DIP flexion         [-0.01, 1.40]
    "middle_mcp_pitch",  # d2:  middle MCP flexion        [0, 1.57]
    "middle_dip",        # d3:  middle DIP flexion        [-0.01, 1.40]
    "pinky_mcp_pitch",   # d4:  pinky MCP flexion         [0, 1.57]
    "pinky_dip",         # d5:  pinky DIP flexion         [-0.01, 1.40]
    "ring_mcp_pitch",    # d6:  ring MCP flexion          [0, 1.57]
    "ring_dip",          # d7:  ring DIP flexion          [-0.01, 1.40]
    "thumb_cmc_yaw",     # d8:  thumb CMC yaw/rotation    [0, 1.54]
    "thumb_cmc_pitch",   # d9:  thumb CMC pitch/proximal  [0, 0.52]
    "thumb_ip",          # d10: thumb IP distal            [0, 0.96]
]

# Mapping to reorder GraspIt DOFs --> MuJoCo hand joint order (qpos[6:17]).
# Usage: mujoco_hand_q[i] = graspit_dofs[GRASPIT_TO_MUJOCO[i]]
GRASPIT_TO_MUJOCO = [8, 9, 10, 0, 1, 2, 3, 6, 7, 4, 5]
