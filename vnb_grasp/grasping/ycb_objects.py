"""YCB Object Configuration and GraspIt Pregrasp Loading.

This module provides configuration for YCB objects used in the belief-MPC grasping
pipeline, including:
- Object body/geom name mapping
- GraspIt pregrasp database loading
- GraspIt-to-MuJoCo coordinate transformations
- Pregrasp selection heuristics

Supported objects (paper set):
- cube (default) - power grasp works well
- potted_meat (010_potted_meat_can) - power grasp works well
- mustard (006_mustard_bottle) - larger object, needs side grasp approach
- soup (005_tomato_soup_can) - cylindrical, needs side grasp approach
- tennis_ball (056_tennis_ball) - spherical, needs precision grasp

Note: The current implementation uses a top-down power grasp approach which works
best with smaller box-like objects. Larger cylindrical objects (soup, mustard) and
spherical objects (tennis_ball) require different grasp strategies for optimal results.

Author: Clinton Enwerem
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# GraspIt uses millimeters, MuJoCo uses meters
MM_TO_M = 0.001

# Path to the grasp database ; relative to vnb_grasp project
GRASP_DB_PATH = Path("grasp_db")


from enum import Enum


class GraspStrategy(Enum):
    """Enumeration of grasp strategies for different object types"""

    TOP_DOWN = "top_down"  # Pinch grasp from above ; works for box shapes
    SIDE_APPROACH = "side_approach"  # Side approach for cylinders
    ENVELOPING = "enveloping"  # Wrap around grasp for spherical objects


@dataclass
class YCBObjectConfig:
    """Configuration for a YCB object"""

    # Human-readable short name ; used in --object argument
    short_name: str

    # YCB object ID ; e.g., "006_mustard_bottle"
    ycb_id: str

    # MuJoCo body name in the scene
    body_name: str

    # MuJoCo collision geom name
    collision_geom: str

    # MuJoCo visual geom name
    visual_geom: str

    # GraspIt database file name
    grasp_db_file: str

    # Default table height for this object ; m
    # Cube center at Z=0.802 ; table at 0.777, cube half-height 0.025
    # For YCB objects, adjust based on object COM height
    table_z: float = 0.777

    # Approximate object half-height ; m for positioning
    half_height: float = 0.05

    # Mass ; kg for reference
    mass: float = 0.1

    # Inertia diagonal ; kg·m²
    inertia: Tuple[float, float, float] = (1e-4, 1e-4, 1e-4)

    # Quaternion ; w, x, y, z to orient the object upright on the table
    # Identity for objects whose mesh Z-axis is "up"
    # For soup can, mesh has Y-axis up, so rotate 90 deg  around X
    upright_quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # Scale factor applied to meshes in MuJoCo ; for GraspIt coordinate alignment
    mesh_scale: float = 1.0

    # OBJECT-SPECIFIC GRASP STRATEGY PARAMETERS
    # Primary grasp strategy for this object geometry
    grasp_strategy: GraspStrategy = GraspStrategy.TOP_DOWN

    # Object-specific pregrasp fraction ; how much of target grasp to use initially
    pregrasp_fraction: float = 0.6

    # Palm height offset above object ; m - varies by strategy
    palm_height_offset: float = 0.08

    # Side approach angle ; degrees for cylindrical objects ; 0 = +Y side
    side_approach_angle: float = 0.0

    # Whether to prefer side grasps from GraspIt database
    prefer_side_grasps: bool = False


# Registry of supported YCB objects
YCB_OBJECTS: Dict[str, YCBObjectConfig] = {
    "mustard": YCBObjectConfig(
        short_name="mustard",
        ycb_id="006_mustard_bottle",
        body_name="006_mustard_bottle",
        collision_geom="006_mustard_bottle_collision",
        visual_geom="006_mustard_bottle_visual",
        grasp_db_file="ycb_mustard_bottle.json",
        half_height=0.10,
        mass=0.052,
        inertia=(1.4e-5, 4.8e-5, 5.6e-5),
        mesh_scale=1.0,
        grasp_strategy=GraspStrategy.TOP_DOWN,
        pregrasp_fraction=0.5,
        palm_height_offset=0.09,
        prefer_side_grasps=False,
    ),
    "soup": YCBObjectConfig(
        short_name="soup",
        ycb_id="005_tomato_soup_can",
        body_name="005_tomato_soup_can",
        collision_geom="005_tomato_soup_can_collision",
        visual_geom="005_tomato_soup_can_visual",
        grasp_db_file="ycb_tomato_soup_can.json",
        half_height=0.0415,
        mass=0.09696,
        inertia=(6.68e-5, 1.25e-4, 1.25e-4),
        upright_quat=(0.707, -0.707, 0.0, 0.0),
        mesh_scale=0.82,
        grasp_strategy=GraspStrategy.SIDE_APPROACH,
        pregrasp_fraction=0.4,
        palm_height_offset=0.06,
        side_approach_angle=90.0,
        prefer_side_grasps=True,
    ),
    "tennis_ball": YCBObjectConfig(
        short_name="tennis_ball",
        ycb_id="056_tennis_ball",
        body_name="056_tennis_ball",
        collision_geom="056_tennis_ball_collision",
        visual_geom="056_tennis_ball_visual",
        grasp_db_file="ycb_tennis_ball_decimated.json",
        half_height=0.0313,
        mass=0.0518,
        inertia=(1.66e-5, 1.66e-5, 1.66e-5),
        mesh_scale=0.52,
        grasp_strategy=GraspStrategy.ENVELOPING,
        pregrasp_fraction=0.3,
        palm_height_offset=0.05,
        side_approach_angle=45.0,
        prefer_side_grasps=True,
    ),
    "potted_meat": YCBObjectConfig(
        short_name="potted_meat",
        ycb_id="010_potted_meat_can",
        body_name="010_potted_meat_can",
        collision_geom="010_potted_meat_can_collision",
        visual_geom="010_potted_meat_can_visual",
        grasp_db_file="ycb_potted_meat_can.json",
        half_height=0.040,
        mass=0.1054,
        inertia=(9.40e-5, 1.18e-4, 1.57e-4),
        mesh_scale=0.80,
        grasp_strategy=GraspStrategy.TOP_DOWN,
        pregrasp_fraction=0.7,
        palm_height_offset=0.07,
        prefer_side_grasps=False,
    ),
    # Keep cube as default
    "cube": YCBObjectConfig(
        short_name="cube",
        ycb_id="cube",
        body_name="cube",
        collision_geom="cube_collision",
        visual_geom="cube_marker",
        grasp_db_file="cube.json",
        half_height=0.025,
        mass=0.05,
        inertia=(1e-4, 1e-4, 1e-4),
        mesh_scale=1.0,
        grasp_strategy=GraspStrategy.TOP_DOWN,
        pregrasp_fraction=0.6,
        palm_height_offset=0.08,
        prefer_side_grasps=False,
    ),
    # GraspIt box primitive: MuJoCo geom size 0.03125 0.03125 0.08125
    "graspit_box": YCBObjectConfig(
        short_name="graspit_box",
        ycb_id="graspit_box",
        body_name="graspit_box",
        collision_geom="graspit_box_geom",
        visual_geom="graspit_box_geom",
        grasp_db_file="box.json",
        half_height=0.08125,  # Z half-size from geom
        mass=0.08,
        inertia=(2e-4, 1e-4, 1e-4),
        mesh_scale=1.0,
        grasp_strategy=GraspStrategy.ENVELOPING,
        pregrasp_fraction=0.4,  # Moderate pre-closure for enveloping grasp
        palm_height_offset=0.07,
        prefer_side_grasps=True,
    ),
    # GraspIt cylinder primitive: mesh from cylinder.stl
    "graspit_cylinder": YCBObjectConfig(
        short_name="graspit_cylinder",
        ycb_id="graspit_cylinder",
        body_name="graspit_cylinder",
        collision_geom="graspit_cylinder_geom",
        visual_geom="graspit_cylinder_geom",
        grasp_db_file="cylinder.json",
        half_height=0.090,  # mesh Z range [-0.09, 0.09]
        mass=0.10,
        inertia=(2e-4, 2e-4, 1e-4),
        mesh_scale=1.0,
        grasp_strategy=GraspStrategy.SIDE_APPROACH,
        pregrasp_fraction=0.4,
        palm_height_offset=0.07,
        side_approach_angle=90.0,
        prefer_side_grasps=True,
    ),
}


@dataclass
class GraspItGrasp:
    """A single grasp from the GraspIt database"""

    # Object pose in GraspIt frame ; mm, quaternion [w,x,y,z]
    object_position_mm: np.ndarray  # ; 3,
    object_orientation: np.ndarray  # ; 4, wxyz

    # Hand DOF values ; radians - 11 DOFs for RealHand L6
    hand_dof_values: np.ndarray  # ; 11,

    # Quality metrics from GraspIt
    epsilon_quality: float
    volume_quality: float
    n_contacts: int

    # Contact points in object frame ; mm
    contact_points_mm: np.ndarray  # ; n_contacts, 3
    contact_normals: np.ndarray  # ; n_contacts, 3


@dataclass
class GraspDatabase:
    """Collection of grasps for an object"""

    object_name: str
    grasps: List[GraspItGrasp]

    def best_grasp(self, metric: str = "epsilon") -> Optional[GraspItGrasp]:
        """Return the grasp with highest quality"""
        if not self.grasps:
            return None
        if metric == "epsilon":
            return max(self.grasps, key=lambda g: g.epsilon_quality)
        elif metric == "volume":
            return max(self.grasps, key=lambda g: g.volume_quality)
        elif metric == "contacts":
            return max(self.grasps, key=lambda g: g.n_contacts)
        else:
            return self.grasps[0]

    def top_k_grasps(self, k: int = 5, metric: str = "epsilon") -> List[GraspItGrasp]:
        """Return top K grasps by quality"""
        if metric == "epsilon":
            sorted_grasps = sorted(
                self.grasps, key=lambda g: g.epsilon_quality, reverse=True
            )
        elif metric == "volume":
            sorted_grasps = sorted(
                self.grasps, key=lambda g: g.volume_quality, reverse=True
            )
        else:
            sorted_grasps = sorted(
                self.grasps, key=lambda g: g.n_contacts, reverse=True
            )
        return sorted_grasps[:k]


def load_grasp_database(object_config: YCBObjectConfig) -> Optional[GraspDatabase]:
    """Load grasp database for an object from JSON file.

    Args:
        object_config: YCB object configuration

    Returns:
        GraspDatabase or None if not found
    """
    db_path = GRASP_DB_PATH / object_config.grasp_db_file

    if not db_path.exists():
        print(f"Warning: Grasp database not found: {db_path}")
        return None

    with open(db_path) as f:
        data = json.load(f)

    grasps = []
    for g in data.get("grasps", []):
        try:
            grasp = GraspItGrasp(
                object_position_mm=np.array(g["object_position_mm"]),
                object_orientation=np.array(g["object_orientation"]),
                hand_dof_values=np.array(g["hand_dof_values"]),
                epsilon_quality=g["epsilon_quality"],
                volume_quality=g["volume_quality"],
                n_contacts=g["n_contacts"],
                contact_points_mm=np.array(g.get("contact_points_mm", [])),
                contact_normals=np.array(g.get("contact_normals", [])),
            )
            grasps.append(grasp)
        except (KeyError, ValueError) as e:
            print(f"Warning: Skipping malformed grasp: {e}")
            continue

    return GraspDatabase(
        object_name=data.get("object_name", object_config.short_name), grasps=grasps
    )


class GraspItToMuJoCoTransform:
    """Transform between GraspIt and MuJoCo coordinate systems.

    GraspIt conventions:
    - Units: millimeters
    - Hand: Palm faces +Y, fingers extend along +Z
    - Object: Object frame depends on mesh origin

    MuJoCo conventions:
    - Units: meters
    - Hand: Palm faces -Z (downward), fingers curl inward
    - World: Z up, robot base at known position

    The transformation involves:
    1. Unit conversion (mm -> m)
    2. Frame rotation (GraspIt palm to MuJoCo palm)
    3. Joint remapping (if needed)

    COORDINATE FRAME ALIGNMENT:
    - GraspIt hand_base frame and MuJoCo hand_base frame share the same
      URDF origin, so no rotation is needed (empirically validated by
      comparing fingertip centroid distance for 5 candidate rotations:
      Identity = 26.5 mm, Rx(180) = 212 mm, Rz(±90) ≈ 50 mm).
    - The GraspIt `object_position_mm` is expressed in the hand_base frame;
      MuJoCo's `palm_link` has +0.07 m Z offset from hand_base (no rotation).
    """

    # GraspIt to MuJoCo rotation matrix : identity (same URDF, same frame)
    R_graspit_to_mujoco = np.eye(3)

    # Joint mapping: GraspIt DOF index --> MuJoCo hand joint index
    #
    # GraspIt RealHand L6 DOF order (from eigen.xml / scene.xml):
    #   d0:  index_mcp_pitch     [0, 1.57]
    #   d1:  index_dip           [-0.01, 1.40]
    #   d2:  middle_mcp_pitch    [0, 1.57]
    #   d3:  middle_dip          [-0.01, 1.40]
    #   d4:  pinky_mcp_pitch     [0, 1.57]
    #   d5:  pinky_dip           [-0.01, 1.40]
    #   d6:  ring_mcp_pitch      [0, 1.57]
    #   d7:  ring_dip            [-0.01, 1.40]
    #   d8:  thumb_cmc_yaw       [0, 1.54]
    #   d9:  thumb_cmc_pitch     [0, 0.52]
    #   d10: thumb_ip            [0, 0.96]
    #
    # MuJoCo RealHand L6 joint order (qpos[6:17]):
    #   mj0: thumb_cmc_yaw      [0, 1.54]
    #   mj1: thumb_cmc_pitch    [0, 0.52]
    #   mj2: thumb_ip           [0, 0.96]
    #   mj3: index_mcp_pitch    [0, 1.57]
    #   mj4: index_dip          [0, 1.40]
    #   mj5: middle_mcp_pitch   [0, 1.57]
    #   mj6: middle_dip         [0, 1.40]
    #   mj7: ring_mcp_pitch     [0, 1.57]
    #   mj8: ring_dip           [0, 1.40]
    #   mj9: pinky_mcp_pitch    [0, 1.57]
    #   mj10: pinky_dip         [0, 1.40]
    #
    # JOINT_MAPPING[mujoco_idx] = graspit_dof_idx
    JOINT_MAPPING = [
        8,   # mj0  thumb_cmc_yaw   ← d8
        9,   # mj1  thumb_cmc_pitch ← d9
        10,  # mj2  thumb_ip        ← d10
        0,   # mj3  index_mcp       ← d0
        1,   # mj4  index_dip       ← d1
        2,   # mj5  middle_mcp      ← d2
        3,   # mj6  middle_dip      ← d3
        6,   # mj7  ring_mcp        ← d6
        7,   # mj8  ring_dip        ← d7
        4,   # mj9  pinky_mcp       ← d4
        5,   # mj10 pinky_dip       ← d5
    ]

    @classmethod
    def convert_position(cls, pos_mm: np.ndarray) -> np.ndarray:
        """Convert position from GraspIt (mm) to MuJoCo (m) with frame rotation"""
        pos_m = pos_mm * MM_TO_M
        return cls.R_graspit_to_mujoco @ pos_m

    @classmethod
    def convert_quaternion(cls, quat_wxyz: np.ndarray) -> np.ndarray:
        """Convert quaternion from GraspIt to MuJoCo.

        Both use wxyz convention. Applies frame rotation.
        """
        # Extract rotation matrix from quaternion
        w, x, y, z = quat_wxyz
        R_obj = np.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * z * w,
                    2 * x * z + 2 * y * w,
                ],
                [
                    2 * x * y + 2 * z * w,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * x * w,
                ],
                [
                    2 * x * z - 2 * y * w,
                    2 * y * z + 2 * x * w,
                    1 - 2 * x * x - 2 * y * y,
                ],
            ]
        )

        # Apply frame transformation
        R_mujoco = cls.R_graspit_to_mujoco @ R_obj @ cls.R_graspit_to_mujoco.T

        # Convert back to quaternion ; wxyz
        tr = np.trace(R_mujoco)
        if tr > 0:
            s = 0.5 / np.sqrt(tr + 1.0)
            w = 0.25 / s
            x = (R_mujoco[2, 1] - R_mujoco[1, 2]) * s
            y = (R_mujoco[0, 2] - R_mujoco[2, 0]) * s
            z = (R_mujoco[1, 0] - R_mujoco[0, 1]) * s
        else:
            if R_mujoco[0, 0] > R_mujoco[1, 1] and R_mujoco[0, 0] > R_mujoco[2, 2]:
                s = 2.0 * np.sqrt(
                    1.0 + R_mujoco[0, 0] - R_mujoco[1, 1] - R_mujoco[2, 2]
                )
                w = (R_mujoco[2, 1] - R_mujoco[1, 2]) / s
                x = 0.25 * s
                y = (R_mujoco[0, 1] + R_mujoco[1, 0]) / s
                z = (R_mujoco[0, 2] + R_mujoco[2, 0]) / s
            elif R_mujoco[1, 1] > R_mujoco[2, 2]:
                s = 2.0 * np.sqrt(
                    1.0 + R_mujoco[1, 1] - R_mujoco[0, 0] - R_mujoco[2, 2]
                )
                w = (R_mujoco[0, 2] - R_mujoco[2, 0]) / s
                x = (R_mujoco[0, 1] + R_mujoco[1, 0]) / s
                y = 0.25 * s
                z = (R_mujoco[1, 2] + R_mujoco[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(
                    1.0 + R_mujoco[2, 2] - R_mujoco[0, 0] - R_mujoco[1, 1]
                )
                w = (R_mujoco[1, 0] - R_mujoco[0, 1]) / s
                x = (R_mujoco[0, 2] + R_mujoco[2, 0]) / s
                y = (R_mujoco[1, 2] + R_mujoco[2, 1]) / s
                z = 0.25 * s

        return np.array([w, x, y, z])

    @classmethod
    def convert_hand_dof(cls, dof_graspit: np.ndarray) -> np.ndarray:
        """Convert hand DOF values from GraspIt to MuJoCo.

        Applies joint remapping and any necessary sign flips.
        """
        if len(dof_graspit) != 11:
            raise ValueError(f"Expected 11 DOFs, got {len(dof_graspit)}")

        # Apply mapping
        dof_mujoco = np.zeros(11)
        for i, j in enumerate(cls.JOINT_MAPPING):
            if j < len(dof_graspit):
                dof_mujoco[i] = dof_graspit[j]

        return dof_mujoco

    @classmethod
    def grasp_to_mujoco(cls, grasp: GraspItGrasp, mesh_scale: float = 1.0) -> dict:
        """Convert a complete GraspIt grasp to MuJoCo format.

        Args:
            grasp: GraspIt grasp configuration
            mesh_scale: Scaling factor applied to the mesh in MuJoCo (affects object positions)

        Returns:
            Dictionary with keys:
            - hand_q: Joint positions for hand (11,)
            - object_pos: Object position in MuJoCo frame (3,)
            - object_quat: Object quaternion in MuJoCo frame (4,)
            - contact_points: Contact points in object frame, meters (N, 3)
        """
        # Convert positions accounting for mesh scaling
        object_pos_m = cls.convert_position(grasp.object_position_mm) * mesh_scale
        contact_points_m = grasp.contact_points_mm * MM_TO_M * mesh_scale

        return {
            "hand_q": cls.convert_hand_dof(grasp.hand_dof_values),
            "object_pos": object_pos_m,
            "object_quat": cls.convert_quaternion(grasp.object_orientation),
            "contact_points": contact_points_m,
            "epsilon_quality": grasp.epsilon_quality,
            "volume_quality": grasp.volume_quality,
            "n_contacts": grasp.n_contacts,
        }


def get_object_config(object_name: str) -> YCBObjectConfig:
    """Get configuration for an object by short name.

    Args:
        object_name: Short name (e.g., "mustard", "soup", "cube")

    Returns:
        YCBObjectConfig

    Raises:
        ValueError if object not found
    """
    if object_name not in YCB_OBJECTS:
        available = ", ".join(YCB_OBJECTS.keys())
        raise ValueError(f"Unknown object '{object_name}'. Available: {available}")
    return YCB_OBJECTS[object_name]


def filter_grasps_by_strategy(
    grasps: List[GraspItGrasp],
    strategy: GraspStrategy,
) -> List[GraspItGrasp]:
    """Filter grasp database by preferred approach strategy.

    Args:
        grasps: List of all available grasps
        strategy: Preferred grasp strategy

    Returns:
        Filtered list of grasps suitable for the strategy
    """
    if strategy == GraspStrategy.TOP_DOWN:
        filtered = []
        for grasp in grasps:
            palm_z_axis = np.array([0, 0, 1])
            palm_rot = quat_to_rotmat(grasp.object_orientation)
            palm_z_world = palm_rot @ palm_z_axis

            if palm_z_world[2] < -0.7:
                filtered.append(grasp)
        return filtered if filtered else grasps[:5]

    elif strategy in [GraspStrategy.SIDE_APPROACH, GraspStrategy.ENVELOPING]:
        filtered = []
        for grasp in grasps:
            palm_z_axis = np.array([0, 0, 1])
            palm_rot = quat_to_rotmat(grasp.object_orientation)
            palm_z_world = palm_rot @ palm_z_axis

            if abs(palm_z_world[2]) < 0.5:
                filtered.append(grasp)
        return filtered if filtered else grasps[:5]

    return grasps


def quat_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion (wxyz) to rotation matrix (3x3)"""
    w, x, y, z = [float(v) for v in quat_wxyz]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def compute_strategy_specific_palm_pose(
    object_world_pos: np.ndarray,
    object_world_quat: np.ndarray,
    config: YCBObjectConfig,
    grasp_offset: np.ndarray = None,
    grasp_quat: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute palm pose using object-specific strategy.

    Args:
        object_world_pos: Object position in world frame
        object_world_quat: Object orientation in world frame
        config: Object configuration with strategy parameters
        grasp_offset: Optional GraspIt object offset relative to palm
        grasp_quat: Optional GraspIt object orientation relative to palm

    Returns:
        (palm_world_pos, palm_world_rot) tuple
    """

    if grasp_offset is not None and config.grasp_strategy != GraspStrategy.TOP_DOWN:
        obj_world_rot = quat_to_rotmat(object_world_quat)
        if grasp_quat is not None:
            obj_palm_rot = quat_to_rotmat(grasp_quat)
        else:
            obj_palm_rot = np.eye(3)

        palm_world_rot = obj_world_rot @ obj_palm_rot.T
        palm_world_pos = object_world_pos - palm_world_rot @ grasp_offset
        return palm_world_pos, palm_world_rot

    if config.grasp_strategy == GraspStrategy.TOP_DOWN:
        palm_world_pos = object_world_pos.copy()
        palm_world_pos[2] += config.palm_height_offset
        palm_world_rot = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)

    elif config.grasp_strategy == GraspStrategy.SIDE_APPROACH:
        angle_rad = np.radians(config.side_approach_angle)
        palm_world_pos = object_world_pos.copy()

        approach_distance = config.palm_height_offset + 0.03
        palm_world_pos[0] += approach_distance * np.cos(angle_rad)
        palm_world_pos[1] += approach_distance * np.sin(angle_rad)

        palm_world_rot = np.array(
            [
                [np.cos(angle_rad + np.pi), -np.sin(angle_rad + np.pi), 0],
                [np.sin(angle_rad + np.pi), np.cos(angle_rad + np.pi), 0],
                [0, 0, -1],
            ],
            dtype=np.float64,
        )

    elif config.grasp_strategy == GraspStrategy.ENVELOPING:
        angle_rad = np.radians(config.side_approach_angle)
        palm_world_pos = object_world_pos.copy()

        approach_distance = config.palm_height_offset + 0.02
        palm_world_pos[0] += approach_distance * np.cos(angle_rad)
        palm_world_pos[1] += approach_distance * np.sin(angle_rad)
        palm_world_pos[2] += approach_distance * 0.2

        palm_world_rot = np.array(
            [
                [np.cos(angle_rad + np.pi), -np.sin(angle_rad + np.pi), 0],
                [np.sin(angle_rad + np.pi), np.cos(angle_rad + np.pi), 0],
                [0, 0, -1],
            ],
            dtype=np.float64,
        )

    else:
        palm_world_pos = object_world_pos.copy()
        palm_world_pos[2] += config.palm_height_offset
        palm_world_rot = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)

    return palm_world_pos, palm_world_rot


def get_pregrasp_hand_config(
    object_name: str,
    grasp_index: int = 0,
    metric: str = "epsilon",
) -> Optional[np.ndarray]:
    """Get pregrasp hand configuration from GraspIt database.

    Args:
        object_name: Object short name
        grasp_index: Index of grasp to use (0 = best by metric)
        metric: Quality metric to rank by ("epsilon", "volume", "contacts")

    Returns:
        Hand joint configuration (11,) or None if not available
    """
    config = get_object_config(object_name)
    db = load_grasp_database(config)

    if db is None or not db.grasps:
        print(f"No grasps available for {object_name}, using default open hand")
        return None

    if config.prefer_side_grasps and config.grasp_strategy != GraspStrategy.TOP_DOWN:
        filtered_grasps = filter_grasps_by_strategy(db.grasps, config.grasp_strategy)
        if filtered_grasps:
            db.grasps = filtered_grasps

    top_grasps = db.top_k_grasps(k=grasp_index + 1, metric=metric)
    if grasp_index >= len(top_grasps):
        grasp = top_grasps[-1]
    else:
        grasp = top_grasps[grasp_index]

    mujoco_grasp = GraspItToMuJoCoTransform.grasp_to_mujoco(grasp, config.mesh_scale)
    hand_q = mujoco_grasp["hand_q"] * config.pregrasp_fraction
    return hand_q


def get_full_grasp_config(
    object_name: str,
    grasp_index: int = 0,
    metric: str = "epsilon",
) -> Optional[dict]:
    """Get full grasp configuration from GraspIt database.

    This returns the complete grasp configuration including object pose
    relative to the palm, which is needed to properly position the object
    for the grasp.

    Args:
        object_name: Object short name
        grasp_index: Index of grasp to use (0 = best by metric)
        metric: Quality metric to rank by ("epsilon", "volume", "contacts")

    Returns:
        Dictionary with:
        - hand_q: Hand joint configuration (11,)
        - object_pos: Object position relative to palm in MuJoCo frame (m)
        - object_quat: Object orientation relative to palm (wxyz)
        - epsilon_quality: GWS epsilon quality
        - n_contacts: Number of contacts in the grasp

        Returns None if no grasps available.
    """
    config = get_object_config(object_name)
    db = load_grasp_database(config)

    if db is None or not db.grasps:
        return None

    if config.prefer_side_grasps and config.grasp_strategy != GraspStrategy.TOP_DOWN:
        filtered_grasps = filter_grasps_by_strategy(db.grasps, config.grasp_strategy)
        if filtered_grasps:
            db.grasps = filtered_grasps

    top_grasps = db.top_k_grasps(k=grasp_index + 1, metric=metric)
    if grasp_index >= len(top_grasps):
        grasp = top_grasps[-1]
    else:
        grasp = top_grasps[grasp_index]

    return GraspItToMuJoCoTransform.grasp_to_mujoco(grasp, config.mesh_scale)


def list_available_objects() -> List[str]:
    """Return list of available object names"""
    return list(YCB_OBJECTS.keys())


# CLI helper
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YCB Object Utilities")
    parser.add_argument("--list", action="store_true", help="List available objects")
    parser.add_argument("--show", type=str, help="Show config for an object")
    parser.add_argument("--grasps", type=str, help="Show top grasps for an object")
    args = parser.parse_args()

    if args.list:
        print("Available objects:")
        for name, cfg in YCB_OBJECTS.items():
            print(f"  {name}: {cfg.ycb_id} ({cfg.body_name})")

    if args.show:
        cfg = get_object_config(args.show)
        print(f"\nConfiguration for '{args.show}':")
        print(f"  YCB ID: {cfg.ycb_id}")
        print(f"  Body name: {cfg.body_name}")
        print(f"  Collision geom: {cfg.collision_geom}")
        print(f"  Half-height: {cfg.half_height}m")
        print(f"  Mass: {cfg.mass}kg")

    if args.grasps:
        cfg = get_object_config(args.grasps)
        db = load_grasp_database(cfg)
        if db:
            print(f"\nTop 5 grasps for '{args.grasps}':")
            for i, g in enumerate(db.top_k_grasps(5)):
                print(
                    f"  {i + 1}. epsilon={g.epsilon_quality:.4f}, V={g.volume_quality:.4f}, n={g.n_contacts}"
                )
                mj_grasp = GraspItToMuJoCoTransform.grasp_to_mujoco(g, cfg.mesh_scale)
                print(f"      hand_q: {mj_grasp['hand_q']}")
