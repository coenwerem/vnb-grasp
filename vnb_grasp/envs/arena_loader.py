"""Arena loader with object pose randomization.

Provides utilities for loading MuJoCo arena XML files and spawning
objects with configurable pose randomization.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import mujoco as mj
import numpy as np


def _repo_root() -> str:
    """Get vnb_grasp repository root"""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(this_dir, "..", ".."))


def list_available_arenas() -> List[str]:
    """List available arena names under arenas/ directory.

    Returns:
        List of arena directory names that contain scene.xml.
    """
    arenas_dir = os.path.join(_repo_root(), "arenas")
    if not os.path.isdir(arenas_dir):
        return []

    arenas = []
    for name in os.listdir(arenas_dir):
        scene_path = os.path.join(arenas_dir, name, "scene.xml")
        if os.path.isfile(scene_path):
            arenas.append(name)
    return sorted(arenas)


@dataclass
class ObjectSpawnConfig:
    """Configuration for spawning a free-body object"""

    body_name: str
    position_range: Tuple[
        Tuple[float, float], Tuple[float, float], Tuple[float, float]
    ] = (
        (-0.1, 0.1),  # x range
        (-0.1, 0.1),  # y range
        (0.0, 0.0),  # z range 
    )
    yaw_range: Tuple[float, float] = (0.0, 360.0)  # degrees
    fixed_position: Optional[Tuple[float, float, float]] = None
    fixed_orientation: Optional[Tuple[float, float, float, float]] = None  # quat wxyz


@dataclass
class ArenaConfig:
    """Configuration for arena loading and object spawning"""

    arena_name: Optional[str] = None
    arena_path: Optional[str] = None

    keyframe: Optional[str] = "home"

    object_body_name: str = "cube"
    spawn_object: bool = True

    randomize_position: bool = False
    randomize_yaw: bool = False

    x_range: Tuple[float, float] = (-0.05, 0.05)
    y_range: Tuple[float, float] = (-0.05, 0.05)
    object_z_offset: float = 0.025
    yaw_range: Tuple[float, float] = (0.0, 360.0)

    # Spawn region on workspace table  --  derived from model geometry at load time.
    # spawn_y_fraction: 0.0 = near edge ; closest to robot, 1.0 = far edge.
    # Default 0.50 places the object near the table center, giving the arm
    # more manoeuvring room than the old 0.30 which was too close to the base.
    workspace_table_body_name: str = "workspace_table"
    workspace_table_collision_geom_name: str = "workspace_table_collision"
    spawn_y_fraction: float = 0.50
    spawn_x_fraction: float = 0.5

    # Fallback if workspace table not found in model  
    default_position: Tuple[float, float, float] = (0.0, 1.0, 0.81)
    default_quat_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    object_configs: List[ObjectSpawnConfig] = field(default_factory=list)


def _collect_includes(xml_text: str) -> List[str]:
    """Extract <include file="..."> paths from XML"""
    root = ET.fromstring(xml_text)
    includes = []
    for elem in root.iter("include"):
        f = elem.get("file")
        if f:
            includes.append(f)
    return includes


def _build_assets_for_scene(scene_xml_path: str) -> Tuple[str, Dict[str, bytes]]:
    """Build assets dict resolving all transitive <include> files.

    Args:
        scene_xml_path: Path to the main scene XML.

    Returns:
        (scene_xml_text, assets_dict)
    """
    assets: Dict[str, bytes] = {}
    seen: set = set()

    def _read_text(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _visit(rel_path: str, base_dir: str):
        if rel_path in seen:
            return
        seen.add(rel_path)
        abs_path = os.path.abspath(os.path.join(base_dir, rel_path))
        txt = _read_text(abs_path)
        assets[rel_path] = txt.encode("utf-8")

        sub_base = os.path.dirname(abs_path)
        for sub in _collect_includes(txt):
            _visit(sub, sub_base)

    scene_dir = os.path.dirname(os.path.abspath(scene_xml_path))
    scene_txt = _read_text(scene_xml_path)
    for rel in _collect_includes(scene_txt):
        _visit(rel, scene_dir)

    return scene_txt, assets


def load_arena_model(config: ArenaConfig) -> Tuple[mj.MjModel, mj.MjData]:
    """Load MuJoCo model from arena configuration.

    Args:
        config: Arena configuration.

    Returns:
        (model, data) tuple.

    Raises:
        FileNotFoundError: If arena cannot be resolved.
    """
    # Resolve XML path
    xml_path = None

    if config.arena_path:
        # Direct path
        p = os.path.expanduser(config.arena_path)
        if os.path.isdir(p):
            p = os.path.join(p, "scene.xml")
        if os.path.isfile(p):
            xml_path = os.path.abspath(p)

    if xml_path is None and config.arena_name:
        # Look in arenas/ folder
        repo_root = _repo_root()
        candidates = [
            os.path.join(repo_root, "arenas", config.arena_name, "scene.xml"),
            os.path.join("arenas", config.arena_name, "scene.xml"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                xml_path = os.path.abspath(c)
                break

    if xml_path is None:
        available = list_available_arenas()
        raise FileNotFoundError(
            f"Could not resolve arena from config. "
            f"arena_name={config.arena_name!r}, arena_path={config.arena_path!r}. "
            f"Available arenas: {available}"
        )

    # Load model
    try:
        model = mj.MjModel.from_xml_path(xml_path)
    except Exception as e:
        # Fallback with include resolution
        scene_txt, assets = _build_assets_for_scene(xml_path)
        scene_dir = os.path.dirname(xml_path)
        old_cwd = os.getcwd()
        try:
            os.chdir(scene_dir)
            model = mj.MjModel.from_xml_string(scene_txt, assets=assets)
        finally:
            os.chdir(old_cwd)
        print(f"[arena_loader] from_xml_path failed ({e}); used fallback")

    data = mj.MjData(model)

    # Apply keyframe if specified
    if config.keyframe is not None and model.nkey > 0:
        key_id = None
        for i in range(model.nkey):
            names = model.names
            adr = model.name_keyadr[i]
            kname = names[adr:].split(b"\x00", 1)[0].decode("utf-8", "ignore")
            if kname == config.keyframe:
                key_id = i
                break
        if key_id is not None:
            mj.mj_resetDataKeyframe(model, data, key_id)
            mj.mj_forward(model, data)

    return model, data


class ArenaLoader:
    """Arena loader with object pose randomization.

    Example:
        >>> loader = ArenaLoader(ArenaConfig(
        ...     arena_name="zarm_realhand_l6_right_arena",
        ...     randomize_position=True,
        ...     randomize_yaw=True,
        ... ))
        >>> model, data = loader.load()
        >>> loader.reset_object_pose(data)  # Randomizes if configured
    """

    def __init__(
        self,
        config: Optional[ArenaConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.config = config or ArenaConfig()
        self.rng = rng or np.random.default_rng()

        self._model: Optional[mj.MjModel] = None
        self._data: Optional[mj.MjData] = None
        self._object_body_id: int = -1
        self._object_qpos_adr: int = -1
        self._spawn_center: Optional[np.ndarray] = None
        self._table_surface_z: Optional[float] = None

    def load(self) -> Tuple[mj.MjModel, mj.MjData]:
        """Load arena model and data.

        Returns:
            (model, data) tuple.
        """
        model, data = load_arena_model(self.config)
        self._model = model
        self._data = data

        # Find object body
        body_name = self.config.object_body_name
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        if bid >= 0:
            self._object_body_id = bid
            jid = model.body_jntadr[bid]
            if jid >= 0 and model.jnt_type[jid] == mj.mjtJoint.mjJNT_FREE:
                self._object_qpos_adr = model.jnt_qposadr[jid]
            else:
                for j in range(model.njnt):
                    if (
                        model.jnt_bodyid[j] == bid
                        and model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE
                    ):
                        self._object_qpos_adr = model.jnt_qposadr[j]
                        break

        self._compute_spawn_center(model)

        return model, data

    def _compute_spawn_center(self, model: mj.MjModel) -> None:
        cfg = self.config
        table_body_id = mj.mj_name2id(
            model, mj.mjtObj.mjOBJ_BODY, cfg.workspace_table_body_name
        )
        table_geom_id = mj.mj_name2id(
            model, mj.mjtObj.mjOBJ_GEOM, cfg.workspace_table_collision_geom_name
        )
        if table_body_id < 0 or table_geom_id < 0:
            self._spawn_center = np.array(cfg.default_position, dtype=np.float64)
            self._table_surface_z = cfg.default_position[2] - cfg.object_z_offset
            return

        body_pos = model.body_pos[table_body_id]
        geom_pos = model.geom_pos[table_geom_id]
        geom_size = model.geom_size[table_geom_id]

        # World-frame position of the collision geom center
        geom_world_x = body_pos[0] + geom_pos[0]
        geom_world_y = body_pos[1] + geom_pos[1]
        geom_world_z = body_pos[2] + geom_pos[2]

        table_half_x = geom_size[0]
        table_half_y = geom_size[1]
        table_half_z = geom_size[2]

        self._table_surface_z = geom_world_z + table_half_z

        near_y = geom_world_y - table_half_y
        far_y = geom_world_y + table_half_y
        left_x = geom_world_x - table_half_x
        right_x = geom_world_x + table_half_x

        spawn_x = left_x + (right_x - left_x) * cfg.spawn_x_fraction
        spawn_y = near_y + (far_y - near_y) * cfg.spawn_y_fraction
        spawn_z = self._table_surface_z + cfg.object_z_offset

        self._spawn_center = np.array([spawn_x, spawn_y, spawn_z], dtype=np.float64)

    @property
    def model(self) -> Optional[mj.MjModel]:
        return self._model

    @property
    def data(self) -> Optional[mj.MjData]:
        return self._data

    @property
    def spawn_center(self) -> Optional[np.ndarray]:
        return self._spawn_center

    @property
    def table_surface_z(self) -> Optional[float]:
        return self._table_surface_z

    def sample_object_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample random object pose based on config.

        Returns:
            (position, quat_wxyz) arrays.
        """
        cfg = self.config
        center = self._spawn_center
        if center is None:
            center = np.array(cfg.default_position, dtype=np.float64)

        if cfg.randomize_position:
            dx = self.rng.uniform(cfg.x_range[0], cfg.x_range[1])
            dy = self.rng.uniform(cfg.y_range[0], cfg.y_range[1])
            pos = center.copy()
            pos[0] += dx
            pos[1] += dy
        else:
            pos = center.copy()

        # Orientation
        if cfg.randomize_yaw:
            yaw_deg = self.rng.uniform(cfg.yaw_range[0], cfg.yaw_range[1])
            yaw_rad = np.deg2rad(yaw_deg)
            # Quaternion for Z rotation: [cos; theta/2, 0, 0, sin; theta/2]
            quat = np.array(
                [
                    np.cos(yaw_rad / 2),
                    0.0,
                    0.0,
                    np.sin(yaw_rad / 2),
                ],
                dtype=np.float64,
            )
        else:
            quat = np.array(cfg.default_quat_wxyz, dtype=np.float64)

        return pos, quat

    def reset_object_pose(
        self,
        data: Optional[mj.MjData] = None,
        position: Optional[np.ndarray] = None,
        quat_wxyz: Optional[np.ndarray] = None,
    ) -> None:
        """Reset object to specified or sampled pose.

        Args:
            data: MjData to modify. Uses self._data if None.
            position: Position XYZ. Samples if None and randomize enabled.
            quat_wxyz: Quaternion WXYZ. Samples if None and randomize enabled.
        """
        data = data or self._data
        if data is None or self._object_qpos_adr < 0:
            return

        if position is None or quat_wxyz is None:
            sampled_pos, sampled_quat = self.sample_object_pose()
            if position is None:
                position = sampled_pos
            if quat_wxyz is None:
                quat_wxyz = sampled_quat

        adr = self._object_qpos_adr
        data.qpos[adr : adr + 3] = position
        data.qpos[adr + 3 : adr + 7] = quat_wxyz

        # Zero velocity
        vadr = self._model.jnt_dofadr[self._model.body_jntadr[self._object_body_id]]
        data.qvel[vadr : vadr + 6] = 0.0

        # Forward kinematics to update body positions
        mj.mj_forward(self._model, data)

    def get_object_pose(
        self, data: Optional[mj.MjData] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get current object pose.

        Returns:
            (position, quat_wxyz) arrays.
        """
        data = data or self._data
        if data is None or self._object_qpos_adr < 0:
            return np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64)

        adr = self._object_qpos_adr
        pos = data.qpos[adr : adr + 3].copy()
        quat = data.qpos[adr + 3 : adr + 7].copy()
        return pos, quat
