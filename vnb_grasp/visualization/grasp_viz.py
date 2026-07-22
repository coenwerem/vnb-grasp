"""Clean grasp visualization module for VNB-Grasp.

Provides publication-quality visualization of grasps with:
- Contact point markers (spheres)
- Contact normal arrows  
- Friction cone visualization
- Semi-transparent object rendering
- Clean camera presets
- Grasp wrench space visualization

Example:
    >>> from vnb_grasp.visualization.grasp_viz import GraspVisualizer
    >>> 
    >>> viz = GraspVisualizer(model, data)
    >>> viz.set_grasp(grasp)
    >>> viz.show()  # Interactive viewer
    >>> viz.render_to_file("grasp.png", camera="front")

Author: Clinton Enwerem
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union
import math
from vnb_grasp.grasping.grasp_sampler import SampledGrasp

import numpy as np
from numpy.typing import NDArray

try:
    import mujoco as mj
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


# ###
# Color Schemes
# ###

@dataclass
class ColorScheme:
    """Color scheme for grasp visualization.
    
    All colors are RGBA floats in [0, 1].
    """
    # Object colors
    object_color: Tuple[float, ...] = (0.65, 0.65, 0.65, 1.0)  # Solid gray
    object_collision: Tuple[float, ...] = (0.7, 0.7, 0.7, 1.0)  # Slightly lighter gray
    
    # Hand colors: ignored when use_original_hand_colors=True (default)
    hand_color: Tuple[float, ...] = (0.3, 0.45, 0.65, 0.95)  # Muted blue (fallback)
    fingertip_color: Tuple[float, ...] = (0.25, 0.55, 0.85, 1.0)  # Brighter blue (fallback)
    
    # Contact visualization
    contact_point: Tuple[float, ...] = (0.95, 0.35, 0.25, 1.0)  # Red-orange
    contact_normal: Tuple[float, ...] = (0.2, 0.75, 0.3, 1.0)  # Green arrow
    target_point: Tuple[float, ...] = (0.9, 0.75, 0.1, 0.9)  # Yellow target
    
    # Friction cone
    cone_fill: Tuple[float, ...] = (0.4, 0.7, 0.9, 0.25)  # Light blue, transparent
    cone_edge: Tuple[float, ...] = (0.3, 0.5, 0.8, 0.6)  # Slightly more opaque edges
    
    # Force/wrench visualization
    net_force: Tuple[float, ...] = (0.9, 0.1, 0.3, 1.0)  # Red
    net_torque: Tuple[float, ...] = (0.1, 0.3, 0.9, 1.0)  # Blue
    
    # Background/scene
    background: Tuple[float, ...] = (1.0, 1.0, 1.0)  # Pure white
    grid_color: Tuple[float, ...] = (0.85, 0.85, 0.85, 0.5)  # Subtle grid
    shadow_color: Tuple[float, ...] = (0.7, 0.7, 0.7, 0.3)  # Soft shadow


# Preset color schemes
SCHEMES = {
    "default": ColorScheme(),
    "dark": ColorScheme(
        object_color=(0.45, 0.45, 0.48, 1.0),
        background=(0.12, 0.12, 0.15),
        contact_point=(1.0, 0.4, 0.2, 1.0),
    ),
    "paper": ColorScheme(
        # High contrast for paper figures: solid gray object, white background
        object_color=(0.65, 0.65, 0.65, 1.0),
        contact_point=(0.9, 0.2, 0.15, 1.0),
        contact_normal=(0.1, 0.7, 0.2, 1.0),
        background=(1.0, 1.0, 1.0),
    ),
    "minimal": ColorScheme(
        # Grayscale except contacts
        object_color=(0.6, 0.6, 0.6, 1.0),
        contact_point=(0.9, 0.3, 0.2, 1.0),
        contact_normal=(0.2, 0.65, 0.3, 1.0),
    ),
    "natural": ColorScheme(
        # Solid gray object, original hand mesh colors
        object_color=(0.65, 0.65, 0.65, 1.0),
        contact_point=(0.9, 0.2, 0.15, 1.0),
        contact_normal=(0.1, 0.7, 0.2, 1.0),
        background=(1.0, 1.0, 1.0),
    ),
}


# ###
# Camera Presets
# ###

@dataclass
class CameraPreset:
    """Camera preset for consistent visualization angles"""
    name: str
    azimuth: float  # degrees
    elevation: float  # degrees
    distance: float  # meters
    lookat: Optional[Tuple[float, float, float]] = None  # None = auto-detect
    
    def apply(self, camera: "mj.MjvCamera", scene_center: Optional[Tuple[float, float, float]] = None) -> None:
        """Apply preset to MuJoCo camera.
        
        Args:
            camera: MjvCamera to configure
            scene_center: Auto-detected scene center used when lookat is None
        """
        camera.type = mj.mjtCamera.mjCAMERA_FREE
        camera.azimuth = self.azimuth
        camera.elevation = self.elevation
        camera.distance = self.distance
        if self.lookat is not None:
            camera.lookat[:] = self.lookat
        elif scene_center is not None:
            camera.lookat[:] = scene_center
        else:
            camera.lookat[:] = (0.0, 0.0, 0.0)


CAMERA_PRESETS = {
    "front": CameraPreset("front", azimuth=180.0, elevation=-18.0, distance=0.28),
    "side": CameraPreset("side", azimuth=90.0, elevation=-12.0, distance=0.28),
    "iso": CameraPreset("iso", azimuth=145.0, elevation=-22.0, distance=0.32),
    "top": CameraPreset("top", azimuth=180.0, elevation=-85.0, distance=0.22),
    "three_quarter": CameraPreset("three_quarter", azimuth=160.0, elevation=-20.0, distance=0.30),
    "close": CameraPreset("close", azimuth=155.0, elevation=-15.0, distance=0.20),
}


# ###
# Geometry Builders
# ###

def _create_sphere_geom(
    scene: "mj.MjvScene",
    pos: NDArray,
    radius: float,
    rgba: Tuple[float, ...],
) -> bool:
    """Add a sphere visualization geom to scene.
    
    Returns True if added successfully, False if scene is full.
    """
    if scene.ngeom >= scene.maxgeom:
        return False
    
    idx = scene.ngeom
    geom = scene.geoms[idx]
    
    geom.type = mj.mjtGeom.mjGEOM_SPHERE
    geom.size[:] = [radius, 0, 0]
    geom.pos[:] = pos
    geom.mat[:] = np.eye(3)
    geom.rgba[:] = rgba
    geom.dataid = -1
    geom.objtype = mj.mjtObj.mjOBJ_UNKNOWN
    geom.objid = -1
    geom.category = mj.mjtCatBit.mjCAT_DECOR
    geom.segid = -1
    
    scene.ngeom += 1
    return True


def _create_arrow_geom(
    scene: "mj.MjvScene",
    start: NDArray,
    direction: NDArray,
    length: float,
    radius: float,
    rgba: Tuple[float, ...],
) -> bool:
    """Add an arrow (cylinder + cone head) to scene.
    
    The arrow points from `start` in `direction` with given length.
    Returns True if added successfully.
    """
    if scene.ngeom >= scene.maxgeom - 1:  # Need 2 geoms
        return False
    
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return False
    direction = direction / norm
    
    # Shaft (cylinder)
    shaft_length = length * 0.75
    shaft_center = start + direction * (shaft_length / 2)
    
    # Build rotation matrix: z-axis aligned with direction
    z_axis = direction
    # Choose orthogonal x-axis
    if abs(z_axis[2]) < 0.9:
        x_axis = np.cross(z_axis, np.array([0, 0, 1]))
    else:
        x_axis = np.cross(z_axis, np.array([0, 1, 0]))
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot_mat = np.column_stack([x_axis, y_axis, z_axis])
    
    # Cylinder (shaft)
    idx = scene.ngeom
    geom = scene.geoms[idx]
    geom.type = mj.mjtGeom.mjGEOM_CYLINDER
    geom.size[:] = [radius, shaft_length / 2, 0]
    geom.pos[:] = shaft_center
    geom.mat[:] = rot_mat
    geom.rgba[:] = rgba
    geom.dataid = -1
    geom.objtype = mj.mjtObj.mjOBJ_UNKNOWN
    geom.objid = -1
    geom.category = mj.mjtCatBit.mjCAT_DECOR
    geom.segid = -1
    scene.ngeom += 1
    
    # Cone (head)
    head_length = length * 0.25
    head_center = start + direction * (shaft_length + head_length / 2)
    head_radius = radius * 2.0
    
    idx = scene.ngeom
    geom = scene.geoms[idx]
    # MuJoCo doesn't have a cone, use a capsule stretched small or sphere
    # Actually we can fake with a small cylinder. For better visuals, just use cylinder.
    geom.type = mj.mjtGeom.mjGEOM_CYLINDER
    geom.size[:] = [head_radius, head_length / 2, 0]
    geom.pos[:] = head_center
    geom.mat[:] = rot_mat
    geom.rgba[:] = rgba
    geom.dataid = -1
    geom.objtype = mj.mjtObj.mjOBJ_UNKNOWN
    geom.objid = -1
    geom.category = mj.mjtCatBit.mjCAT_DECOR
    geom.segid = -1
    scene.ngeom += 1
    
    return True


def _create_friction_cone_geoms(
    scene: "mj.MjvScene",
    contact_pos: NDArray,
    normal: NDArray,
    friction_coef: float,
    rgba_fill: Tuple[float, ...],
    rgba_edge: Tuple[float, ...],
    n_segments: int = 8,
    cone_height: float = 0.02,
) -> int:
    """Create friction cone visualization at contact point.
    
    Returns number of geoms added.
    """
    normal = np.asarray(normal, dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return 0
    normal = normal / norm
    
    # Cone opening angle from friction coefficient
    cone_angle = math.atan(friction_coef)
    cone_radius = cone_height * math.tan(cone_angle)
    
    # Build rotation: z aligned with normal
    z_axis = normal
    if abs(z_axis[2]) < 0.9:
        x_axis = np.cross(z_axis, np.array([0, 0, 1]))
    else:
        x_axis = np.cross(z_axis, np.array([0, 1, 0]))
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    # Draw cone edges as thin cylinders
    n_added = 0
    for i in range(n_segments):
        if scene.ngeom >= scene.maxgeom:
            break
            
        theta = 2 * math.pi * i / n_segments
        # Direction on cone surface
        edge_dir = (
            math.sin(cone_angle) * (math.cos(theta) * x_axis + math.sin(theta) * y_axis)
            + math.cos(cone_angle) * z_axis
        )
        edge_end = contact_pos + edge_dir * cone_height
        
        idx = scene.ngeom
        geom = scene.geoms[idx]
        geom.type = mj.mjtGeom.mjGEOM_CAPSULE
        
        # Capsule from contact_pos to edge_end
        midpoint = (contact_pos + edge_end) / 2
        length = np.linalg.norm(edge_end - contact_pos)
        
        # Rotation for capsule
        cap_dir = (edge_end - contact_pos) / length
        if abs(cap_dir[2]) < 0.999:
            cap_x = np.cross(cap_dir, np.array([0, 0, 1]))
        else:
            cap_x = np.cross(cap_dir, np.array([0, 1, 0]))
        cap_x = cap_x / np.linalg.norm(cap_x)
        cap_y = np.cross(cap_dir, cap_x)
        cap_rot = np.column_stack([cap_x, cap_y, cap_dir])
        
        geom.size[:] = [0.0005, length / 2, 0]  # Very thin
        geom.pos[:] = midpoint
        geom.mat[:] = cap_rot
        geom.rgba[:] = rgba_edge
        geom.dataid = -1
        geom.objtype = mj.mjtObj.mjOBJ_UNKNOWN
        geom.objid = -1
        geom.category = mj.mjtCatBit.mjCAT_DECOR
        geom.segid = -1
        
        scene.ngeom += 1
        n_added += 1
    
    return n_added


# ###
# Main GraspVisualizer Class
# ###

@dataclass
class ContactVizData:
    """Data for visualizing a single contact"""
    position: NDArray
    normal: NDArray
    finger: str = ""
    is_target: bool = False  # If True, this is target vs achieved contact


class GraspVisualizer:
    """Publication-quality grasp visualization.
    
    Creates visualizations with contact points,
    normals, friction cones, and semi-transparent objects.
    
    Camera lookat is auto-detected from the scene geometry
    (hand + object positions) so presets work across different
    arena layouts without manual tuning.
    """
    
    def __init__(
        self,
        model: "mj.MjModel",
        data: "mj.MjData",
        scheme: Union[str, ColorScheme] = "default",
        maxgeom: int = 5000,
        use_original_hand_colors: bool = True,
    ):
        """Initialize visualizer.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            scheme: Color scheme name or ColorScheme instance
            maxgeom: Maximum visualization geometries
            use_original_hand_colors: If True, keep the hand's original mesh
                materials instead of overriding them to the scheme color.
        """
        if not HAS_MUJOCO:
            raise ImportError("mujoco is required")
        
        self.model = model
        self.data = data
        self.maxgeom = maxgeom
        self._use_original_hand_colors = use_original_hand_colors
        
        # Color scheme
        if isinstance(scheme, str):
            self.scheme = SCHEMES.get(scheme, SCHEMES["default"])
        else:
            self.scheme = scheme
        
        # Visualization state
        self._contacts: List[ContactVizData] = []
        self._friction_coef: float = 0.5
        self._show_friction_cones: bool = True
        self._show_normals: bool = True
        self._show_targets: bool = True
        self._arrow_length: float = 0.015
        self._contact_radius: float = 0.0025
        self._object_alpha: float = 1.0  # Solid opaque object by default
        
        # Rendering context (lazy init)
        self._opt: Optional["mj.MjvOption"] = None
        self._scene: Optional["mj.MjvScene"] = None
        self._camera: Optional["mj.MjvCamera"] = None
        self._renderer = None  # mj.Renderer
        self._render_width: int = 0
        self._render_height: int = 0
        
        # Auto-detected scene center (computed once, updated on set_grasp)
        self._scene_center: Optional[Tuple[float, float, float]] = None
        
        # Apply clean rendering style to model
        self._apply_clean_style()
        
    def __del__(self) -> None:
        """Clean up GL resources"""
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
    
    def _apply_clean_style(self) -> None:
        """        
        Modifies model visual parameters for publication-quality output:
        - Soft headlight for uniform lighting
        - Object made semi-transparent so contacts/penetration visible
        - Floor/table moved to hidden geom group (excluded from both RGB + depth)
        """
        # Headlight for clean uniform lighting: high ambient prevents harsh
        # pitch-black shadows that can be mistaken for artifacts
        self.model.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
        self.model.vis.headlight.diffuse[:] = [0.5, 0.5, 0.5]
        self.model.vis.headlight.specular[:] = [0.1, 0.1, 0.1]
        
        # Move floor, table, marker, and hand collision geoms to group 4 (hidden).
        # This properly excludes them from BOTH RGB and depth rendering.
        # Hand collision geoms (named *_collision*) add no visual value and can
        # cause black pixel artifacts even with alpha=0.
        hide_patterns = ["floor", "table", "ground", "plane", "marker"]
        for i in range(self.model.ngeom):
            geom_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, i) or ""
            body_id = self.model.geom_bodyid[i]
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body_id) or ""
            combined = (geom_name + " " + body_name).lower()
            
            # Hide floor/table/marker
            if any(p in combined for p in hide_patterns):
                self.model.geom_group[i] = 4
                continue
            
            # Hide hand collision geoms (not object collision geoms)
            # These are invisible or semi-transparent debug markers (alpha<0.5)
            # and can cause visual noise or depth artifacts
            if "collision" in geom_name.lower() and self.model.geom_rgba[i, 3] < 0.5:
                self.model.geom_group[i] = 4
        
        # Override hand materials to scheme colors (unless user wants originals).
        if not self._use_original_hand_colors:
            hand_color = self.scheme.hand_color
            for i in range(self.model.nmat):
                mat_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_MATERIAL, i) or ""
                mat_lower = mat_name.lower()
                if "palm" in mat_lower or "base" in mat_lower:
                    self.model.mat_rgba[i, :] = hand_color[:4]
                elif "visual" in mat_lower and "collision" not in mat_lower:
                    self.model.mat_rgba[i, :] = hand_color[:4]
        
        # Make object geoms semi-transparent so contacts are visible through them
        self._make_object_transparent()
    
    def _make_object_transparent(self) -> None:
        """Set object geom colors to scheme object_color.
        
        Finds geoms attached to object bodies (not hand, not world)
        and sets their RGBA to scheme.object_color.
        """
        hand_patterns = [
            "thumb", "index", "middle", "ring", "pinky", "little",
            "finger", "palm", "hand", "eef", "link", "base",
        ]
        world_patterns = ["floor", "table", "ground", "plane"]
        
        obj_rgba = self.scheme.object_color
        
        for i in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[i]
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body_id) or ""
            geom_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, i) or ""
            name_lower = (body_name + " " + geom_name).lower()
            
            # Skip hand / world geoms
            is_hand = any(p in name_lower for p in hand_patterns)
            is_world = body_id == 0 or any(p in name_lower for p in world_patterns)
            if is_hand or is_world:
                continue
            
            # Skip invisible collision geoms (alpha already 0)
            if self.model.geom_rgba[i, 3] < 0.01:
                continue
            
            # This is an object geom: set to scheme color
            self.model.geom_rgba[i, :] = obj_rgba[:4]
            # Also override material if one is assigned
            mat_id = self.model.geom_matid[i]
            if mat_id >= 0:
                self.model.mat_rgba[mat_id, :] = obj_rgba[:4]
        
    def _compute_scene_center(self) -> Tuple[float, float, float]:
        """Compute the center of the grasp scene from body positions.
        
        Uses fingertip sites/bodies + object body to find the centroid
        of the manipulation workspace. Falls back to scanning all
        non-world bodies if no hand bodies are identified.
        
        Returns:
            (x, y, z) center of the scene
        """
        mj.mj_forward(self.model, self.data)
        
        positions = []
        
        # Collect positions from known hand-related body name patterns
        hand_patterns = [
            "thumb", "index", "middle", "ring", "pinky", "little",
            "finger", "palm", "hand", "tip",
        ]
        object_patterns = [
            "cube", "box", "cylinder", "sphere", "ball",
            "bottle", "can", "object", "ycb",
        ]
        
        for i in range(1, self.model.nbody):  # Skip world body (0)
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            if name is None:
                continue
            name_lower = name.lower()
            is_relevant = any(p in name_lower for p in hand_patterns + object_patterns)
            if is_relevant:
                positions.append(self.data.xpos[i].copy())
        
        # Fallback: if we found nothing specific, use all non-world bodies
        # within a reasonable bounding box
        if len(positions) < 2:
            for i in range(1, self.model.nbody):
                positions.append(self.data.xpos[i].copy())
        
        if not positions:
            return (0.0, 0.0, 0.0)
        
        center = np.mean(positions, axis=0)
        return tuple(float(c) for c in center)
        
    def _ensure_render_context(self) -> None:
        """Initialize rendering context if needed"""
        if self._opt is None:
            self._opt = mj.MjvOption()
            # Configure visualization options
            self._opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = False  # We draw our own
            self._opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = False
            self._opt.flags[mj.mjtVisFlag.mjVIS_CONVEXHULL] = False
            self._opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = False
            self._opt.flags[mj.mjtVisFlag.mjVIS_ACTUATOR] = False
            self._opt.flags[mj.mjtVisFlag.mjVIS_COM] = False
            self._opt.flags[mj.mjtVisFlag.mjVIS_CONSTRAINT] = False
            self._opt.flags[mj.mjtVisFlag.mjVIS_ACTIVATION] = False
            self._opt.flags[mj.mjtVisFlag.mjVIS_SELECT] = False
            
        if self._scene is None:
            self._scene = mj.MjvScene(self.model, self.maxgeom)
            
        if self._camera is None:
            self._camera = mj.MjvCamera()
            # Auto-detect scene center
            if self._scene_center is None:
                self._scene_center = self._compute_scene_center()
            CAMERA_PRESETS["iso"].apply(self._camera, self._scene_center)
    
    def set_contacts(
        self,
        positions: NDArray,
        normals: NDArray,
        fingers: Optional[List[str]] = None,
        is_targets: Optional[List[bool]] = None,
    ) -> "GraspVisualizer":
        """Set contact points for visualization.
        
        Args:
            positions: (N, 3) array of contact positions
            normals: (N, 3) array of contact normals (pointing into object)
            fingers: Optional finger names for each contact
            is_targets: Whether each contact is target (vs achieved)
            
        Returns:
            self for chaining
        """
        positions = np.atleast_2d(positions)
        normals = np.atleast_2d(normals)
        n = len(positions)
        
        if fingers is None:
            fingers = [""] * n
        if is_targets is None:
            is_targets = [False] * n
            
        self._contacts = [
            ContactVizData(
                position=positions[i],
                normal=normals[i],
                finger=fingers[i] if i < len(fingers) else "",
                is_target=is_targets[i] if i < len(is_targets) else False,
            )
            for i in range(n)
        ]
        return self
    
    def set_grasp(self, grasp: "SampledGrasp") -> "GraspVisualizer":
        """Set grasp from a SampledGrasp result.
        
        Args:
            grasp: SampledGrasp from grasp_sampler
            
        Returns:
            self for chaining
        """
        # Apply grasp qpos
        self.data.qpos[:len(grasp.hand_qpos)] = grasp.hand_qpos
        mj.mj_forward(self.model, self.data)
        
        # Recompute scene center now that hand is in grasp pose
        self._scene_center = self._compute_scene_center()
        # Update camera if already initialized
        if self._camera is not None:
            self._camera.lookat[:] = self._scene_center
        
        # Extract contacts
        positions = []
        normals = []
        fingers = []
        is_targets = []
        
        # Target contacts
        for finger, pos in grasp.target_contacts.items():
            positions.append(pos)
            normal = grasp.target_normals.get(finger, np.array([0, 0, 1]))
            normals.append(normal)
            fingers.append(finger)
            is_targets.append(True)
        
        # Achieved fingertip positions
        for finger, pos in grasp.fingertip_positions.items():
            positions.append(pos)
            # Use same normal as target if available, else default
            if finger in grasp.target_normals:
                normals.append(grasp.target_normals[finger])
            else:
                normals.append(np.array([0, 0, 1]))
            fingers.append(finger)
            is_targets.append(False)
        
        if positions:
            self.set_contacts(
                np.array(positions),
                np.array(normals),
                fingers=fingers,
                is_targets=is_targets,
            )
        
        return self
    
    def set_camera(
        self,
        preset: Union[str, CameraPreset, None] = None,
        azimuth: Optional[float] = None,
        elevation: Optional[float] = None,
        distance: Optional[float] = None,
        lookat: Optional[Tuple[float, float, float]] = None,
    ) -> "GraspVisualizer":
        """Configure camera.
        
        Args:
            preset: Camera preset name or CameraPreset instance
            azimuth: Override azimuth angle (degrees)
            elevation: Override elevation angle (degrees)  
            distance: Override camera distance (meters)
            lookat: Override lookat point
            
        Returns:
            self for chaining
        """
        self._ensure_render_context()
        
        if preset is not None:
            if isinstance(preset, str):
                preset_obj = CAMERA_PRESETS.get(preset, CAMERA_PRESETS["iso"])
            else:
                preset_obj = preset
            preset_obj.apply(self._camera, self._scene_center)
        
        if azimuth is not None:
            self._camera.azimuth = azimuth
        if elevation is not None:
            self._camera.elevation = elevation
        if distance is not None:
            self._camera.distance = distance
        if lookat is not None:
            self._camera.lookat[:] = lookat
            
        return self
    
    def _update_scene(self) -> None:
        """Update the scene with current state and overlays.
        
        This is used for the interactive viewer (show method).
        For offscreen rendering, we use Renderer.update_scene + _add_contact_visuals.
        """
        self._ensure_render_context()
        
        # Update base scene
        mj.mjv_updateScene(
            self.model, self.data, self._opt, None, self._camera,
            mj.mjtCatBit.mjCAT_ALL, self._scene,
        )
        
        # Add contact visualizations
        self._add_contact_visuals_to_scene(self._scene)
    
    def _add_contact_visuals_to_scene(self, scene: "mj.MjvScene") -> None:
        """Add contact point visualizations to a given scene"""
        if scene is None:
            return
            
        for contact in self._contacts:
            if contact.is_target and not self._show_targets:
                continue
                
            # Contact point sphere
            color = (
                self.scheme.target_point if contact.is_target
                else self.scheme.contact_point
            )
            _create_sphere_geom(
                scene,
                contact.position,
                self._contact_radius,
                color,
            )
            
            # Contact normal arrow
            if self._show_normals and not contact.is_target:
                _create_arrow_geom(
                    scene,
                    contact.position,
                    contact.normal,
                    self._arrow_length,
                    self._contact_radius * 0.4,
                    self.scheme.contact_normal,
                )
            
            # Friction cone
            if self._show_friction_cones and not contact.is_target:
                _create_friction_cone_geoms(
                    scene,
                    contact.position,
                    contact.normal,
                    self._friction_coef,
                    self.scheme.cone_fill,
                    self.scheme.cone_edge,
                    n_segments=6,
                    cone_height=0.010,
                )
    
    def render(
        self,
        width: int = 1280,
        height: int = 960,
    ) -> NDArray:
        """Render to numpy array.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            RGB image as (height, width, 3) uint8 array
        """
        self._ensure_render_context()
        
        # Ensure model's offscreen buffer is large enough
        if self.model.vis.global_.offwidth < width:
            self.model.vis.global_.offwidth = width
        if self.model.vis.global_.offheight < height:
            self.model.vis.global_.offheight = height
        
        # Create renderer at max resolution if not yet created.
        # NEVER recreate: mj.Renderer.close() + new() causes segfaults
        # from GL context reallocation. Instead, always render at the
        # maximum resolution ever requested and downscale when needed.
        need_w = max(width, self._render_width)
        need_h = max(height, self._render_height)
        
        if self._renderer is None:
            if self.model.vis.global_.offwidth < need_w:
                self.model.vis.global_.offwidth = need_w
            if self.model.vis.global_.offheight < need_h:
                self.model.vis.global_.offheight = need_h
            self._renderer = mj.Renderer(self.model, height=need_h, width=need_w)
            self._render_width = need_w
            self._render_height = need_h
        
        # Update the renderer's scene with current data + clean vis options
        self._renderer.update_scene(self.data, camera=self._camera, scene_option=self._opt)
        
        # Apply clean rendering flags to the scene
        scene = self._renderer.scene
        scene.flags[mj.mjtRndFlag.mjRND_SKYBOX] = False
        scene.flags[mj.mjtRndFlag.mjRND_HAZE] = False
        scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = True
        scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = False
        
        # Add our custom contact visualization geometry
        self._add_contact_visuals_to_scene(scene)
        
        # Render RGB
        rgb = self._renderer.render().copy()
        
        # Depth-based background masking: find pixels at far plane
        # and replace with clean background color
        try:
            self._renderer.enable_depth_rendering()
            depth = self._renderer.render()
            self._renderer.disable_depth_rendering()
            
            bg = self.scheme.background
            bg_color = np.array([int(c * 255) for c in bg[:3]], dtype=np.uint8)
            depth_max = depth.max()
            is_bg = depth >= (depth_max - 1e-6)
            rgb[is_bg] = bg_color
        except Exception:
            # Fallback: replace only truly-black pixels at image borders
            bg = self.scheme.background
            bg_color = np.array([int(c * 255) for c in bg[:3]], dtype=np.uint8)
            # Use flood-fill from corners to identify background
            # Simple heuristic: top rows are guaranteed background
            corner_color = rgb[0, 0]
            if np.all(corner_color == 0):
                # Only replace pixels matching the corner color
                is_bg = np.all(rgb == corner_color, axis=2)
                rgb[is_bg] = bg_color
        
        # Downscale if we rendered at higher resolution than requested
        if rgb.shape[1] != width or rgb.shape[0] != height:
            try:
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray(rgb)
                pil_img = pil_img.resize((width, height), PILImage.LANCZOS)
                rgb = np.array(pil_img)
            except ImportError:
                # Nearest-neighbor resize fallback
                row_idx = np.linspace(0, rgb.shape[0] - 1, height).astype(int)
                col_idx = np.linspace(0, rgb.shape[1] - 1, width).astype(int)
                rgb = rgb[np.ix_(row_idx, col_idx)]
        
        return rgb
    
    def render_to_file(
        self,
        path: str,
        width: int = 1280,
        height: int = 960,
        camera: Optional[str] = None,
    ) -> str:
        """Render and save to file.
        
        Args:
            path: Output path (PNG/JPG)
            width: Image width
            height: Image height
            camera: Camera preset name to use
            
        Returns:
            Path to saved file
        """
        if camera:
            self.set_camera(preset=camera)
            
        rgb = self.render(width, height)
        
        # Save using PIL or imageio
        try:
            from PIL import Image
            Image.fromarray(rgb).save(path)
        except ImportError:
            try:
                import imageio
                imageio.imwrite(path, rgb)
            except ImportError:
                import cv2
                cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        return path
    
    def render_multi_view(
        self,
        views: Sequence[str] = ("front", "side", "iso"),
        width: int = 1280,
        height: int = 960,
        padding: int = 4,
    ) -> NDArray:
        """Render multiple views in a grid.
        
        Args:
            views: List of camera preset names
            width: Width per view
            height: Height per view
            padding: Padding between views
            
        Returns:
            Combined RGB image
        """
        images = []
        for view in views:
            self.set_camera(preset=view)
            img = self.render(width, height)
            images.append(img)
        
        # Stack horizontally
        n = len(images)
        total_width = n * width + (n - 1) * padding
        combined = np.ones((height, total_width, 3), dtype=np.uint8) * 255
        
        for i, img in enumerate(images):
            x_start = i * (width + padding)
            combined[:, x_start:x_start + width] = img
        
        return combined
    
    def show(self, block: bool = True) -> Optional["mj.viewer.Handle"]:
        """Launch interactive viewer.
        
        Args:
            block: If True, block until viewer closes
            
        Returns:
            Viewer handle if non-blocking
        """
        # Update scene once for overlays
        self._update_scene()
        
        try:
            viewer = mj.viewer.launch_passive(self.model, self.data)
            if block:
                print("Close the viewer window to continue.")
                while viewer.is_running():
                    viewer.sync()
                return None
            return viewer
        except Exception as e:
            print(f"Could not launch viewer: {e}")
            return None
    
    # Fluent configuration methods
    def with_friction_coef(self, mu: float) -> "GraspVisualizer":
        """Set friction coefficient for cone visualization"""
        self._friction_coef = mu
        return self
    
    def with_friction_cones(self, show: bool = True) -> "GraspVisualizer":
        """Enable/disable friction cone visualization"""
        self._show_friction_cones = show
        return self
    
    def with_normals(self, show: bool = True) -> "GraspVisualizer":
        """Enable/disable contact normal arrows"""
        self._show_normals = show
        return self
    
    def with_targets(self, show: bool = True) -> "GraspVisualizer":
        """Enable/disable target contact visualization"""
        self._show_targets = show
        return self
    
    def with_scheme(self, scheme: Union[str, ColorScheme]) -> "GraspVisualizer":
        """Change color scheme"""
        if isinstance(scheme, str):
            self.scheme = SCHEMES.get(scheme, SCHEMES["default"])
        else:
            self.scheme = scheme
        return self
    
    def with_object_alpha(self, alpha: float) -> "GraspVisualizer":
        """Set object transparency (0=invisible, 1=opaque).
        
        Updates the alpha channel of the scheme object_color and
        re-applies to all object geoms. Default is 1.0 (solid).
        """
        alpha = max(0.0, min(1.0, alpha))
        r, g, b, _ = self.scheme.object_color
        self.scheme.object_color = (r, g, b, alpha)
        self._make_object_transparent()
        return self


# ###
# Convenience Functions
# ###

def visualize_grasp(
    model: "mj.MjModel",
    data: "mj.MjData",
    grasp: "SampledGrasp",
    camera: str = "iso",
    scheme: str = "default",
    friction_coef: float = 0.5,
    show_friction_cones: bool = True,
    output_path: Optional[str] = None,
    show: bool = True,
    width: int = 2560,
    height: int = 1920,
) -> Optional[NDArray]:
    """Convenience function to visualize a grasp.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        grasp: SampledGrasp result
        camera: Camera preset name
        scheme: Color scheme name
        friction_coef: Friction coefficient for cone visualization
        show_friction_cones: Whether to show friction cones
        output_path: If provided, save image to this path
        show: If True and output_path not set, launch viewer
        width: Render width
        height: Render height
        
    Returns:
        RGB numpy array if output_path provided or show=False, else None
    """
    viz = (
        GraspVisualizer(model, data, scheme=scheme, use_original_hand_colors=True)
        .set_grasp(grasp)
        .with_friction_coef(friction_coef)
        .with_friction_cones(show_friction_cones)
        .set_camera(preset=camera)
    )
    
    if output_path:
        viz.render_to_file(output_path, width, height)
        return viz.render(width, height)
    elif show:
        viz.show()
        return None
    else:
        return viz.render(width, height)


def create_grasp_figure(
    model: "mj.MjModel",
    data: "mj.MjData",
    grasps: Sequence["SampledGrasp"],
    output_path: str,
    views: Sequence[str] = ("front", "side"),
    scheme: str = "paper",
    width_per_grasp: int = 1280,
    height: int = 960,
) -> str:
    """Create a publication figure comparing multiple grasps.
    
    Renders each grasp from multiple views with no text or stats.
    Metrics should be saved separately to JSON.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        grasps: List of grasps to compare
        output_path: Output path for figure
        views: Camera views per grasp
        scheme: Color scheme (paper recommended)
        width_per_grasp: Width per grasp in pixels
        height: Height per view
        
    Returns:
        Path to saved figure
    """
    viz = GraspVisualizer(model, data, scheme=scheme, use_original_hand_colors=True)
    
    n_grasps = len(grasps)
    n_views = len(views)
    
    # Render each grasp from each view
    all_images = []
    for grasp in grasps:
        viz.set_grasp(grasp)
        grasp_images = []
        for view in views:
            viz.set_camera(preset=view)
            img = viz.render(width_per_grasp, height)
            grasp_images.append(img)
        # Stack views vertically for this grasp
        grasp_col = np.vstack(grasp_images)
        all_images.append(grasp_col)
    
    # Stack grasps horizontally
    figure = np.hstack(all_images)
    
    # Save
    try:
        from PIL import Image
        Image.fromarray(figure).save(output_path)
    except ImportError:
        import imageio
        imageio.imwrite(output_path, figure)
    
    return output_path
