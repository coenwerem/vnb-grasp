"""Multi-camera rendering utilities for VNB-Grasp.

Provides helpers for setting up multiple camera views (main, wrist, scene)
with inset rendering and RGBD capture.

Example:
    >>> from vnb_grasp.visualization.multi_camera import (
    ...     MultiCameraRenderer, CameraConfig
    ... )
    >>> 
    >>> renderer = MultiCameraRenderer(model, data, context)
    >>> renderer.add_camera("wrist", "wrist_camera", inset=(0.27, 0.27), position="top-right")
    >>> renderer.add_camera("scene", "external_camera", inset=(0.27, 0.27), position="top-left")
    >>> 
    >>> # In render loop:
    >>> renderer.render_all(scene, opt, framebuffer_size)
    >>> rgb, depth = renderer.read_rgbd("wrist")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import mujoco as mj
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


@dataclass
class CameraConfig:
    """Configuration for a camera view.
    
    Attributes:
        name: Unique identifier for this camera view
        camera_name: MuJoCo camera name in XML, or None for free camera
        inset_size: (width_fraction, height_fraction) of frame buffer
        position: Inset position ("top-right", "top-left", "bottom-right", "bottom-left")
        margin: Pixel margin from edge
        label: Text label to show on viewport
    """
    name: str
    camera_name: Optional[str] = None
    inset_size: Tuple[float, float] = (0.27, 0.27)
    position: str = "top-right"
    margin: int = 14
    label: Optional[str] = None
    
    @property
    def label_text(self) -> str:
        return self.label or self.camera_name or self.name.upper()


@dataclass
class CameraView:
    """Internal state for a camera view"""
    config: CameraConfig
    mjv_camera: "mj.MjvCamera"
    mjv_scene: "mj.MjvScene"
    camera_id: int = -1
    is_fixed: bool = False


class MultiCameraRenderer:
    """Multi-viewport renderer for MuJoCo scenes.
    
    Manages multiple camera views with automatic viewport layout,
    RGBD capture, and overlay labels.
    """
    
    def __init__(
        self,
        model: "mj.MjModel",
        data: "mj.MjData",
        context: "mj.MjrContext",
        context_small: Optional["mj.MjrContext"] = None,
        maxgeom: int = 10000,
    ):
        """Initialize renderer.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            context: MuJoCo rendering context
            context_small: Optional smaller font context for overlays
            maxgeom: Maximum geometries per scene
        """
        self.model = model
        self.data = data
        self.context = context
        self.context_small = context_small or context
        self.maxgeom = maxgeom
        
        self._views: Dict[str, CameraView] = {}
        self._main_camera: Optional["mj.MjvCamera"] = None
        self._main_scene: Optional["mj.MjvScene"] = None
        
    def setup_main_camera(
        self,
        azimuth: float = 90,
        elevation: float = -25,
        distance: float = 3.2,
        lookat: Tuple[float, float, float] = (0.0, 0.0, 1.5),
    ) -> "mj.MjvCamera":
        """Set up the main (free) camera view.
        
        Args:
            azimuth: Camera azimuth in degrees
            elevation: Camera elevation in degrees
            distance: Distance from lookat point
            lookat: 3D point to look at
            
        Returns:
            MjvCamera object
        """
        cam = mj.MjvCamera()
        mj.mjv_defaultCamera(cam)
        cam.azimuth = azimuth
        cam.elevation = elevation
        cam.distance = distance
        cam.lookat[:] = np.array(lookat)
        
        self._main_camera = cam
        self._main_scene = mj.MjvScene(self.model, maxgeom=self.maxgeom)
        
        return cam
    
    def add_camera(
        self,
        name: str,
        camera_name: str,
        inset_size: Tuple[float, float] = (0.27, 0.27),
        position: str = "top-right",
        margin: int = 14,
        label: Optional[str] = None,
    ) -> bool:
        """Add an inset camera view.
        
        Args:
            name: Unique identifier
            camera_name: MuJoCo camera name from XML
            inset_size: (width_fraction, height_fraction) relative to framebuffer
            position: One of "top-right", "top-left", "bottom-right", "bottom-left"
            margin: Pixel margin from edge
            label: Optional label text
            
        Returns:
            True if camera was found and added
        """
        config = CameraConfig(
            name=name,
            camera_name=camera_name,
            inset_size=inset_size,
            position=position,
            margin=margin,
            label=label,
        )
        
        cam = mj.MjvCamera()
        mj.mjv_defaultCamera(cam)
        
        scene = mj.MjvScene(self.model, maxgeom=self.maxgeom)
        
        camera_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, camera_name)
        is_fixed = camera_id >= 0
        
        if is_fixed:
            cam.fixedcamid = camera_id
            cam.type = mj.mjtCamera.mjCAMERA_FIXED
        
        view = CameraView(
            config=config,
            mjv_camera=cam,
            mjv_scene=scene,
            camera_id=camera_id,
            is_fixed=is_fixed,
        )
        
        self._views[name] = view
        return is_fixed
    
    def _compute_viewport(
        self,
        config: CameraConfig,
        fb_w: int,
        fb_h: int,
        index: int = 0,
    ) -> "mj.MjrRect":
        """Compute viewport rectangle for an inset camera"""
        w = int(config.inset_size[0] * fb_w)
        h = int(config.inset_size[1] * fb_h)
        m = config.margin
        
        # Compute base position
        if "right" in config.position:
            x = fb_w - w - m
        else:
            x = m
            
        if "top" in config.position:
            y = fb_h - h - m
        else:
            y = m
        
        # Offset for multiple cameras on same side
        if "right" in config.position:
            x -= index * (w + m)
        else:
            x += index * (w + m)
            
        return mj.MjrRect(x, y, w, h)
    
    def render_main(
        self,
        opt: "mj.MjvOption",
        fb_w: int,
        fb_h: int,
    ) -> "mj.MjrRect":
        """Render the main viewport.
        
        Returns:
            Main viewport rectangle
        """
        if self._main_camera is None or self._main_scene is None:
            raise RuntimeError("Main camera not set up. Call setup_main_camera() first.")
        
        viewport = mj.MjrRect(0, 0, fb_w, fb_h)
        
        mj.mjv_updateScene(
            self.model, self.data, opt, None,
            self._main_camera, mj.mjtCatBit.mjCAT_ALL.value,
            self._main_scene
        )
        mj.mjr_render(viewport, self._main_scene, self.context)
        
        return viewport
    
    def render_insets(
        self,
        opt: "mj.MjvOption",
        fb_w: int,
        fb_h: int,
        show_labels: bool = True,
    ) -> Dict[str, "mj.MjrRect"]:
        """Render all inset camera views.
        
        Args:
            opt: MjvOption for rendering
            fb_w: Framebuffer width
            fb_h: Framebuffer height
            show_labels: Whether to show camera labels
            
        Returns:
            Dictionary mapping camera names to their viewport rectangles
        """
        viewports = {}
        
        # Group by position for offset computation
        positions: Dict[str, List[str]] = {}
        for name, view in self._views.items():
            pos = view.config.position
            if pos not in positions:
                positions[pos] = []
            positions[pos].append(name)
        
        for pos, names in positions.items():
            for idx, name in enumerate(names):
                view = self._views[name]
                vp = self._compute_viewport(view.config, fb_w, fb_h, idx)
                viewports[name] = vp
                
                # Update and render scene
                mj.mjv_updateScene(
                    self.model, self.data, opt, None,
                    view.mjv_camera, mj.mjtCatBit.mjCAT_ALL.value,
                    view.mjv_scene
                )
                mj.mjr_render(vp, view.mjv_scene, self.context)
                
                # Add label overlay
                if show_labels:
                    label_pos = mj.mjtGridPos.mjGRID_TOP.value
                    if "right" in pos:
                        label_pos = mj.mjtGridPos.mjGRID_TOPRIGHT.value
                    elif "left" in pos:
                        label_pos = mj.mjtGridPos.mjGRID_TOPLEFT.value
                    
                    mj.mjr_overlay(
                        mj.mjtFontScale.mjFONTSCALE_150,
                        label_pos,
                        vp,
                        view.config.label_text,
                        "",
                        self.context,
                    )
        
        return viewports
    
    def render_all(
        self,
        opt: "mj.MjvOption",
        fb_w: int,
        fb_h: int,
        show_labels: bool = True,
    ) -> Dict[str, "mj.MjrRect"]:
        """Render main view and all insets.
        
        Returns:
            Dictionary with "main" and all camera names mapped to viewports
        """
        viewports = {}
        viewports["main"] = self.render_main(opt, fb_w, fb_h)
        viewports.update(self.render_insets(opt, fb_w, fb_h, show_labels))
        return viewports
    
    def read_rgbd(self, name: str, viewport: "mj.MjrRect") -> Tuple[np.ndarray, np.ndarray]:
        """Read RGB and depth from a rendered viewport.
        
        Args:
            name: Camera name
            viewport: Viewport rectangle (from render_all)
            
        Returns:
            (rgb, depth) arrays with shape (H, W, 3) and (H, W)
        """
        rgb = np.zeros((viewport.height, viewport.width, 3), dtype=np.uint8)
        depth = np.zeros((viewport.height, viewport.width), dtype=np.float32)
        
        mj.mjr_readPixels(rgb, depth, viewport, self.context)
        
        # MuJoCo returns images bottom-to-top
        return np.flipud(rgb), np.flipud(depth)
    
    @property
    def main_camera(self) -> Optional["mj.MjvCamera"]:
        """Get the main camera for external manipulation"""
        return self._main_camera
    
    @property
    def main_scene(self) -> Optional["mj.MjvScene"]:
        """Get the main scene"""
        return self._main_scene
    
    def get_camera(self, name: str) -> Optional["mj.MjvCamera"]:
        """Get a camera by name for external manipulation"""
        view = self._views.get(name)
        return view.mjv_camera if view else None
    
    def has_camera(self, name: str) -> bool:
        """Check if a camera exists and was found in the model"""
        view = self._views.get(name)
        return view is not None and view.is_fixed


def setup_fixed_camera(
    model: "mj.MjModel",
    mjv_cam: "mj.MjvCamera",
    camera_name: str,
) -> bool:
    """Set up a fixed camera from a MuJoCo model.
    
    Args:
        model: MuJoCo model
        mjv_cam: MjvCamera to configure
        camera_name: Camera name from XML
        
    Returns:
        True if camera was found
    """
    cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id == -1:
        return False
    mjv_cam.fixedcamid = cam_id
    mjv_cam.type = mj.mjtCamera.mjCAMERA_FIXED
    return True


def read_viewport_rgbd(
    viewport: "mj.MjrRect",
    context: "mj.MjrContext",
) -> Tuple[np.ndarray, np.ndarray]:
    """Read RGB and depth from a viewport.
    
    Args:
        viewport: Viewport rectangle
        context: MuJoCo rendering context
        
    Returns:
        (rgb, depth) with rgb shape (H, W, 3) and depth shape (H, W)
    """
    rgb = np.zeros((viewport.height, viewport.width, 3), dtype=np.uint8)
    depth = np.zeros((viewport.height, viewport.width), dtype=np.float32)
    mj.mjr_readPixels(rgb, depth, viewport, context)
    return np.flipud(rgb), np.flipud(depth)
