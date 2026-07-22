"""Visualization utilities for VNB-Grasp.

Provides multi-camera rendering, HUD overlays, grasp visualization, and viewport management.

Modules:
    multi_camera: Multi-viewport rendering with insets
    hud_overlay: Configurable HUD text overlays
    grasp_viz: Clean grasp visualization with contact points and friction cones
"""

from vnb_grasp.visualization.multi_camera import (
    CameraConfig,
    CameraView,
    MultiCameraRenderer,
    read_viewport_rgbd,
    setup_fixed_camera,
)
from vnb_grasp.visualization.hud_overlay import (
    HUDOverlay,
    OverlayLine,
    OverlaySection,
    build_contact_stats_section,
    build_grasp_status_section,
    build_joint_status_section,
)
from vnb_grasp.visualization.grasp_viz import (
    GraspVisualizer,
    ColorScheme,
    CameraPreset,
    SCHEMES,
    CAMERA_PRESETS,
    visualize_grasp,
    create_grasp_figure,
)

__all__ = [
    # multi_camera
    "CameraConfig",
    "CameraView",
    "MultiCameraRenderer",
    "read_viewport_rgbd",
    "setup_fixed_camera",
    # hud_overlay
    "HUDOverlay",
    "OverlayLine",
    "OverlaySection",
    "build_contact_stats_section",
    "build_grasp_status_section",
    "build_joint_status_section",
    # grasp_viz
    "GraspVisualizer",
    "ColorScheme",
    "CameraPreset",
    "SCHEMES",
    "CAMERA_PRESETS",
    "visualize_grasp",
    "create_grasp_figure",
]
