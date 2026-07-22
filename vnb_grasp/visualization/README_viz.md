# vnb_grasp.visualization

Multi-camera rendering, HUD overlays, grasp visualization, and viewport management utilities for MuJoCo simulations.

## Modules

### multi_camera.py

Multi-viewport rendering with inset cameras for wrist/scene views.

```python
from vnb_grasp.visualization import MultiCameraRenderer, CameraConfig

# Set up multi-camera renderer
renderer = MultiCameraRenderer(model, data, context)
renderer.setup_main_camera(azimuth=90, elevation=-25, distance=3.2)
renderer.add_camera("wrist", "wrist_camera", position="top-right")
renderer.add_camera("scene", "external_camera", position="top-left")

# In render loop
viewports = renderer.render_all(opt, fb_width, fb_height)

# Read RGBD from inset
rgb, depth = renderer.read_rgbd("wrist", viewports["wrist"])
```

**Key classes:**
- `MultiCameraRenderer`: Manages multiple camera views with automatic viewport layout
- `CameraConfig`: Configuration for camera views (position, inset size, labels)
- `CameraView`: Internal state for each camera

**Key functions:**
- `setup_fixed_camera()`: Configure a fixed camera from model XML
- `read_viewport_rgbd()`: Capture RGB and depth from a viewport

### hud_overlay.py

Configurable heads-up display text overlays for status information.

```python
from vnb_grasp.visualization import HUDOverlay, OverlayLine

# Create HUD
hud = HUDOverlay(position="top-left", font_scale="tiny")

# Add static controls section
hud.add_static_section("controls", [
    "Keys: 1-6 select joint",
    "      G/Space grasp",
])

# Add dynamic status section
hud.add_section("status", [
    OverlayLine("Mode", lambda: "AUTO" if auto_enabled else "MANUAL"),
    OverlayLine("Grasp", lambda: "CLOSE" if closed else "OPEN"),
])

# In render loop
hud.render(viewport, context)
```

**Key classes:**
- `HUDOverlay`: Main overlay manager with sections and visibility control
- `OverlayLine`: Single line with label and dynamic content
- `OverlaySection`: Group of lines with optional header

**Convenience builders:**
- `build_contact_stats_section()`: Contact force/count statistics
- `build_joint_status_section()`: Joint selection and value display
- `build_grasp_status_section()`: Grasp mode/state display

### grasp_viz.py

```python
from vnb_grasp.visualization import GraspVisualizer, visualize_grasp

# Method 1: Fluent API
viz = (
    GraspVisualizer(model, data, scheme="paper")
    .set_grasp(best_grasp)
    .with_friction_coef(0.5)
    .with_friction_cones(True)
    .with_normals(True)
    .set_camera(preset="iso")
)

# Render to file for paper
viz.render_to_file("grasp.png", width=1280, height=960)

# Or launch interactive viewer
viz.show()

# Multi-view rendering
rgb = viz.render_multi_view(views=["front", "side", "iso"])

# Method 2: Convenience function
visualize_grasp(model, data, grasp, camera="iso", scheme="paper")
```

**Features:**
- **Contact point markers**: Red/yellow spheres at contact locations
- **Contact normal arrows**: Green arrows showing surface normals
- **Friction cone visualization**: Semi-transparent cones showing friction constraints
- **Semi-transparent objects**: See through to contact points
- **Clean camera presets**: front, side, iso, top, three_quarter, close
- **Multiple color schemes**: default, paper, dark, minimal
- **Multi-view rendering**: Side-by-side views for comparison figures

**Key classes:**
- `GraspVisualizer`: Main visualization class with fluent API
- `ColorScheme`: RGBA color configuration for all elements
- `CameraPreset`: Named camera angle configurations

**Color schemes:**
- `default`: Clean light theme
- `paper`: High contrast for publications (white background)
- `dark`: Dark theme for presentations
- `minimal`: Grayscale except contacts (emphasizes contact geometry)

**Camera presets:**
- `front`, `side`, `iso`: Standard orthographic views
- `top`: Plan view from above
- `three_quarter`: Appealing visualization angle
- `close`: Zoomed in on contact region

**Convenience functions:**
- `visualize_grasp()`: One-liner for quick grasp visualization
- `create_grasp_figure()`: Generate publication figure comparing multiple grasps

## Related Modules

- `vnb_grasp.control.ibvs`: IBVS visual servoing using camera geometry
- `vnb_grasp.perception`: Point cloud and pose estimation

## Source

Extracted and modularized from `utils/launch_mj_env.py` multi-camera and HUD features.
