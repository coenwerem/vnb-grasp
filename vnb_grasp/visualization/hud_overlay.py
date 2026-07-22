"""HUD overlay utilities for VNB-Grasp.

Provides helpers for rendering text overlays with status information,
contact statistics, and key bindings on top of MuJoCo renders.

Example:
    >>> from vnb_grasp.visualization.hud_overlay import HUDOverlay, OverlayLine
    >>> 
    >>> hud = HUDOverlay(position="top-left", font_scale="small")
    >>> hud.add_section("status", [
    ...     OverlayLine("Mode", lambda: "AUTO" if auto_enabled else "MANUAL"),
    ...     OverlayLine("State", lambda: current_state),
    ... ])
    >>> hud.add_static_section("controls", [
    ...     "Keys: 1-6 select joint | Up/Down move",
    ...     "      G/Space grasp | H reset",
    ... ])
    >>> 
    >>> # In render loop:
    >>> hud.render(viewport, context)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

try:
    import mujoco as mj
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


# Type for dynamic line content
LineContent = Union[str, Callable[[], str]]


@dataclass
class OverlayLine:
    """A single line in the overlay.
    
    Attributes:
        label: Label prefix (e.g., "Mode")
        content: Static string or callable returning current value
        separator: Separator between label and content
    """
    label: str
    content: LineContent = ""
    separator: str = ": "
    
    def render(self) -> str:
        """Render the line to a string"""
        if callable(self.content):
            value = self.content()
        else:
            value = self.content
        
        if self.label and value:
            return f"{self.label}{self.separator}{value}"
        elif self.label:
            return self.label
        else:
            return str(value)


@dataclass
class OverlaySection:
    """A section of the overlay with optional header"""
    name: str
    lines: List[Union[OverlayLine, str]]
    header: Optional[str] = None
    visible: bool = True


def _clip_line(line: str, max_chars: int = 60) -> str:
    """Clip a line to maximum characters"""
    return line[:max_chars] if len(line) > max_chars else line


class HUDOverlay:
    """Configurable heads-up display overlay.
    
    Manages multiple sections of text that can be updated dynamically
    and rendered as an overlay on MuJoCo viewports.
    """
    
    FONT_SCALES = {
        "tiny": "mjFONTSCALE_50",
        "small": "mjFONTSCALE_100",
        "medium": "mjFONTSCALE_150",
        "large": "mjFONTSCALE_200",
        "huge": "mjFONTSCALE_250",
        "giant": "mjFONTSCALE_300",
    }
    
    POSITIONS = {
        "top-left": "mjGRID_TOPLEFT",
        "top-right": "mjGRID_TOPRIGHT",
        "bottom-left": "mjGRID_BOTTOMLEFT",
        "bottom-right": "mjGRID_BOTTOMRIGHT",
        "top": "mjGRID_TOP",
        "bottom": "mjGRID_BOTTOM",
    }
    
    def __init__(
        self,
        position: str = "top-left",
        font_scale: str = "tiny",
        max_line_chars: int = 60,
        section_separator: str = "",
    ):
        """Initialize HUD overlay.
        
        Args:
            position: Overlay position on screen
            font_scale: Font size ("tiny", "small", "medium", "large")
            max_line_chars: Maximum characters per line
            section_separator: Text between sections (empty for single newline)
        """
        self.position = position
        self.font_scale = font_scale
        self.max_line_chars = max_line_chars
        self.section_separator = section_separator
        
        self._sections: Dict[str, OverlaySection] = {}
        self._section_order: List[str] = []
    
    def add_section(
        self,
        name: str,
        lines: List[Union[OverlayLine, str]],
        header: Optional[str] = None,
        visible: bool = True,
    ) -> None:
        """Add a section to the overlay.
        
        Args:
            name: Unique section identifier
            lines: List of OverlayLine or static strings
            header: Optional section header
            visible: Whether section is initially visible
        """
        section = OverlaySection(
            name=name,
            lines=lines,
            header=header,
            visible=visible,
        )
        self._sections[name] = section
        if name not in self._section_order:
            self._section_order.append(name)
    
    def add_static_section(
        self,
        name: str,
        lines: List[str],
        header: Optional[str] = None,
    ) -> None:
        """Add a section with static text lines"""
        self.add_section(name, lines, header)
    
    def add_dynamic_section(
        self,
        name: str,
        content_fn: Callable[[], List[str]],
        header: Optional[str] = None,
    ) -> None:
        """Add a section with dynamically generated lines.
        
        Args:
            name: Section identifier
            content_fn: Callable returning list of strings
            header: Optional header
        """
        # Wrap the callable in a single line that calls it
        section = OverlaySection(
            name=name,
            lines=[OverlayLine("", content_fn)],
            header=header,
            visible=True,
        )
        self._sections[name] = section
        if name not in self._section_order:
            self._section_order.append(name)
    
    def set_visibility(self, name: str, visible: bool) -> None:
        """Set section visibility"""
        if name in self._sections:
            self._sections[name].visible = visible
    
    def toggle_visibility(self, name: str) -> bool:
        """Toggle section visibility, returns new state"""
        if name in self._sections:
            self._sections[name].visible = not self._sections[name].visible
            return self._sections[name].visible
        return False
    
    def update_line(
        self,
        section_name: str,
        line_index: int,
        content: LineContent,
    ) -> None:
        """Update a specific line's content"""
        section = self._sections.get(section_name)
        if section and 0 <= line_index < len(section.lines):
            line = section.lines[line_index]
            if isinstance(line, OverlayLine):
                line.content = content
            else:
                section.lines[line_index] = str(content)
    
    def render_text(self) -> str:
        """Render all sections to a single text string"""
        all_lines: List[str] = []
        
        for name in self._section_order:
            section = self._sections.get(name)
            if not section or not section.visible:
                continue
            
            if all_lines and self.section_separator:
                all_lines.append(self.section_separator)
            elif all_lines:
                all_lines.append("")
            
            if section.header:
                all_lines.append(_clip_line(section.header, self.max_line_chars))
            
            for line in section.lines:
                if isinstance(line, OverlayLine):
                    rendered = line.render()
                    # Handle multi-line dynamic content
                    if callable(line.content):
                        result = line.content()
                        if isinstance(result, list):
                            for sub_line in result:
                                all_lines.append(_clip_line(str(sub_line), self.max_line_chars))
                            continue
                    all_lines.append(_clip_line(rendered, self.max_line_chars))
                else:
                    all_lines.append(_clip_line(str(line), self.max_line_chars))
        
        return "\n".join(all_lines)
    
    def render(
        self,
        viewport: "mj.MjrRect",
        context: "mj.MjrContext",
    ) -> None:
        """Render the overlay onto a viewport.
        
        Args:
            viewport: Target viewport rectangle
            context: MuJoCo rendering context
        """
        font_attr = self.FONT_SCALES.get(self.font_scale, "mjFONTSCALE_100")
        pos_attr = self.POSITIONS.get(self.position, "mjGRID_TOPLEFT")
        
        font_scale = getattr(mj.mjtFontScale, font_attr)
        grid_pos = getattr(mj.mjtGridPos, pos_attr).value
        
        text = self.render_text()
        
        mj.mjr_overlay(
            font_scale,
            grid_pos,
            viewport,
            text,
            "",
            context,
        )


# --- Convenience builders for common HUD sections ---

def build_contact_stats_section(
    model: "mj.MjModel",
    data: "mj.MjData",
    object_body_name: str = "cube",
    hand_geom_prefixes: Sequence[str] = ("thumb_", "index_", "middle_", "ring_", "pinky_", "palm_"),
) -> Callable[[], List[str]]:
    """Build a callable that generates contact statistics lines.
    
    Returns a function suitable for add_dynamic_section().
    """
    def get_contact_lines() -> List[str]:
        ncon = int(data.ncon)
        if ncon == 0:
            return ["Contacts: ncon=0"]
        
        max_fn = 0.0
        ncon_obj = 0
        ncon_table = 0
        ncon_hand = 0
        ncon_obj_hand = 0
        sample_pairs: List[str] = []
        
        force6 = np.zeros(6, dtype=np.float64)
        obj_key = f"{object_body_name.lower()}_collision"
        
        for ci in range(min(ncon, 100)):  # Limit to avoid slowdown
            try:
                mj.mj_contactForce(model, data, ci, force6)
                fn = abs(float(force6[0]))
                if fn > max_fn:
                    max_fn = fn
                
                c = data.contact[ci]
                g1 = int(c.geom1)
                g2 = int(c.geom2)
                n1 = (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g1) or "").lower()
                n2 = (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g2) or "").lower()
                
                is_obj = (obj_key in n1) or (obj_key in n2)
                is_table = "table" in n1 or "table" in n2
                is_hand = any(k in n1 or k in n2 for k in hand_geom_prefixes)
                
                if is_obj:
                    ncon_obj += 1
                if is_table:
                    ncon_table += 1
                if is_hand:
                    ncon_hand += 1
                if is_obj and is_hand:
                    ncon_obj_hand += 1
                
                if len(sample_pairs) < 2:
                    n1_disp = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g1) or str(g1)
                    n2_disp = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g2) or str(g2)
                    sample_pairs.append(f"{n1_disp} x {n2_disp}")
            except Exception:
                continue
        
        lines = [
            f"Contacts: ncon={ncon}  maxFn={max_fn:.2f}",
            f"  obj={ncon_obj}  obj-hand={ncon_obj_hand}",
            f"  table={ncon_table}  hand={ncon_hand}",
        ]
        for i, pair in enumerate(sample_pairs):
            lines.append(f"  sample[{i+1}]: {pair[:40]}")
        
        return lines
    
    return get_contact_lines


def build_joint_status_section(
    joint_names: List[str],
    get_joint_index: Callable[[], int],
    get_joint_value: Callable[[int], float],
    get_joint_speed: Callable[[], float],
) -> Callable[[], List[str]]:
    """Build a callable that generates joint control status lines"""
    def get_joint_lines() -> List[str]:
        idx = get_joint_index()
        name = joint_names[idx] if 0 <= idx < len(joint_names) else f"joint_{idx}"
        value = get_joint_value(idx)
        speed = get_joint_speed()
        return [
            f"Joint: {idx} ({name})",
            f"q_des={value:+.3f}   speed={speed:.2f} rad/s",
        ]
    return get_joint_lines


def build_grasp_status_section(
    get_grasp_closed: Callable[[], bool],
    get_mode: Callable[[], str],
    get_state: Callable[[], str],
) -> Callable[[], List[str]]:
    """Build a callable that generates grasp status lines"""
    def get_grasp_lines() -> List[str]:
        grasp = "CLOSE" if get_grasp_closed() else "OPEN"
        mode = get_mode()
        state = get_state()
        return [
            f"Mode: {mode}   state={state}",
            f"Grasp: {grasp}",
        ]
    return get_grasp_lines
