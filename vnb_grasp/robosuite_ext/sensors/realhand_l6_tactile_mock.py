from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

@dataclass(frozen=True)
class LinkerL6TactileSpec:
    """Minimal spec needed to emulate Linker L6 matrix tactile.

    Hardware-facing semantics we mimic:
    - 6x12 taxel array per finger pad (spec), but Linker SDK streams it as 12 rows of 6 values.
    - Output is a scalar per taxel.
    - Counts saturate at 4095 for full-scale.

    Notes:
    - This is not a coordinate-frame convention (no ROS / TF). It is purely array layout and does not take the hand's kinematics into account.
    """

    rows: int = 12  # SDK layout rows
    cols: int = 6   # SDK layout cols
    width_m: float = 0.0144   # 14.4mm
    height_m: float = 0.0096  # 9.6mm
    full_scale_n: float = 20.0
    full_scale_counts: int = 4095


@dataclass(frozen=True)
class TactilePadCalibration:
    """Per-pad calibration mapping normal force to analog counts.

    counts_per_newton sets the linear gain (k_i) in the hardware scaling law
    A_i = k_i * f_i. max_force_n clips both the force and the resulting counts
    to match the physical fingertip limits.
    """

    counts_per_newton: float
    max_force_n: float


class LinkerL6TactileMock:
    """Emulates Linker L6 per-finger matrix tactile from MuJoCo contacts.

    The implementation bins normal contact forces into a rectangular grid in each fingertip site's frame.

    Output layout:
    - `matrix_12x6`: matches Linker SDK internal layout (12 rows, 6 cols)
    - `matrix_6x12`: transposed view matching the spec wording (6 rows, 12 cols)

    This is intended as a *mock* for learning pipelines; it does not require MuJoCo plugins.
    """

    def __init__(
        self,
        *,
        sim,
        finger_tip_sites: Dict[str, str],
        finger_geom_substrings: Dict[str, str],
        spec: LinkerL6TactileSpec | None = None,
        pad_specs: Dict[str, LinkerL6TactileSpec] | None = None,
        pad_calibration: Dict[str, TactilePadCalibration] | None = None,
        spatial_spread: str = "bilinear",
        ema_beta: float | None = 0.85,
    ):
        self.sim = sim
        self.default_spec = spec or LinkerL6TactileSpec()
        self.pad_specs = dict(pad_specs) if pad_specs is not None else {}
        self.pad_calibration = dict(pad_calibration) if pad_calibration else {}
        self.spatial_spread = str(spatial_spread)
        self.ema_alpha = None if ema_alpha is None else float(ema_alpha)
        self._ema_counts: dict[str, np.ndarray] = {}
        self._last_force_grids: dict[str, np.ndarray] = {}
        self.fingers = list(finger_geom_substrings.keys())
        self._finger_geom_ids = {
            f: self._find_geom_ids_by_substring(substr) for f, substr in finger_geom_substrings.items()
        }

        # Tip sites
        self._tip_site_ids = {f: self._site_name_to_id(name) for f, name in finger_tip_sites.items()}

        # Per-finger frame source: site if available else first matched geom.
        self._frame_site_id: dict[str, int] = {}
        self._frame_geom_id: dict[str, int] = {}
        for f in list(self.fingers):
            gids = self._finger_geom_ids.get(f, ())
            if len(gids) == 0:
                continue
            sid = self._tip_site_ids.get(f)
            if isinstance(sid, int) and sid >= 0:
                self._frame_site_id[f] = int(sid)
            else:
                self._frame_geom_id[f] = int(gids[0])

        # Keep only fingers that have at least one geom match.
        self.fingers = [f for f in self.fingers if len(self._finger_geom_ids.get(f, ())) > 0]

        # Initialize EMA buffers 
        for f in self.fingers:
            s = self.pad_specs.get(f, self.default_spec)
            self._ema_counts[f] = np.zeros((s.rows, s.cols), dtype=np.float32)


    def reset_filter(self):
        """Clears temporal filtering state (EMA buffers)"""
        for f in self.fingers:
            buf = self._ema_counts.get(f)
            if buf is not None:
                buf.fill(0.0)

    def _site_name_to_id(self, name: str) -> int:
        """Best-effort site name -> id.

        Returns None if the site doesn't exist in this compiled model.
        """
        model = getattr(self.sim.model, "_model", self.sim.model)

        # mujoco>=3 python binding supports model.site; name
        try:
            return int(model.site(name).id)
        except Exception:
            pass

        # Fallback: use mj_name2id
        try:
            import mujoco

            sid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))
            return sid if sid != -1 else None
        except Exception:
            return None

    def _find_geom_ids_by_substring(self, substr: str) -> tuple[int, ...]:
        ids: list[int] = []
        model = getattr(self.sim.model, "_model", self.sim.model)
        ngeom = int(model.ngeom)
        for gid in range(ngeom):
            name = None
            try:
                name = model.geom(gid).name
            except Exception:
                try:
                    import mujoco

                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                except Exception:
                    name = None
            if name and substr in name:
                ids.append(int(gid))
        return tuple(ids)

    def _mj_model_data(self):
        m = getattr(self.sim.model, "_model", self.sim.model)
        d = getattr(self.sim.data, "_data", self.sim.data)
        return m, d

    def compute_counts(self) -> Dict[str, np.ndarray]:
        """Returns per-finger tactile matrices in SDK layout (12x6) as float32 counts"""

        try:
            import mujoco
        except Exception:
            # If mujoco python isn't available, return zeros rather than failing.
            out = {}
            for f in self.fingers:
                s = self.pad_specs.get(f, self.default_spec)
                out[f] = np.zeros((s.rows, s.cols), dtype=np.float32)
            return out

        m, d = self._mj_model_data()

        # Initialize output grids ; per-pad shape and per-pad parameters
        grids: Dict[str, np.ndarray] = {}
        pad_params: dict[str, tuple[int, int, float, float, float, float, float]] = {}
        # pad_params[f] = ; rows, cols, w, h, inv_dx, inv_dy, counts_per_newton
        for f in self.fingers:
            s = self.pad_specs.get(f, self.default_spec)
            rows, cols = int(s.rows), int(s.cols)
            grids[f] = np.zeros((rows, cols), dtype=np.float32)
            w = float(s.width_m)
            h = float(s.height_m)
            inv_dx = float(cols) / w
            inv_dy = float(rows) / h
            cal = self.pad_calibration.get(f)
            if cal is not None:
                scale = float(cal.counts_per_newton)
            else:
                scale = float(s.full_scale_counts) / max(1e-9, float(s.full_scale_n))
            pad_params[f] = (rows, cols, w, h, inv_dx, inv_dy, scale)

        # Precompute per-finger geom id sets for quick membership
        geom_sets = {f: set(self._finger_geom_ids[f]) for f in self.fingers}

        force6 = np.zeros(6, dtype=np.float64)

        # Iterate contacts
        for ci in range(int(self.sim.data.ncon)):
            c = self.sim.data.contact[ci]
            g1 = int(c.geom1)
            g2 = int(c.geom2)

            finger = None
            for f, gids in geom_sets.items():
                if g1 in gids or g2 in gids:
                    finger = f
                    break
            if finger is None:
                continue

            rows, cols, w, h, inv_dx, inv_dy, scale = pad_params[finger]

            force6[:] = 0.0
            mujoco.mj_contactForce(m, d, ci, force6)
            fn = abs(float(force6[0]))  # normal force in contact frame
            if fn <= 0.0:
                continue

            # Contact point in world
            p_w = np.array(c.pos, dtype=np.float64)

            # Fingertip frame in world: prefer site if available, else use a distal collision geom frame.
            if finger in self._frame_site_id:
                sid = self._frame_site_id[finger]
                origin_w = np.array(self.sim.data.site_xpos[sid], dtype=np.float64)
                R_wf = np.array(self.sim.data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
            else:
                gid = self._frame_geom_id.get(finger)
                if gid is None:
                    continue
                origin_w = np.array(self.sim.data.geom_xpos[gid], dtype=np.float64)
                R_wf = np.array(self.sim.data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)

            # Express contact point in site-local coordinates
            p_s = R_wf.T @ (p_w - origin_w)

            # Use local x/y as patch plane coordinates; assume patch centered at frame origin
            u = float(p_s[0])
            v = float(p_s[1])

            # Convert to continuous bin coordinates in [0, cols) and [0, rows)
            xf = (u + 0.5 * w) * inv_dx
            yf = (v + 0.5 * h) * inv_dy

            if self.spatial_spread == "none":
                col = int(np.floor(xf))
                row = int(np.floor(yf))
                if 0 <= row < rows and 0 <= col < cols:
                    grids[finger][row, col] += fn
            else:
                # Bilinear spread to 4 neighbors to smooth single-taxel impulses
                x0 = int(np.floor(xf))
                y0 = int(np.floor(yf))
                tx = float(xf - x0)
                ty = float(yf - y0)

                for yy, wy in ((y0, 1.0 - ty), (y0 + 1, ty)):
                    if not (0 <= yy < rows):
                        continue
                    for xx, wx in ((x0, 1.0 - tx), (x0 + 1, tx)):
                        if not (0 <= xx < cols):
                            continue
                        wgt = float(wx * wy)
                        if wgt > 0.0:
                            grids[finger][yy, xx] += fn * wgt

        # Convert N -> counts and saturate
        out_counts: Dict[str, np.ndarray] = {}
        self._last_force_grids.clear()
        for f, grid_n in grids.items():
            s = self.pad_specs.get(f, self.default_spec)
            _, _, _, _, _, _, scale = pad_params[f]

            cal: Optional[TactilePadCalibration] = self.pad_calibration.get(f)
            force_grid = grid_n
            if cal is not None:
                force_grid = np.clip(force_grid, 0.0, float(cal.max_force_n))
                max_counts = float(cal.counts_per_newton * cal.max_force_n)
            else:
                max_counts = float(s.full_scale_counts)

            self._last_force_grids[f] = force_grid.astype(np.float32)

            counts_raw = np.clip(force_grid * scale, 0.0, max_counts).astype(np.float32)

            # Temporal smoothing to mimic piezoresistive dynamics
            if self.ema_alpha is not None:
                a = float(np.clip(self.ema_alpha, 0.0, 0.999))
                prev = self._ema_counts.get(f)
                if prev is None or prev.shape != counts_raw.shape:
                    prev = np.zeros_like(counts_raw)
                counts = a * prev + (1.0 - a) * counts_raw
                self._ema_counts[f] = counts.astype(np.float32)
            else:
                counts = counts_raw

            # Return as counts-like values
            out_counts[f] = np.clip(counts, 0.0, float(s.full_scale_counts)).astype(np.float32)

        return out_counts

    def compute_tensor(self, *, layout: str = "sdk", order: list[str] | None = None) -> np.ndarray:
        """Returns stacked tensor for an ordered list of pads.

        layout:
        - "sdk": (N, rows, cols)
        - "spec": (N, cols, rows) (transpose of sdk)
        """
        # TODO: May need to add noise, hysteresis, or per-taxel bias to emulate sensor imperfection.

        per_finger = self.compute_counts()
        ordered = order if order is not None else ["thumb", "index", "middle", "ring", "pinky"]
        mats = []
        for f in ordered:
            s = self.pad_specs.get(f, self.default_spec)
            mats.append(per_finger.get(f, np.zeros((s.rows, s.cols), dtype=np.float32)))

        # If pad shapes mismatch, return object array rather than raising.
        try:
            tensor = np.stack(mats, axis=0).astype(np.float32)
        except Exception:
            return np.array(mats, dtype=object)

        if layout == "spec":
            tensor = np.transpose(tensor, (0, 2, 1))
        return tensor

    def compute_force_tensor(self, *, order: list[str] | None = None) -> np.ndarray:
        """Return per-pad normal forces in Newtons with the same stacking as compute_tensor"""

        # Ensure we have a recent force grid
        if not self._last_force_grids:
            _ = self.compute_counts()

        ordered = order if order is not None else ["thumb", "index", "middle", "ring", "pinky"]
        mats = []
        for f in ordered:
            s = self.pad_specs.get(f, self.default_spec)
            mats.append(self._last_force_grids.get(f, np.zeros((s.rows, s.cols), dtype=np.float32)))

        try:
            tensor = np.stack(mats, axis=0).astype(np.float32)
        except Exception:
            tensor = np.array(mats, dtype=object)
        return tensor
