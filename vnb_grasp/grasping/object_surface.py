"""Object surface representation and contact-point sampling.

Given either a MuJoCo mesh geom or a primitive geom (box, sphere, cylinder,
capsule, ellipsoid) this module provides:

1.  Surface point sampling, uniform or weighted by an energy functional.
2.  Surface normal estimation at any sampled point.
3.  Primitive-geometry procedural sampling without external mesh files.

For mesh geoms the vertices and faces are read directly from
``mujoco.MjModel`` arrays (mesh_vert, mesh_face).  Open3D is used
only when available and only for mesh-based sampling; the module
falls back to rejection sampling on the triangle soup when Open3D is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None

try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:  # pragma: no cover
    HAS_OPEN3D = False


# 
# Public data types
# 

class GeomKind(Enum):
    """Supported MuJoCo geom types for surface sampling"""

    BOX = auto()
    SPHERE = auto()
    CYLINDER = auto()
    CAPSULE = auto()
    ELLIPSOID = auto()
    MESH = auto()


@dataclass
class SurfaceSample:
    """A batch of points sampled on an object surface.

    Attributes
    ----------
    points : (N, 3) float, positions in **object-local** frame.
    normals : (N, 3) float, outward unit normals.
    weights : (N,) float, sampling weights (1 / N for uniform).
    """

    points: NDArray
    normals: NDArray
    weights: NDArray


@dataclass
class ObjectSurface:
    """Surface representation for a single rigid body in MuJoCo.

    Use the factory methods from_model or from_primitive to construct.

    Once built, call ``sample()`` to draw contact-point candidates.
    """

    kind: GeomKind
    size: NDArray  # MuJoCo geom size (interpretation depends on kind)

    # For MESH kind only
    vertices: NDArray = field(default_factory=lambda: np.zeros((0, 3)))
    faces: NDArray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int32))
    face_areas: NDArray = field(default_factory=lambda: np.zeros(0))
    face_normals: NDArray = field(default_factory=lambda: np.zeros((0, 3)))

    # Object pose in world frame (updated externally)
    position: NDArray = field(default_factory=lambda: np.zeros(3))
    rotation: NDArray = field(default_factory=lambda: np.eye(3))

    # 
    # Factories
    # 

    @classmethod
    def from_model(
        cls,
        model,
        geom_name: Optional[str] = None,
        geom_id: Optional[int] = None,
        body_name: Optional[str] = None,
    ) -> "ObjectSurface":
        """Create an ObjectSurface from a MuJoCo model.

        Exactly one of geom_name, geom_id, or body_name must be given.
        When body_name is used the first geom attached to that body is
        selected.

        Parameters
        ----------
        model : mujoco.MjModel
        geom_name : str, optional
        geom_id : int, optional
        body_name : str, optional
        """
        if mujoco is None:
            raise ImportError("mujoco is required")

        # Resolve geom id
        if sum(x is not None for x in (geom_name, geom_id, body_name)) != 1:
            raise ValueError("Provide exactly one of geom_name, geom_id, body_name")

        if geom_name is not None:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if gid < 0:
                raise ValueError(f"Geom '{geom_name}' not found in model")
        elif body_name is not None:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid < 0:
                raise ValueError(f"Body '{body_name}' not found in model")
            # First geom belonging to body
            gid = -1
            for g in range(model.ngeom):
                if int(model.geom_bodyid[g]) == bid:
                    gid = g
                    break
            if gid < 0:
                raise ValueError(f"Body '{body_name}' has no geoms")
        else:
            gid = int(geom_id)

        geom_type = int(model.geom_type[gid])
        geom_size = np.array(model.geom_size[gid], dtype=np.float64).copy()

        # Map MuJoCo geom type int --> GeomKind
        _MJ_TYPE = {
            6: GeomKind.BOX,        # mjGEOM_BOX = 6
            2: GeomKind.SPHERE,     # mjGEOM_SPHERE = 2
            5: GeomKind.CYLINDER,   # mjGEOM_CYLINDER = 5
            3: GeomKind.CAPSULE,    # mjGEOM_CAPSULE = 3
            4: GeomKind.ELLIPSOID,  # mjGEOM_ELLIPSOID = 4
            7: GeomKind.MESH,       # mjGEOM_MESH = 7
        }
        kind = _MJ_TYPE.get(geom_type)
        if kind is None:
            raise ValueError(
                f"Unsupported geom type {geom_type} for geom id {gid}"
            )

        obj = cls(kind=kind, size=geom_size)

        if kind == GeomKind.MESH:
            mesh_id = int(model.geom_dataid[gid])
            if mesh_id < 0:
                raise ValueError(f"Geom {gid} is type MESH but has no mesh data")
            va = int(model.mesh_vertadr[mesh_id])
            vn = int(model.mesh_vertnum[mesh_id])
            fa = int(model.mesh_faceadr[mesh_id])
            fn = int(model.mesh_facenum[mesh_id])
            obj.vertices = np.array(model.mesh_vert[va: va + vn], dtype=np.float64)
            obj.faces = np.array(model.mesh_face[fa: fa + fn], dtype=np.int32)
            obj._precompute_mesh()

        return obj

    @classmethod
    def from_primitive(
        cls,
        kind: GeomKind,
        size: Sequence[float],
    ) -> "ObjectSurface":
        """Create an ObjectSurface from a known primitive shape"""
        return cls(kind=kind, size=np.asarray(size, dtype=np.float64))

    # 
    # Sampling
    # 

    def sample(
        self,
        n: int = 500,
        *,
        rng: Optional[np.random.Generator] = None,
        energy_fn: Optional[Callable[[NDArray, NDArray], NDArray]] = None,
    ) -> SurfaceSample:
        """Sample n contact-candidate points on the surface.

        Parameters
        ----------
        n : int
            Number of points to sample.
        rng : numpy Generator, optional
            Random state.  A fresh one is created when None.
        energy_fn : callable (points, normals) --> weights, optional
            An energy functional that assigns a **non-negative** weight to each
            point.  Higher weight --> more likely to be selected in a downstream
            importance-sampling step.  When None all points are weighted
            uniformly.

        Returns
        -------
        SurfaceSample
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.kind == GeomKind.MESH:
            pts, nrm = self._sample_mesh(n, rng)
        elif self.kind == GeomKind.BOX:
            pts, nrm = self._sample_box(n, rng)
        elif self.kind == GeomKind.SPHERE:
            pts, nrm = self._sample_sphere(n, rng)
        elif self.kind == GeomKind.CYLINDER:
            pts, nrm = self._sample_cylinder(n, rng)
        elif self.kind == GeomKind.CAPSULE:
            pts, nrm = self._sample_capsule(n, rng)
        elif self.kind == GeomKind.ELLIPSOID:
            pts, nrm = self._sample_ellipsoid(n, rng)
        else:
            raise ValueError(f"Unsupported kind: {self.kind}")

        if energy_fn is not None:
            w = np.asarray(energy_fn(pts, nrm), dtype=np.float64).ravel()
            w = np.maximum(w, 0.0)
            total = w.sum()
            if total > 0:
                w /= total
            else:
                w = np.ones(n, dtype=np.float64) / n
        else:
            w = np.ones(n, dtype=np.float64) / n

        return SurfaceSample(points=pts, normals=nrm, weights=w)

    def to_world(self, local_points: NDArray) -> NDArray:
        """Transform (N, 3) object-local points to world frame"""
        return (self.rotation @ local_points.T).T + self.position

    def normal_to_world(self, local_normals: NDArray) -> NDArray:
        """Rotate (N, 3) object-local normals to world frame"""
        return (self.rotation @ local_normals.T).T

    # Signed distance field (SDF), negative inside the object

    def signed_distance(self, world_points: NDArray) -> NDArray:
        """Signed distance from world-frame points to the object surface.

        Parameters
        ----------
        world_points : (N, 3) or (3,)
            Query positions in world frame.

        Returns
        -------
        (N,) float, signed distance. Negative means inside the object.
        """
        pts = np.atleast_2d(world_points).astype(np.float64)
        # Transform to object-local frame
        local = (self.rotation.T @ (pts - self.position).T).T

        if self.kind == GeomKind.BOX:
            return self._sdf_box(local)
        elif self.kind == GeomKind.SPHERE:
            return self._sdf_sphere(local)
        elif self.kind == GeomKind.CYLINDER:
            return self._sdf_cylinder(local)
        elif self.kind == GeomKind.CAPSULE:
            return self._sdf_capsule(local)
        elif self.kind == GeomKind.ELLIPSOID:
            return self._sdf_ellipsoid(local)
        elif self.kind == GeomKind.MESH:
            return self._sdf_mesh_approx(local)
        else:
            return np.zeros(len(local))

    def outward_direction(self, world_points: NDArray) -> NDArray:
        """Outward surface normal at the closest surface point.

        For a point inside the object this gives the direction to push it
        back out of the geometry.

        Parameters
        ----------
        world_points : (N, 3) or (3,)

        Returns
        -------
        (N, 3), unit outward directions in world frame.
        """
        pts = np.atleast_2d(world_points).astype(np.float64)
        local = (self.rotation.T @ (pts - self.position).T).T

        if self.kind == GeomKind.BOX:
            local_dirs = self._outward_dir_box(local)
        elif self.kind == GeomKind.SPHERE:
            local_dirs = self._outward_dir_sphere(local)
        elif self.kind == GeomKind.CYLINDER:
            local_dirs = self._outward_dir_cylinder(local)
        elif self.kind == GeomKind.CAPSULE:
            local_dirs = self._outward_dir_capsule(local)
        elif self.kind == GeomKind.ELLIPSOID:
            local_dirs = self._outward_dir_ellipsoid(local)
        elif self.kind == GeomKind.MESH:
            local_dirs = self._outward_dir_mesh(local)
        else:
            local_dirs = self._outward_dir_sphere(local)  # fallback

        # Rotate back to world frame
        return (self.rotation @ local_dirs.T).T

    # SDF primitives (all operate in object-local frame)

    def _sdf_box(self, p: NDArray) -> NDArray:
        half = self.size[:3]
        q = np.abs(p) - half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    def _sdf_sphere(self, p: NDArray) -> NDArray:
        return np.linalg.norm(p, axis=1) - self.size[0]

    def _sdf_cylinder(self, p: NDArray) -> NDArray:
        r, h = self.size[0], self.size[1]
        d_radial = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - r
        d_axial = np.abs(p[:, 2]) - h
        q = np.column_stack([d_radial, d_axial])
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    def _sdf_capsule(self, p: NDArray) -> NDArray:
        r, h = self.size[0], self.size[1]
        # Closest point on central segment [-h, h] along Z
        z_clamped = np.clip(p[:, 2], -h, h)
        closest = np.column_stack([np.zeros(len(p)), np.zeros(len(p)), z_clamped])
        return np.linalg.norm(p - closest, axis=1) - r

    def _sdf_ellipsoid(self, p: NDArray) -> NDArray:
        semi = self.size[:3]
        scaled = p / semi
        r = np.linalg.norm(scaled, axis=1)
        # Approximate (exact ellipsoid SDF has no closed form)
        return (r - 1.0) * np.min(semi)

    def _sdf_mesh_approx(self, p: NDArray) -> NDArray:
        """True (approximate) mesh SDF via closest-point distance + sign.

        For each query point:
        1. Find the closest point on the triangle soup (exact unsigned distance).
        2. Determine inside/outside via the pseudo-normal sign test:
           dot(p - closest, face_normal). If negative, the point is inside, so SDF < 0.

        This replaces the old bounding-sphere surrogate which was
        fundamentally wrong: a point could be deeply inside the actual
        mesh yet outside the bounding sphere, making the penetration
        barrier, SDF refinement, and contact validation all useless.
        """
        if len(self.vertices) == 0 or len(self.faces) == 0:
            return np.zeros(len(p))

        n = len(p)
        v0 = self.vertices[self.faces[:, 0]]  # (F, 3)
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        sdf = np.zeros(n)
        for i in range(n):
            # Vector-based closest-point on each triangle
            q = p[i]  # (3,)
            e0 = v1 - v0  # (F, 3)
            e1 = v2 - v0
            v = q[None, :] - v0  # (F, 3)

            d00 = np.sum(e0 * e0, axis=1)  # (F,)
            d01 = np.sum(e0 * e1, axis=1)
            d11 = np.sum(e1 * e1, axis=1)
            d20 = np.sum(v * e0, axis=1)
            d21 = np.sum(v * e1, axis=1)

            denom = d00 * d11 - d01 * d01
            denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
            s = (d11 * d20 - d01 * d21) / denom
            t = (d00 * d21 - d01 * d20) / denom

            # Clamp barycentric coords to triangle
            s = np.clip(s, 0.0, 1.0)
            t = np.clip(t, 0.0, 1.0)
            excess = s + t - 1.0
            mask = excess > 0
            if np.any(mask):
                s[mask] -= excess[mask] * 0.5
                t[mask] -= excess[mask] * 0.5
                s = np.clip(s, 0.0, 1.0)
                t = np.clip(t, 0.0, 1.0)

            # Closest point on each triangle
            closest = v0 + s[:, None] * e0 + t[:, None] * e1  # (F, 3)
            diff = q[None, :] - closest  # (F, 3)
            dist_sq = np.sum(diff * diff, axis=1)  # (F,)

            best = int(np.argmin(dist_sq))
            unsigned_dist = np.sqrt(dist_sq[best])

            # Sign, dot product of (p - closest) with face normal
            # Positive means outside, negative means inside
            sign_dot = np.dot(diff[best], self.face_normals[best])
            sign = 1.0 if sign_dot >= 0.0 else -1.0

            sdf[i] = sign * unsigned_dist

        return sdf

    def _outward_dir_mesh(self, p: NDArray) -> NDArray:
        """Outward direction for mesh: gradient of the SDF.

        Uses the face normal of the closest triangle as the outward
        direction, which is the correct surface normal at the closest
        point.
        """
        if len(self.vertices) == 0 or len(self.faces) == 0:
            return self._outward_dir_sphere(p)

        n = len(p)
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        dirs = np.zeros((n, 3))
        for i in range(n):
            q = p[i]
            e0 = v1 - v0
            e1 = v2 - v0
            v = q[None, :] - v0

            d00 = np.sum(e0 * e0, axis=1)
            d01 = np.sum(e0 * e1, axis=1)
            d11 = np.sum(e1 * e1, axis=1)
            d20 = np.sum(v * e0, axis=1)
            d21 = np.sum(v * e1, axis=1)

            denom = d00 * d11 - d01 * d01
            denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
            s = (d11 * d20 - d01 * d21) / denom
            t = (d00 * d21 - d01 * d20) / denom

            s = np.clip(s, 0.0, 1.0)
            t = np.clip(t, 0.0, 1.0)
            excess = s + t - 1.0
            mask = excess > 0
            if np.any(mask):
                s[mask] -= excess[mask] * 0.5
                t[mask] -= excess[mask] * 0.5
                s = np.clip(s, 0.0, 1.0)
                t = np.clip(t, 0.0, 1.0)

            closest = v0 + s[:, None] * e0 + t[:, None] * e1
            diff = q[None, :] - closest
            dist_sq = np.sum(diff * diff, axis=1)
            best = int(np.argmin(dist_sq))

            # Use the face normal of the nearest triangle
            normal = self.face_normals[best].copy()
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                dirs[i] = normal / norm
            else:
                # Fallback: direction from closest point to query
                d = diff[best]
                dn = np.linalg.norm(d)
                dirs[i] = d / dn if dn > 1e-10 else np.array([0, 0, 1.0])

        return dirs

    # Outward direction primitives (object-local)

    def _outward_dir_box(self, p: NDArray) -> NDArray:
        half = self.size[:3]
        n = len(p)
        dirs = np.zeros((n, 3))
        for i in range(n):
            # Signed distance to each of the 6 faces
            dists = np.array([
                half[0] - p[i, 0],   # +x
                half[0] + p[i, 0],   # -x
                half[1] - p[i, 1],   # +y
                half[1] + p[i, 1],   # -y
                half[2] - p[i, 2],   # +z
                half[2] + p[i, 2],   # -z
            ])
            face = int(np.argmin(np.abs(dists)))
            axis = face // 2
            sign = 1.0 if face % 2 == 0 else -1.0
            dirs[i, axis] = sign
        return dirs

    def _outward_dir_sphere(self, p: NDArray) -> NDArray:
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return p / norms

    def _outward_dir_cylinder(self, p: NDArray) -> NDArray:
        r, h = self.size[0], self.size[1]
        n = len(p)
        dirs = np.zeros((n, 3))
        for i in range(n):
            rxy = np.sqrt(p[i, 0] ** 2 + p[i, 1] ** 2)
            d_lat = abs(rxy - r)
            d_top = abs(p[i, 2] - h)
            d_bot = abs(p[i, 2] + h)
            nearest = int(np.argmin([d_lat, d_top, d_bot]))
            if nearest == 0:
                if rxy > 1e-10:
                    dirs[i, 0] = p[i, 0] / rxy
                    dirs[i, 1] = p[i, 1] / rxy
                else:
                    dirs[i, 0] = 1.0
            elif nearest == 1:
                dirs[i, 2] = 1.0
            else:
                dirs[i, 2] = -1.0
        return dirs

    def _outward_dir_capsule(self, p: NDArray) -> NDArray:
        h = self.size[1]
        z_clamped = np.clip(p[:, 2], -h, h)
        closest = np.column_stack([np.zeros(len(p)), np.zeros(len(p)), z_clamped])
        diff = p - closest
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return diff / norms

    def _outward_dir_ellipsoid(self, p: NDArray) -> NDArray:
        semi = self.size[:3]
        # Gradient of the implicit F(p) = (x/a)^2 + (y/b)^2 + (z/c)^2 - 1
        grad = 2.0 * p / (semi ** 2)
        norms = np.linalg.norm(grad, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return grad / norms

    # 
    # Primitive samplers
    # 

    def _sample_box(self, n: int, rng: np.random.Generator) -> Tuple[NDArray, NDArray]:
        """Uniform sampling on the 6 faces of an axis-aligned box.

        MuJoCo box size = half-extents (sx, sy, sz).
        """
        sx, sy, sz = self.size[:3]
        areas = np.array([
            sy * sz,  # +x
            sy * sz,  # -x
            sx * sz,  # +y
            sx * sz,  # -y
            sx * sy,  # +z
            sx * sy,  # -z
        ]) * 4  # each face is 2*sa x 2*sb
        probs = areas / areas.sum()

        face_idx = rng.choice(6, size=n, p=probs)
        u = rng.uniform(-1, 1, size=n)
        v = rng.uniform(-1, 1, size=n)

        pts = np.zeros((n, 3))
        nrm = np.zeros((n, 3))

        for fi, (fixed_ax, sign, a_ax, b_ax, ha, hb) in enumerate([
            (0, +1, 1, 2, sy, sz),   # +x face
            (0, -1, 1, 2, sy, sz),   # -x face
            (1, +1, 0, 2, sx, sz),   # +y face
            (1, -1, 0, 2, sx, sz),   # -y face
            (2, +1, 0, 1, sx, sy),   # +z face
            (2, -1, 0, 1, sx, sy),   # -z face
        ]):
            mask = face_idx == fi
            k = mask.sum()
            if k == 0:
                continue
            pts[mask, fixed_ax] = sign * self.size[fixed_ax]
            pts[mask, a_ax] = u[mask] * ha
            pts[mask, b_ax] = v[mask] * hb
            nrm[mask, fixed_ax] = sign

        return pts, nrm

    def _sample_sphere(self, n: int, rng: np.random.Generator) -> Tuple[NDArray, NDArray]:
        """Uniform sampling on a sphere of radius size[0]"""
        r = self.size[0]
        # Marsaglia method for uniform sphere surface
        z = rng.uniform(-1, 1, size=n)
        phi = rng.uniform(0, 2 * np.pi, size=n)
        rho = np.sqrt(1 - z ** 2)
        pts = np.column_stack([rho * np.cos(phi), rho * np.sin(phi), z]) * r
        nrm = pts / r  # unit outward normals
        return pts, nrm

    def _sample_cylinder(self, n: int, rng: np.random.Generator) -> Tuple[NDArray, NDArray]:
        """Uniform sampling on a cylinder (MuJoCo: radius=size[0], half-height=size[1]).

        The cylinder axis is Z in the geom frame.
        """
        r = self.size[0]
        h = self.size[1]  # half-height

        # Areas: top cap, bottom cap, lateral
        a_cap = np.pi * r ** 2
        a_lat = 2 * np.pi * r * (2 * h)
        total = 2 * a_cap + a_lat
        p_top = a_cap / total
        p_bot = a_cap / total
        region = rng.uniform(size=n)
        is_top = region < p_top
        is_bot = (region >= p_top) & (region < p_top + p_bot)
        is_lat = ~(is_top | is_bot)

        pts = np.zeros((n, 3))
        nrm = np.zeros((n, 3))

        # Caps (uniform disk sampling)
        for mask, sign in [(is_top, +1), (is_bot, -1)]:
            k = mask.sum()
            if k == 0:
                continue
            # Concentric disk mapping
            u = rng.uniform(0, 1, size=k)
            theta = rng.uniform(0, 2 * np.pi, size=k)
            rr = r * np.sqrt(u)
            pts[mask, 0] = rr * np.cos(theta)
            pts[mask, 1] = rr * np.sin(theta)
            pts[mask, 2] = sign * h
            nrm[mask, 2] = sign

        # Lateral
        k = is_lat.sum()
        if k > 0:
            theta = rng.uniform(0, 2 * np.pi, size=k)
            z = rng.uniform(-h, h, size=k)
            pts[is_lat, 0] = r * np.cos(theta)
            pts[is_lat, 1] = r * np.sin(theta)
            pts[is_lat, 2] = z
            nrm[is_lat, 0] = np.cos(theta)
            nrm[is_lat, 1] = np.sin(theta)

        return pts, nrm

    def _sample_capsule(self, n: int, rng: np.random.Generator) -> Tuple[NDArray, NDArray]:
        """Uniform sampling on a capsule (cylinder + two hemispherical caps)"""
        r = self.size[0]
        h = self.size[1]  # half-height of cylindrical part

        a_hemi = 2 * np.pi * r ** 2        # each hemisphere
        a_lat = 2 * np.pi * r * (2 * h)    # cylinder lateral
        total = 2 * a_hemi + a_lat
        p_top = a_hemi / total
        p_bot = a_hemi / total

        region = rng.uniform(size=n)
        is_top = region < p_top
        is_bot = (region >= p_top) & (region < p_top + p_bot)
        is_lat = ~(is_top | is_bot)

        pts = np.zeros((n, 3))
        nrm = np.zeros((n, 3))

        # Hemispheres
        for mask, sign in [(is_top, +1), (is_bot, -1)]:
            k = mask.sum()
            if k == 0:
                continue
            z = rng.uniform(0, 1, size=k)  # hemisphere uses [0,1]
            phi = rng.uniform(0, 2 * np.pi, size=k)
            rho = np.sqrt(1 - z ** 2)
            pts[mask, 0] = r * rho * np.cos(phi)
            pts[mask, 1] = r * rho * np.sin(phi)
            pts[mask, 2] = sign * (h + r * z)
            nrm[mask, 0] = rho * np.cos(phi)
            nrm[mask, 1] = rho * np.sin(phi)
            nrm[mask, 2] = sign * z

        # Lateral
        k = is_lat.sum()
        if k > 0:
            theta = rng.uniform(0, 2 * np.pi, size=k)
            z = rng.uniform(-h, h, size=k)
            pts[is_lat, 0] = r * np.cos(theta)
            pts[is_lat, 1] = r * np.sin(theta)
            pts[is_lat, 2] = z
            nrm[is_lat, 0] = np.cos(theta)
            nrm[is_lat, 1] = np.sin(theta)

        return pts, nrm

    def _sample_ellipsoid(self, n: int, rng: np.random.Generator) -> Tuple[NDArray, NDArray]:
        """Approximate-uniform sampling on an ellipsoid (size = semi-axes)"""
        # Sample on unit sphere, scale by semi-axes.  Not perfectly uniform
        # (area element is scaled by the Jacobian) but acceptable for contact
        # candidate generation.
        a, b, c = self.size[:3]
        z = rng.uniform(-1, 1, size=n)
        phi = rng.uniform(0, 2 * np.pi, size=n)
        rho = np.sqrt(1 - z ** 2)
        unit = np.column_stack([rho * np.cos(phi), rho * np.sin(phi), z])
        pts = unit * np.array([a, b, c])
        # Normal on ellipsoid surface, gradient of F = (2x/a^2, 2y/b^2, 2z/c^2)
        nrm_raw = unit / np.array([a, b, c])
        nrm = nrm_raw / np.linalg.norm(nrm_raw, axis=1, keepdims=True)
        return pts, nrm

    # 
    # Mesh sampling helpers
    # 

    def _precompute_mesh(self) -> None:
        """Compute per-face areas and normals from vertices/faces"""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        area = np.linalg.norm(cross, axis=1) * 0.5
        # Guard against degenerate faces
        nrm = np.zeros_like(cross)
        ok = area > 1e-12
        nrm[ok] = cross[ok] / (2 * area[ok, None])
        self.face_areas = area
        self.face_normals = nrm

    def _sample_mesh(self, n: int, rng: np.random.Generator) -> Tuple[NDArray, NDArray]:
        """Sample uniformly on a triangle mesh surface"""
        if HAS_OPEN3D and len(self.vertices) > 0:
            return self._sample_mesh_open3d(n, rng)
        return self._sample_mesh_bary(n, rng)

    def _sample_mesh_bary(self, n: int, rng: np.random.Generator) -> Tuple[NDArray, NDArray]:
        """Barycentric triangle sampling (no Open3D)"""
        if len(self.face_areas) == 0:
            return np.zeros((n, 3)), np.zeros((n, 3))
        probs = self.face_areas / self.face_areas.sum()
        fi = rng.choice(len(self.faces), size=n, p=probs)
        # Random barycentric coords
        r1 = rng.uniform(size=n)
        r2 = rng.uniform(size=n)
        sqrt_r1 = np.sqrt(r1)
        u = 1 - sqrt_r1
        v = r2 * sqrt_r1
        w = 1 - u - v
        v0 = self.vertices[self.faces[fi, 0]]
        v1 = self.vertices[self.faces[fi, 1]]
        v2 = self.vertices[self.faces[fi, 2]]
        pts = u[:, None] * v0 + v[:, None] * v1 + w[:, None] * v2
        nrm = self.face_normals[fi]
        return pts, nrm

    def _sample_mesh_open3d(self, n: int, rng: np.random.Generator) -> Tuple[NDArray, NDArray]:
        """Sample via Open3D (more robust normal estimation)"""
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(self.vertices),
            triangles=o3d.utility.Vector3iVector(self.faces),
        )
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=n)
        pts = np.asarray(pcd.points)
        nrm = np.asarray(pcd.normals)

        # Orient normals outward via centroid heuristic
        centroid = pts.mean(axis=0)
        dirs = pts - centroid
        flip = np.sum(dirs * nrm, axis=1) < 0
        nrm[flip] *= -1

        return pts, nrm
