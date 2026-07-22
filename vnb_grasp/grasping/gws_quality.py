"""Grasp Wrench Space (GWS) quality metric for MuJoCo contacts.

Port of the GraspIt! GWS analysis to work with MuJoCo contact data.

The GWS is the set of all wrenches (forces and torques) that can be
applied to the object through the grasp contacts. A grasp is force-closure
if the GWS contains the origin in its interior.

Quality metrics:
- epsilon: Largest inscribed ball radius in GWS (Ferrari-Canny metric)
- volume: Volume of the GWS convex hull
- min_singular: Smallest singular value of the grasp matrix

References:
- Ferrari & Canny, "Planning Optimal Grasps", ICRA 1992
- Miller & Allen, "GraspIt!", IEEE R&A Magazine 2004

Author: Clinton Enwerem
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.spatial import ConvexHull, QhullError
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..belief.mujoco_rollout import ContactInfo


@dataclass
class GWSResult:
    """Result of GWS analysis"""
    epsilon: float          # Ferrari-Canny quality ; largest inscribed ball
    volume: float           # GWS volume
    min_singular: float     # Minimum singular value of grasp matrix
    is_force_closure: bool  # True if origin is inside GWS
    n_contacts: int         # Number of contacts used
    
    def quality(self) -> float:
        """Primary quality metric (normalized epsilon)"""
        if not self.is_force_closure:
            return 0.0
        # Normalize by typical max value
        return min(self.epsilon / 0.5, 1.0)


def build_grasp_matrix(
    contacts: List[ContactInfo],
    object_center: NDArray,
    friction_coef: float = 0.5,
    n_friction_edges: int = 8,
) -> NDArray:
    """Build the grasp matrix G from contacts.

    For each contact with friction cone approximation, we get primitive
    wrenches that span the achievable wrench space at that contact.

    Args:
        contacts: List of contact info from MuJoCo
        object_center: Object center of mass position
        friction_coef: Friction coefficient for cone linearization
        n_friction_edges: Number of edges to approximate friction cone

    Returns:
        G: (6, n_wrenches) grasp matrix where each column is a primitive wrench
    """
    if len(contacts) == 0:
        return np.zeros((6, 1))

    # Generate friction cone edge directions
    angles = np.linspace(0, 2 * np.pi, n_friction_edges, endpoint=False)
    
    all_wrenches = []

    for contact in contacts:
        # Contact normal and tangent directions
        normal = contact.normal
        
        # Build local frame at contact
        if abs(normal[2]) < 0.9:
            tangent1 = np.cross(normal, np.array([0, 0, 1]))
        else:
            tangent1 = np.cross(normal, np.array([1, 0, 0]))
        tangent1 /= np.linalg.norm(tangent1)
        tangent2 = np.cross(normal, tangent1)

        # Friction cone edges
        for angle in angles:
            # Direction on friction cone
            f_dir = (
                normal +
                friction_coef * np.cos(angle) * tangent1 +
                friction_coef * np.sin(angle) * tangent2
            )
            f_dir /= np.linalg.norm(f_dir)

            # Moment arm from object center to contact
            r = contact.pos - object_center

            # Wrench = [force; torque]
            torque = np.cross(r, f_dir)
            wrench = np.concatenate([f_dir, torque])
            all_wrenches.append(wrench)

    return np.array(all_wrenches).T  # ; 6, n_wrenches


def compute_gws_epsilon(
    grasp_matrix: NDArray,
    max_iter: int = 1000,
) -> Tuple[float, bool]:
    """Compute Ferrari-Canny epsilon quality metric.

    Epsilon is the radius of the largest ball centered at origin
    that fits inside the GWS convex hull.

    Args:
        grasp_matrix: (6, n_wrenches) matrix of primitive wrenches
        max_iter: Max iterations for convex hull

    Returns:
        (epsilon, is_force_closure)
    """
    if not HAS_SCIPY:
        # Fallback: use minimum singular value as proxy
        _, s, _ = np.linalg.svd(grasp_matrix, full_matrices=False)
        min_sv = s[-1] if len(s) > 0 else 0.0
        return min_sv, min_sv > 1e-6

    try:
        # Build convex hull of wrench columns in 6D
        points = grasp_matrix.T  # ; n_wrenches, 6
        
        if points.shape[0] < 7:
            # Not enough points for 6D hull
            return 0.0, False

        hull = ConvexHull(points)

        # Check if origin is inside hull
        # For each facet, check if origin is on the inside
        origin = np.zeros(6)
        
        min_dist = float('inf')
        for eq in hull.equations:
            # eq is [a1, a2, ..., a6, offset] where a·x + offset <= 0
            normal = eq[:-1]
            offset = eq[-1]
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-15:
                continue

            # Signed distance from origin to hyperplane:
            #   dist = (a·0 + offset) / ||a|| = offset / ||a||
            # Negative when origin is on the feasible (interior) side.
            signed_dist = offset / normal_norm

            if signed_dist > 1e-10:
                # Origin is outside this facet --> not force closure
                return 0.0, False

            min_dist = min(min_dist, abs(signed_dist))

        return min_dist, True

    except QhullError:
        return 0.0, False


def analyze_gws(
    contacts: List[ContactInfo],
    object_center: NDArray,
    friction_coef: float = 0.5,
) -> GWSResult:
    """Analyze grasp wrench space for a set of contacts.

    Args:
        contacts: List of MuJoCo contacts
        object_center: Object center of mass
        friction_coef: Friction coefficient

    Returns:
        GWSResult with quality metrics
    """
    if len(contacts) < 2:
        return GWSResult(
            epsilon=0.0,
            volume=0.0,
            min_singular=0.0,
            is_force_closure=False,
            n_contacts=len(contacts),
        )

    # Build grasp matrix
    G = build_grasp_matrix(contacts, object_center, friction_coef)

    # Singular value analysis
    _, s, _ = np.linalg.svd(G, full_matrices=False)
    min_singular = float(s[-1]) if len(s) > 0 else 0.0

    # Epsilon quality
    epsilon, is_fc = compute_gws_epsilon(G)

    # Compute Volume ; scale to readable range
    volume = 0.0
    if HAS_SCIPY and G.shape[1] >= 7:
        try:
            hull = ConvexHull(G.T)
            # Scale volume for readability ; 6D hull volumes are tiny
            volume = hull.volume * 1e6  # Scale by 10^6
        except QhullError:
            pass

    return GWSResult(
        epsilon=epsilon,
        volume=volume,
        min_singular=min_singular,
        is_force_closure=is_fc,
        n_contacts=len(contacts),
    )


def ferrari_canny_quality(
    contacts: List[ContactInfo],
    object_center: NDArray,
    friction_coef: float = 0.5,
) -> float:
    """Convenience function for Ferrari-Canny epsilon quality.

    Args:
        contacts: MuJoCo contacts
        object_center: Object COM position
        friction_coef: Friction coefficient

    Returns:
        Quality in [0, 1] (0 if not force-closure)
    """
    result = analyze_gws(contacts, object_center, friction_coef)
    return result.quality()
