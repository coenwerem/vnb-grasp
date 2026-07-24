"""Contact-specific belief representations for dexterous grasping.

Defines the latent state space and observation model for belief-space
grasp planning in MuJoCo.

Key differences from GraspIt! version:
- MuJoCo provides continuous contact forces (not quasi-static solve)
- Can use tactile sensor readings directly
- Contact dynamics are richer (friction cone, slip velocity)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .particle_filter import ParticleBelief


class ContactMode(str, Enum):
    """Discrete contact mode m_t^i.

    In MuJoCo, these map to the contact state from mj_contact:
    - separated: no contact (distance > 0)
    - sticking: contact with static friction (slip velocity  ~=  0)
    - sliding: contact with kinetic friction (slip velocity > threshold)
    """
    separated = "separated"
    sticking = "sticking"
    sliding = "sliding"


@dataclass(frozen=True)
class LatentContactState:
    """Latent per-contact parameters theta_t^i = (mu, kappa, m).

    These are the unobserved quantities that govern grasp stability.
    The belief is a distribution over these parameters.

    Attributes:
        mu: Coefficient of friction (Coulomb model)
        kappa: Contact stiffness/compliance (N/m)
        mode: Discrete contact mode (sticking/sliding/separated)
        slip: Slip velocity magnitude (m/s), observable proxy
    """
    mu: float           # Friction coefficient
    kappa: float        # Stiffness
    mode: ContactMode = ContactMode.sticking
    slip: float = 0.0   # Slip indicator


@dataclass(frozen=True)
class GraspParticle:
    """One particle = one hypothesis about latent contact parameters.

    The full state x_t = (x_t^obs, x_t^lat) where:
    - x_t^obs is observed (joint positions, object pose, contact geometry)
    - x_t^lat = theta_t is the tuple of LatentContactState per contact

    In MuJoCo, observable state comes from env.obs; we only track latent.
    """
    contacts: Tuple[LatentContactState, ...]

    @property
    def n_contacts(self) -> int:
        return len(self.contacts)

    def friction_array(self) -> NDArray:
        """Extract friction coefficients as array"""
        return np.array([c.mu for c in self.contacts], dtype=np.float64)

    def stiffness_array(self) -> NDArray:
        """Extract stiffness values as array"""
        return np.array([c.kappa for c in self.contacts], dtype=np.float64)


@dataclass
class GraspObservation:
    """Multimodal observation o_t = (o_t^v, o_t^tac, o_t^p).

    This wraps the VNB-Grasp MultimodalObs for belief updates.

    Attributes:
        q: Joint positions (proprioception)
        dq: Joint velocities (proprioception)
        contact_forces: Normal force at each contact (from MuJoCo)
        contact_points: Contact locations in world frame
        contact_normals: Surface normals at contacts
        slip_velocity: Tangential slip velocity per contact (d/2 * ang_vel)
        tactile_pressure: Tactile sensor readings
        gripper_effort: Motor effort / commanded torque
    """
    q: NDArray                                      # ; n_dof,
    dq: NDArray                                     # ; n_dof,
    contact_forces: Optional[NDArray] = None        # ; n_contacts,
    contact_points: Optional[NDArray] = None        # ; n_contacts, 3
    contact_normals: Optional[NDArray] = None       # ; n_contacts, 3
    slip_velocity: Optional[NDArray] = None         # ; n_contacts,
    tactile_pressure: Optional[NDArray] = None      # ; n_fingers, rows, cols
    gripper_effort: Optional[NDArray] = None        # ; n_finger_joints,

    @classmethod
    def from_mujoco_data(cls, data, model, config=None) -> "GraspObservation":
        """Extract observation from MuJoCo data structure.

        Args:
            data: mujoco.MjData
            model: mujoco.MjModel
            config: Optional MultimodalObsConfig

        Returns:
            GraspObservation populated from simulator state
        """
        q = data.qpos.copy()
        dq = data.qvel.copy()

        n_contacts = data.ncon
        if n_contacts > 0:
            contact_forces = np.zeros(n_contacts, dtype=np.float64)
            contact_points = np.zeros((n_contacts, 3), dtype=np.float64)
            contact_normals = np.zeros((n_contacts, 3), dtype=np.float64)

            for i in range(n_contacts):
                contact = data.contact[i]
                contact_points[i] = contact.pos
                # Frame columns: normal is first column
                contact_normals[i] = contact.frame[:3]
        else:
            contact_forces = None
            contact_points = None
            contact_normals = None

        return cls(
            q=q,
            dq=dq,
            contact_forces=contact_forces,
            contact_points=contact_points,
            contact_normals=contact_normals,
        )


def initialize_grasp_belief(
    n_particles: int,
    n_contacts: int,
    *,
    mu_prior: Tuple[float, float] = (0.6, 0.15),
    kappa_prior: Tuple[float, float] = (1000.0, 250.0),
    mu_prior_mixture: Optional[Tuple[Tuple[float, float], Tuple[float, float], float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> ParticleBelief[GraspParticle]:
    """Create initial belief over latent contact parameters.

    Samples friction and compliance from Gaussian priors.
    All particles start with sticking mode and zero slip.

    Args:
        n_particles: Number of particles (N_p in paper, default 100)
        n_contacts: Number of potential contact sites (fingertips)
        mu_prior: (mean, std) for friction prior (used if mu_prior_mixture is None)
        kappa_prior: (mean, std) for stiffness prior
        mu_prior_mixture: Optional mixture prior for friction:
            ((mu_normal, std_normal), (mu_tail, std_tail), p_tail)
            E.g., ((0.85, 0.05), (0.30, 0.05), 0.2) for 80% normal, 20% tail
        rng: Random generator for reproducibility

    Returns:
        Initial ParticleBelief with uniform weights
    """
    if rng is None:
        rng = np.random.default_rng(42)

    kappa_mean, kappa_std = kappa_prior
    
    # Determine friction sampling strategy
    if mu_prior_mixture is not None:
        (mu_normal, std_normal), (mu_tail, std_tail), p_tail = mu_prior_mixture
        use_mixture = True
    else:
        mu_mean, mu_std = mu_prior
        use_mixture = False

    particles = []
    for _ in range(n_particles):
        contacts = []
        for _ in range(n_contacts):
            # Sample friction from mixture or single Gaussian
            if use_mixture:
                if rng.random() < p_tail:
                    mu = float(np.clip(rng.normal(mu_tail, std_tail), 0.05, 2.0))
                else:
                    mu = float(np.clip(rng.normal(mu_normal, std_normal), 0.05, 2.0))
            else:
                mu = float(np.clip(rng.normal(mu_mean, mu_std), 0.05, 2.0))
            
            kappa = float(max(10.0, rng.normal(kappa_mean, kappa_std)))
            contacts.append(LatentContactState(
                mu=mu,
                kappa=kappa,
                mode=ContactMode.sticking,
                slip=0.0,
            ))
        particles.append(GraspParticle(contacts=tuple(contacts)))

    weights = np.ones(n_particles, dtype=np.float64)
    return ParticleBelief(
        particles=np.array(particles, dtype=object),
        weights=weights,
    )


def default_observation_likelihood(
    obs: GraspObservation,
    particle: GraspParticle,
    *,
    sigma_force: float = 0.5,
    beta_slip: float = 10.0,
) -> float:
    """Observation likelihood p(o_t | theta_t) for belief update.

    Evaluates how consistent the observed contact behavior is with the
    particle's friction hypothesis. This is the primary driver of belief
    contraction  --  particles whose friction predictions disagree with
    observed slip/force behavior get down-weighted.

    High friction particles should predict:
    - Low slip velocities (contacts are sticking)
    - High sustainable normal forces

    Low friction particles should predict:
    - Higher slip velocities (contacts slide more easily)
    - Lower sustainable forces before slip

    The likelihood must be DISCRIMINATIVE: different friction hypotheses
    must yield meaningfully different likelihood values for the same
    observation. Otherwise weights stay uniform and entropy never contracts.

    Args:
        obs: Current observation
        particle: Latent state hypothesis
        sigma_force: Force observation noise scale
        beta_slip: Sensitivity of slip penalty (higher = more discriminative)

    Returns:
        Likelihood p(obs | particle)  in  (0, 1]
    """
    if particle.n_contacts == 0:
        return 1.0

    log_lik = 0.0

    if obs.slip_velocity is not None and len(obs.slip_velocity) > 0:
        # slip_velocity here is actually friction_ratio = ||f_t|| / f_n
        # This is the ratio of tangential to normal force at each contact.
        #
        # In MuJoCo, the solver enforces Coulomb friction:
        #   friction_ratio  <=  mu_true
        # So observing friction_ratio = r implies mu_true  >=  r.
        #
        # Likelihood model ; soft friction cone constraint:
        #   p; r | mu is proportional to sigmoid; beta * (mu - r)
        #
        # - mu >> r: contact safely inside friction cone  ->  high likelihood
        # - mu  ~=  r: contact at cone boundary  ->  moderate likelihood
        # - mu < r: friction cone VIOLATED  ->  very low likelihood
        #
        # This creates correct discrimination:
        # - High friction_ratio eliminates low-mu particles
        # - Low friction_ratio is consistent with any mu  ->  less informative

        k = min(len(obs.slip_velocity), particle.n_contacts)
        mu = particle.friction_array()[:k]
        slip = np.asarray(obs.slip_velocity)[:k]

        for j in range(k):
            # margin > 0: inside cone ; consistent, < 0: violated ; inconsistent
            margin = mu[j] - slip[j]
            # log; sigmoid(x) = -log; 1 + exp(-x)
            # Clip to prevent overflow: exp(-x) overflows when x < -709
            clipped_arg = np.clip(-beta_slip * margin, -700, 700)
            log_lik += -np.log1p(np.exp(clipped_arg))

    elif obs.contact_forces is not None and len(obs.contact_forces) > 0:
        # Use force magnitudes as proxy
        k = min(len(obs.contact_forces), particle.n_contacts)
        mu = particle.friction_array()[:k]
        forces = np.asarray(obs.contact_forces)[:k]

        # High forces sustained without slip  ->  consistent with high friction
        f_max = np.max(np.abs(forces)) + 1e-6
        forces_norm = np.abs(forces) / f_max

        for j in range(k):
            # Higher friction particles can sustain higher force ratios
            # p; high_force | high_mu > p; high_force | low_mu
            log_lik += forces_norm[j] * np.log(mu[j] + 0.1) * 2.0

    else:
        # No contact observations  --  slightly favor lower-friction hypotheses
        # ; prior regularization: without evidence, be cautious
        mu_mean = float(np.mean(particle.friction_array()))
        log_lik = -0.1 * mu_mean

    return float(np.exp(np.clip(log_lik, -50.0, 10.0)))


def friction_violation_likelihood(
    obs: GraspObservation,
    particle: GraspParticle,
    *,
    beta_violation: float = 10.0,
) -> float:
    """Likelihood based on friction cone violation.

    If observed tangent/normal force ratio exceeds particle's friction
    coefficient, the particle is penalized (it should have slipped).

    Args:
        obs: Current observation with contact forces
        particle: Latent state hypothesis
        beta_violation: Penalty scale for violations

    Returns:
        Likelihood value
    """
    if obs.contact_forces is None or particle.n_contacts == 0:
        return 1.0

    # In MuJoCo, we'd compute mu_required = ||f_t|| / f_n per contact
    # For now, use slip_velocity as proxy for friction requirement
    if obs.slip_velocity is None:
        return 1.0

    k = min(len(obs.slip_velocity), particle.n_contacts)
    mu_particle = particle.friction_array()[:k]

    # Higher slip velocity suggests friction is insufficient
    # If particle.mu is low and slip is high, that's consistent ; likelihood = 1
    # If particle.mu is high but slip is high, that's inconsistent ; likelihood < 1
    slip = np.asarray(obs.slip_velocity)[:k]
    slip_threshold = 0.01  # m/s

    # Particles that predict sticking but observe slip are penalized
    violation = np.sum(
        np.maximum(0.0, slip - slip_threshold) * mu_particle
    )

    if violation < 1e-9:
        return 1.0

    return float(np.exp(-beta_violation * violation))

# Belief statistics for cost functions
def belief_mean_friction(belief: ParticleBelief[GraspParticle]) -> NDArray:
    """Posterior mean friction per contact: E_b[mu^i]"""
    if belief.n == 0:
        return np.array([])

    n_contacts = belief.particles[0].n_contacts
    if n_contacts == 0:
        return np.array([])

    mus = np.zeros((belief.n, n_contacts), dtype=np.float64)
    for i, p in enumerate(belief.particles):
        mus[i] = p.friction_array()

    return np.average(mus, weights=belief.weights, axis=0)

def belief_variance_friction(belief: ParticleBelief[GraspParticle]) -> NDArray:
    """Posterior variance of friction per contact: Var_b[mu^i]"""
    if belief.n == 0:
        return np.array([])

    n_contacts = belief.particles[0].n_contacts
    if n_contacts == 0:
        return np.array([])

    mus = np.zeros((belief.n, n_contacts), dtype=np.float64)
    for i, p in enumerate(belief.particles):
        mus[i] = p.friction_array()

    mean = np.average(mus, weights=belief.weights, axis=0)
    diff = mus - mean
    return np.average(diff ** 2, weights=belief.weights, axis=0)


def estimate_slip_probability(
    belief: ParticleBelief[GraspParticle],
    required_mu: NDArray,
    margin: float = 0.0,
) -> float:
    """Estimate P(slip) = P(mu < mu_required + margin) under belief.

    Used for the failure cost term c_F(b_t).

    Args:
        belief: Current belief
        required_mu: Required friction per contact (from force analysis)
        margin: Safety margin

    Returns:
        Probability of slip
    """
    required = np.asarray(required_mu, dtype=np.float64) + margin
    p_slip = 0.0

    for w, particle in zip(belief.weights, belief.particles):
        k = min(len(required), particle.n_contacts)
        mu = particle.friction_array()[:k]
        # Slip if any contact has insufficient friction
        if np.any(mu < required[:k]):
            p_slip += w

    return float(np.clip(p_slip, 0.0, 1.0))
