"""Belief-space planning for VNB-Grasp.

Port of the GraspIt! belief-space grasping framework to MuJoCo dynamics.

Modules:
- particle_filter: Generic particle belief with CVaR/VaR risk metrics
- contact_belief: Contact-specific belief for dexterous grasping
- belief_mpc: Risk-sensitive receding-horizon planner
- mujoco_rollout: MuJoCo-specific rollout and contact extraction
- gws_quality: Grasp Wrench Space quality metrics
- differentiable_metrics: JAX-based differentiable grasp quality metrics
"""

from .particle_filter import ParticleBelief, cvar, var, failure_probability
from .contact_belief import (
    ContactMode,
    LatentContactState,
    GraspParticle,
    GraspObservation,
    initialize_grasp_belief,
    default_observation_likelihood,
    friction_violation_likelihood,
    belief_mean_friction,
)
from .belief_mpc import BeliefMPCConfig, BeliefMPCPlanner, GraspAction, ActionType

# Variational Neural Beliefs ; new continuous belief representations
try:
    from .variational_belief import (
        VariationalBeliefConfig,
        VariationalBelief,
        GaussianMixtureBelief,
        ImplicitNeuralBelief,
        SIRENNetwork,
        NeuralBeliefFilter,
        AdaptiveBeliefManager,
        RiskAwareNeuralPolicy,
    )
    HAS_VARIATIONAL_BELIEFS = True
except ImportError:
    HAS_VARIATIONAL_BELIEFS = False
from .mujoco_rollout import (
    SimState,
    ContactInfo,
    extract_contacts,
    get_fingertip_geom_ids,
    ParticleRolloutEngine,
    compute_grasp_quality_from_contacts,
    friction_cone_violation,
)
from ..grasping.gws_quality import GWSResult, analyze_gws, ferrari_canny_quality

# Differentiable metrics ; requires JAX
try:
    from .differentiable_metrics import (
        # Data structures
        ContactState,
        GraspQuality,
        DifferentiableMetricsConfig,
        # Soft operations
        soft_min,
        soft_max,
        soft_indicator,
        soft_relu,
        # Core metrics
        compute_wrench_space,
        soft_epsilon_metric,
        soft_volume_metric,
        soft_slip_margin,
        soft_disturbance_margin,
        soft_contact_count,
        grasp_fragility,
        # Belief-integrated metrics
        cvar_metric,
        belief_robust_epsilon,
        expected_quality,
        quality_variance,
        failure_probability as diff_failure_probability,
        # End-to-end evaluation
        make_differentiable_grasp_evaluator,
        # Optimization
        optimize_grasp_gradient,
        optimize_grasp_adam,
        check_gradients,
    )
    HAS_DIFFERENTIABLE_METRICS = True
except ImportError:
    HAS_DIFFERENTIABLE_METRICS = False

__all__ = [
    # Particle filter
    "ParticleBelief",
    "cvar",
    "var", 
    "failure_probability",
    # Contact belief
    "ContactMode",
    "LatentContactState",
    "GraspParticle",
    "GraspObservation",
    "initialize_grasp_belief",
    "default_observation_likelihood",
    "friction_violation_likelihood",
    "belief_mean_friction",
    # MPC
    "BeliefMPCConfig",
    "BeliefMPCPlanner",
    "GraspAction",
    "ActionType",
    # Variational Neural Beliefs ; new
    "HAS_VARIATIONAL_BELIEFS",
    "VariationalBeliefConfig",
    "VariationalBelief",
    "GaussianMixtureBelief",
    "ImplicitNeuralBelief",
    "SIRENNetwork",
    "NeuralBeliefFilter",
    "AdaptiveBeliefManager",
    "RiskAwareNeuralPolicy",
    # MuJoCo rollout
    "SimState",
    "ContactInfo",
    "extract_contacts",
    "get_fingertip_geom_ids",
    "ParticleRolloutEngine",
    "compute_grasp_quality_from_contacts",
    "friction_cone_violation",
    # GWS quality
    "GWSResult",
    "analyze_gws",
    "ferrari_canny_quality",
    # Differentiable metrics ; JAX
    "HAS_DIFFERENTIABLE_METRICS",
    "ContactState",
    "GraspQuality",
    "DifferentiableMetricsConfig",
    "soft_min",
    "soft_max",
    "soft_indicator",
    "soft_relu",
    "compute_wrench_space",
    "soft_epsilon_metric",
    "soft_volume_metric",
    "soft_slip_margin",
    "soft_disturbance_margin",
    "soft_contact_count",
    "grasp_fragility",
    "cvar_metric",
    "belief_robust_epsilon",
    "expected_quality",
    "quality_variance",
    "diff_failure_probability",
    "make_differentiable_grasp_evaluator",
    "optimize_grasp_gradient",
    "optimize_grasp_adam",
    "check_gradients",
]
