"""Grasp synthesis, quality metrics, and pregrasp planning for VNB-Grasp.

Modules
    gws_quality               Grasp wrench space analysis (Ferrari-Canny, GWS volume)
    rs_quality                Risk-sensitive quality metrics built on CVaR
    ycb_objects               YCB object definitions and pose transform utilities
    pregrasp_planner          Arm and hand pregrasp pose planning
    graspit_loader            Load grasps from the bundled grasp database
    naive_executor            Position-control grasp execution
    object_surface            Object surface sampling from MuJoCo geometry
    grasp_sampler             Sampling-based grasp candidate generation
    grasp_optimizer           SQP refinement of sampled grasps
    precision_grip_optimizer  Standoff-based precision grip synthesis
    contact_first_optimizer   Contact-first grasp synthesis
    arm_grasp_optimizer       Joint arm and hand grasp synthesis
    closure_grasp_solver      Force-closure solver helpers
"""

from .gws_quality import (
    GWSResult,
    analyze_gws,
    ferrari_canny_quality,
)

from .rs_quality import (
    ForceClosureMargin,
    SlipMargin,
    RiskSensitiveGraspQuality,
    MultiObjectiveGraspQuality,
    create_default_evaluator,
)

from .ycb_objects import (
    GraspItToMuJoCoTransform,
    YCBObjectConfig,
    get_object_config,
)

from .pregrasp_planner import (
    PregraspPlan,
    plan_pregrasp,
    plan_all_pregrasps,
)

# Grasp database loading, from graspit_loader.py
from .graspit_loader import (
    GraspItGrasp,
    GraspDatabase,
    GraspLoader,
    load_grasps,
    list_available_objects,
    transform_grasp_to_current_pose,
    REALHAND_L6_DOF_NAMES,
    GRASP_DB_PATH,
)

# Grasp execution
from .naive_executor import (
    NaiveGraspExecutor,
    ExecutionConfig,
    ExecutionResult,
    ExecutionPhase,
    execute_grasp,
)

# Sampling-based grasp solver
from .object_surface import (
    ObjectSurface,
    SurfaceSample,
    GeomKind,
)
from .grasp_sampler import (
    GraspSampler,
    SampledGrasp,
    SamplerConfig,
    ContactIKSolver,
    FingerAssigner,
    DEFAULT_FINGER_MAP,
)

# SQP-based grasp optimizer
from .grasp_optimizer import (
    GraspOptimizer,
    OptimizerConfig,
)

# Precision-grip optimizer (standoff-based, zero proximal penetration)
from .precision_grip_optimizer import (
    PrecisionGripOptimizer,
    PrecisionGripConfig,
    FingerOnlyIKSolver,
)

__all__ = [
    # GWS quality
    "GWSResult",
    "analyze_gws",
    "ferrari_canny_quality",
    # Risk-sensitive quality
    "ForceClosureMargin",
    "SlipMargin",
    "RiskSensitiveGraspQuality",
    "MultiObjectiveGraspQuality",
    "create_default_evaluator",
    # Object configuration
    "GraspItToMuJoCoTransform",
    "YCBObjectConfig",
    "get_object_config",
    # Pregrasp planning
    "PregraspPlan",
    "plan_pregrasp",
    "plan_all_pregrasps",
    # Grasp database
    "GraspItGrasp",
    "GraspDatabase",
    "GraspLoader",
    "load_grasps",
    "list_available_objects",
    "transform_grasp_to_current_pose",
    "REALHAND_L6_DOF_NAMES",
    "GRASP_DB_PATH",
    # Execution
    "NaiveGraspExecutor",
    "ExecutionConfig",
    "ExecutionResult",
    "ExecutionPhase",
    "execute_grasp",
    # Sampling-based solver
    "ObjectSurface",
    "SurfaceSample",
    "GeomKind",
    "GraspSampler",
    "SampledGrasp",
    "SamplerConfig",
    "ContactIKSolver",
    "FingerAssigner",
    "DEFAULT_FINGER_MAP",
    # SQP optimizer
    "GraspOptimizer",
    "OptimizerConfig",
    # Precision-grip optimizer
    "PrecisionGripOptimizer",
    "PrecisionGripConfig",
    "FingerOnlyIKSolver",
]
