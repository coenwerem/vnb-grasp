"""
Differentiable grasp quality metrics via JAX.

Key features:
- Smooth approximations of classical metrics (epsilon, volume, slip margin)
- Gradients through MJX simulation
- Batched computation for belief-weighted expectations
- End-to-end grasp optimization

This module enables gradient-based grasp synthesis by providing differentiable
relaxations of classical grasp quality metrics. The key insight is that hard
operations (min, max, indicator functions) can be replaced with smooth
approximations (log-sum-exp, sigmoid) that preserve gradients while closely
approximating the original metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    # Provide stubs for type hints
    jnp = None  # type: ignore
    jit = lambda x: x
    grad = lambda x: x
    vmap = lambda x: x

try:
    import mujoco.mjx as mjx
    HAS_MJX = True
except Exception:  # optional GPU accel; tolerate missing/version-skewed mjx
    HAS_MJX = False
    mjx = None  # type: ignore


# Type aliases for array types
# Use Union to handle both JAX and NumPy arrays
if TYPE_CHECKING:
    # For type checkers, use a flexible type
    Array = Union[npt.NDArray[np.floating[Any]], Any]  # Accepts both jnp and np arrays
else:
    # At runtime, just use whatever is available
    Array = Any


class ContactState(NamedTuple):
    """Differentiable contact representation from MJX.
    
    Attributes:
        positions: (n_contacts, 3) contact point locations in world frame
        normals: (n_contacts, 3) contact normal directions (into object)
        forces: (n_contacts, 3) contact forces [normal, tangent1, tangent2]
        depths: (n_contacts,) penetration depths (negative = penetrating)
        active: (n_contacts,) soft activation values in [0, 1]
    """
    positions: Array      # ; n_contacts, 3
    normals: Array        # ; n_contacts, 3
    forces: Array         # ; n_contacts, 3 - normal + tangent
    depths: Array         # ; n_contacts, - penetration
    active: Array         # ; n_contacts, - soft activation [0,1]


class GraspQuality(NamedTuple):
    """Differentiable grasp quality outputs.
    
    All fields are JAX arrays that support autodiff.
    
    Attributes:
        epsilon_soft: Soft Ferrari-Canny metric (largest inscribed ball)
        volume_soft: Soft wrench hull volume approximation
        slip_margin: Minimum distance to friction cone boundary
        disturbance_margin: Maximum resistable external disturbance
        fragility: Sensitivity of quality to latent parameter changes
        contact_count_soft: Soft count of active contacts
    """
    epsilon_soft: Array           # Soft Ferrari-Canny metric
    volume_soft: Array            # Soft wrench hull volume
    slip_margin: Array            # Distance to friction cone boundary
    disturbance_margin: Array     # Soft critical disturbance magnitude
    fragility: Array              # Sensitivity to parameter perturbation
    contact_count_soft: Array     # Soft contact count


@dataclass
class DifferentiableMetricsConfig:
    """Configuration for differentiable metric computation.
    
    Attributes:
        temperature: Softmax temperature (lower = sharper approximation)
        n_wrench_directions: Number of directions for epsilon-metric sampling
        sharpness: Sigmoid sharpness for indicator functions
        friction_cone_segments: Discretization of friction cone
        use_fixed_seed: Whether to use fixed random seed for reproducibility
        seed: Random seed value if use_fixed_seed is True
    """
    temperature: float = 0.1
    n_wrench_directions: int = 64
    sharpness: float = 10.0
    friction_cone_segments: int = 8
    use_fixed_seed: bool = True
    seed: int = 42


# Soft Operations

def soft_min(x: Array, temperature: float = 0.1) -> Array:
    """Differentiable approximation to min via negative log-sum-exp.
    
    soft_min(x)  ~=  min(x) as temperature  ->  0
    
    The approximation is:
        soft_min(x) = -temperature * log(Sigma exp(-x_i / temperature))
    
    Args:
        x: Input array of values
        temperature: Smoothing parameter (smaller = closer to hard min)
    
    Returns:
        Scalar approximation to min(x)
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for differentiable metrics")
    return -temperature * jax.nn.logsumexp(-x / temperature)


def soft_max(x: Array, temperature: float = 0.1) -> Array:
    """Differentiable approximation to max via log-sum-exp.
    
    soft_max(x)  ~=  max(x) as temperature  ->  0
    
    Args:
        x: Input array of values
        temperature: Smoothing parameter (smaller = closer to hard max)
    
    Returns:
        Scalar approximation to max(x)
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for differentiable metrics")
    return temperature * jax.nn.logsumexp(x / temperature)


def soft_indicator(
    x: Array,
    threshold: float = 0.0,
    sharpness: float = 10.0,
) -> Array:
    """Smooth indicator function via sigmoid.
    
    Returns values in (0, 1) that approximate:
        1 if x > threshold
        0 if x < threshold
    
    Args:
        x: Input values
        threshold: Decision boundary
        sharpness: Sigmoid steepness (higher = closer to step function)
    
    Returns:
        Soft indicator values in (0, 1)
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for differentiable metrics")
    return jax.nn.sigmoid(sharpness * (x - threshold))


def soft_relu(x: Array, beta: float = 10.0) -> Array:
    """Smooth ReLU approximation via softplus.
    
    soft_relu(x)  ~=  max(0, x) as beta  ->   inf 
    
    Args:
        x: Input values
        beta: Sharpness parameter
    
    Returns:
        Softplus approximation to ReLU
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for differentiable metrics")
    return jax.nn.softplus(x * beta) / beta


def soft_abs(x: Array, epsilon: float = 1e-6) -> Array:
    """Smooth absolute value via sqrt(x^2 + epsilon).
    
    Args:
        x: Input values
        epsilon: Smoothing parameter to avoid non-differentiability at 0
    
    Returns:
        Smooth approximation to |x|
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for differentiable metrics")
    return jnp.sqrt(x**2 + epsilon)


def soft_norm(x: Array, axis: int = -1, epsilon: float = 1e-6) -> Array:
    """Smooth vector norm via sqrt(||x||^2 + epsilon).
    
    Args:
        x: Input array
        axis: Axis along which to compute norm
        epsilon: Smoothing parameter
    
    Returns:
        Smooth approximation to ||x||
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for differentiable metrics")
    return jnp.sqrt(jnp.sum(x**2, axis=axis) + epsilon)


# Wrench Space Computation

def compute_wrench_space(
    contact_positions: Array,  # ; n_contacts, 3
    contact_normals: Array,    # ; n_contacts, 3
    contact_forces: Array,     # ; n_contacts, normal force magnitudes
    friction_coefs: Array,     # ; n_contacts, or scalar
    com: Array,                # ; 3, center of mass
    n_friction_edges: int = 8,
) -> Array:
    """Compute grasp wrench space basis vectors.
    
    For each contact, generates wrench vectors corresponding to the
    friction cone edges. This forms the primitive wrenches that span
    the grasp wrench space.
    
    Args:
        contact_positions: Contact point locations in world frame
        contact_normals: Contact normal directions
        contact_forces: Normal force magnitudes at each contact
        friction_coefs: Friction coefficients (per-contact or scalar)
        com: Object center of mass
        n_friction_edges: Number of edges to discretize friction cone
    
    Returns:
        wrenches: (n_contacts * n_friction_edges, 6) primitive wrenches
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for differentiable metrics")
    
    n_contacts = contact_positions.shape[0]

    if friction_coefs.ndim == 0:
        friction_coefs = jnp.full(n_contacts, friction_coefs)

    # Position relative to COM
    r = contact_positions - com[None, :]
    
    # Generate friction cone edges for each contact
    angles = jnp.linspace(0, 2 * jnp.pi, n_friction_edges, endpoint=False)
    
    def contact_wrenches(pos, normal, fn, mu):
        """Generate wrenches for one contact's friction cone"""
        # Build local tangent frame from an arbitrary perpendicular vector
        arbitrary = jnp.where(
            jnp.abs(normal[0]) < 0.9,
            jnp.array([1.0, 0.0, 0.0]),
            jnp.array([0.0, 1.0, 0.0]),
        )
        t1 = jnp.cross(normal, arbitrary)
        t1 = t1 / (soft_norm(t1) + 1e-8)
        t2 = jnp.cross(normal, t1)

        # Friction cone edges
        def edge_wrench(angle):
            tangent = jnp.cos(angle) * t1 + jnp.sin(angle) * t2
            force_dir = normal + mu * tangent
            force_dir = force_dir / (soft_norm(force_dir) + 1e-8)
            force = fn * force_dir
            torque = jnp.cross(pos - com, force)

            return jnp.concatenate([force, torque])

        return vmap(edge_wrench)(angles)

    all_wrenches = vmap(contact_wrenches)(r, contact_normals, contact_forces, friction_coefs)
    
    # Reshape to ; n_contacts * n_friction_edges, 6
    return all_wrenches.reshape(-1, 6)


def compute_simple_wrenches(
    contact_positions: Array,  # ; n_contacts, 3
    contact_normals: Array,    # ; n_contacts, 3
    contact_forces: Array,     # ; n_contacts, normal magnitudes
    com: Array,                # ; 3, center of mass
) -> Array:
    """Compute simple normal-force-only wrenches (no friction cone).
    
    Useful for quick approximations when friction is not the focus.
    
    Args:
        contact_positions: Contact locations
        contact_normals: Contact normal directions
        contact_forces: Normal force magnitudes
        com: Object center of mass
    
    Returns:
        wrenches: (n_contacts, 6) wrench per contact
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for differentiable metrics")
    
    # Position relative to COM
    r = contact_positions - com[None, :]
    
    # Force contribution ; normal direction scaled by magnitude
    forces = contact_normals * contact_forces[:, None]
    
    # Torque contribution: tau = r  x  f
    torques = jnp.cross(r, forces)

    wrenches = jnp.concatenate([forces, torques], axis=-1)
    
    return wrenches


# Core Differentiable Metrics

def soft_epsilon_metric(
    wrenches: Array,           # ; n_wrenches, 6
    active: Array,             # ; n_wrenches, soft activations
    n_directions: int = 64,    # Discretization of unit sphere
    temperature: float = 0.1,
    seed: int = 42,
) -> Array:
    """Differentiable approximation to Ferrari-Canny epsilon metric.
    
    The Ferrari-Canny epsilon metric is defined as:
        epsilon = min_{||w||=1} max{ alpha : alphaw  in  GWS }
    
    i.e., the radius of the largest ball centered at origin inscribed
    in the Grasp Wrench Space. This is approximated via:
    
    1. Sample directions uniformly on 6D unit sphere
    2. For each direction, compute the support function h(d) = max_{w  in  GWS} <d, w>
    3. Take soft-min over directions to find the worst case
    
    Args:
        wrenches: Primitive wrench vectors spanning the GWS
        active: Soft activation weights for each wrench
        n_directions: Number of random directions to sample
        temperature: Softmax temperature
        seed: Random seed for direction sampling
    
    Returns:
        epsilon: Differentiable approximation to epsilon-metric
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for differentiable metrics")
    
    # Generate uniform directions on 6D unit sphere
    key = jax.random.PRNGKey(seed)
    directions = jax.random.normal(key, (n_directions, 6))
    directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
    
    # For each direction, compute support function
    # h_GWS; d = max_{w  in  GWS} <d, w>
    # With soft weighting by contact activation

    def support_fn(d: Array) -> Array:
        projections = jnp.dot(wrenches, d)

        # Weight by activation ; inactive wrenches contribute less
        weighted_proj = projections * active

        return soft_max(weighted_proj, temperature)

    supports = vmap(support_fn)(directions)
    
    # epsilon is the minimum support ; worst-case direction
    epsilon = soft_min(supports, temperature)
    
    return epsilon


def soft_volume_metric(
    wrenches: Array,           # ; n_wrenches, 6
    active: Array,             # ; n_wrenches,
    n_projections: int = 32,
    temperature: float = 0.1,
    seed: int = 42,
) -> Array:
    """Differentiable approximation to wrench space volume.
    
    Uses random projections to estimate the "spread" of the GWS,
    which correlates with the convex hull volume.
    
    Args:
        wrenches: Primitive wrenches
        active: Soft activations
        n_projections: Number of random projections
        temperature: Softmax temperature
        seed: Random seed
    
    Returns:
        Soft volume estimate (higher = larger GWS)
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for differentiable metrics")
    
    key = jax.random.PRNGKey(seed)
    projections = jax.random.normal(key, (n_projections, 6))
    projections = projections / jnp.linalg.norm(projections, axis=-1, keepdims=True)
    
    def projection_spread(proj_dir):
        """Compute spread of GWS along a projection direction"""
        proj_values = jnp.dot(wrenches, proj_dir) * active
        max_proj = soft_max(proj_values, temperature)
        min_proj = soft_min(proj_values, temperature)
        return max_proj - min_proj
    
    spreads = vmap(projection_spread)(projections)
    
    # Geometric mean of spreads approximates volume^; 1/6
    log_volume = jnp.mean(jnp.log(spreads + 1e-8))
    
    return jnp.exp(log_volume)


def soft_slip_margin(
    normal_forces: Array,      # ; n_contacts,
    tangent_forces: Array,     # ; n_contacts, 2
    friction_coefs: Array,     # ; n_contacts,
    active: Array,             # ; n_contacts,
    temperature: float = 0.1,
) -> Array:
    """Differentiable margin to friction cone boundary.
    
    For each contact, the slip margin is:
        margin_i = mu_i |f_n^i| - ||f_t^i||
    
    Positive margin means inside friction cone (no slip).
    Negative margin means outside (slipping).
    
    Returns the soft-minimum over all active contacts.
    
    Args:
        normal_forces: Normal force magnitudes
        tangent_forces: Tangential force components [t1, t2]
        friction_coefs: Friction coefficients
        active: Soft contact activations
        temperature: Softmax temperature
    
    Returns:
        slip_margin: Minimum margin across contacts (higher = more stable)
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for differentiable metrics")
    
    tangent_mag = soft_norm(tangent_forces, axis=-1)
    cone_radius = friction_coefs * soft_abs(normal_forces)
    margins = cone_radius - tangent_mag

    # Inactive contacts get large margin so they don't affect the minimum
    inactive_penalty = (1.0 - active) * 1e6
    weighted_margins = margins + inactive_penalty
    
    return soft_min(weighted_margins, temperature)


def soft_disturbance_margin(
    wrenches: Array,           # ; n_wrenches, 6
    active: Array,             # ; n_wrenches,
    task_wrench: Array,        # ; 6, external disturbance direction
    temperature: float = 0.1,
) -> Array:
    """Maximum disturbance magnitude before grasp failure.
    
    Computes how much external wrench the grasp can resist along
    a specified direction (e.g., gravity, lateral shear).
    
    d* = max{ d : grasp can resist ||w_ext|| = d along task_wrench }
    
    This is the support function of the GWS evaluated at the
    negative task wrench direction.
    
    Args:
        wrenches: Primitive wrenches
        active: Soft activations
        task_wrench: Direction of external disturbance
        temperature: Softmax temperature
    
    Returns:
        Maximum resistable disturbance magnitude
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for differentiable metrics")
    
    task_dir = task_wrench / (soft_norm(task_wrench) + 1e-8)

    # Positive projection means we can resist wrench in that direction
    projections = jnp.dot(wrenches, -task_dir) * active
    
    # Maximum projection is disturbance we can resist
    d_star = soft_max(projections, temperature)
    
    return d_star


def soft_contact_count(
    active: Array,
    threshold: float = 0.5,
    sharpness: float = 10.0,
) -> Array:
    """Differentiable count of active contacts.
    
    Args:
        active: Soft activation values in [0, 1]
        threshold: Activation threshold for counting
        sharpness: Sigmoid sharpness
    
    Returns:
        Soft count of contacts with activation > threshold
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for differentiable metrics")
    
    above_threshold = soft_indicator(active, threshold, sharpness)
    
    return jnp.sum(above_threshold)


# Fragility and Sensitivity Metrics

def grasp_fragility(
    epsilon_fn: Callable[[Array], Array],
    theta: Array,
    epsilon_step: float = 1e-3,
) -> Array:
    """Compute gradient magnitude of epsilon w.r.t. latent parameters.
    
    High fragility means grasp quality is sensitive to small
    parameter changes -- the grasp is "brittle" even if nominally good.
    
    F(g; theta) = || grad _theta epsilon(g; theta)||
    
    Args:
        epsilon_fn: Function theta  ->  epsilon that computes epsilon for given params
        theta: Current latent parameter values
        epsilon_step: Not used (JAX autodiff handles gradients)
    
    Returns:
        Gradient magnitude (fragility score)
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for differentiable metrics")
    
    grad_epsilon = grad(epsilon_fn)(theta)
    return soft_norm(grad_epsilon)


def parameter_sensitivity_matrix(
    quality_fn: Callable[[Array], GraspQuality],
    theta: Array,
) -> Array:
    """Compute Jacobian of all quality metrics w.r.t. parameters.
    
    Returns matrix where entry (i, j) is  partial Q_i/ partial theta_j.
    
    Args:
        quality_fn: Function theta  ->  GraspQuality
        theta: Current latent parameters
    
    Returns:
        Jacobian matrix (n_metrics, n_params)
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for differentiable metrics")
    
    # Stack quality outputs into single array for Jacobian computation
    def stacked_quality(t):
        q = quality_fn(t)
        return jnp.array([
            q.epsilon_soft,
            q.volume_soft,
            q.slip_margin,
            q.disturbance_margin,
        ])
    
    return jax.jacfwd(stacked_quality)(theta)


# Belief-Integrated Metrics

def cvar_metric(
    values: Array,             # ; n_particles, metric values
    weights: Array,            # ; n_particles, belief weights
    beta: float,               # Risk level ; 0 = expectation, 1 = worst-case
    temperature: float = 0.01,
) -> Array:
    """Differentiable Conditional Value-at-Risk computation.
    
    CVaR_beta is the expected value in the beta-worst-case tail of the
    distribution. For grasp quality, low CVaR means the grasp is
    robust to worst-case parameter realizations.
    
    Uses implicit differentiation through sorting.
    
    Args:
        values: Metric values for each belief particle
        weights: Normalized belief weights (must sum to 1)
        beta: Risk level in (0, 1]. Higher = more conservative.
        temperature: Smoothing for differentiability
    
    Returns:
        CVaR estimate of the metric under belief
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for differentiable metrics")
    
    n = values.shape[0]
    
    # Sort values ; ascending - worst cases first for minimization
    sorted_idx = jnp.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    
    cum_weights = jnp.cumsum(sorted_weights)
    
    # Soft indicator for being in the beta-tail ; lowest beta fraction
    in_tail = soft_indicator(beta - cum_weights, 0.0, sharpness=100.0)
    
    # Shift to get indicator for values below VaR
    # We want particles where cum_weight <= beta
    in_tail = jnp.concatenate([jnp.array([1.0]), in_tail[:-1]])

    tail_weights = in_tail * sorted_weights
    tail_sum = jnp.sum(tail_weights) + 1e-8
    tail_weights = tail_weights / tail_sum
    
    cvar = jnp.sum(sorted_values * tail_weights)
    
    return cvar


def belief_robust_epsilon(
    epsilons: Array,           # ; n_particles, epsilon per particle
    weights: Array,            # ; n_particles, belief weights
    beta: float = 0.9,
) -> Array:
    """Belief-weighted robust epsilon metric.
    
    Returns CVaR_beta[epsilon] - the beta-worst-case grasp quality under belief.
    Lower values indicate the grasp is fragile under uncertainty.
    
    Args:
        epsilons: Epsilon values computed for each belief particle
        weights: Belief weights (normalized)
        beta: Risk level (0.9 = focus on worst 10%)
    
    Returns:
        Robust epsilon value
    """
    return cvar_metric(epsilons, weights, beta)


def expected_quality(
    values: Array,
    weights: Array,
) -> Array:
    """Expected metric value under belief.
    
    Args:
        values: Metric values per particle
        weights: Belief weights
    
    Returns:
        E[Q] under belief
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for differentiable metrics")
    return jnp.sum(values * weights)


def quality_variance(
    values: Array,
    weights: Array,
) -> Array:
    """Variance of metric under belief.
    
    High variance indicates sensitivity to uncertain parameters.
    
    Args:
        values: Metric values per particle
        weights: Belief weights
    
    Returns:
        Var[Q] under belief
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for differentiable metrics")
    
    mean = expected_quality(values, weights)
    return jnp.sum(weights * (values - mean)**2)


def failure_probability(
    epsilons: Array,
    weights: Array,
    threshold: float = 0.0,
    sharpness: float = 100.0,
) -> Array:
    """Probability of grasp failure under belief.
    
    Failure is defined as epsilon < threshold (not force-closure).
    
    Args:
        epsilons: Epsilon values per particle
        weights: Belief weights
        threshold: Epsilon threshold for success
        sharpness: Sigmoid sharpness
    
    Returns:
        P(epsilon < threshold) under belief
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for differentiable metrics")
    
    failed = soft_indicator(threshold - epsilons, 0.0, sharpness)
    
    return jnp.sum(weights * failed)


# Contact Extraction from MJX

def extract_contacts_mjx(
    mjx_data,
    max_contacts: int = 20,
    activation_threshold: float = 1e-4,
) -> ContactState:
    """Extract differentiable contact state from MJX simulation data.
    
    Args:
        mjx_data: MJX data object after simulation step
        max_contacts: Maximum contacts to track
        activation_threshold: Minimum force for contact to be active
    
    Returns:
        ContactState with differentiable contact information
    """
    if not HAS_MJX:
        raise RuntimeError("MJX required for contact extraction")
    
    # Note: Exact API depends on MJX version
    ncon = mjx_data.ncon
    
    # Pad or truncate to max_contacts
    positions = jnp.zeros((max_contacts, 3))
    normals = jnp.zeros((max_contacts, 3))
    forces = jnp.zeros((max_contacts, 3))
    depths = jnp.zeros(max_contacts)

    n_copy = jnp.minimum(ncon, max_contacts)
    
    # Contact positions and frames
    if hasattr(mjx_data, 'contact'):
        contact = mjx_data.contact
        positions = positions.at[:n_copy].set(contact.pos[:n_copy])
        # Normal is first row of contact frame
        normals = normals.at[:n_copy].set(contact.frame[:n_copy, :3])
        depths = depths.at[:n_copy].set(contact.dist[:n_copy])
    
    # Contact forces ; from constraint solver
    if hasattr(mjx_data, 'efc_force'):
        # Simplified - actual extraction depends on contact constraint mapping
        forces = forces.at[:n_copy, 0].set(
            jnp.abs(mjx_data.efc_force[:n_copy])
        )
    
    # Activation based on normal force magnitude
    active = soft_indicator(forces[:, 0], activation_threshold, sharpness=100.0)
    
    return ContactState(
        positions=positions,
        normals=normals,
        forces=forces,
        depths=depths,
        active=active,
    )


# End-to-End Differentiable Grasp Evaluation

def make_differentiable_grasp_evaluator(
    mjx_model,
    com: Array,
    config: Optional[DifferentiableMetricsConfig] = None,
    n_simulation_steps: int = 50,
):
    """Create end-to-end differentiable grasp quality function.
    
    Returns a JIT-compiled function that:
    1. Simulates grasp execution in MJX
    2. Extracts final contact state
    3. Computes differentiable quality metrics
    
    The returned function is fully differentiable via JAX autodiff,
    enabling gradient-based grasp optimization.
    
    Args:
        mjx_model: MJX model object
        com: Object center of mass (3,)
        config: Metric computation configuration
        n_simulation_steps: Physics steps per control step
    
    Returns:
        evaluate_grasp: Function (qpos, qvel, ctrl, friction)  ->  GraspQuality
    """
    if not HAS_MJX:
        raise RuntimeError("MJX required for differentiable evaluation")
    
    if config is None:
        config = DifferentiableMetricsConfig()
    
    @jit
    def evaluate_grasp(
        initial_qpos: Array,
        initial_qvel: Array,
        ctrl_sequence: Array,      # ; horizon, n_ctrl
        friction_coefs: Array,     # ; n_contacts, or scalar
    ) -> GraspQuality:
        """Simulate grasp and compute differentiable quality metrics.
        
        Args:
            initial_qpos: Initial joint positions
            initial_qvel: Initial joint velocities
            ctrl_sequence: Control inputs over horizon
            friction_coefs: Friction coefficients
        
        Returns:
            GraspQuality tuple with all differentiable metrics
        """
        mjx_data = mjx.make_data(mjx_model)
        mjx_data = mjx_data.replace(qpos=initial_qpos, qvel=initial_qvel)
        
        # Forward simulate with control sequence
        def step_fn(carry, ctrl):
            data = carry
            data = data.replace(ctrl=ctrl)
            # Multiple physics steps per control step
            def physics_step(d, _):
                return mjx.step(mjx_model, d), None
            data, _ = jax.lax.scan(physics_step, data, None, length=n_simulation_steps)
            return data, data
        
        final_data, trajectory = jax.lax.scan(step_fn, mjx_data, ctrl_sequence)

        contacts = extract_contacts_mjx(final_data)

        wrenches = compute_wrench_space(
            contacts.positions,
            contacts.normals,
            contacts.forces[:, 0],  # Normal component
            friction_coefs,
            com,
            n_friction_edges=config.friction_cone_segments,
        )
        
        # Expand activations to match wrench count
        n_contacts = contacts.active.shape[0]
        n_edges = config.friction_cone_segments
        wrench_active = jnp.repeat(contacts.active, n_edges)

        eps = soft_epsilon_metric(
            wrenches,
            wrench_active,
            n_directions=config.n_wrench_directions,
            temperature=config.temperature,
            seed=config.seed if config.use_fixed_seed else 0,
        )
        
        vol = soft_volume_metric(
            wrenches,
            wrench_active,
            temperature=config.temperature,
            seed=config.seed if config.use_fixed_seed else 0,
        )
        
        slip = soft_slip_margin(
            contacts.forces[:, 0],
            contacts.forces[:, 1:3],
            friction_coefs if friction_coefs.ndim > 0 else jnp.full(n_contacts, friction_coefs),
            contacts.active,
            temperature=config.temperature,
        )
        
        # Gravity disturbance ; downward force on object
        gravity_wrench = jnp.array([0., 0., -9.81, 0., 0., 0.])
        dist = soft_disturbance_margin(
            wrenches,
            wrench_active,
            gravity_wrench,
            temperature=config.temperature,
        )
        
        n_contacts_soft = soft_contact_count(contacts.active)
        
        return GraspQuality(
            epsilon_soft=eps,
            volume_soft=vol,
            slip_margin=slip,
            disturbance_margin=dist,
            fragility=jnp.array(0.0),  # Computed separately with grad
            contact_count_soft=n_contacts_soft,
        )
    
    return evaluate_grasp


# Gradient-Based Grasp Optimization

def optimize_grasp_gradient(
    evaluator: Callable,
    initial_qpos: Array,
    initial_qvel: Array,
    initial_ctrl: Array,
    friction_belief: Tuple[Array, Array],  # ; particles, weights
    beta: float = 0.9,
    learning_rate: float = 0.01,
    n_iterations: int = 100,
    verbose: bool = True,
) -> Tuple[Array, list]:
    """Optimize control sequence to maximize belief-robust epsilon.
    
    Uses gradient ascent on CVaR_beta[epsilon(ctrl; theta)] where theta ~ belief.
    
    This enables finding control sequences that are robust to
    uncertainty in friction and other latent parameters.
    
    Args:
        evaluator: Differentiable grasp evaluator function
        initial_qpos: Starting joint positions
        initial_qvel: Starting joint velocities
        initial_ctrl: Initial control sequence to optimize
        friction_belief: Tuple of (friction_particles, weights)
        beta: CVaR risk level (higher = more conservative)
        learning_rate: Gradient descent step size
        n_iterations: Number of optimization iterations
        verbose: Whether to print progress
    
    Returns:
        optimized_ctrl: Optimized control sequence
        loss_history: List of loss values during optimization
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for gradient optimization")
    
    friction_particles, weights = friction_belief
    
    @jit
    def loss_fn(ctrl: Array) -> Array:
        """Compute negative CVaR-epsilon (for minimization)"""
        def eval_particle(mu):
            quality = evaluator(initial_qpos, initial_qvel, ctrl, mu)
            return quality.epsilon_soft
        
        epsilons = vmap(eval_particle)(friction_particles)
        
        # CVaR loss ; negate for maximization via minimization
        return -cvar_metric(epsilons, weights, beta)
    
    grad_fn = jit(grad(loss_fn))
    
    ctrl = initial_ctrl
    loss_history = []
    
    for i in range(n_iterations):
        loss = loss_fn(ctrl)
        loss_history.append(float(loss))
        
        g = grad_fn(ctrl)
        ctrl = ctrl - learning_rate * g
        
        if verbose and i % 10 == 0:
            print(f"Iter {i:4d}: CVaR-epsilon = {-loss:.4f}")
    
    return ctrl, loss_history


def optimize_grasp_adam(
    evaluator: Callable,
    initial_qpos: Array,
    initial_qvel: Array,
    initial_ctrl: Array,
    friction_belief: Tuple[Array, Array],
    beta: float = 0.9,
    learning_rate: float = 0.001,
    n_iterations: int = 200,
    verbose: bool = True,
) -> Tuple[Array, list]:
    """Optimize grasp using Adam optimizer.
    
    More stable than vanilla gradient descent for this problem.
    
    Args:
        evaluator: Differentiable grasp evaluator
        initial_qpos: Starting positions
        initial_qvel: Starting velocities
        initial_ctrl: Initial controls
        friction_belief: (particles, weights)
        beta: CVaR level
        learning_rate: Adam learning rate
        n_iterations: Optimization steps
        verbose: Print progress
    
    Returns:
        optimized_ctrl: Result
        loss_history: Loss values
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for gradient optimization")
    
    try:
        import optax
    except ImportError:
        raise RuntimeError("optax required for Adam optimization: pip install optax")
    
    friction_particles, weights = friction_belief
    
    @jit
    def loss_fn(ctrl: Array) -> Array:
        def eval_particle(mu):
            quality = evaluator(initial_qpos, initial_qvel, ctrl, mu)
            return quality.epsilon_soft
        
        epsilons = vmap(eval_particle)(friction_particles)
        return -cvar_metric(epsilons, weights, beta)
    
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(initial_ctrl)
    
    @jit
    def step(ctrl, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(ctrl)
        updates, opt_state = optimizer.update(grads, opt_state, ctrl)
        ctrl = optax.apply_updates(ctrl, updates)
        return ctrl, opt_state, loss
    
    ctrl = initial_ctrl
    loss_history = []
    
    for i in range(n_iterations):
        ctrl, opt_state, loss = step(ctrl, opt_state)
        loss_history.append(float(loss))
        
        if verbose and i % 20 == 0:
            print(f"Iter {i:4d}: CVaR-epsilon = {-loss:.4f}")
    
    return ctrl, loss_history


# Utility Functions

def check_gradients(
    evaluator: Callable,
    qpos: Array,
    qvel: Array,
    ctrl: Array,
    friction: Array,
    eps: float = 1e-4,
) -> dict:
    """Numerical gradient check for debugging.
    
    Compares autodiff gradients to finite differences.
    
    Args:
        evaluator: Grasp evaluator function
        qpos, qvel, ctrl, friction: Evaluation inputs
        eps: Finite difference step size
    
    Returns:
        Dictionary with gradient comparison results
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required")
    
    def epsilon_fn(c):
        return evaluator(qpos, qvel, c, friction).epsilon_soft

    auto_grad = grad(epsilon_fn)(ctrl)

    fd_grad = jnp.zeros_like(ctrl)
    flat_ctrl = ctrl.flatten()
    
    for i in range(flat_ctrl.shape[0]):
        ctrl_plus = flat_ctrl.at[i].add(eps).reshape(ctrl.shape)
        ctrl_minus = flat_ctrl.at[i].add(-eps).reshape(ctrl.shape)
        fd_grad = fd_grad.at[jnp.unravel_index(i, ctrl.shape)].set(
            (epsilon_fn(ctrl_plus) - epsilon_fn(ctrl_minus)) / (2 * eps)
        )

    diff = jnp.abs(auto_grad - fd_grad)
    rel_diff = diff / (jnp.abs(auto_grad) + 1e-8)
    
    return {
        "autodiff_grad": auto_grad,
        "finite_diff_grad": fd_grad,
        "max_abs_diff": float(jnp.max(diff)),
        "max_rel_diff": float(jnp.max(rel_diff)),
        "mean_abs_diff": float(jnp.mean(diff)),
    }
