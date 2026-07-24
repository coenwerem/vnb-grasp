#!/usr/bin/env python3
"""
Test script for differentiable grasp quality metrics.

Demonstrates:
1. Soft metric approximations vs. discrete counterparts
2. Gradient computation through metrics
3. Belief-weighted CVaR computation
4. Simple gradient-based grasp optimization

Usage:
    python examples/test_differentiable_metrics.py
"""

from __future__ import annotations

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, vmap
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("JAX not available. Install with: pip install jax jaxlib")
    exit(1)

# Import differentiable metrics
from vnb_grasp.belief.differentiable_metrics import (
    # Soft operations
    soft_min,
    soft_max,
    soft_indicator,
    soft_relu,
    # Core metrics
    compute_simple_wrenches,
    soft_epsilon_metric,
    soft_volume_metric,
    soft_slip_margin,
    soft_disturbance_margin,
    soft_contact_count,
    grasp_fragility,
    # Belief metrics
    cvar_metric,
    belief_robust_epsilon,
    expected_quality,
    quality_variance,
    failure_probability,
    # Config
    DifferentiableMetricsConfig,
)


def test_soft_operations():
    """Test soft approximations match hard operations"""
    print("\n" + "=" * 60)
    print("Testing Soft Operations")
    print("=" * 60)
    
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    for temp in [1.0, 0.1, 0.01]:
        sm = soft_min(x, temperature=temp)
        print(f"soft_min(x, temp={temp:.2f}) = {sm:.4f}  (hard min = {x.min():.4f})")

    for temp in [1.0, 0.1, 0.01]:
        sm = soft_max(x, temperature=temp)
        print(f"soft_max(x, temp={temp:.2f}) = {sm:.4f}  (hard max = {x.max():.4f})")

    print("\nSoft indicator (threshold=2.5):")
    for sharpness in [1.0, 10.0, 100.0]:
        indicators = soft_indicator(x, threshold=2.5, sharpness=sharpness)
        print(f"  sharpness={sharpness:5.1f}: {indicators}")

    print("\nGradient flow through soft_min:")
    grad_fn = grad(lambda y: soft_min(y, temperature=0.1).sum())
    grads = grad_fn(x)
    print(f"   grad soft_min = {grads}")
    print("  (Should concentrate on minimum element)")


def test_wrench_computation():
    """Test wrench space computation"""
    print("\n" + "=" * 60)
    print("Testing Wrench Space Computation")
    print("=" * 60)
    
    # Simple 3-contact grasp on a cube
    # Contacts on opposite faces
    contact_positions = jnp.array([
        [0.05, 0.0, 0.0],   # Right face
        [-0.05, 0.0, 0.0],  # Left face
        [0.0, 0.05, 0.0],   # Top face
    ])
    
    contact_normals = jnp.array([
        [-1.0, 0.0, 0.0],   # Points inward, from the left face
        [1.0, 0.0, 0.0],    # Points inward, from the right face
        [0.0, -1.0, 0.0],   # Points inward, from the top face
    ])
    
    contact_forces = jnp.array([1.0, 1.0, 1.0])
    com = jnp.array([0.0, 0.0, 0.0])
    
    wrenches = compute_simple_wrenches(
        contact_positions,
        contact_normals,
        contact_forces,
        com,
    )
    
    print(f"Contact positions:\n{contact_positions}")
    print(f"Contact normals:\n{contact_normals}")
    print(f"\nResulting wrenches (force, torque):")
    for i, w in enumerate(wrenches):
        print(f"  Contact {i}: f=[{w[0]:6.3f}, {w[1]:6.3f}, {w[2]:6.3f}], "
              f"tau=[{w[3]:6.3f}, {w[4]:6.3f}, {w[5]:6.3f}]")


def test_epsilon_metric():
    """Test differentiable epsilon metric"""
    print("\n" + "=" * 60)
    print("Testing Soft Epsilon Metric")
    print("=" * 60)
    
    # 6 contacts creating a roughly symmetric GWS
    n_contacts = 6
    wrenches = jax.random.normal(jax.random.PRNGKey(42), (n_contacts, 6))
    wrenches = wrenches / jnp.linalg.norm(wrenches, axis=-1, keepdims=True)
    active = jnp.ones(n_contacts)
    
    config = DifferentiableMetricsConfig(temperature=0.1, n_wrench_directions=64)
    
    epsilon = soft_epsilon_metric(
        wrenches,
        active,
        n_directions=config.n_wrench_directions,
        temperature=config.temperature,
    )
    
    print(f"Number of wrenches: {n_contacts}")
    print(f"Soft epsilon metric: {epsilon:.4f}")
    
    def eps_fn(w):
        return soft_epsilon_metric(w, active, n_directions=64, temperature=0.1)

    grad_eps = grad(eps_fn)(wrenches)
    print(f"Gradient shape: {grad_eps.shape}")
    print(f"Gradient norm: {jnp.linalg.norm(grad_eps):.4f}")

    print("\nEpsilon vs. number of active contacts:")
    for n_active in range(1, n_contacts + 1):
        active_mask = jnp.zeros(n_contacts).at[:n_active].set(1.0)
        eps = soft_epsilon_metric(wrenches, active_mask, n_directions=64, temperature=0.1)
        print(f"  {n_active} active contacts: epsilon = {eps:.4f}")


def test_slip_margin():
    """Test slip margin computation"""
    print("\n" + "=" * 60)
    print("Testing Soft Slip Margin")
    print("=" * 60)
    
    n_contacts = 4

    normal_forces = jnp.array([10.0, 8.0, 12.0, 5.0])
    friction_coefs = jnp.array([0.5, 0.5, 0.5, 0.5])

    test_cases = [
        ("Safe (well inside cone)", jnp.array([[1.0, 0.5], [0.5, 0.5], [1.0, 1.0], [0.5, 0.5]])),
        ("Critical (near boundary)", jnp.array([[4.8, 0.5], [3.8, 0.5], [5.8, 0.5], [2.3, 0.5]])),
        ("Slipping (outside cone)", jnp.array([[6.0, 2.0], [5.0, 1.0], [7.0, 2.0], [4.0, 1.0]])),
    ]
    
    active = jnp.ones(n_contacts)
    
    for name, tangent_forces in test_cases:
        margin = soft_slip_margin(
            normal_forces,
            tangent_forces,
            friction_coefs,
            active,
            temperature=0.1,
        )
        print(f"{name}: margin = {margin:.4f}")
    
    print("\nGradient of slip margin w.r.t. friction:")
    tangent_forces = jnp.array([[3.0, 1.0], [2.0, 1.0], [4.0, 1.0], [1.5, 1.0]])
    
    def margin_fn(mu):
        return soft_slip_margin(normal_forces, tangent_forces, mu, active, temperature=0.1)
    
    grad_margin = grad(margin_fn)(friction_coefs)
    print(f"   grad _mu(margin) = {grad_margin}")
    print("  (Positive gradient indicates increasing margin with friction)")


def test_cvar_metric():
    """Test CVaR computation for belief-weighted metrics"""
    print("\n" + "=" * 60)
    print("Testing CVaR Metric")
    print("=" * 60)
    
    # Simulate belief particles with varying grasp quality
    n_particles = 100
    key = jax.random.PRNGKey(123)

    # Bimodal distribution: mostly good grasps, some bad
    key1, key2 = jax.random.split(key)
    good_eps = jax.random.normal(key1, (80,)) * 0.05 + 0.3  # mu=0.3, sigma=0.05
    bad_eps = jax.random.normal(key2, (20,)) * 0.05 + 0.05   # mu=0.05, sigma=0.05
    epsilons = jnp.concatenate([good_eps, bad_eps])

    weights = jnp.ones(n_particles) / n_particles
    
    print(f"Epsilon distribution: mean={epsilons.mean():.3f}, std={epsilons.std():.3f}")
    print(f"  min={epsilons.min():.3f}, max={epsilons.max():.3f}")
    
    print("\nCVaR at different risk levels:")
    for beta in [0.1, 0.25, 0.5, 0.9]:
        cvar = cvar_metric(epsilons, weights, beta, temperature=0.01)
        print(f"  CVaR_{beta:.2f} = {cvar:.4f}")

    exp_val = expected_quality(epsilons, weights)
    robust_val = belief_robust_epsilon(epsilons, weights, beta=0.9)
    var_val = quality_variance(epsilons, weights)
    fail_prob = failure_probability(epsilons, weights, threshold=0.1)
    
    print(f"\nSummary statistics:")
    print(f"  E[epsilon]         = {exp_val:.4f}")
    print(f"  CVaR_0.9[epsilon]  = {robust_val:.4f}")
    print(f"  Var[epsilon]       = {var_val:.4f}")
    print(f"  P(epsilon < 0.1)   = {fail_prob:.4f}")
    
    print("\nGradient of CVaR w.r.t. epsilon values:")
    grad_cvar = grad(lambda e: cvar_metric(e, weights, 0.9))(epsilons)
    print(f"  Gradient concentrates on worst particles")
    print(f"  Max | grad | indices: {jnp.argsort(jnp.abs(grad_cvar))[-5:]}")


def test_gradient_optimization():
    """Demonstrate simple gradient-based optimization"""
    print("\n" + "=" * 60)
    print("Testing Gradient-Based Optimization")
    print("=" * 60)
    
    # Simplified problem: optimize wrench magnitudes to maximize epsilon
    n_contacts = 5
    key = jax.random.PRNGKey(999)
    
    # Fixed wrench directions (unit vectors)
    directions = jax.random.normal(key, (n_contacts, 6))
    directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)

    initial_magnitudes = jnp.ones(n_contacts) * 0.5
    
    def epsilon_from_magnitudes(mags):
        wrenches = directions * mags[:, None]
        active = jnp.ones(n_contacts)
        return soft_epsilon_metric(wrenches, active, n_directions=64, temperature=0.1)
    
    learning_rate = 0.1
    magnitudes = initial_magnitudes
    
    print("Gradient ascent to maximize epsilon:")
    print(f"  Initial epsilon = {epsilon_from_magnitudes(magnitudes):.4f}")
    
    grad_fn = jax.jit(grad(epsilon_from_magnitudes))
    
    for i in range(50):
        g = grad_fn(magnitudes)
        magnitudes = magnitudes + learning_rate * g
        magnitudes = jnp.clip(magnitudes, 0.1, 2.0)
        
        if i % 10 == 9:
            eps = epsilon_from_magnitudes(magnitudes)
            print(f"  Iter {i+1:3d}: epsilon = {eps:.4f}, mags = {magnitudes}")
    
    final_eps = epsilon_from_magnitudes(magnitudes)
    print(f"  Final epsilon = {final_eps:.4f}")
    print(f"  Improvement: {(final_eps / epsilon_from_magnitudes(initial_magnitudes) - 1) * 100:.1f}%")


def test_fragility():
    """Test grasp fragility computation"""
    print("\n" + "=" * 60)
    print("Testing Grasp Fragility")
    print("=" * 60)
    
    n_contacts = 4
    
    # Fixed contact geometry
    contact_positions = jnp.array([
        [0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [0.0, 0.05, 0.0],
        [0.0, -0.05, 0.0],
    ])
    
    contact_normals = jnp.array([
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    
    com = jnp.array([0.0, 0.0, 0.0])
    contact_forces = jnp.ones(n_contacts)
    active = jnp.ones(n_contacts)
    
    def epsilon_from_friction(friction_coefs):
        """Compute epsilon as function of friction coefficients"""
        # Friction doesn't affect simple wrenches, but would affect full GWS
        # For demo, we'll perturb force magnitudes based on friction
        effective_forces = contact_forces * (1.0 + 0.5 * friction_coefs)
        wrenches = compute_simple_wrenches(
            contact_positions,
            contact_normals,
            effective_forces,
            com,
        )
        return soft_epsilon_metric(wrenches, active, n_directions=64, temperature=0.1)
    
    friction_values = jnp.array([0.3, 0.5, 0.7, 0.9])

    print("Grasp fragility (|| grad _mu epsilon||) at different friction levels:")
    for mu in [0.3, 0.5, 0.7, 0.9]:
        friction = jnp.full(n_contacts, mu)
        frag = grasp_fragility(epsilon_from_friction, friction)
        eps = epsilon_from_friction(friction)
        print(f"  mu = {mu:.1f}: epsilon = {eps:.4f}, fragility = {frag:.4f}")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print(" Differentiable Grasp Metrics Test Suite")
    print("=" * 60)
    
    print(f"\nJAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")

    test_soft_operations()
    test_wrench_computation()
    test_epsilon_metric()
    test_slip_margin()
    test_cvar_metric()
    test_gradient_optimization()
    test_fragility()
    
    print("\n" + "=" * 60)
    print(" All tests completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Integrate with MJX for end-to-end differentiable simulation")
    print("  2. Use optimize_grasp_adam() for policy optimization")
    print("  3. Connect to belief-MPC for gradient-enhanced planning")


if __name__ == "__main__":
    main()
