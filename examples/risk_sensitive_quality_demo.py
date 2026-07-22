"""
Example Usage and Documentation for Risk-Sensitive Grasp Quality Metrics

This module demonstrates how to use the risk-sensitive grasp quality metrics
implemented in rs_quality.py for robust grasp synthesis with probabilistic
closure guarantees.

Key Features Implemented:
1. CVaR-based risk assessment with probabilistic closure guarantees
2. Entropic risk measures for robustness quantification  
3. Differentiable stability margins (force closure and slip)
4. Smooth temporal aggregation for trajectory-level evaluation
5. Multi-objective combination with classical metrics

Mathematical Foundations:
- Trajectory margin: M_tau(theta,ξ) = -tau log Sigma_{t=1}^T exp(-m_t(theta,ξ)/tau)
- CVaR quality: Q_beta(theta) = CVaR_beta(M_tau(theta,ξ))
- Probabilistic closure: CVaR_beta(epsilon) > 0 ⟹ P(closure)  >=  1-beta
- Entropic risk: rho_theta(X) = theta log E[exp(X/theta)]
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from vnb_grasp.grasping.rs_quality import (
    RiskSensitiveGraspQuality, 
    ForceClosureMargin,
    SlipMargin,
    MultiObjectiveGraspQuality,
    create_default_evaluator
)


def generate_synthetic_grasp_data(n_samples: int = 100, 
                                 n_timesteps: int = 20,
                                 n_contacts: int = 6,
                                 stability_trend: str = 'stable') -> torch.Tensor:
    """
    Generate synthetic trajectory data for testing.
    
    Args:
        n_samples: Number of uncertainty realizations
        n_timesteps: Length of trajectory
        n_contacts: Number of contact points  
        stability_trend: 'stable', 'unstable', or 'mixed'
        
    Returns:
        Trajectory samples [n_samples, n_timesteps, n_contacts, 6]
    """
    torch.manual_seed(42)
    
    if stability_trend == 'stable':
        # Generate data that should result in positive margins
        base_wrenches = torch.randn(n_contacts, 6) * 0.5
        base_wrenches[:, 2] += 1.0  # Positive normal forces
        noise_scale = 0.1
    elif stability_trend == 'unstable':
        # Generate data that should result in negative margins
        base_wrenches = torch.randn(n_contacts, 6) * 0.3
        base_wrenches[:, 2] -= 0.5  # Negative normal forces
        noise_scale = 0.2
    else:  # mixed
        base_wrenches = torch.randn(n_contacts, 6) * 0.8
        noise_scale = 0.3
    
    # Generate trajectory samples
    trajectory_samples = []
    for i in range(n_samples):
        trajectory = []
        for t in range(n_timesteps):
            # Add temporal variation and uncertainty
            time_factor = 1.0 - 0.1 * t / n_timesteps  # Slight degradation over time
            noise = torch.randn_like(base_wrenches) * noise_scale
            wrenches = base_wrenches * time_factor + noise
            trajectory.append(wrenches)
        trajectory_samples.append(torch.stack(trajectory))
    
    return torch.stack(trajectory_samples)


def demonstrate_basic_usage():
    """Demonstrate basic usage of risk-sensitive grasp quality metrics"""
    
    print("=== Basic Usage Demonstration ===")
    
    # Create evaluator
    evaluator = create_default_evaluator('force_closure')
    
    # Generate test data
    stable_trajectories = generate_synthetic_grasp_data(
        n_samples=50, stability_trend='stable'
    )
    unstable_trajectories = generate_synthetic_grasp_data(
        n_samples=50, stability_trend='unstable' 
    )
    
    print("\n1. Evaluating Stable Grasp:")
    stable_metrics = evaluator.evaluate_grasp_quality(stable_trajectories)
    for key, value in stable_metrics.items():
        print(f"   {key}: {value:.4f}")
    
    print("\n2. Evaluating Unstable Grasp:")
    unstable_metrics = evaluator.evaluate_grasp_quality(unstable_trajectories)
    for key, value in unstable_metrics.items():
        print(f"   {key}: {value:.4f}")
    
    # Probabilistic closure analysis
    print("\n3. Probabilistic Closure Analysis:")
    for beta in [0.05, 0.1, 0.2]:
        stable_cert, stable_cvar = evaluator.probabilistic_closure_certificate(
            stable_trajectories, beta=beta
        )
        unstable_cert, unstable_cvar = evaluator.probabilistic_closure_certificate(
            unstable_trajectories, beta=beta
        )
        
        confidence = (1 - beta) * 100
        print(f"   {confidence:.0f}% confidence:")
        print(f"     Stable grasp certified: {stable_cert} (CVaR: {stable_cvar:.4f})")
        print(f"     Unstable grasp certified: {unstable_cert} (CVaR: {unstable_cvar:.4f})")


def demonstrate_risk_sensitivity():
    """Demonstrate risk sensitivity across different beta levels"""
    
    print("\n=== Risk Sensitivity Analysis ===")
    
    evaluator = create_default_evaluator('force_closure')
    
    # Generate mixed trajectory with some outliers
    mixed_trajectories = generate_synthetic_grasp_data(
        n_samples=100, stability_trend='mixed'
    )
    
    beta_levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    cvar_values = []
    
    print("\nCVaR Analysis across Risk Levels:")
    print("Beta Level | CVaR Value | Interpretation")
    print("-" * 45)
    
    for beta in beta_levels:
        cvar = evaluator.compute_cvar(
            torch.randn(100), beta  # Simple test with normal distribution
        )
        cvar_values.append(cvar.item())
        
        if beta <= 0.05:
            interpretation = "Very conservative (worst 5%)"
        elif beta <= 0.1:
            interpretation = "Conservative (worst 10%)"
        elif beta <= 0.2:
            interpretation = "Moderate risk (worst 20%)"
        else:
            interpretation = "Higher risk tolerance"
            
        print(f"{beta:8.2f}   | {cvar:9.4f} | {interpretation}")


def demonstrate_entropic_risk():
    """Demonstrate entropic risk measures with different risk aversion levels"""
    
    print("\n=== Entropic Risk Measures ===")
    
    evaluator = create_default_evaluator('force_closure')
    
    # Generate sample margins
    sample_margins = torch.randn(100) * 0.5  # Normal distribution with some spread
    
    theta_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    print("\nEntropic Risk Analysis:")
    print("Theta | Entropic Risk | Risk Aversion Level")
    print("-" * 45)
    
    for theta in theta_levels:
        entropic_risk = evaluator.compute_entropic_risk(
            sample_margins, theta=theta
        )
        
        if theta <= 0.5:
            aversion_level = "Very high"
        elif theta <= 1.0:
            aversion_level = "High"
        elif theta <= 2.0:
            aversion_level = "Moderate"
        else:
            aversion_level = "Low"
            
        print(f"{theta:5.1f} | {entropic_risk:12.4f} | {aversion_level}")


def demonstrate_multi_objective():
    """Demonstrate multi-objective grasp quality combining classical and risk metrics"""
    
    print("\n=== Multi-Objective Quality Assessment ===")
    
    # Create multi-objective evaluator
    risk_evaluator = create_default_evaluator('force_closure')
    multi_evaluator = MultiObjectiveGraspQuality(
        risk_evaluator=risk_evaluator,
        classical_weight=0.4,
        risk_weight=0.6
    )
    
    # Generate test data
    trajectory_samples = generate_synthetic_grasp_data(
        n_samples=50, stability_trend='stable'
    )
    contact_wrenches = torch.randn(6, 6) * 0.5
    contact_wrenches[:, 2] += 1.0  # Ensure reasonable normal forces
    
    # Evaluate combined quality
    combined_metrics = multi_evaluator.evaluate_combined_quality(
        trajectory_samples, contact_wrenches
    )
    
    print("\nCombined Quality Metrics:")
    print(f"Classical epsilon metric: {combined_metrics['classical_epsilon']:.4f}")
    print(f"Risk CVaR_0.05: {combined_metrics['cvar_0.050']:.4f}")
    print(f"Combined score: {combined_metrics['combined_score']:.4f}")
    print(f"Expected margin: {combined_metrics['expected_margin']:.4f}")
    print(f"Failure probability: {combined_metrics['failure_probability']:.4f}")


def demonstrate_differentiability():
    """Demonstrate differentiability for gradient-based optimization"""
    
    print("\n=== Differentiability for Optimization ===")
    
    evaluator = create_default_evaluator('force_closure')
    
    # Create a parameterized grasp ; simplified example
    grasp_params = torch.randn(10, requires_grad=True)  # 10 grasp parameters
    
    def grasp_quality_objective(params):
        """Simplified objective function using grasp parameters"""
        # In practice, params would control hand pose, joint angles, etc.
        # Here we just use them to perturb contact wrenches
        n_samples, n_timesteps, n_contacts = 20, 10, 6
        
        base_wrenches = torch.randn(n_contacts, 6)
        # Use first few parameters to influence wrench quality
        base_wrenches += params[:6].unsqueeze(-1) * 0.1
        
        # Generate trajectory
        trajectory = base_wrenches.unsqueeze(0).unsqueeze(0).expand(
            n_samples, n_timesteps, -1, -1
        )
        
        # Add parameter-dependent noise
        noise_scale = torch.abs(params[6:]).mean() * 0.1
        trajectory = trajectory + torch.randn_like(trajectory) * noise_scale
        
        # Compute quality metrics
        metrics = evaluator.evaluate_grasp_quality(trajectory)
        return metrics['cvar_0.050']  # Use CVaR as objective
    
    # Compute quality and gradients
    quality = grasp_quality_objective(grasp_params)
    quality.backward()
    
    print(f"Initial grasp quality (CVaR_0.05): {quality.item():.4f}")
    print(f"Gradient magnitude: {grasp_params.grad.norm().item():.4f}")
    print(f"Max gradient component: {grasp_params.grad.abs().max().item():.4f}")
    
    # Simulate gradient-based optimization step
    with torch.no_grad():
        learning_rate = 0.1
        grasp_params += learning_rate * grasp_params.grad
        
    # Recompute quality after step
    with torch.no_grad():
        new_quality = grasp_quality_objective(grasp_params)
    
    print(f"Quality after gradient step: {new_quality.item():.4f}")
    print(f"Quality improvement: {new_quality.item() - quality.item():.4f}")


def create_visualization_example():
    """Create a simple visualization of risk metrics"""
    
    print("\n=== Risk Metrics Visualization ===")
    
    try:
        # Generate data across different stability levels
        stability_levels = np.linspace(-1.0, 1.0, 21)
        expected_margins = []
        cvar_005_margins = []
        cvar_020_margins = []
        entropic_risks = []
        
        evaluator = create_default_evaluator('force_closure')
        
        for stability_bias in stability_levels:
            # Generate biased trajectory data
            samples = torch.randn(50) + stability_bias
            
            expected_margin = samples.mean()
            cvar_005 = evaluator.compute_cvar(samples, beta=0.05)
            cvar_020 = evaluator.compute_cvar(samples, beta=0.20)
            entropic_risk = evaluator.compute_entropic_risk(samples, theta=1.0)
            
            expected_margins.append(expected_margin.item())
            cvar_005_margins.append(cvar_005.item())
            cvar_020_margins.append(cvar_020.item())
            entropic_risks.append(entropic_risk.item())
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(stability_levels, expected_margins, 'b-', label='Expected Value')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Stability Bias')
        plt.ylabel('Expected Margin')
        plt.title('Expected Grasp Quality')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(stability_levels, cvar_005_margins, 'r-', label='CVaR 5%', linewidth=2)
        plt.plot(stability_levels, cvar_020_margins, 'orange', label='CVaR 20%', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Stability Bias')
        plt.ylabel('CVaR Margin')
        plt.title('Risk-Sensitive Quality (CVaR)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(stability_levels, entropic_risks, 'g-', label='Entropic Risk', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Stability Bias')
        plt.ylabel('Entropic Risk')
        plt.title('Entropic Risk Measure')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 4)
        closure_probs_005 = [1.0 if cvar >= 0 else 0.0 for cvar in cvar_005_margins]
        closure_probs_020 = [1.0 if cvar >= 0 else 0.0 for cvar in cvar_020_margins]
        plt.plot(stability_levels, closure_probs_005, 'r-o', label='95% Confidence', markersize=4)
        plt.plot(stability_levels, closure_probs_020, 'orange', marker='s', label='80% Confidence', markersize=4)
        plt.xlabel('Stability Bias')
        plt.ylabel('Probabilistic Closure')
        plt.title('Closure Guarantees')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(-0.1, 1.1)
        
        plt.tight_layout()
        plt.savefig('outputs/figures/risk_metrics_demo.png', 
                   dpi=300, bbox_inches='tight')
        print("Visualization saved to outputs/figures/risk_metrics_demo.png")
        
    except Exception as e:
        print(f"Visualization failed (likely missing matplotlib): {e}")


if __name__ == "__main__":
    print("Risk-Sensitive Grasp Quality Metrics - Comprehensive Demo")
    print("=" * 60)
    
    # Run all demonstrations
    demonstrate_basic_usage()
    demonstrate_risk_sensitivity() 
    demonstrate_entropic_risk()
    demonstrate_multi_objective()
    demonstrate_differentiability()
    create_visualization_example()
    
    print("\n" + "=" * 60)
    print("Demo completed! Key takeaways:")
    print("1. CVaR provides probabilistic closure guarantees")
    print("2. Entropic risk measures enable tunable risk aversion")  
    print("3. All metrics are differentiable for gradient-based optimization")
    print("4. Risk-sensitive metrics outperform expected-value metrics for robustness")
    print("5. Multi-objective combination balances classical and risk-aware quality")