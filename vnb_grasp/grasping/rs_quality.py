"""
Risk-Sensitive Grasp Quality Metrics

Implementation of differentiable, risk-sensitive grasp quality metrics that provide
probabilistic closure guarantees using Conditional Value-at-Risk (CVaR) and entropic
risk measures.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Union
from scipy import stats
from abc import ABC, abstractmethod


class StabilityMargin(ABC):
    """Abstract base class for stability margin computations"""
    
    @abstractmethod
    def compute_margin(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute stability margin for a given state.
        
        Args:
            state: System state tensor
            
        Returns:
            Stability margin (positive = stable, negative = unstable)
        """
        pass


class ForceClosureMargin(StabilityMargin):
    """Differentiable force closure margin based on epsilon metric surrogate"""
    
    def __init__(self, eps_threshold: float = 0.01):
        self.eps_threshold = eps_threshold
    
    def compute_margin(self, contact_wrenches: torch.Tensor, 
                      friction_coeffs: torch.Tensor = None) -> torch.Tensor:
        """Compute differentiable force closure margin.
        
        Args:
            contact_wrenches: Contact wrench vectors [N_contacts, 6]
            friction_coeffs: Friction coefficients [N_contacts] (optional)
            
        Returns:
            Force closure margin
        """
        # Compute Grasp Wrench Space (GWS) approximation
        if contact_wrenches.dim() == 2:
            # Single grasp case
            gws_volume = self._compute_gws_volume(contact_wrenches, friction_coeffs)
            return gws_volume - self.eps_threshold
        else:
            # Batch case
            batch_size = contact_wrenches.shape[0]
            margins = []
            for i in range(batch_size):
                fc = friction_coeffs[i] if friction_coeffs is not None else None
                volume = self._compute_gws_volume(contact_wrenches[i], fc)
                margins.append(volume - self.eps_threshold)
            return torch.stack(margins)
    
    def _compute_gws_volume(self, wrenches: torch.Tensor, 
                           friction_coeffs: torch.Tensor = None) -> torch.Tensor:
        """Compute differentiable GWS volume approximation"""
        if friction_coeffs is None:
            friction_coeffs = torch.ones(wrenches.shape[0], device=wrenches.device) * 0.7
        
        scaled_wrenches = wrenches * friction_coeffs.unsqueeze(-1)

        # Compute convex hull volume approximation using determinant
        if scaled_wrenches.shape[0] >= 6:  # Need at least 6 wrenches for 6D space
            # Select 6 linearly independent wrenches (approximation)
            gram_matrix = torch.matmul(scaled_wrenches, scaled_wrenches.T)
            eigenvals = torch.linalg.eigvals(gram_matrix)
            # Volume proportional to sqrt(det(Gram matrix))
            volume = torch.sqrt(torch.prod(torch.clamp(eigenvals.real, min=1e-8)))
        else:
            # Insufficient contacts, use norm-based approximation
            volume = torch.norm(scaled_wrenches.flatten())
        
        return volume


class SlipMargin(StabilityMargin):
    """Slip margin based on friction cone constraints"""
    
    def __init__(self, safety_factor: float = 1.2):
        self.safety_factor = safety_factor
    
    def compute_margin(self, contact_forces: torch.Tensor, 
                      normal_forces: torch.Tensor,
                      friction_coeffs: torch.Tensor) -> torch.Tensor:
        """Compute slip margin from friction cone constraints.
        
        Args:
            contact_forces: Tangential contact forces [N_contacts, 3]
            normal_forces: Normal contact forces [N_contacts]
            friction_coeffs: Friction coefficients [N_contacts]
            
        Returns:
            Slip margin (positive = no slip, negative = slip)
        """
        tangential_force_mag = torch.norm(contact_forces[..., :2], dim=-1)
        max_tangential_force = friction_coeffs * torch.abs(normal_forces)
        
        # Margin = (max allowable - current) / safety factor
        return (max_tangential_force - tangential_force_mag) / self.safety_factor


class RiskSensitiveGraspQuality:
    """
    Risk-sensitive grasp quality metrics using CVaR and entropic risk measures.
    
    Implements the mathematical formulations from the paper including:
    - Smooth temporal aggregation via soft minimum
    - CVaR-based risk assessment
    - Entropic risk measures
    - Probabilistic closure guarantees
    """
    
    def __init__(self, 
                 stability_margin: StabilityMargin,
                 tau: float = 0.1,
                 beta_levels: List[float] = [0.05, 0.1, 0.2],
                 entropy_coeff: float = 1.0):
        """
        Initialize risk-sensitive grasp quality evaluator.
        
        Args:
            stability_margin: StabilityMargin instance for margin computation
            tau: Temperature parameter for smooth minimum approximation
            beta_levels: Risk levels for CVaR computation (higher = more conservative)
            entropy_coeff: Coefficient for entropic risk measures
        """
        self.stability_margin = stability_margin
        self.tau = tau
        self.beta_levels = beta_levels
        self.entropy_coeff = entropy_coeff
    
    def compute_trajectory_margin(self, 
                                trajectory_states: torch.Tensor,
                                **margin_kwargs) -> torch.Tensor:
        """
        Compute trajectory-level stability margin using smooth minimum.
        
        Implements M_tau(theta, xi) = -tau log Sigma_{t=1}^T exp(-m_t(theta, xi)/tau)
        
        Args:
            trajectory_states: States over time [T, ...] or [batch, T, ...]
            **margin_kwargs: Additional arguments for margin computation
            
        Returns:
            Trajectory stability margin
        """
        if trajectory_states.dim() == 3:
            # Single trajectory case [T, N_contacts, 6]
            margins_t = []
            for t in range(trajectory_states.shape[0]):
                margin = self.stability_margin.compute_margin(
                    contact_wrenches=trajectory_states[t], **margin_kwargs
                )
                margins_t.append(margin)
            margins_t = torch.stack(margins_t)
        else:
            # Batch trajectory case [batch, T, N_contacts, 6]
            batch_size, T = trajectory_states.shape[:2]
            margins_t = []
            for t in range(T):
                margin = self.stability_margin.compute_margin(
                    contact_wrenches=trajectory_states[:, t], **margin_kwargs
                )
                margins_t.append(margin)
            margins_t = torch.stack(margins_t, dim=1)  # [batch, T]
        
        # Smooth minimum approximation
        trajectory_margin = -self.tau * torch.logsumexp(-margins_t / self.tau, dim=-1)
        
        return trajectory_margin
    
    def compute_cvar(self, 
                    samples: torch.Tensor, 
                    beta: float,
                    weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Conditional Value-at-Risk (CVaR) for risk-sensitive assessment.
        
        Args:
            samples: Sample values [N_samples] or [batch, N_samples]
            beta: Risk level (e.g., 0.05 for 5% worst cases)
            weights: Optional sample weights [N_samples]
            
        Returns:
            CVaR value
        """
        if weights is None:
            weights = torch.ones_like(samples) / samples.shape[-1]
        
        if samples.dim() == 1:
            # Single case
            sorted_samples, indices = torch.sort(samples)
            sorted_weights = weights[indices]
            cumsum_weights = torch.cumsum(sorted_weights, dim=0)
            
            # Find VaR (Value-at-Risk) threshold
            var_idx = torch.searchsorted(cumsum_weights, beta)
            var_value = sorted_samples[var_idx] if var_idx < len(sorted_samples) else sorted_samples[-1]
            
            # CVaR is expected value of samples <= VaR
            mask = samples <= var_value
            if mask.sum() > 0:
                cvar = torch.sum(samples * weights * mask) / torch.sum(weights * mask)
            else:
                cvar = var_value
        else:
            # Batch case
            batch_cvars = []
            for i in range(samples.shape[0]):
                sample_weights = weights[i] if weights.dim() > 1 else weights
                batch_cvar = self.compute_cvar(samples[i], beta, sample_weights)
                batch_cvars.append(batch_cvar)
            cvar = torch.stack(batch_cvars)
        
        return cvar
    
    def compute_entropic_risk(self, 
                             samples: torch.Tensor,
                             theta: float = None,
                             weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute entropic risk measure.
        
        Implements: rho_theta(X) = theta log E[exp(X/theta)] for theta > 0
        
        Args:
            samples: Sample values [N_samples] or [batch, N_samples]
            theta: Risk aversion parameter (defaults to entropy_coeff)
            weights: Optional sample weights
            
        Returns:
            Entropic risk value
        """
        if theta is None:
            theta = self.entropy_coeff
        
        if weights is None:
            weights = torch.ones_like(samples) / samples.shape[-1]
        
        # Compute weighted expectation of exp(X / theta)
        exp_scaled = torch.exp(samples / theta)
        
        if samples.dim() == 1:
            weighted_exp_mean = torch.sum(exp_scaled * weights)
        else:
            # Batch case
            if weights.dim() == 1:
                weights = weights.unsqueeze(0).expand_as(samples)
            weighted_exp_mean = torch.sum(exp_scaled * weights, dim=-1)
        
        entropic_risk = theta * torch.log(weighted_exp_mean)
        
        return entropic_risk
    
    def evaluate_grasp_quality(self,
                              trajectory_samples: torch.Tensor,
                              uncertainty_weights: Optional[torch.Tensor] = None,
                              **margin_kwargs) -> dict:
        """
        Comprehensive risk-sensitive grasp quality evaluation.
        
        Args:
            trajectory_samples: Trajectory states [N_samples, T, ...] 
            uncertainty_weights: Weights for uncertainty realizations [N_samples]
            **margin_kwargs: Arguments for stability margin computation
            
        Returns:
            Dictionary containing quality metrics
        """
        N_samples = trajectory_samples.shape[0]
        
        if uncertainty_weights is None:
            uncertainty_weights = torch.ones(N_samples) / N_samples
        
        trajectory_margins = []
        for i in range(N_samples):
            margin = self.compute_trajectory_margin(
                trajectory_samples[i], **margin_kwargs
            )
            trajectory_margins.append(margin)
        trajectory_margins = torch.stack(trajectory_margins)

        results = {
            'expected_margin': torch.sum(trajectory_margins * uncertainty_weights),
            'margin_std': torch.sqrt(torch.sum(
                (trajectory_margins - torch.sum(trajectory_margins * uncertainty_weights))**2 
                * uncertainty_weights
            ))
        }
        
        # CVaR for different risk levels
        for beta in self.beta_levels:
            cvar_value = self.compute_cvar(trajectory_margins, beta, uncertainty_weights)
            results[f'cvar_{beta:.3f}'] = cvar_value
            # Probabilistic closure guarantee
            results[f'prob_closure_{1-beta:.3f}'] = (cvar_value > 0).float()
        
        # Entropic risk measures
        results['entropic_risk'] = self.compute_entropic_risk(
            trajectory_margins, weights=uncertainty_weights
        )
        
        results['worst_case_margin'] = torch.min(trajectory_margins)
        results['failure_probability'] = (trajectory_margins <= 0).float().mean()
        
        return results
    
    def probabilistic_closure_certificate(self,
                                        trajectory_samples: torch.Tensor,
                                        beta: float = 0.05,
                                        **margin_kwargs) -> Tuple[bool, float]:
        """
        Check probabilistic closure guarantee.
        
        Returns True if CVaR_beta(epsilon) > 0, certifying force closure with 
        probability at least 1-beta.
        
        Args:
            trajectory_samples: Trajectory states [N_samples, T, ...]
            beta: Risk level
            **margin_kwargs: Arguments for stability margin computation
            
        Returns:
            (closure_certified, cvar_value)
        """
        trajectory_margins = []
        for i in range(trajectory_samples.shape[0]):
            margin = self.compute_trajectory_margin(
                trajectory_samples[i], **margin_kwargs
            )
            trajectory_margins.append(margin)
        trajectory_margins = torch.stack(trajectory_margins)

        cvar_value = self.compute_cvar(trajectory_margins, beta)

        closure_certified = cvar_value > 0
        
        return closure_certified.item(), cvar_value.item()


class MultiObjectiveGraspQuality:
    """
    Multi-objective grasp quality combining risk measures with classical metrics.
    """
    
    def __init__(self, 
                 risk_evaluator: RiskSensitiveGraspQuality,
                 classical_weight: float = 0.3,
                 risk_weight: float = 0.7):
        self.risk_evaluator = risk_evaluator
        self.classical_weight = classical_weight
        self.risk_weight = risk_weight
    
    def compute_classical_epsilon_metric(self, contact_wrenches: torch.Tensor) -> torch.Tensor:
        """Compute classical epsilon (Ferrari-Canny) metric"""
        # Simplified implementation, in practice would use proper GWS computation
        if contact_wrenches.shape[0] < 4:
            return torch.tensor(0.0)

        # Minimum distance from origin to convex hull boundary (approximation)
        gram_matrix = torch.matmul(contact_wrenches, contact_wrenches.T)
        eigenvals, _ = torch.linalg.eig(gram_matrix)
        epsilon = torch.min(eigenvals.real)
        
        return epsilon
    
    def evaluate_combined_quality(self,
                                 trajectory_samples: torch.Tensor,
                                 contact_wrenches: torch.Tensor,
                                 **margin_kwargs) -> dict:
        """
        Evaluate combined classical and risk-sensitive grasp quality.
        
        Args:
            trajectory_samples: Trajectory states for risk evaluation
            contact_wrenches: Contact wrenches for classical metric
            **margin_kwargs: Arguments for stability margin computation
            
        Returns:
            Combined quality metrics
        """
        epsilon_metric = self.compute_classical_epsilon_metric(contact_wrenches)

        risk_metrics = self.risk_evaluator.evaluate_grasp_quality(
            trajectory_samples, **margin_kwargs
        )

        combined_score = (
            self.classical_weight * epsilon_metric + 
            self.risk_weight * risk_metrics['cvar_0.050']
        )
        
        results = {
            'combined_score': combined_score,
            'classical_epsilon': epsilon_metric,
            **risk_metrics
        }
        
        return results


def create_default_evaluator(margin_type: str = 'force_closure') -> RiskSensitiveGraspQuality:
    """
    Create default risk-sensitive grasp quality evaluator.
    
    Args:
        margin_type: Type of stability margin ('force_closure' or 'slip')
        
    Returns:
        Configured evaluator instance
    """
    if margin_type == 'force_closure':
        stability_margin = ForceClosureMargin(eps_threshold=0.01)
    elif margin_type == 'slip':
        stability_margin = SlipMargin(safety_factor=1.2)
    else:
        raise ValueError(f"Unknown margin type: {margin_type}")
    
    evaluator = RiskSensitiveGraspQuality(
        stability_margin=stability_margin,
        tau=0.1,
        beta_levels=[0.05, 0.1, 0.2],
        entropy_coeff=1.0
    )
    
    return evaluator


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    torch.manual_seed(42)
    N_samples, T, n_contacts = 100, 20, 6

    trajectory_samples = torch.randn(N_samples, T, n_contacts, 6) * 0.1
    contact_wrenches = torch.randn(n_contacts, 6)

    evaluator = create_default_evaluator('force_closure')

    quality_metrics = evaluator.evaluate_grasp_quality(
        trajectory_samples
    )

    print("Risk-Sensitive Grasp Quality Metrics:")
    for key, value in quality_metrics.items():
        print(f"{key}: {value:.4f}")

    closure_certified, cvar_value = evaluator.probabilistic_closure_certificate(
        trajectory_samples,
        beta=0.05
    )
    
    print(f"\nProbabilistic Closure (95% confidence): {closure_certified}")
    print(f"CVaR value: {cvar_value:.4f}")
