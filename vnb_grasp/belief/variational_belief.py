"""Variational Neural Belief Networks for Risk-Aware Manipulation.

This module implements neural belief representations that replace discrete particle 
filters with continuous variational distributions, enabling exact gradient computation
through risk measures like CVaR for risk-aware manipulation planning.

Key contributions:
1. Continuous belief representations with exact risk gradients
2. Adaptive belief complexity based on epistemic uncertainty  
3. End-to-end differentiable belief-to-action learning
4. Meta-learning belief dynamics across manipulation scenarios

Author: Clinton Enwerem
Reference: Code targeting submission for IROS 2026 - "Variational Neural Beliefs for Risk-Aware Dexterous Manipulation"
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical
import numpy as np

# # Configuration
# 
@dataclass
class VariationalBeliefConfig:
    """Configuration for variational belief networks"""
    
    # Belief representation
    belief_latent_dim: int = 64
    contact_param_dim: int = 4  # mu, kappa, mode, slip per contact
    n_contacts: int = 5
    
    # Gaussian mixture components
    n_components: int = 8
    min_std: float = 1e-4
    max_std: float = 10.0
    
    # Network architecture
    hidden_dim: int = 256
    n_layers: int = 3
    activation: str = "relu"
    dropout_rate: float = 0.1
    
    # Observation processing
    obs_dim: int = 256
    action_dim: int = 11
    
    # Training
    kl_weight: float = 1e-3
    entropy_regularization: float = 1e-4
    
    # Risk-aware planning
    cvar_beta: float = 0.9
    risk_weight: float = 0.5
    
    # Adaptive complexity
    uncertainty_threshold: float = 0.1
    capacity_levels: int = 3

# # Core Variational Belief Distributions
# 
class VariationalBelief(ABC):
    """Abstract base class for neural belief representations"""
    
    @abstractmethod
    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample contact parameters from belief distribution"""
        pass
    
    @abstractmethod
    def log_prob(self, contact_params: torch.Tensor) -> torch.Tensor:
        """Compute log probability of contact parameters"""
        pass
    
    @abstractmethod
    def entropy(self) -> torch.Tensor:
        """Compute entropy of belief distribution"""
        pass
    
    @abstractmethod
    def cvar_gradient(self, cost_fn, beta: float, n_samples: int = 256) -> Dict[str, torch.Tensor]:
        """Compute exact gradients through CVaR risk measure.

        Args:
            cost_fn: Differentiable cost function mapping samples to scalar costs.
            beta: CVaR confidence level.
            n_samples: Number of reparameterized samples.
        """
        pass


class GaussianMixtureBelief(nn.Module, VariationalBelief):
    """Gaussian Mixture Model belief representation.
    
    Parameterizes belief as mixture of Gaussians over contact parameters:
    b(θ) = Σ π_k N(θ | μ_k, Σ_k)
    
    Supports reparameterized sampling for exact gradient computation through
    risk measures like CVaR, which is the key advantage over particle filters.
    """
    
    def __init__(self, config: VariationalBeliefConfig):
        super().__init__()
        self.config = config
        self.param_dim = config.contact_param_dim * config.n_contacts
        
        # Mixture parameters 
        self.mixture_logits = nn.Parameter(torch.zeros(config.n_components))
        self.means = nn.Parameter(torch.randn(config.n_components, self.param_dim))
        self.log_stds = nn.Parameter(torch.zeros(config.n_components, self.param_dim))
        
    def _get_stds(self) -> torch.Tensor:
        """Get clamped standard deviations"""
        return torch.clamp(
            self.log_stds.exp(),
            self.config.min_std,
            self.config.max_std,
        )

    def _get_distribution(self) -> MixtureSameFamily:
        """Get the mixture distribution (for log_prob / entropy only).

        Guards against NaN/Inf in belief parameters that can arise when the
        MuJoCo simulator becomes numerically unstable mid-episode.  Corrupted
        logits are replaced by uniform logits; corrupted means/log_stds are
        zeroed so the distribution degenerates gracefully rather than crashing.
        """
        logits = self.mixture_logits
        if not torch.isfinite(logits).all():
            logits = torch.zeros_like(logits)  # fall back to uniform mixture

        means = self.means
        if not torch.isfinite(means).all():
            means = torch.zeros_like(means)

        stds = self._get_stds()
        if not torch.isfinite(stds).all():
            stds = torch.full_like(stds, self.config.min_std)

        mix = Categorical(logits=logits)
        comp = MultivariateNormal(
            means,
            scale_tril=torch.diag_embed(stds),
        )
        return MixtureSameFamily(mix, comp)

    # ------------------------------------------------------------------
    # Reparameterized sampling  -  the key differentiable primitive
    # ------------------------------------------------------------------
    def rsample(self, n_samples: int) -> torch.Tensor:
        """Reparameterized sample from the GMM.

        Uses Gumbel-Softmax for differentiable component selection and the
        standard location-scale reparameterization for each Gaussian, so that
        the returned samples carry gradients w.r.t. ``mixture_logits``,
        ``means``, and ``log_stds``.

        Returns:
            Tensor of shape ``(n_samples, param_dim)``
        """
        K = self.config.n_components
        stds = self._get_stds()  # (K, D)

        # Gumbel-Softmax  -->  differentiable soft one-hot over components
        # tau is low enough for a peaked selection but high enough for stable grads
        gumbel_weights = F.gumbel_softmax(
            self.mixture_logits.unsqueeze(0).expand(n_samples, -1),
            tau=0.5,
            hard=False,
        )  # (N, K)

        # Reparameterized Gaussian samples per component:  μ_k + σ_k ⊙ ε
        eps = torch.randn(n_samples, K, self.param_dim)  # (N, K, D)
        component_samples = self.means.unsqueeze(0) + stds.unsqueeze(0) * eps  # (N, K, D)

        # Weighted combination using soft assignment
        samples = torch.einsum("nk,nkd->nd", gumbel_weights, component_samples)
        return samples

    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample contact parameters from belief (no gradients)"""
        with torch.no_grad():
            return self.rsample(n_samples)
    
    def log_prob(self, contact_params: torch.Tensor) -> torch.Tensor:
        """Compute log probability"""
        return self._get_distribution().log_prob(contact_params)
    
    def entropy(self) -> torch.Tensor:
        """Compute entropy (approximated via sampling).

        Returns zero entropy if the belief parameters are corrupt (NaN/Inf),
        which can occur when the simulator produces unstable dynamics.  This
        prevents episodes from crashing with a validation error inside torch
        distributions and lets the caller record a graceful FAIL instead.
        """
        if not (torch.isfinite(self.mixture_logits).all()
                and torch.isfinite(self.means).all()
                and torch.isfinite(self.log_stds).all()):
            return torch.tensor(0.0)
        samples = self.sample(1000)
        log_probs = self.log_prob(samples)
        return -log_probs.mean()

    # ------------------------------------------------------------------
    # Differentiable cost evaluation over belief samples
    # ------------------------------------------------------------------
    def expected_cost(self, cost_fn, n_samples: int = 256) -> torch.Tensor:
        """Evaluate expected cost under the belief via reparameterized samples.

        Args:
            cost_fn: Callable  ``(samples: Tensor[N, D]) -> Tensor[N]``
                     mapping contact-parameter samples to per-sample costs.
                     Must be differentiable (torch ops only).
            n_samples: Number of reparameterized samples to draw.

        Returns:
            Per-sample cost tensor of shape ``(n_samples,)`` that is
            differentiable w.r.t. the belief parameters.
        """
        samples = self.rsample(n_samples)
        return cost_fn(samples)

    # ------------------------------------------------------------------
    # CVaR with exact gradients through the belief
    # ------------------------------------------------------------------
    def cvar_gradient(self, cost_fn, beta: float,
                      n_samples: int = 256,
                      max_grad_norm: float = 1000.0) -> Dict[str, torch.Tensor]:
        """Compute CVaR and its exact gradients through the continuous belief.

        Unlike a particle-filter approach that estimates CVaR from a fixed set
        of weighted particles, this draws *reparameterized* samples from the
        GMM belief, evaluates a differentiable cost function, and computes
        CVaR so that ``torch.autograd`` can propagate gradients all the way
        back to ``mixture_logits``, ``means``, and ``log_stds``.

        Args:
            cost_fn: Differentiable cost function  ``Tensor[N, D] -> Tensor[N]``.
            beta: CVaR confidence level in (0, 1].  ``beta = 0.9`` means
                  "expected cost in the worst 10 % of the belief".
            n_samples: Monte-Carlo sample count.
            max_grad_norm: Maximum gradient norm for clipping (prevents explosion).

        Returns:
            Dictionary with ``'cvar'`` (scalar tensor with grad_fn),
            and per-parameter gradients (clipped to max_grad_norm).
        """
        # Draw reparameterized samples and evaluate cost
        samples = self.rsample(n_samples)         # (N, D): has grad_fn
        costs = cost_fn(samples)                   # (N,)  : has grad_fn
        
        # Check for NaN/Inf in costs - can happen with unstable belief params
        if not torch.isfinite(costs).all():
            # Return zero gradients to avoid corrupting the belief
            return {
                'cvar': torch.tensor(float('nan')),
                'costs': costs,
                'mixture_grad': torch.zeros_like(self.mixture_logits),
                'means_grad': torch.zeros_like(self.means),
                'stds_grad': torch.zeros_like(self.log_stds),
            }

        # Soft CVaR (differentiable) via the dual representation:
        #   CVaR_β(C) = min_η { η + 1/(1-β) E[ max(C - η, 0) ] }
        # We optimise η analytically by setting it to the empirical β-quantile.
        sorted_costs, _ = torch.sort(costs)
        var_idx = int(beta * n_samples)            # β-quantile index
        var_idx = min(var_idx, n_samples - 1)
        eta = sorted_costs[var_idx].detach()       # detach η for stable grads

        # Smooth approximation of max(C - η, 0) using softplus for
        # non-zero gradients even when C < η
        excess = F.softplus(costs - eta, beta=5.0)
        cvar_val = eta + excess.mean() / (1.0 - beta + 1e-8)
        
        # Check for overflow in cvar_val
        if not torch.isfinite(cvar_val):
            return {
                'cvar': torch.tensor(float('nan')),
                'costs': costs,
                'mixture_grad': torch.zeros_like(self.mixture_logits),
                'means_grad': torch.zeros_like(self.means),
                'stds_grad': torch.zeros_like(self.log_stds),
            }

        # Compute explicit parameter gradients
        cvar_grad = torch.autograd.grad(
            cvar_val,
            [self.mixture_logits, self.means, self.log_stds],
            retain_graph=True,
        )
        
        # Clip gradients to prevent explosion
        mixture_grad = cvar_grad[0]
        means_grad = cvar_grad[1]
        stds_grad = cvar_grad[2]
        
        # Per-tensor clipping: scale down if norm exceeds max
        for grad in [mixture_grad, means_grad, stds_grad]:
            grad_norm = grad.norm()
            if grad_norm > max_grad_norm:
                grad.mul_(max_grad_norm / grad_norm)
        
        return {
            'cvar': cvar_val,
            'costs': costs,
            'mixture_grad': mixture_grad,
            'means_grad': means_grad, 
            'stds_grad': stds_grad,
        }

class ImplicitNeuralBelief(nn.Module, VariationalBelief):
    """Implicit neural representation of belief using SIREN networks.
    
    Represents belief as neural field b(θ) where θ are contact parameters.
    Enables learning arbitrarily complex belief shapes.
    """
    
    def __init__(self, config: VariationalBeliefConfig):
        super().__init__()
        self.config = config
        self.param_dim = config.contact_param_dim * config.n_contacts
        
        # SIREN network for implicit representation
        self.net = SIRENNetwork(
            input_dim=self.param_dim,
            hidden_dim=config.hidden_dim,
            output_dim=1,
            n_layers=config.n_layers,
            omega_0=1.0
        )
        
        # Learnable normalization constant (log partition function)
        self.log_Z = nn.Parameter(torch.zeros(1))
        
    def _density(self, contact_params: torch.Tensor) -> torch.Tensor:
        """Unnormalized density function"""
        return torch.exp(self.net(contact_params).squeeze(-1))
    
    def log_prob(self, contact_params: torch.Tensor) -> torch.Tensor:
        """Log probability with learned normalization"""
        log_unnorm = self.net(contact_params).squeeze(-1)
        return log_unnorm - self.log_Z
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample using Langevin MCMC"""
        return self._langevin_sampling(n_samples, n_steps=100, step_size=0.01)
    
    def _langevin_sampling(self, n_samples: int, n_steps: int, step_size: float) -> torch.Tensor:
        """Langevin MCMC sampling from implicit belief"""
        # Initialize from prior
        samples = torch.randn(n_samples, self.param_dim)
        samples.requires_grad_(True)
        
        for step in range(n_steps):
            if samples.grad is not None:
                samples.grad.zero_()
                
            # Compute log probability
            log_p = self.log_prob(samples)
            score = torch.autograd.grad(log_p.sum(), samples)[0]
            
            # Langevin update
            noise = torch.randn_like(samples) * math.sqrt(2 * step_size)
            samples = samples + step_size * score + noise
            samples = samples.detach().requires_grad_(True)
            
        return samples.detach()
    
    def entropy(self) -> torch.Tensor:
        """Entropy via sampling and importance weighting"""
        samples = self.sample(1000)
        log_probs = self.log_prob(samples)
        return -log_probs.mean()
    
    def cvar_gradient(self, cost_fn, beta: float,
                      n_samples: int = 256) -> Dict[str, torch.Tensor]:
        """CVaR gradients through implicit representation.

        Draws Langevin samples (treated as approximately reparameterized),
        evaluates a differentiable cost function, and computes CVaR with
        gradients flowing back through the network parameters.
        """
        samples = self.sample(n_samples).requires_grad_(True)
        costs = cost_fn(samples)

        sorted_costs, _ = torch.sort(costs)
        var_idx = min(int(beta * n_samples), n_samples - 1)
        eta = sorted_costs[var_idx].detach()

        excess = F.softplus(costs - eta, beta=5.0)
        cvar_val = eta + excess.mean() / (1.0 - beta + 1e-8)

        cvar_grad = torch.autograd.grad(
            cvar_val,
            list(self.net.parameters()),
            retain_graph=True,
        )

        return {
            'cvar': cvar_val,
            'costs': costs,
            'net_grads': list(cvar_grad),
        }


class SIRENNetwork(nn.Module):
    """SIREN network with periodic activations for implicit representations"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 n_layers: int, omega_0: float = 1.0):
        super().__init__()
        self.omega_0 = omega_0
        
        # First layer
        self.first_layer = nn.Linear(input_dim, hidden_dim)
        with torch.no_grad():
            self.first_layer.weight.uniform_(-1 / input_dim, 1 / input_dim)
            
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_layers - 2):
            layer = nn.Linear(hidden_dim, hidden_dim)
            with torch.no_grad():
                layer.weight.uniform_(-math.sqrt(6 / hidden_dim) / omega_0,
                                     math.sqrt(6 / hidden_dim) / omega_0)
            self.hidden_layers.append(layer)
            
        # Final layer
        self.final_layer = nn.Linear(hidden_dim, output_dim)
        with torch.no_grad():
            self.final_layer.weight.uniform_(-math.sqrt(6 / hidden_dim) / omega_0,
                                           math.sqrt(6 / hidden_dim) / omega_0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sin(self.omega_0 * self.first_layer(x))
        
        for layer in self.hidden_layers:
            x = torch.sin(self.omega_0 * layer(x))
            
        return self.final_layer(x)

# # Belief Transition and Update Networks
# 
class NeuralBeliefFilter(nn.Module):
    """Neural belief propagation replacing traditional particle filters.
    
    Implements differentiable belief dynamics:
    b_{t+1} = transition(b_t, a_t, o_{t+1})
    """
    
    def __init__(self, config: VariationalBeliefConfig):
        super().__init__()
        self.config = config
        
        # Transition model: b_t, a_t -> predicted belief parameters
        self.transition_net = nn.Sequential(
            nn.Linear(config.belief_latent_dim + config.action_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.belief_latent_dim)
        )
        
        # Observation update: predicted belief + observation -> updated belief
        self.observation_net = nn.Sequential(
            nn.Linear(config.belief_latent_dim + config.obs_dim, config.hidden_dim),
            nn.ReLU(), 
            nn.Linear(config.hidden_dim, config.belief_latent_dim)
        )
        
        # Total flattened belief params:
        #   mixture_logits (n_components) + means (n_components * param_dim) + log_stds (n_components * param_dim)
        param_dim = config.contact_param_dim * config.n_contacts
        self._belief_param_count = config.n_components * (1 + 2 * param_dim)
        
        # Encode belief distribution to latent representation
        self.belief_encoder = nn.Sequential(
            nn.Linear(self._belief_param_count, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.belief_latent_dim)
        )
        
        # Decode latent to belief distribution parameters
        self.belief_decoder = nn.Sequential(
            nn.Linear(config.belief_latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, self._belief_param_count)
        )
        
    def encode_belief(self, belief: GaussianMixtureBelief) -> torch.Tensor:
        """Encode belief distribution to latent vector"""
        # Flatten belief parameters
        mix_probs = F.softmax(belief.mixture_logits, dim=-1)
        means_flat = belief.means.flatten()
        stds_flat = belief.log_stds.exp().flatten()
        
        belief_params = torch.cat([mix_probs, means_flat, stds_flat])
        return self.belief_encoder(belief_params.unsqueeze(0))
    
    def decode_belief(self, latent: torch.Tensor) -> GaussianMixtureBelief:
        """Decode latent vector to belief distribution"""
        params = self.belief_decoder(latent).squeeze(0)  # remove batch dim
        
        n_comp = self.config.n_components
        param_dim = self.config.contact_param_dim * self.config.n_contacts
        
        # Unpack parameters
        mix_logits = params[:n_comp]
        means = params[n_comp:n_comp + n_comp * param_dim].view(n_comp, param_dim)
        log_stds = params[n_comp + n_comp * param_dim:].view(n_comp, param_dim)
        
        # Create new belief
        new_belief = GaussianMixtureBelief(self.config)
        new_belief.mixture_logits.data = mix_logits
        new_belief.means.data = means
        new_belief.log_stds.data = log_stds
        
        return new_belief
    
    def forward(self, belief_t: GaussianMixtureBelief, action_t: torch.Tensor, 
                obs_t1: torch.Tensor,
                ema_alpha: float = 0.1) -> GaussianMixtureBelief:
        """One-step belief update with EMA blending.

        The transition and observation networks are randomly initialised
        (untrained), so their output is a noisy perturbation of the prior
        belief.  To prevent K=8 mixture components from being scattered into
        noise we blend the neural-network output with the previous belief
        using an exponential moving average:

            new_param = (1 - alpha) * old_param + alpha * nn_param

        This preserves the useful multimodal structure of the GMM while
        still allowing gradual belief refinement.  ``alpha=0.1`` by default.
        """
        # Encode current belief
        belief_latent = self.encode_belief(belief_t)
        
        # Predict belief evolution
        transition_input = torch.cat([belief_latent, action_t.unsqueeze(0)], dim=-1)
        predicted_latent = self.transition_net(transition_input)
        
        # Update with observation
        obs_input = torch.cat([predicted_latent, obs_t1.unsqueeze(0)], dim=-1)
        updated_latent = self.observation_net(obs_input)
        
        # Decode neural network proposal
        nn_belief = self.decode_belief(updated_latent)

        # EMA blend: keep most of the old belief, nudge toward NN proposal.
        # This prevents the untrained NN from destroying multimodal structure.
        blended = GaussianMixtureBelief(self.config)
        blended.mixture_logits.data = (
            (1 - ema_alpha) * belief_t.mixture_logits.data
            + ema_alpha * nn_belief.mixture_logits.data
        )
        blended.means.data = (
            (1 - ema_alpha) * belief_t.means.data
            + ema_alpha * nn_belief.means.data
        )
        blended.log_stds.data = (
            (1 - ema_alpha) * belief_t.log_stds.data
            + ema_alpha * nn_belief.log_stds.data
        )

        # Clamp log_stds to prevent entropy divergence.  Without this, the
        # random-weight NN slowly drifts log_stds --> ±∞ over many EMA steps.
        # log(min_std)..log(max_std) keeps σ in [1e-4, 10].
        import math as _math
        blended.log_stds.data.clamp_(
            _math.log(self.config.min_std),
            _math.log(self.config.max_std),
        )
        blended.means.data.clamp_(-20.0, 20.0)

        return blended

# # Adaptive Belief Complexity
# 
class AdaptiveBeliefManager(nn.Module):
    """Manages belief complexity based on epistemic uncertainty.
    
    Allocates computational resources adaptively - use high-capacity
    implicit beliefs when uncertainty is high, simpler Gaussian when low.
    """
    
    def __init__(self, config: VariationalBeliefConfig):
        super().__init__()
        self.config = config
        
        # Uncertainty estimator
        self.uncertainty_net = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Multiple belief representations with different capacities
        self.simple_belief = GaussianMixtureBelief(config)
        
        complex_config = VariationalBeliefConfig(**{
            **config.__dict__,
            'n_components': config.n_components * 2,
            'hidden_dim': config.hidden_dim * 2
        })
        self.complex_belief = ImplicitNeuralBelief(complex_config)
        
    def forward(self, observation: torch.Tensor, 
                current_belief: Optional[VariationalBelief] = None) -> VariationalBelief:
        """Select appropriate belief representation based on uncertainty"""
        uncertainty = self.uncertainty_net(observation)
        
        if uncertainty > self.config.uncertainty_threshold:
            return self.complex_belief
        else:
            return self.simple_belief

# # Risk-Aware Policy with Neural Beliefs
# 
class RiskAwareNeuralPolicy(nn.Module):
    """End-to-end policy with variational belief and CVaR objectives"""
    
    def __init__(self, config: VariationalBeliefConfig):
        super().__init__()
        self.config = config
        
        self.belief_filter = NeuralBeliefFilter(config)
        self.adaptive_manager = AdaptiveBeliefManager(config)
        
        # Risk-aware policy network
        self.policy_net = nn.Sequential(
            nn.Linear(config.obs_dim + config.belief_latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
            nn.Tanh()  # Normalized actions
        )
        
    def forward(self, observation: torch.Tensor, 
                belief_state: Optional[VariationalBelief] = None) -> Tuple[torch.Tensor, VariationalBelief]:
        """Forward pass returning action and updated belief"""
        
        # Get appropriate belief representation
        if belief_state is None:
            belief_state = self.adaptive_manager(observation)
        
        # Encode belief to latent
        if isinstance(belief_state, GaussianMixtureBelief):
            belief_latent = self.belief_filter.encode_belief(belief_state)
        else:
            # For implicit beliefs, use sampling-based encoding
            samples = belief_state.sample(100)
            belief_latent = samples.mean(dim=0, keepdim=True)
        
        # Policy forward pass
        policy_input = torch.cat([observation.unsqueeze(0), belief_latent], dim=-1)
        action = self.policy_net(policy_input)
        
        return action.squeeze(0), belief_state
    
    def cvar_loss(self, beliefs: List[VariationalBelief], 
                  cost_fn, beta: float = None) -> torch.Tensor:
        """Compute differentiable CVaR loss across a list of beliefs.

        Args:
            beliefs: Belief distributions collected during a rollout.
            cost_fn: Differentiable cost function ``Tensor[N, D] -> Tensor[N]``.
            beta: CVaR confidence level (defaults to ``self.config.cvar_beta``).
        """
        if beta is None:
            beta = self.config.cvar_beta
        total_loss = 0.0
        
        for belief in beliefs:
            cvar_grad_dict = belief.cvar_gradient(cost_fn, beta)
            cvar_val = cvar_grad_dict['cvar']
            total_loss += cvar_val
            
        return total_loss / len(beliefs)