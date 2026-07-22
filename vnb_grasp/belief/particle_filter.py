"""Particle filter belief representation.

Simulator-agnostic implementation that works with both GraspIt! (quasi-static)
and MuJoCo (dynamic) backends.

Based on: graspit_python_wrapper/core/pomdp.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray


StateT = TypeVar("StateT")


@dataclass
class ParticleBelief(Generic[StateT]):
    """Particle approximation of a belief distribution over latent contact states.

    This is the core representation for belief-space grasp planning.
    Each particle represents one hypothesis about the unobserved contact
    parameters (friction, compliance, contact mode).

    Attributes:
        particles: Array of state samples (dtype=object), shape (N,)
        weights: Normalized probability weights, shape (N,)
    """

    particles: NDArray  # dtype=object, shape ; N,
    weights: NDArray    # float64, shape ; N,

    def __post_init__(self) -> None:
        self.particles = np.asarray(self.particles, dtype=object)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        if self.particles.ndim != 1:
            raise ValueError(f"particles must be 1D; got {self.particles.shape}")
        if self.weights.shape != self.particles.shape:
            raise ValueError(
                f"weights shape {self.weights.shape} != particles shape {self.particles.shape}"
            )
        self.normalize_inplace()

    @property
    def n(self) -> int:
        """Number of particles"""
        return int(self.particles.shape[0])

    def normalize_inplace(self) -> None:
        """Normalize weights to sum to 1"""
        s = float(np.sum(self.weights))
        if not np.isfinite(s) or s <= 0:
            self.weights = np.ones_like(self.weights) / max(1, self.n)
            return
        self.weights = self.weights / s

    def ess(self) -> float:
        """Effective sample size.

        Low ESS indicates particle degeneracy; trigger resampling when
        ESS < threshold * N (typically threshold = 0.5).
        """
        return float(1.0 / max(1e-12, np.sum(self.weights ** 2)))

    def entropy(self) -> float:
        r"""Entropy of the weight distribution: H(b) = -\sum_i w_i log w_i.

        This is the uncertainty metric from the paper's Eq. (4.3).
        Maximum entropy = log(N) when weights are uniform.
        Entropy decreases as observations eliminate hypotheses.
        """
        w = np.clip(self.weights, 1e-12, 1.0)
        return float(-np.sum(w * np.log(w)))

    def resample_inplace(self, rng: np.random.Generator) -> None:
        """Systematic resampling to combat particle degeneracy.

        After resampling, all weights are reset to 1/N.
        """
        self.normalize_inplace()
        n = self.n
        cdf = np.cumsum(self.weights)
        u0 = rng.random() / n
        us = u0 + np.arange(n) / n
        idx = np.searchsorted(cdf, us, side="left")
        idx = np.clip(idx, 0, n - 1)
        self.particles = self.particles[idx].copy()
        self.weights = np.ones(n, dtype=np.float64) / n

    def resample_if_needed(
        self,
        rng: np.random.Generator,
        ess_threshold: float = 0.5,
    ) -> bool:
        """Resample if ESS drops below threshold * N.

        Returns:
            True if resampling was performed.
        """
        if self.ess() < ess_threshold * self.n:
            self.resample_inplace(rng)
            return True
        return False

    def update_inplace(
        self,
        obs: any,
        likelihood_fn: Callable[[any, StateT], float],
    ) -> None:
        """Bayesian belief update: reweight particles by observation likelihood.

        w_i' ∝ w_i · p(obs | state_i)

        Args:
            obs: The observation to condition on
            likelihood_fn: Function computing p(obs | state)
        """
        for i in range(self.n):
            w = float(likelihood_fn(obs, self.particles[i]))
            self.weights[i] *= max(0.0, w)
        self.normalize_inplace()

    def mean(self, extract_fn: Callable[[StateT], NDArray]) -> NDArray:
        """Compute weighted mean of a feature extracted from particles"""
        features = np.array([extract_fn(s) for s in self.particles])
        return np.average(features, weights=self.weights, axis=0)

    def variance(self, extract_fn: Callable[[StateT], NDArray]) -> NDArray:
        """Compute weighted variance of a feature"""
        features = np.array([extract_fn(s) for s in self.particles])
        mean = np.average(features, weights=self.weights, axis=0)
        diff = features - mean
        return np.average(diff ** 2, weights=self.weights, axis=0)

    def copy(self) -> "ParticleBelief[StateT]":
        """Deep copy of the belief"""
        return ParticleBelief(
            particles=self.particles.copy(),
            weights=self.weights.copy(),
        )



# Risk metrics ; from paper Eq. 4.7-4.8


def cvar(samples: Sequence[float], weights: Optional[Sequence[float]] = None, *, beta: float) -> float:
    """Conditional Value-at-Risk (CVaR_beta).

    CVaR_beta = E[X | X  >=  VaR_beta]

    This is the expected value in the worst (1-beta) fraction of outcomes.
    Used for risk-sensitive action selection in the MPC objective.

    Standard convention: higher beta = more risk-averse (focus on smaller tail)
    - beta=0.5  ->  worst 50% (median and above)
    - beta=0.9  ->  worst 10% (90th percentile and above)
    - beta=0.95  ->  worst 5%
    - beta=0.99  ->  worst 1%

    Args:
        samples: Cost samples from Monte Carlo rollouts
        weights: Optional importance weights for each sample. If *None*,
            uniform weights are assumed (original behaviour).
        beta: Risk level in (0, 1). beta=0.9 means worst 10% (1-beta).

    Returns:
        CVaR value
    """
    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must be in (0, 1); got {beta}")
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        raise ValueError("samples must be non-empty")

    if weights is None:
        # Uniform weights - fast path ; original behaviour
        x_sorted = np.sort(x)
        k = max(1, int(np.ceil((1.0 - beta) * x_sorted.size)))
        return float(np.mean(x_sorted[-k:]))

    # Weighted CVaR: sort by value, accumulate normalised weights from the
    # right ; worst tail until we have covered mass ; 1 - beta.
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != x.shape:
        raise ValueError(f"weights shape {w.shape} != samples shape {x.shape}")
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = w[order]
    w_sorted = w_sorted / w_sorted.sum()           # normalise
    tail_mass = 1.0 - beta
    # Walk from the worst ; right end
    cum = 0.0
    tail_val = 0.0
    for i in range(len(x_sorted) - 1, -1, -1):
        if cum + w_sorted[i] <= tail_mass:
            tail_val += w_sorted[i] * x_sorted[i]
            cum += w_sorted[i]
        else:
            remaining = tail_mass - cum
            tail_val += remaining * x_sorted[i]
            cum = tail_mass
            break
    if cum > 0:
        return float(tail_val / cum)
    # Fallback: return the maximum value
    return float(x_sorted[-1])


def var(samples: Sequence[float], *, beta: float) -> float:
    """Value-at-Risk (VaR_beta).

    VaR_beta is the (1-beta) quantile of the cost distribution.

    Args:
        samples: Cost samples
        beta: Risk level in (0, 1)

    Returns:
        VaR value
    """
    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must be in (0, 1); got {beta}")
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        raise ValueError("samples must be non-empty")
    return float(np.percentile(x, 100 * (1 - beta)))


def failure_probability(
    indicators: Sequence[bool],
    weights: Optional[Sequence[float]] = None,
) -> float:
    r"""Estimate failure probability from particle outcomes.

    phat_F = \sum w_i · 1[failure_i]

    Args:
        indicators: Boolean failure indicator per particle
        weights: Particle weights (uniform if None)

    Returns:
        Estimated P(failure)
    """
    ind = np.asarray(indicators, dtype=np.float64)
    if weights is None:
        return float(np.mean(ind))
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.sum(w)
    return float(np.sum(w * ind))


def info_gain(entropy_before: float, entropy_after: float) -> float:
    """Information gain = entropy reduction.

    Used in the information-seeking cost term c_I(b_t, u_t) from Eq. (3.13).
    """
    return max(0.0, entropy_before - entropy_after)
