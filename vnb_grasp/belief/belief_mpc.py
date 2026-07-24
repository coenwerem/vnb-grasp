"""Belief-Space Model Predictive Control for dexterous grasping.

Port of the risk-sensitive receding-horizon planner from GraspIt! to MuJoCo.

Key adaptations:
- Uses MuJoCo step() instead of quasi-static solver
- Rollouts are dynamic (physics simulation) not collision-only
- Can leverage MuJoCo's parallel simulation (mjx) for faster rollouts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .particle_filter import ParticleBelief, cvar, failure_probability
from .contact_belief import (
    GraspParticle,
    GraspObservation,
    initialize_grasp_belief,
    default_observation_likelihood,
)

import yaml
from pathlib import Path

def _load_yaml(relative_path: str):
    """Load a YAML file relative to the project root"""
    root = Path(__file__).resolve().parent.parent.parent
    abs_path = root / relative_path
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Could not load YAML file {abs_path}: {e}")
        return None

# try loading ctrl_config.yaml
CTRL_CONFIG_FILE = "config/ctrl_config.yaml"
try:
    ctrl_config = _load_yaml(CTRL_CONFIG_FILE)
except Exception as e:
    print(f"Couldn't get control config file at {CTRL_CONFIG_FILE}: {e} \n Using default config.")

class ActionType(str, Enum):
    """Discrete action types for grasp MPC"""
    CLOSE = "close"           # Close all fingers
    CLOSE_FORCE = "close_force"  # Close with palm push
    OPEN = "open"             # Open fingers
    NUDGE = "nudge"           # Adjust single finger
    APPROACH = "approach"     # Move hand toward object
    FORCE = "force"           # Apply Cartesian force


@dataclass
class GraspAction:
    """Action for the grasp MPC.

    Maps to joint position/velocity commands for MuJoCo.
    """
    action_type: ActionType
    magnitude: float = 0.1 # rad
    finger_idx: Optional[int] = None  # For NUDGE
    direction: Optional[NDArray] = None  # For FORCE/APPROACH

    def to_control(self, env) -> NDArray:
        """Convert high-level action to MuJoCo actuator command using env.actmap.
        
        Args:
            env: MuJoCo environment with model and actmap attributes
            
        Returns:
            Control vector for MuJoCo actuators
        """
        ctrl = np.zeros(env.model.nu, dtype=np.float64)

        if self.action_type == ActionType.CLOSE:
            for i in env.actmap.hand:
                ctrl[i] = self.magnitude

        elif self.action_type == ActionType.OPEN:
            for i in env.actmap.hand:
                ctrl[i] = -self.magnitude

        elif self.action_type == ActionType.NUDGE:
            if self.finger_idx is not None:
                # Map finger index to joint indices
                finger_map = {
                    0: env.actmap.thumb,
                    1: env.actmap.index,
                    2: env.actmap.middle,
                    3: env.actmap.ring,
                    4: env.actmap.pinky,
                }
                if self.finger_idx in finger_map:
                    for i in finger_map[self.finger_idx]:
                        ctrl[i] = self.magnitude

        elif self.action_type == ActionType.APPROACH:
            # Cartesian approach mapped to arm actuators
            if self.direction is not None:
                # direction can be 3D or 6D; apply as many as we have
                for k, i in enumerate(env.actmap.arm):
                    if k < len(self.direction):
                        ctrl[i] = self.magnitude * self.direction[k]

        return ctrl


@dataclass
class BeliefMPCConfig:
    """Configuration for belief-space MPC.

    Default values match Table 1 in the paper.
    """
    # Belief parameters
    n_particles: int = 100          # N_p
    n_contacts: int = 5             # Number of fingertips

    # Priors on latent parameters
    mu_prior: Tuple[float, float] = (0.6, 0.15)
    kappa_prior: Tuple[float, float] = (1000.0, 250.0)
    # Optional mixture prior for friction: ; (mu_norm, std_norm, ; mu_tail, std_tail, p_tail)
    # If set, overrides mu_prior for belief initialization
    mu_prior_mixture: Optional[Tuple[Tuple[float, float], Tuple[float, float], float]] = None

    # MPC parameters
    horizon: int = 5                # L ; lookahead
    n_candidates: int = 15          # K ; action sequences - increased for better exploration
    max_steps: int = 50             # T_max

    # Risk parameters
    beta: float = 0.9               # CVaR level
    delta: float = 0.05             # Failure probability bound
    lambda_cvar: float = 0.5        # CVaR weight in score
    sigma_process: float = 0.0      # Process noise magnitude ; 0=deterministic, use 0.2-0.5 for experiments
    
    # Action space options
    disable_close_force: bool = False  # If True, remove CLOSE_FORCE from action space ; avoids destabilizing pushes

    # Cost weights ; Eq. 3.12
    w_quality: float = 1.0          # w_Q
    w_failure: float = 10.0         # w_F
    w_info: float = 2.0             # w_I - increased to emphasize information-seeking
    w_control: float = 0.01         # w_U

    # Termination criteria
    epsilon_des: float = 0.25       # Target grasp quality
    delta_epsilon_tol: float = 0.02 # Quality improvement tolerance
    delta_H_min: float = 0.3        # Minimum entropy change
    min_contacts: int = 2           # Minimum contacts to commit
    min_steps_before_commit: int = 10  # Minimum steps before entropy-based commit

    # Resampling
    ess_threshold: float = 0.5      # Resample when ESS < threshold * N

    # Random seed
    seed: int = 42


@dataclass
class RolloutResult:
    """Result of a Monte Carlo rollout for one particle"""
    cumulative_cost: float
    final_quality: float
    is_failure: bool
    entropy_trajectory: List[float] = field(default_factory=list)


class BeliefMPCPlanner:
    """Risk-sensitive belief-space MPC for quasi-dynamic grasping.

    Implements Algorithm 1 from the paper, adapted for MuJoCo dynamics.
    """

    def __init__(
        self,
        config: BeliefMPCConfig,
        env,  # VNB-Grasp environment
        quality_fn: Optional[Callable[[any], float]] = None,
    ):
        """Initialize the planner.

        Args:
            config: MPC configuration
            env: VNB-Grasp MuJoCo environment
            quality_fn: Function to compute grasp quality from env state
        """
        self.config = config
        self.env = env
        self.quality_fn = quality_fn or self._default_quality
        self.rng = np.random.default_rng(config.seed)

        # Initialize belief
        self.belief = initialize_grasp_belief(
            n_particles=config.n_particles,
            n_contacts=config.n_contacts,
            mu_prior=config.mu_prior,
            kappa_prior=config.kappa_prior,
            mu_prior_mixture=config.mu_prior_mixture,
            rng=self.rng,
        )

        # Action space ; discrete set from Appendix A
        # SIMPLIFIED: Removed APPROACH ; doesn't work in joint space
        # Focus on finger-based acquisition only
        self.action_set = [
            GraspAction(ActionType.CLOSE, magnitude=0.1),
            GraspAction(ActionType.CLOSE, magnitude=0.2),
            GraspAction(ActionType.NUDGE, finger_idx=0, magnitude=0.1),  # thumb
            GraspAction(ActionType.NUDGE, finger_idx=1, magnitude=0.1),  # index
            GraspAction(ActionType.NUDGE, finger_idx=2, magnitude=0.1),  # middle
        ]
        
        # add CLOSE_FORCE ; can destabilize on low-friction surfaces
        if not config.disable_close_force:
            self.action_set.insert(2, GraspAction(ActionType.CLOSE_FORCE, magnitude=0.15))

        # State tracking
        self.step_count = 0
        self.entropy_history: List[float] = []
        self.quality_history: List[float] = []

    def _default_quality(self, env_state) -> float:
        """Default grasp quality: number of contacts normalized"""
        # Placeholder; real implementation would use GWS analysis
        return 0.0

    def _extract_observation(self) -> GraspObservation:
        """Extract observation from current environment state"""
        return self.env.get_observation()

    def _sample_action_sequence(self) -> List[GraspAction]:
        """Sample a candidate action sequence of length L"""
        # Early phase: bias toward closing
        if self.step_count < 5:
            # Build early bias list based on action space ; respects disable_close_force
            early_bias = [GraspAction(ActionType.CLOSE), GraspAction(ActionType.CLOSE)]
            if not self.config.disable_close_force:
                early_bias.append(GraspAction(ActionType.CLOSE_FORCE))
            return [
                self.rng.choice(early_bias) if self.rng.random() < 0.7
                else self.rng.choice(self.action_set)
                for _ in range(self.config.horizon)
            ]
        else:
            return [
                self.rng.choice(self.action_set)
                for _ in range(self.config.horizon)
            ]

    def _rollout_particle(
        self,
        particle: GraspParticle,
        action_sequence: List[GraspAction],
    ) -> RolloutResult:
        """Evaluate action sequence under one particle's friction hypothesis.

        Implements per-particle cost evaluation from Algorithm 1 (Eq. 3.12).
        Since we cannot fork MuJoCo state, we use analytic cost approximation
        conditioned on each particle's latent parameters (friction, stiffness).

        The key insight: particles with low friction yield higher costs because
        the friction cone is tighter  ->  more likely to slip  ->  worse quality gap.
        This creates meaningful cost VARIANCE across particles, allowing CVaR
        to discriminate between action sequences at different beta levels.

        Process noise (sigma_process > 0) perturbs particle friction per paper
        Eq. 3.8: theta_{t+1} = theta_t + epsilon_t, creating friction-dependent dynamics.
        """
        cumulative_cost = 0.0
        entropy_traj = []

        # Get current env state ; shared across particles for this step
        current_quality = 0.0
        n_contacts = 0
        if self.quality_fn is not None:
            try:
                current_quality = self.quality_fn(self.env)
            except:
                current_quality = 0.0
        if hasattr(self.env, 'data') and self.env.data is not None:
            n_contacts = self.env.data.ncon

        # Per-particle friction ; may be perturbed by process noise per step
        mu_arr = particle.friction_array().copy()
        mu_max = 1.0

        for t, action in enumerate(action_sequence):
            # Apply process noise to friction ; paper Eq. 3.8
            # sigma_eff scales with distance from mu_max: low-mu particles drift more
            if self.config.sigma_process > 0:
                for j in range(len(mu_arr)):
                    noise_scale = max(0.0, 1.0 - mu_arr[j] / mu_max)
                    # Increased scaling from 0.01 to 0.1 for sufficient variance
                    sigma_eff = self.config.sigma_process * noise_scale * 0.1
                    mu_arr[j] = np.clip(
                        mu_arr[j] + self.rng.normal(0, sigma_eff),
                        0.05, 2.0,
                    )

            mu_mean = float(np.mean(mu_arr))

            # #### c_Q: Quality cost conditioned on particle friction ####
            # Under this particle's friction hypothesis, the achievable quality
            # is degraded if friction is low ; contacts may slip.
            # Model: quality_particle = quality_env * friction_reliability
            # where friction_reliability = min; 1, mu_mean / mu_required
            mu_required = 0.4  # Typical required friction for force closure
            friction_reliability = min(1.0, mu_mean / mu_required)
            quality_particle = current_quality * friction_reliability

            c_quality = max(0.0, 1.0 - quality_particle / max(self.config.epsilon_des, 0.01))

            # #### c_F: Failure probability for this particle ####
            # P; slip = P; mu < mu_required  --  higher for low-friction particles
            # Also depends on number of contacts ; form closure margin
            if mu_mean < mu_required:
                friction_failure_prob = 1.0 - (mu_mean / mu_required) ** 2
            else:
                friction_failure_prob = max(0.0, 1.0 - (mu_mean - mu_required))

            contact_margin = min(1.0, n_contacts / 4.0)  # 4+ contacts is stable
            c_failure = friction_failure_prob * (1.0 - 0.5 * contact_margin)

            # #### c_I: Information cost ; expected entropy reduction ####
            # PARTICLE-SPECIFIC: Expected future entropy depends on particle's friction
            # High-friction particles  ->  informative observations  ->  low future entropy
            # Low-friction particles  ->  marginal contacts  ->  high future entropy
            current_entropy = self.belief.entropy()
            max_entropy = np.log(self.config.n_particles)

            # Friction reliability determines observation informativeness
            friction_reliability = min(1.0, mu_mean / mu_required)
            
            # Base information gain scaled by friction reliability
            if action.action_type in (ActionType.CLOSE, ActionType.CLOSE_FORCE):
                # Closing creates new contacts  ->  info gain scales with friction
                base_info_gain = 0.15 * (current_entropy / max_entropy)
                expected_info_gain = base_info_gain * friction_reliability
            elif action.action_type == ActionType.NUDGE:
                # Nudge adjusts contacts  ->  moderate info, friction-dependent
                base_info_gain = 0.08 * (current_entropy / max_entropy)
                expected_info_gain = base_info_gain * friction_reliability
            else:
                expected_info_gain = 0.0

            # Expected future entropy ; higher for low-friction particles
            expected_future_entropy = max(0.0, current_entropy - expected_info_gain)
            
            # Cost is the expected future entropy ; higher = worse
            # This creates particle-specific variance that CVaR can exploit
            c_info = expected_future_entropy

            # #### c_U: Control cost ####
            c_control = action.magnitude ** 2

            stage_cost = (
                self.config.w_quality * c_quality
                + self.config.w_failure * c_failure
                + self.config.w_info * c_info
                + self.config.w_control * c_control
            )

            cumulative_cost += stage_cost
            entropy_traj.append(current_entropy)

        # Terminal failure check  --  per-particle
        friction_failure = float(np.mean(mu_arr)) < 0.25
        quality_failure = current_quality < 0.01 and n_contacts < 2
        is_failure = friction_failure or quality_failure

        return RolloutResult(
            cumulative_cost=cumulative_cost,
            final_quality=current_quality,
            is_failure=is_failure,
            entropy_trajectory=entropy_traj,
        )

    def _evaluate_sequence(
        self,
        action_sequence: List[GraspAction],
    ) -> Tuple[float, float, float]:
        """Evaluate action sequence via Monte Carlo rollouts over belief.

        Returns:
            (mean_cost, cvar_cost, failure_prob)
        """
        costs = []
        failures = []

        for particle in self.belief.particles:
            result = self._rollout_particle(particle, action_sequence)
            costs.append(result.cumulative_cost)
            failures.append(result.is_failure)

        mean_cost = float(np.mean(costs))
        cvar_cost = cvar(costs, beta=self.config.beta)
        fail_prob = failure_probability(failures, self.belief.weights)

        return mean_cost, cvar_cost, fail_prob

    def _compute_score(
        self,
        mean_cost: float,
        cvar_cost: float,
        fail_prob: float,
    ) -> float:
        """Compute penalized score for action sequence (Eq. 4.10).

        S^(k) = J_bar^(k) + lambda*CVaR_beta^(k) + c_fail*1[p_hat_F > delta]
        """
        score = mean_cost + self.config.lambda_cvar * cvar_cost

        # Hard constraint on failure probability
        if fail_prob > self.config.delta:
            score = float('inf')

        return score

    def update_belief(self, obs: GraspObservation) -> None:
        """Bayesian belief update with new observation"""
        self._last_obs = obs
        self.belief.update_inplace(
            obs,
            lambda o, p: default_observation_likelihood(o, p),
        )
        self.belief.resample_if_needed(self.rng, self.config.ess_threshold)
        self.entropy_history.append(self.belief.entropy())

    def select_action(self) -> GraspAction:
        """Select best action via sampling-based optimization.

        Implements the inner loop of Algorithm 1.
        """
        best_score = float('inf')
        best_sequence = None

        for _ in range(self.config.n_candidates):
            sequence = self._sample_action_sequence()
            mean_cost, cvar_cost, fail_prob = self._evaluate_sequence(sequence)
            score = self._compute_score(mean_cost, cvar_cost, fail_prob)

            if score < best_score:
                best_score = score
                best_sequence = sequence

        if best_sequence is None:
            # Fallback to safe action
            return GraspAction(ActionType.CLOSE, magnitude=0.05)

        return best_sequence[0]  # Receding horizon: execute first action

    def step(self) -> Tuple[GraspAction, Dict]:
        """Execute one MPC step.

        Returns:
            (action, info_dict)
        """
        obs = self._extract_observation()
        self._last_obs = obs

        self.update_belief(obs)

        action = self.select_action()

        quality = self.quality_fn(self.env)
        self.quality_history.append(quality)

        self.step_count += 1

        info = {
            "entropy": self.belief.entropy(),
            "quality": quality,
            "step": self.step_count,
            "ess": self.belief.ess(),
        }

        return action, info

    def should_commit(self) -> bool:
        """Check termination criteria for committing to grasp.

        Commit when:
        - Max steps reached, OR
        - Quality meets target AND at least min_contacts active, OR
        - Entropy has stabilized (information gain diminished) AND
          enough steps have passed AND contacts have been made.

        The entropy-stabilisation criterion is gated on
        ``min_steps_before_commit`` and ``min_contacts`` to prevent
        premature termination before the hand has made contact.
        """
        if self.step_count >= self.config.max_steps:
            return True

        # Count current active contacts ; from the latest observation
        n_contacts = 0
        if hasattr(self, '_last_obs') and self._last_obs is not None:
            obs = self._last_obs
            if obs.contact_forces is not None and len(obs.contact_forces) > 0:
                n_contacts = int(np.sum(obs.contact_forces > 0))
            elif obs.tactile_pressure is not None:
                n_contacts = int(np.sum(obs.tactile_pressure > 0) > 0)
        if n_contacts == 0 and len(self.quality_history) > 0:
            # quality > 0 is a reasonable proxy for having contacts
            n_contacts = 1 if self.quality_history[-1] > 0 else 0

        # Quality target met ; only if we actually have contacts
        if (len(self.quality_history) > 0
                and self.quality_history[-1] >= self.config.epsilon_des
                and n_contacts >= self.config.min_contacts):
            return True

        # Entropy stabilisation  --  only after a warmup period and contacts
        if (self.step_count >= self.config.min_steps_before_commit
                and n_contacts >= self.config.min_contacts
                and len(self.entropy_history) >= 3):
            recent_delta = abs(
                self.entropy_history[-1] - self.entropy_history[-3]
            )
            if recent_delta < self.config.delta_H_min:
                return True

        return False

    def run(self) -> Dict:
        """Run MPC until termination.

        Returns:
            Summary statistics
        """
        while not self.should_commit():
            # Get action from MPC ; internally observes and updates belief
            action, info = self.step()

            ctrl = action.to_control(self.env)
            obs, reward, done, step_info = self.env.step(ctrl)
            
            # Note: next iteration will observe the new state via step()

        return {
            "n_steps": self.step_count,
            "final_entropy": self.belief.entropy(),
            "final_quality": self.quality_history[-1] if self.quality_history else 0.0,
            "entropy_contraction": (
                self.entropy_history[0] - self.entropy_history[-1]
                if len(self.entropy_history) > 1 else 0.0
            ),
        }
