"""Environment wrapper integrating belief-space MPC with VNB-Grasp envs.

This wraps a robosuite environment (e.g., VNBGraspLift) and provides:
- Automatic belief updates from contact observations
- Risk-sensitive action selection via BeliefMPCPlanner
- Logging of entropy, quality, and CVaR trajectories

Author: Clinton Enwerem
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import mujoco as mj
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

from .belief_mpc import BeliefMPCConfig, BeliefMPCPlanner, GraspAction
from .contact_belief import GraspObservation, initialize_grasp_belief
from .mujoco_rollout import (
    ContactInfo,
    SimState,
    extract_contacts,
    get_fingertip_geom_ids,
)
from ..grasping.gws_quality import ferrari_canny_quality

@dataclass
class BeliefEnvConfig:
    """Configuration for belief-wrapped environment"""
    # Inherit MPC config
    mpc_config: BeliefMPCConfig = field(default_factory=BeliefMPCConfig)

    # Environment settings
    max_episode_steps: int = 200
    action_repeat: int = 5  # Physics steps per high-level action

    # Object info ; for GWS quality
    object_name: str = "cube"
    object_geom_names: List[str] = field(default_factory=lambda: ["cube_g0"])

    # Hand info
    fingertip_geom_names: Optional[List[str]] = None  # None = auto-detect

    # Logging
    log_belief_stats: bool = True
    log_interval: int = 10


class BeliefGraspingEnv:
    """Belief-augmented grasping environment.

    Wraps a robosuite manipulation environment and adds:
    - Particle belief over contact parameters (mu, kappa)
    - Risk-sensitive MPC for action selection
    - GWS-based grasp quality metric
    """

    def __init__(
        self,
        base_env,  # robosuite environment ; e.g., VNBGraspLift
        config: BeliefEnvConfig = BeliefEnvConfig(),
    ):
        """Initialize belief environment wrapper.

        Args:
            base_env: Underlying robosuite environment
            config: Environment configuration
        """
        self.env = base_env
        self.config = config

        # MuJoCo model/data references
        self.model = base_env.sim.model._model
        self.data = base_env.sim.data._data

        # Identify relevant geoms
        self.fingertip_geoms = get_fingertip_geom_ids(
            self.model,
            config.fingertip_geom_names,
        )
        self.object_geoms = self._get_object_geom_ids()

        # Initialize planner
        self.planner = BeliefMPCPlanner(
            config.mpc_config,
            base_env,
            quality_fn=self._compute_quality,
        )

        # Episode state
        self.step_count = 0
        self.episode_logs: Dict[str, List[float]] = {
            "entropy": [],
            "quality": [],
            "ess": [],
            "n_contacts": [],
        }

    def _get_object_geom_ids(self) -> set:
        """Get geom IDs for the target object"""
        geom_ids = set()
        for name in self.config.object_geom_names:
            try:
                gid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name)
                if gid >= 0:
                    geom_ids.add(gid)
            except Exception:
                pass
        return geom_ids

    def _get_object_center(self) -> NDArray:
        """Get object center of mass position"""
        try:
            body_id = mj.mj_name2id(
                self.model,
                mj.mjtObj.mjOBJ_BODY,
                self.config.object_name,
            )
            return self.data.xpos[body_id].copy()
        except Exception:
            return np.zeros(3)

    def _extract_fingertip_contacts(self) -> List[ContactInfo]:
        """Extract contacts between fingertips and object"""
        all_contacts = extract_contacts(
            self.model,
            self.data,
            geom_filter=self.fingertip_geoms,
        )

        # Filter to only object contacts
        relevant = []
        for c in all_contacts:
            if c.geom1 in self.object_geoms or c.geom2 in self.object_geoms:
                relevant.append(c)

        return relevant

    def _compute_quality(self, env=None) -> float:
        """Compute grasp quality using GWS analysis"""
        contacts = self._extract_fingertip_contacts()
        if len(contacts) < 2:
            return 0.0

        object_center = self._get_object_center()

        # Use belief mean friction for GWS analysis
        from .contact_belief import belief_mean_friction
        mu_mean = belief_mean_friction(self.planner.belief)

        return ferrari_canny_quality(contacts, object_center, mu_mean)

    def _observation_from_contacts(
        self,
        contacts: List[ContactInfo],
    ) -> GraspObservation:
        """Build GraspObservation from MuJoCo contacts"""
        obs = self.env.get_observations()

        q = obs.get("robot0_joint_pos", np.zeros(17))
        dq = obs.get("robot0_joint_vel", np.zeros(17))

        contact_forces = np.array([
            c.normal_force for c in contacts[:self.config.mpc_config.n_contacts]
        ]) if contacts else None

        contact_points = np.array([
            c.pos for c in contacts[:self.config.mpc_config.n_contacts]
        ]) if contacts else None

        return GraspObservation(
            q=q,
            dq=dq,
            contact_forces=contact_forces,
            contact_points=contact_points,
        )

    def reset(self) -> Dict[str, Any]:
        """Reset environment and belief"""
        obs = self.env.reset()
        self.step_count = 0

        # Reinitialize belief
        self.planner = BeliefMPCPlanner(
            self.config.mpc_config,
            self.env,
            quality_fn=self._compute_quality,
        )

        # Clear logs
        self.episode_logs = {k: [] for k in self.episode_logs}

        return obs

    def step(self, action: Optional[GraspAction] = None) -> Tuple[Dict, float, bool, Dict]:
        """Execute one step with belief update.

        Args:
            action: If None, use MPC to select action

        Returns:
            (obs, reward, done, info)
        """
        # Select action via MPC if not provided
        if action is None:
            action, mpc_info = self.planner.step()
        else:
            mpc_info = {}

        # Convert to control
        ctrl = action.to_control(n_joints=self.env.action_dim)

        # Execute with action repeat
        cumulative_reward = 0.0
        for _ in range(self.config.action_repeat):
            obs, reward, done, info = self.env.step(ctrl)
            cumulative_reward += reward
            if done:
                break

        # Extract contacts and update belief
        contacts = self._extract_fingertip_contacts()
        grasp_obs = self._observation_from_contacts(contacts)
        self.planner.update_belief(grasp_obs)

        # Compute quality
        quality = self._compute_quality()

        # Logging
        if self.config.log_belief_stats:
            self.episode_logs["entropy"].append(self.planner.belief.entropy())
            self.episode_logs["quality"].append(quality)
            self.episode_logs["ess"].append(self.planner.belief.ess())
            self.episode_logs["n_contacts"].append(len(contacts))

        self.step_count += 1

        # Augment info
        info.update({
            "belief_entropy": self.planner.belief.entropy(),
            "grasp_quality": quality,
            "n_contacts": len(contacts),
            "step": self.step_count,
            **mpc_info,
        })

        return obs, cumulative_reward, done, info

    def run_episode(
        self,
        max_steps: Optional[int] = None,
        render: bool = False,
    ) -> Dict[str, Any]:
        """Run a complete grasping episode with belief MPC.

        Args:
            max_steps: Override max episode steps
            render: If True, render each step

        Returns:
            Episode summary statistics
        """
        max_steps = max_steps or self.config.max_episode_steps
        self.reset()

        total_reward = 0.0
        success = False

        while self.step_count < max_steps:
            obs, reward, done, info = self.step()
            total_reward += reward

            if render:
                self.env.render()

            # Check termination
            if self.planner.should_commit():
                success = info.get("grasp_quality", 0) > 0.2
                break

            if done:
                break

        return {
            "success": success,
            "total_reward": total_reward,
            "n_steps": self.step_count,
            "final_entropy": self.planner.belief.entropy(),
            "final_quality": self.episode_logs["quality"][-1] if self.episode_logs["quality"] else 0.0,
            "entropy_trajectory": self.episode_logs["entropy"],
            "quality_trajectory": self.episode_logs["quality"],
        }


def make_belief_env(
    env_name: str = "VNBGraspLift",
    **env_kwargs,
) -> BeliefGraspingEnv:
    """Factory function for belief-augmented environment.

    Args:
        env_name: Name of base environment
        **env_kwargs: Passed to base environment

    Returns:
        BeliefGraspingEnv instance
    """
    # Import here to avoid circular deps
    import robosuite as suite
    from vnb_grasp.robosuite_ext.register_all import register_all_environments

    register_all_environments()

    base_env = suite.make(
        env_name,
        robots=["ZArmRealhandL6Right"],
        has_renderer=env_kwargs.pop("has_renderer", False),
        has_offscreen_renderer=env_kwargs.pop("has_offscreen_renderer", False),
        use_camera_obs=env_kwargs.pop("use_camera_obs", False),
        **env_kwargs,
    )

    return BeliefGraspingEnv(base_env)
