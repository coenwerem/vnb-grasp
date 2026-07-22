"""
Naive grasp executor that drives hand joints to target positions.

This executor:
1. Takes a grasp from the grasp database
2. Drives hand joints to the target configuration using position control
3. Optionally monitors contact formation and attempts a lift

No arm planning, no collision avoidance, no fancy stuff.
Assumes the hand is already positioned near the object.
Perfect for data collection and baseline experiments.

Usage:
    from vnb_grasp.grasping import load_grasps, NaiveGraspExecutor

    # Load grasps
    db = load_grasps("cube")
    grasp = db.best_grasp()

    # Execute in environment
    executor = NaiveGraspExecutor(env)
    result = executor.execute(grasp)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Any

import numpy as np

from .graspit_loader import GraspItGrasp, REALHAND_L6_DOF_NAMES

if TYPE_CHECKING:
    pass


class ExecutionPhase(Enum):
    """Phases of grasp execution"""

    IDLE = auto()
    OPENING = auto()       # Open hand to pre-grasp
    POSITIONING = auto()   # Move fingers toward target config
    CLOSING = auto()       # Close fingers incrementally
    GRASPING = auto()      # Maintain grasp, check contacts
    LIFTING = auto()       # Attempt to lift object
    SUCCESS = auto()
    FAILED = auto()


@dataclass
class ExecutionConfig:
    """Configuration for grasp execution"""

    # Timing
    open_duration: float = 0.5      # seconds to open hand
    position_duration: float = 1.0  # seconds to reach pre-grasp
    close_duration: float = 2.0     # seconds to close fingers
    hold_duration: float = 0.5      # seconds to hold grasp
    lift_duration: float = 1.5      # seconds for lift attempt

    # Finger control
    pregrasp_offset: float = 0.15   # radians - open from target for pre-grasp
    close_increment: float = 0.03   # radians per step during closing
    max_finger_value: float = 1.57  # radians - max finger curl ; ~90 deg
    min_finger_value: float = 0.0   # radians - fully open

    # Success criteria
    min_contacts_for_grasp: int = 2    # minimum finger contacts
    lift_height_threshold: float = 0.02  # meters above table for success

    # Control frequency ; should match env
    control_freq: int = 20  # Hz


@dataclass
class ExecutionResult:
    """Result of a grasp execution attempt"""

    success: bool
    final_phase: ExecutionPhase
    finger_contacts: int = 0
    lift_height_achieved: float = 0.0
    execution_time: float = 0.0
    error_message: str = ""

    # Data for learning
    n_steps: int = 0
    joint_trajectory: List[np.ndarray] = field(default_factory=list)
    contact_sequence: List[int] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)


class NaiveGraspExecutor:
    """
    Execute grasps in a robosuite environment using simple position control.

    This executor assumes:
    - The environment has a robot with a dexterous hand
    - Hand actuators are position-controlled
    - The object is within reach of the hand

    It does NOT:
    - Plan arm motion
    - Check for collisions
    - Handle object pose changes during execution
    """

    def __init__(
        self,
        env,
        config: Optional[ExecutionConfig] = None,
        hand_actuator_prefix: str = "",
        verbose: bool = False,
    ):
        """
        Initialize executor.

        Args:
            env: Robosuite environment instance
            config: Execution configuration
            hand_actuator_prefix: Prefix for hand actuator names (e.g., "robot0_")
            verbose: Print execution progress
        """
        self.env = env
        self.config = config or ExecutionConfig()
        self.hand_actuator_prefix = hand_actuator_prefix
        self.verbose = verbose

        # Map DOF names to actuator indices
        self._build_actuator_map()

        # Current state
        self.phase = ExecutionPhase.IDLE
        self._target_dofs: np.ndarray = np.zeros(11)

    def _build_actuator_map(self):
        """Build mapping from DOF index to actuator index"""
        self.dof_to_actuator: Dict[int, int] = {}
        self.actuator_to_dof: Dict[int, int] = {}

        model = self.env.sim.model

        for dof_idx, dof_name in enumerate(REALHAND_L6_DOF_NAMES):
            # Try to find actuator with this name
            full_name = f"{self.hand_actuator_prefix}{dof_name}"

            for act_idx in range(model.nu):
                act_name = model.actuator(act_idx).name.lower()
                # Match by suffix or full name
                if dof_name in act_name or act_name.endswith(dof_name):
                    self.dof_to_actuator[dof_idx] = act_idx
                    self.actuator_to_dof[act_idx] = dof_idx
                    break

        if self.verbose:
            print(f"Mapped {len(self.dof_to_actuator)}/11 DOFs to actuators")

    def execute(
        self,
        grasp: GraspItGrasp,
        record: bool = True,
        render: bool = False,
    ) -> ExecutionResult:
        """
        Execute a grasp.

        Args:
            grasp: Grasp to execute
            record: Whether to record trajectory data
            render: Whether to render during execution

        Returns:
            ExecutionResult with success status and recorded data
        """
        start_time = time.time()
        result = ExecutionResult(success=False, final_phase=ExecutionPhase.IDLE)

        # Store target DOF values
        self._target_dofs = grasp.hand_dof_values.copy()

        try:
            # Phase 1: Open hand
            self.phase = ExecutionPhase.OPENING
            self._execute_open(result if record else None, render)

            # Phase 2: Move to pre-grasp configuration
            self.phase = ExecutionPhase.POSITIONING
            self._execute_pregrasp(result if record else None, render)

            # Phase 3: Close fingers toward target + beyond
            self.phase = ExecutionPhase.CLOSING
            self._execute_close(result if record else None, render)

            # Phase 4: Check grasp quality
            self.phase = ExecutionPhase.GRASPING
            contacts = self._count_contacts()
            result.finger_contacts = contacts

            if self.verbose:
                print(f"Grasp contacts: {contacts}")

            if contacts < self.config.min_contacts_for_grasp:
                self.phase = ExecutionPhase.FAILED
                result.final_phase = ExecutionPhase.FAILED
                result.error_message = f"Insufficient contacts: {contacts}"
            else:
                # Phase 5: Hold; can also lift
                self._execute_hold(result if record else None, render)

                # Check success
                self.phase = ExecutionPhase.SUCCESS
                result.success = True
                result.final_phase = ExecutionPhase.SUCCESS

        except Exception as e:
            self.phase = ExecutionPhase.FAILED
            result.final_phase = ExecutionPhase.FAILED
            result.error_message = str(e)

        result.execution_time = time.time() - start_time
        return result

    def _execute_open(
        self,
        result: Optional[ExecutionResult],
        render: bool,
    ):
        """Open all fingers to minimum position"""
        n_steps = int(self.config.open_duration * self.config.control_freq)

        # Target: all fingers open
        target = np.full(11, self.config.min_finger_value)

        for step in range(n_steps):
            action = self._create_action(target)
            obs, _, _, _ = self.env.step(action)

            if result is not None:
                result.n_steps += 1
                self._record_step(result, obs)

            if render:
                self.env.render()

    def _execute_pregrasp(
        self,
        result: Optional[ExecutionResult],
        render: bool,
    ):
        """Move fingers to pre-grasp position (slightly open from target)"""
        n_steps = int(self.config.position_duration * self.config.control_freq)

        # Target: slightly open from grasp target
        pregrasp = np.maximum(
            self._target_dofs - self.config.pregrasp_offset,
            self.config.min_finger_value,
        )

        for step in range(n_steps):
            action = self._create_action(pregrasp)
            obs, _, _, _ = self.env.step(action)

            if result is not None:
                result.n_steps += 1
                self._record_step(result, obs)

            if render:
                self.env.render()

    def _execute_close(
        self,
        result: Optional[ExecutionResult],
        render: bool,
    ):
        """Close fingers incrementally toward and past target"""
        n_steps = int(self.config.close_duration * self.config.control_freq)

        # Start from pre-grasp
        current = np.maximum(
            self._target_dofs - self.config.pregrasp_offset,
            self.config.min_finger_value,
        )

        for step in range(n_steps):
            # Increment toward max
            current = np.minimum(
                current + self.config.close_increment,
                self.config.max_finger_value,
            )

            action = self._create_action(current)
            obs, _, _, _ = self.env.step(action)

            if result is not None:
                result.n_steps += 1
                contacts = self._count_contacts()
                result.contact_sequence.append(contacts)
                self._record_step(result, obs)

            if render:
                self.env.render()

            # Early stop if enough contacts
            if self._count_contacts() >= 4:
                break

    def _execute_hold(
        self,
        result: Optional[ExecutionResult],
        render: bool,
    ):
        """Hold current grasp configuration"""
        n_steps = int(self.config.hold_duration * self.config.control_freq)

        # Get current finger positions from sim
        current = self._get_current_finger_positions()

        for step in range(n_steps):
            action = self._create_action(current)
            obs, _, _, _ = self.env.step(action)

            if result is not None:
                result.n_steps += 1
                self._record_step(result, obs)

            if render:
                self.env.render()

    def _create_action(self, finger_targets: np.ndarray) -> np.ndarray:
        """
        Create action array for the environment.

        This maps the 11 finger DOF targets to the environment's action space.
        Assumes action space includes arm + hand actuators.
        """
        # Start with zero action ; hold arm position
        action = np.zeros(self.env.action_dim)

        # Fill in hand actuator targets
        for dof_idx, act_idx in self.dof_to_actuator.items():
            if dof_idx < len(finger_targets):
                # Map to action space - robosuite typically uses [-1, 1]
                # Assuming position control with range [0, max_finger_value]
                normalized = (
                    2.0 * finger_targets[dof_idx] / self.config.max_finger_value - 1.0
                )
                action[act_idx] = np.clip(normalized, -1.0, 1.0)

        return action

    def _get_current_finger_positions(self) -> np.ndarray:
        """Get current finger joint positions from simulation"""
        positions = np.zeros(11)
        model = self.env.sim.model
        data = self.env.sim.data

        for dof_idx, dof_name in enumerate(REALHAND_L6_DOF_NAMES):
            # Find joint with this name
            for jnt_idx in range(model.njnt):
                jnt_name = model.joint(jnt_idx).name.lower()
                if dof_name in jnt_name:
                    qpos_adr = model.jnt_qposadr[jnt_idx]
                    positions[dof_idx] = data.qpos[qpos_adr]
                    break

        return positions

    def _count_contacts(self) -> int:
        """Count contacts between hand and object"""
        # Use environment's contact counting if available
        if hasattr(self.env, "_num_gripper_object_contacts"):
            return self.env._num_gripper_object_contacts()

        # Fallback: count all contacts
        return self.env.sim.data.ncon

    def _record_step(self, result: ExecutionResult, obs: Dict):
        """Record observation and joint positions"""
        # Record joint positions
        joint_pos = self._get_current_finger_positions()
        result.joint_trajectory.append(joint_pos.copy())

        # Store compact observation info
        if "robot0_proprio-state" in obs:
            result.observations.append({
                "proprio": obs["robot0_proprio-state"].copy(),
            })


def execute_grasp(
    env,
    grasp: GraspItGrasp,
    config: Optional[ExecutionConfig] = None,
    verbose: bool = False,
    render: bool = False,
) -> ExecutionResult:
    """
    Convenience function to execute a grasp in an environment.

    Args:
        env: Robosuite environment
        grasp: Grasp to execute
        config: Execution configuration
        verbose: Print progress
        render: Render during execution

    Returns:
        ExecutionResult
    """
    executor = NaiveGraspExecutor(env, config, verbose=verbose)
    return executor.execute(grasp, record=True, render=render)
