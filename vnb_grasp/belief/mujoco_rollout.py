"""MuJoCo-specific rollout utilities for belief-space planning.

Supports:
- State save/restore for particle rollouts
- Batched forward simulation via mjx (JAX)
- Contact extraction for belief updates
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import mujoco as mj
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

try:
    import mujoco.mjx as mjx
    import jax
    import jax.numpy as jnp
    HAS_MJX = True
except Exception:
    # mjx is an optional GPU accelerator. A missing OR version-skewed install
    # (e.g. mujoco 3.9 core vs newer mujoco_warp, which raises AttributeError on
    # import) must degrade to the CPU rollout path, not break the whole
    # `vnb_grasp` package import. CPU rollouts use mj_step, not mjx.
    mjx = None
    HAS_MJX = False


@dataclass
class SimState:
    """Snapshot of MuJoCo simulation state for rollback"""
    time: float
    qpos: NDArray
    qvel: NDArray
    act: NDArray
    ctrl: NDArray

    @classmethod
    def from_data(cls, data: "mj.MjData") -> "SimState":
        """Capture current simulation state"""
        return cls(
            time=data.time,
            qpos=data.qpos.copy(),
            qvel=data.qvel.copy(),
            act=data.act.copy() if data.act.size > 0 else np.array([]),
            ctrl=data.ctrl.copy(),
        )

    def restore(self, data: "mj.MjData") -> None:
        """Restore simulation to this state"""
        data.time = self.time
        data.qpos[:] = self.qpos
        data.qvel[:] = self.qvel
        if self.act.size > 0:
            data.act[:] = self.act
        data.ctrl[:] = self.ctrl
        mj.mj_forward(data.model, data)


@dataclass 
class ContactInfo:
    """Contact information extracted from MuJoCo"""
    geom1: int
    geom2: int
    pos: NDArray       # 3D contact position
    frame: NDArray     # 3x3 contact frame ; normal, tangent1, tangent2
    dist: float        # Penetration depth ; negative = penetrating
    force: NDArray     # 3D contact force ; normal + friction

    @property
    def normal(self) -> NDArray:
        """Contact normal (first row of frame)"""
        return self.frame[0]

    @property
    def normal_force(self) -> float:
        """Magnitude of normal force"""
        return float(self.force[0])

    @property
    def tangent_force(self) -> NDArray:
        """Tangential (friction) force components"""
        return self.force[1:3]

    @property
    def friction_ratio(self) -> float:
        """Ratio of tangent to normal force (slip indicator)"""
        fn = abs(self.normal_force)
        if fn < 1e-9:
            return 0.0
        return float(np.linalg.norm(self.tangent_force) / fn)


def extract_contacts(
    model: "mj.MjModel",
    data: "mj.MjData",
    geom_filter: Optional[set] = None,
) -> List[ContactInfo]:
    """Extract contact information from MuJoCo simulation.

    Args:
        model: MuJoCo model
        data: MuJoCo data (after mj_step)
        geom_filter: If provided, only include contacts where at least
                     one geom is in this set

    Returns:
        List of ContactInfo for active contacts
    """
    contacts = []

    for i in range(data.ncon):
        c = data.contact[i]

        if geom_filter is not None:
            if c.geom1 not in geom_filter and c.geom2 not in geom_filter:
                continue

        force = np.zeros(6)
        mj.mj_contactForce(model, data, i, force)

        contacts.append(ContactInfo(
            geom1=c.geom1,
            geom2=c.geom2,
            pos=c.pos.copy(),
            frame=c.frame.reshape(3, 3),
            dist=c.dist,
            force=force[:3],  # Normal + 2 tangent components
        ))

    return contacts


def get_fingertip_geom_ids(
    model: "mj.MjModel",
    tip_names: Optional[List[str]] = None,
) -> set:
    """Get geom IDs for fingertip geoms.

    Args:
        model: MuJoCo model
        tip_names: List of geom names. If None, uses RealHand L6 defaults.

    Returns:
        Set of geom IDs for fingertips
    """
    if tip_names is None:
        # RealHand L6 default fingertip geom names
        tip_names = [
            "thumb_tip_collision",
            "index_tip_collision", 
            "middle_tip_collision",
            "ring_tip_collision",
            "pinky_tip_collision",
        ]

    geom_ids = set()
    for name in tip_names:
        try:
            gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                geom_ids.add(gid)
        except Exception:
            pass

    return geom_ids


class ParticleRolloutEngine:
    """Engine for rolling out action sequences across belief particles.

    For each particle (representing a hypothesis about friction, stiffness, etc.),
    we fork the simulation state, apply the action sequence, and measure outcomes.
    """

    def __init__(
        self,
        model: "mj.MjModel",
        data: "mj.MjData",
        n_steps_per_action: int = 10,
        use_mjx: bool = False,
    ):
        """Initialize rollout engine.

        Args:
            model: MuJoCo model
            data: MuJoCo data (will be forked for rollouts)
            n_steps_per_action: Physics steps per high-level action
            use_mjx: If True and mjx available, use JAX-accelerated rollouts
        """
        self.model = model
        self.data = data
        self.n_steps_per_action = n_steps_per_action
        self.use_mjx = use_mjx and HAS_MJX

        self.fingertip_geoms = get_fingertip_geom_ids(model)

        if self.use_mjx:
            self._mjx_model = mjx.put_model(model)

    def rollout_single(
        self,
        ctrl_sequence: List[NDArray],
        cost_fn: Callable[[ContactInfo], float],
        initial_state: Optional[SimState] = None,
    ) -> Tuple[float, List[ContactInfo]]:
        """Rollout a control sequence and compute cumulative cost.

        Args:
            ctrl_sequence: List of control vectors to apply
            cost_fn: Function mapping ContactInfo list to stage cost
            initial_state: Starting state; if None, uses current data state

        Returns:
            (cumulative_cost, final_contacts)
        """
        if initial_state is None:
            initial_state = SimState.from_data(self.data)

        # Fork to rollout state
        rollout_data = mj.MjData(self.model)
        initial_state.restore(rollout_data)

        cumulative_cost = 0.0

        for ctrl in ctrl_sequence:
            rollout_data.ctrl[:] = ctrl

            for _ in range(self.n_steps_per_action):
                mj.mj_step(self.model, rollout_data)

            contacts = extract_contacts(
                self.model, rollout_data, self.fingertip_geoms
            )
            cumulative_cost += cost_fn(contacts)

        final_contacts = extract_contacts(
            self.model, rollout_data, self.fingertip_geoms
        )

        return cumulative_cost, final_contacts

    def rollout_batch_mjx(
        self,
        ctrl_sequences: NDArray,  # ; n_particles, horizon, n_ctrl
        cost_fn: Callable,
    ) -> NDArray:
        """Batched rollout using mjx for GPU acceleration.

        Args:
            ctrl_sequences: Control sequences for each particle
            cost_fn: Vectorized cost function

        Returns:
            costs: (n_particles,) array of cumulative costs
        """
        if not self.use_mjx:
            raise RuntimeError("mjx not available or disabled")

        n_particles, horizon, n_ctrl = ctrl_sequences.shape

        # Replicate initial state across particles
        initial_state = SimState.from_data(self.data)
        qpos_batch = jnp.tile(initial_state.qpos[None, :], (n_particles, 1))
        qvel_batch = jnp.tile(initial_state.qvel[None, :], (n_particles, 1))

        mjx_data = mjx.make_data(self._mjx_model)
        mjx_data = mjx_data.replace(
            qpos=qpos_batch,
            qvel=qvel_batch,
        )

        @jax.vmap
        def step_particle(data, ctrl):
            data = data.replace(ctrl=ctrl)
            for _ in range(self.n_steps_per_action):
                data = mjx.step(self._mjx_model, data)
            return data

        costs = jnp.zeros(n_particles)
        for t in range(horizon):
            ctrl_t = jnp.array(ctrl_sequences[:, t, :])
            mjx_data = step_particle(mjx_data, ctrl_t)

        return np.array(costs)


def compute_grasp_quality_from_contacts(
    contacts: List[ContactInfo],
    min_contacts: int = 2,
    target_contacts: int = 4,
) -> float:
    """Compute a simple grasp quality metric from contacts.

    Quality = (n_contacts / target_contacts) * mean_normal_force_ratio

    For a proper GWS-based quality, see gws_quality.py.

    Args:
        contacts: List of active fingertip contacts
        min_contacts: Minimum contacts for non-zero quality
        target_contacts: Target number of contacts

    Returns:
        Quality in [0, 1]
    """
    if len(contacts) < min_contacts:
        return 0.0

    count_factor = min(len(contacts) / target_contacts, 1.0)

    # Force distribution factor ; prefer balanced forces
    if len(contacts) > 0:
        normal_forces = [abs(c.normal_force) for c in contacts]
        mean_force = np.mean(normal_forces)
        std_force = np.std(normal_forces)
        balance_factor = 1.0 / (1.0 + std_force / (mean_force + 1e-6))
    else:
        balance_factor = 0.0

    return float(count_factor * balance_factor)


def friction_cone_violation(
    contacts: List[ContactInfo],
    mu_hypothesis: float,
) -> float:
    """Check how much contacts violate friction cone for a given mu.

    Returns fraction of contacts where |f_t| / |f_n| > mu.

    Args:
        contacts: List of contacts
        mu_hypothesis: Friction coefficient hypothesis

    Returns:
        Violation ratio in [0, 1]
    """
    if len(contacts) == 0:
        return 0.0

    violations = 0
    for c in contacts:
        if c.friction_ratio > mu_hypothesis:
            violations += 1

    return violations / len(contacts)
