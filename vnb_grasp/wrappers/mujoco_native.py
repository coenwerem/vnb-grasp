"""
MuJoCo-native environment adapter for interfacing with external grasp proposal providers.
"""

from __future__ import annotations

import numpy as np
import mujoco as mj

from vnb_grasp.belief.contact_belief import GraspObservation
from vnb_grasp.belief.mujoco_rollout import extract_contacts, get_fingertip_geom_ids, SimState
from vnb_grasp.control.actuator_map import ActuatorMap

class RawMujocoEnv:
    """
    Minimal MuJoCo backend for BeliefMPCPlanner and BeliefGraspingEnv
    """
    def __init__(
        self,
        xml_path: str,
        fingertip_geom_names=None,
        object_geom_names=None,
        n_substeps: int = 10,
    ):
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.n_substeps = n_substeps
        self.actmap = ActuatorMap(self.model)

        self.action_dim = self.model.nu  # actuator controls
        if self.action_dim == 0:
            raise ValueError("Model has no actuators (nu=0). Add actuators or use torque/position actuators.")

        # Identify fingertip geoms
        self.fingertip_geoms = get_fingertip_geom_ids(self.model, fingertip_geom_names)

        # object geoms to filter contacts
        self.object_geoms = set()
        if object_geom_names is not None:
            for name in object_geom_names:
                gid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name)
                if gid >= 0:
                    self.object_geoms.add(gid)

    def reset(self):
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        return self.get_observation()

    def step(self, ctrl: np.ndarray):
        ctrl = np.asarray(ctrl, dtype=np.float64)
        if ctrl.shape != (self.action_dim,):
            raise ValueError(f"ctrl must be shape ({self.action_dim},), got {ctrl.shape}")

        self.data.ctrl[:] = ctrl

        for _ in range(self.n_substeps):
            mj.mj_step(self.model, self.data)

        obs = self.get_observation()
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info

    def fork_state(self) -> SimState:
        return SimState.from_data(self.data)

    def restore_state(self, state: SimState):
        state.restore(self.data)

    def _get_contacts(self):
        # fingertip-filtered contacts; TODO: may extend to include patches or fingertip sites
        contacts = extract_contacts(self.model, self.data, geom_filter=self.fingertip_geoms)
        if len(self.object_geoms) == 0:
            return contacts
        return [c for c in contacts if (c.geom1 in self.object_geoms or c.geom2 in self.object_geoms)]

    def get_observation(self) -> GraspObservation:
        # Basic proprioception: use MuJoCo qpos/qvel directly
        # TODO: Add camera and tactile observations later
        q = self.data.qpos.copy()
        dq = self.data.qvel.copy()

        # Contacts
        contacts = self._get_contacts()
        if len(contacts) > 0:
            # Normal force proxy is ContactInfo.normal_force
            contact_forces = np.array([c.normal_force for c in contacts], dtype=np.float64)
            contact_points = np.array([c.pos for c in contacts], dtype=np.float64)
            contact_normals = np.array([c.normal for c in contacts], dtype=np.float64)

            # Get slip proxy from tangent/normal force ratio
            slip_proxy = np.array([c.friction_ratio for c in contacts], dtype=np.float64)
        else:
            contact_forces = None
            contact_points = None
            contact_normals = None
            slip_proxy = None

        return GraspObservation(
            q=q,
            dq=dq,
            contact_forces=contact_forces,
            contact_points=contact_points,
            contact_normals=contact_normals,
            slip_velocity=slip_proxy,  # not true velocity, but works as a slip indicator
        )
