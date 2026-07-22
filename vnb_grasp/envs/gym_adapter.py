"""Gymnasium adapter for RawMujocoEnv.

Provides a ``make_env`` factory that returns environments with the standard
Gymnasium interface (dict observations, 5-tuple step returns) used by the
example scripts.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from vnb_grasp.wrappers.mujoco_native import RawMujocoEnv


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# Default fingertip geom names for the ZArm hand
_DEFAULT_FINGERTIP_GEOMS = [
    "thumb_metacarpals_base2_collision_0",
    "thumb_metacarpals_collision_0",
    "thumb_distal_collision_0",
    "index_proximal_collision_0",
    "index_distal_collision_0",
    "middle_proximal_collision_0",
    "middle_distal_collision_0",
    "ring_proximal_collision_0",
    "ring_distal_collision_0",
    "pinky_proximal_collision_0",
    "pinky_distal_collision_0",
    "hand_base_link_collision",
    "palm_link_collision",
]

# Mapping from short env name to (arena_subdir, object_geom_names)
_ENV_REGISTRY: Dict[str, Dict[str, Any]] = {
    "zarm_grasp": {
        "arena": "zarm_realhand_l6_right_arena",
        "object_geoms": ["cube_collision"],
    },
    "hand_object_testbed": {
        "arena": "hand_object_testbed",
        "object_geoms": ["cube_collision"],
    },
}


class GymEnvAdapter:
    """Thin adapter giving ``RawMujocoEnv`` a Gymnasium-like interface.

    * ``reset()`` returns a dict ``{"observation": ..., "state": ...}``
    * ``step(action)`` returns ``(obs_dict, reward, terminated, truncated, info)``
    """

    def __init__(self, raw_env: RawMujocoEnv, max_steps: int = 200) -> None:
        self._env = raw_env
        self._max_steps = max_steps
        self._step_count = 0

    # ------------------------------------------------------------------
    # Gymnasium-like interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None) -> Dict[str, Any]:
        raw_obs = self._env.reset()
        self._step_count = 0
        return self._wrap_obs(raw_obs)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        raw_obs, reward, done, info = self._env.step(action)
        self._step_count += 1
        terminated = done
        truncated = self._step_count >= self._max_steps
        return self._wrap_obs(raw_obs), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Convenience proxies
    # ------------------------------------------------------------------

    @property
    def model(self):
        return self._env.model

    @property
    def data(self):
        return self._env.data

    @property
    def action_dim(self) -> int:
        return self._env.action_dim

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_obs(raw_obs) -> Dict[str, Any]:
        """Convert ``GraspObservation`` to a flat dict"""
        obs_vec = np.concatenate([raw_obs.q, raw_obs.dq])

        state = obs_vec.copy()

        return {"observation": obs_vec, "state": state}


def make_env(
    env_name: str = "zarm_grasp",
    *,
    arena_xml: Optional[str] = None,
    max_steps: int = 200,
) -> GymEnvAdapter:
    """Create a Gymnasium MuJoCo environment.

    Parameters
    ----------
    env_name:
        Short name registered in ``_ENV_REGISTRY``, e.g. ``"zarm_grasp"``.
    arena_xml:
        Explicit path to a scene XML. Overrides the registry lookup.
    max_steps:
        Episode horizon for truncation.

    Returns
    -------
    GymEnvAdapter
        Environment with ``reset()`` / ``step()`` matching the Gymnasium API.
    """
    entry = _ENV_REGISTRY.get(env_name)

    if arena_xml is None:
        if entry is None:
            available = list(_ENV_REGISTRY.keys())
            raise ValueError(
                f"Unknown env_name {env_name!r}. Available: {available}. "
                "Or pass arena_xml= explicitly."
            )
        arena_xml = os.path.join(
            _repo_root(), "arenas", entry["arena"], "scene.xml"
        )

    object_geoms = entry["object_geoms"] if entry else []

    raw_env = RawMujocoEnv(
        xml_path=arena_xml,
        fingertip_geom_names=_DEFAULT_FINGERTIP_GEOMS,
        object_geom_names=object_geoms,
    )

    return GymEnvAdapter(raw_env, max_steps=max_steps)
