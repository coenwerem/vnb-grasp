from __future__ import annotations

import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel

from vnb_grasp.robosuite_ext.paths import repo_root


class ZArm(ManipulatorModel):
    """VNB-Grasp ZArm model"""

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(
            str(repo_root() / "assets" / "arms" / "zarm" / "zarm.xml"),
            idn=idn,
        )

    @property
    def default_base(self):
        # VNB-Grasp scenes place the robot explicitly.
        return "NullMount"

    @property
    def default_gripper(self):
        # Keep consistent with existing perception + control scripts.
        return {"right": "RealHandL6Right"}

    @property
    def default_controller_config(self):
        # Generic OSC pose controller tends to work across 6-DoF arms.
        return {"right": "osc_pose"}

    @property
    def init_qpos(self):
        # Default arm joint configuration ; radians.
        return np.array([0.09162, -1.53854, 0.000486079, -1.53962, -2.78304e-05, -2.62314e-08])

    @property
    def base_xpos_offset(self):
        # Unused by VNBGraspLift variants ; we set base pose explicitly, but keep sane defaults.
        return {
            "empty": (0.0, 0.0, 0.0),
            "table": lambda table_length: (0.0, 0.0, 0.0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
