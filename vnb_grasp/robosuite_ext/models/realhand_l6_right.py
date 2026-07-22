from __future__ import annotations

import numpy as np

from robosuite.models.grippers import GripperModel

from vnb_grasp.robosuite_ext.paths import repo_root


class RealHandL6RightGripper(GripperModel):
    """RealHand L6 (right) wrapped for robosuite gripper conventions"""

    def __init__(self, idn=0):
        super().__init__(
            str(
                repo_root()
                / "assets"
                / "end_effectors"
                / "realhand_l6_right"
                / "realhand_l6_right.xml"
            ),
            idn=idn,
        )

    def format_action(self, action):
        # RealHand actuators use a small torque range ; ctrlrange="-0.2 0.2"
        # Map robosuite ; -1, 1 action space into that range
        action = np.array(action, dtype=np.float32).reshape(-1)
        return np.clip(action, -1.0, 1.0) * 0.2

    @property
    def naming_prefix(self):
        return f"gripper{self.idn}_"

    @property
    def init_qpos(self):
        return np.zeros(self.dof)


def register_realhand_l6_right():
    """Registers the gripper in robosuite's gripper factory"""

    # robosuite uses a global registry behind gripper_factory; importing the module is
    # usually enough if it registers itself, but robosuite doesn't expose a stable
    # registration API. Therefore, we support both:
    from robosuite.models.grippers import GRIPPER_MAPPING

    GRIPPER_MAPPING["RealHandL6Right"] = RealHandL6RightGripper


# ensure this module import registers the mapping
register_realhand_l6_right()
