from __future__ import annotations

"""Simple script for registration of VNB-Grasp models and envs into robosuite.

robosuite discovers robots, grippers, and envs by import-time registration.
This module ensures those imports happen and updates robosuite's mappings
"""


def register_all() -> None:
    # Importing these modules is what triggers registration into robosuite registries
    # i.e., RobotModelMeta  register_gripper, and register_env
    from vnb_grasp.robosuite_ext.models import zarm  # noqa: F401
    from vnb_grasp.robosuite_ext.models import realhand_l6_right  # noqa: F401
    from vnb_grasp.robosuite_ext.environments import vnb_grasp_lift  # noqa: F401

    from robosuite.robots import FixedBaseRobot, ROBOT_CLASS_MAPPING

    ROBOT_CLASS_MAPPING.setdefault("ZArm", FixedBaseRobot)

    # Load MuJoCo plugin libraries if present
    try:
        import mujoco

        # Best-effort: load all bundled plugins shipped with the mujoco pip package
        if hasattr(mujoco, "_load_all_bundled_plugins"):
            mujoco._load_all_bundled_plugins()
        elif hasattr(mujoco, "mj_loadAllPluginLibraries") and hasattr(
            mujoco, "PLUGINS_DIR"
        ):
            mujoco.mj_loadAllPluginLibraries(str(mujoco.PLUGINS_DIR))
    except Exception:
        pass
