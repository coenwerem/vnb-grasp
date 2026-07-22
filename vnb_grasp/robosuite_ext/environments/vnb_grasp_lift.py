from __future__ import annotations

import inspect

import numpy as np

from robosuite.environments.base import register_env
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor

from vnb_grasp.robosuite_ext.arenas.zarm_table_astra_arena import (
    ZArmRealhandL6RightArena,
)


class VNBGraspLift(Lift):
    """Lift variant that uses VNB-Grasp arena and an explicit robot base pose"""

    def __init__(
        self,
        *args,
        robot_base_pos: np.ndarray | None = None,
        **kwargs,
    ):
        """Accepts robosuite Lift kwargs, ignoring unsupported extras.

        The training YAML may include keys that exist in other robosuite envs and forks
        (e.g., `early_termination`). Robosuite will pass them through to the env
        constructor, so we drop unknown keys here to avoid errors
        """
        # VNB-Grasp arena uses different camera naming than robosuite defaults.
        # Robosuite may still attempt to resolve the render camera during init, even if
        # rendering is disabled, so we must ensure a valid camera name.
        kwargs.setdefault("render_camera", "agent-view")
        kwargs.setdefault("camera_names", "agent-view")

        sig = inspect.signature(Lift.__init__)
        allowed = set(sig.parameters.keys())
        allowed.discard("self")

        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}

        # NOTE: Lift.__init__ calls _load_model(), so any state used by _load_model()
        # must be initialized before calling super().__init__.
        self._robot_base_pos = (
            np.array(robot_base_pos, dtype=float)
            if robot_base_pos is not None
            else np.array([0.0, 0.405, 0.775], dtype=float)
        )

        super().__init__(*args, **filtered_kwargs)

        # Filled in once the MuJoCo sim is created ; see _setup_references
        self.gripper_geom_ids = set()
        self.object_geom_ids = set()

    def _num_gripper_object_contacts(self):
        """
        Counts active contacts between gripper and object geoms.
        Returns an integer >= 0.
        """
        count = 0
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            g1 = c.geom1
            g2 = c.geom2

            if (g1 in self.gripper_geom_ids and g2 in self.object_geom_ids) or (
                g2 in self.gripper_geom_ids and g1 in self.object_geom_ids
            ):
                # require actual contact, not near-miss
                if c.dist < 0.0:
                    count += 1

        return count

    def reward(self, action=None):
        # Monkey-patch reward function for collision-aware training
        reward = 0.0

        # Success
        if self._check_success():
            reward = 2.25

        elif self.reward_shaping:
            # Reaching
            gripper = self.robots[0].gripper
            if isinstance(gripper, dict):
                gripper = next(iter(gripper.values()))
            dist = self._gripper_to_target(
                gripper=gripper,
                target=self.cube.root_body,
                target_type="body",
                return_distance=True,
            )
            reward += 1.0 / (1.0 + 5.0 * dist)

            # Grasp formation
            num_contacts = self._num_gripper_object_contacts()
            reward += min(num_contacts / 4.0, 1.0) * 0.25

            # Lift shaping
            if num_contacts > 0:
                height = self.cube.root_body.pos[2]
                reward += np.clip((height - self.table_height) / 0.10, 0.0, 1.0)

        # Scale
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def _load_model(self):
        # Call ManipulationEnv._load_model() via Lift
        super(Lift, self)._load_model()

        # Place robot explicitly on the zarm_table surface.
        self.robots[0].robot_model.set_base_xpos(self._robot_base_pos)

        mujoco_arena = ZArmRealhandL6RightArena(table_friction=self.table_friction)
        mujoco_arena.set_origin([0, 0, 0])

        # Use the table_top site from the VNB-Grasp arena XML as our placement reference
        self.table_offset = mujoco_arena.table_offset

        # Object of interest; TODO: parameterize
        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],
            size_max=[0.022, 0.022, 0.022],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )

        # Placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    def _setup_references(self):
        super()._setup_references()

        try:
            from vnb_grasp.robosuite_ext.sensors.realhand_l6_tactile_mock import (
                LinkerL6TactileMock,
                LinkerL6TactileSpec,
            )

            gripper = self.robots[0].gripper
            if isinstance(gripper, dict):
                gripper = next(iter(gripper.values()))
            gripper_prefix = getattr(gripper, "naming_prefix", "")

            # set limits and force multipliers based on Linker L6 datasheet
            self._tactile_tip_force_max = {
                "thumb": 10.0,
                "index": 1.25,
                "middle": 1.25,
                "ring": 1.25,
                "pinky": 1.25,
            }
            self._tactile_tip_k = {
                "thumb": 25.5,
                "index": 204.0,
                "middle": 204.0,
                "ring": 204.0,
                "pinky": 204.0,
            }

            default_spec = LinkerL6TactileSpec(full_scale_n=1.25)
            tip_specs = {
                finger: LinkerL6TactileSpec(full_scale_n=force)
                for finger, force in self._tactile_tip_force_max.items()
            }
            mid_spec = LinkerL6TactileSpec(
                rows=12, cols=6, width_m=0.026, height_m=0.012, full_scale_n=1.25
            )
            thumb_mid_spec = LinkerL6TactileSpec(
                rows=12, cols=6, width_m=0.026, height_m=0.012, full_scale_n=10.0
            )
            palm_spec = LinkerL6TactileSpec(
                rows=12, cols=6, width_m=0.045, height_m=0.030, full_scale_n=1.25
            )

            self._realhand_l6_tactile = LinkerL6TactileMock(
                sim=self.sim,
                finger_tip_sites={
                    "thumb": f"{gripper_prefix}thumb_tip_site",
                    "index": f"{gripper_prefix}index_tip_site",
                    "middle": f"{gripper_prefix}middle_tip_site",
                    "ring": f"{gripper_prefix}ring_tip_site",
                    "pinky": f"{gripper_prefix}pinky_tip_site",
                    "thumb_mid": f"{gripper_prefix}thumb_mid_site",
                    "index_mid": f"{gripper_prefix}index_mid_site",
                    "middle_mid": f"{gripper_prefix}middle_mid_site",
                    "ring_mid": f"{gripper_prefix}ring_mid_site",
                    "pinky_mid": f"{gripper_prefix}pinky_mid_site",
                },
                finger_geom_substrings={
                    "thumb": f"{gripper_prefix}thumb_distal_collision_0",
                    "index": f"{gripper_prefix}index_distal_collision_0",
                    "middle": f"{gripper_prefix}middle_distal_collision_0",
                    "ring": f"{gripper_prefix}ring_distal_collision_0",
                    "pinky": f"{gripper_prefix}pinky_distal_collision_0",
                    "thumb_mid": f"{gripper_prefix}thumb_metacarpals_collision_0",
                    "index_mid": f"{gripper_prefix}index_proximal_collision_0",
                    "middle_mid": f"{gripper_prefix}middle_proximal_collision_0",
                    "ring_mid": f"{gripper_prefix}ring_proximal_collision_0",
                    "pinky_mid": f"{gripper_prefix}pinky_proximal_collision_0",
                    "palm": f"{gripper_prefix}hand_base_link_collision",
                },
                spec=default_spec,
                pad_specs={
                    **tip_specs,
                    "thumb_mid": thumb_mid_spec,
                    "index_mid": mid_spec,
                    "middle_mid": mid_spec,
                    "ring_mid": mid_spec,
                    "pinky_mid": mid_spec,
                },
            )
        except Exception:
            self._realhand_l6_tactile = None

    def _tactile_pad_to_finger(self, pad_name: str) -> str:
        if pad_name.endswith("_mid"):
            return pad_name[: -len("_mid")]
        return pad_name

    def _tactile_counts_to_force(self, counts, order):
        per_pad = []
        is_object = getattr(counts, "dtype", None) == object
        for idx, pad in enumerate(order):
            spec = self._realhand_l6_tactile.pad_specs.get(
                pad, self._realhand_l6_tactile.default_spec
            )
            scale = float(spec.full_scale_n) / max(1e-9, float(spec.full_scale_counts))
            pad_counts = counts[idx] if not is_object else np.asarray(counts[idx])
            per_pad.append(np.asarray(pad_counts, dtype=np.float32) * scale)
        try:
            return np.stack(per_pad, axis=0).astype(np.float32)
        except Exception:
            return np.array(per_pad, dtype=object)

    def _tactile_force_to_taxel(self, force, order):
        per_pad = []
        is_object = getattr(force, "dtype", None) == object
        for idx, pad in enumerate(order):
            base = self._tactile_pad_to_finger(pad)
            k = float(self._tactile_tip_k.get(base, 204.0))
            pad_force = force[idx] if not is_object else np.asarray(force[idx])
            per_pad.append(np.asarray(pad_force, dtype=np.float32) * k)
        try:
            return np.stack(per_pad, axis=0).astype(np.float32)
        except Exception:
            return np.array(per_pad, dtype=object)

    def _setup_observables(self):
        """Adds a Realhand-L6 tactile matrix observation"""

        observables = super()._setup_observables()

        if getattr(self, "_realhand_l6_tactile", None) is None:
            return observables

        pf = self.robots[0].robot_model.naming_prefix
        modality = f"{pf}tactile"

        @sensor(modality=modality)
        def tactile_l6_matrix(obs_cache):
            return self._realhand_l6_tactile.compute_tensor(layout="sdk").reshape(-1)

        observables[f"{pf}tactile_l6_matrix"] = Observable(
            name=f"{pf}tactile_l6_matrix",
            sensor=tactile_l6_matrix,
            sampling_rate=self.control_freq,
        )

        @sensor(modality=modality)
        def tactile_l6_matrix_ext(obs_cache):
            order = [
                "thumb",
                "index",
                "middle",
                "ring",
                "pinky",
                "thumb_mid",
                "index_mid",
                "middle_mid",
                "ring_mid",
                "pinky_mid",
            ]
            return self._realhand_l6_tactile.compute_tensor(
                layout="sdk", order=order
            ).reshape(-1)

        observables[f"{pf}tactile_l6_matrix_ext"] = Observable(
            name=f"{pf}tactile_l6_matrix_ext",
            sensor=tactile_l6_matrix_ext,
            sampling_rate=self.control_freq,
        )

        @sensor(modality=modality)
        def tactile_l6_force(obs_cache):
            order = ["thumb", "index", "middle", "ring", "pinky"]
            counts = self._realhand_l6_tactile.compute_tensor(layout="sdk", order=order)
            force = self._tactile_counts_to_force(counts, order)
            return force.reshape(-1)

        observables[f"{pf}tactile_l6_force"] = Observable(
            name=f"{pf}tactile_l6_force",
            sensor=tactile_l6_force,
            sampling_rate=self.control_freq,
        )

        @sensor(modality=modality)
        def tactile_l6_force_ext(obs_cache):
            order = [
                "thumb",
                "index",
                "middle",
                "ring",
                "pinky",
                "thumb_mid",
                "index_mid",
                "middle_mid",
                "ring_mid",
                "pinky_mid",
                "palm",
            ]
            counts = self._realhand_l6_tactile.compute_tensor(layout="sdk", order=order)
            force = self._tactile_counts_to_force(counts, order)
            return force.reshape(-1)

        observables[f"{pf}tactile_l6_force_ext"] = Observable(
            name=f"{pf}tactile_l6_force_ext",
            sensor=tactile_l6_force_ext,
            sampling_rate=self.control_freq,
        )

        @sensor(modality=modality)
        def tactile_l6_taxel(obs_cache):
            order = ["thumb", "index", "middle", "ring", "pinky"]
            counts = self._realhand_l6_tactile.compute_tensor(layout="sdk", order=order)
            force = self._tactile_counts_to_force(counts, order)
            taxel = self._tactile_force_to_taxel(force, order)
            return taxel.reshape(-1)

        observables[f"{pf}tactile_l6_taxel"] = Observable(
            name=f"{pf}tactile_l6_taxel",
            sensor=tactile_l6_taxel,
            sampling_rate=self.control_freq,
        )

        @sensor(modality=modality)
        def tactile_l6_taxel_ext(obs_cache):
            order = [
                "thumb",
                "index",
                "middle",
                "ring",
                "pinky",
                "thumb_mid",
                "index_mid",
                "middle_mid",
                "ring_mid",
                "pinky_mid",
            ]
            counts = self._realhand_l6_tactile.compute_tensor(layout="sdk", order=order)
            force = self._tactile_counts_to_force(counts, order)
            taxel = self._tactile_force_to_taxel(force, order)
            return taxel.reshape(-1)

        observables[f"{pf}tactile_l6_taxel_ext"] = Observable(
            name=f"{pf}tactile_l6_taxel_ext",
            sensor=tactile_l6_taxel_ext,
            sampling_rate=self.control_freq,
        )

        return observables


register_env(VNBGraspLift)
