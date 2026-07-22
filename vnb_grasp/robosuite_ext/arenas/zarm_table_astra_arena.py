from __future__ import annotations

import numpy as np

from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import new_element, new_geom

from vnb_grasp.robosuite_ext.paths import repo_root


class ZArmTableAstraArena(Arena):
    """VNB-Grasp arena: ZArm base table, workspace table, and an external Astra camera"""

    def __init__(
        self,
        xml: str | None = None,
        *,
        table_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001),
    ):
        if xml is None:
            xml = str(
                repo_root()
                / "arenas"
                / "zarm_realhand_l6_right_arena"
                / "zarm_realhand_l6_right_arena.xml"
            )

        super().__init__(xml)
        self._add_floor()

        self._table_friction = table_friction

        # Lift expects a site named table_top and uses table_offset as placement reference
        self._table_top_site = self.worldbody.find("site[@name='table_top']")

        # We treat the workspace table collision geom as the primary table surface
        self.table_body = self.worldbody.find("body[@name='workspace_table']")
        self.table_collision_geom = self.worldbody.find(
            "body[@name='workspace_table']/geom[@name='workspace_table_collision']"
        )

        if (
            self._table_top_site is None
            or self.table_body is None
            or self.table_collision_geom is None
        ):
            raise ValueError(
                "Arena XML must define body 'workspace_table', geom 'workspace_table_collision', and site 'table_top'"
            )

        table_top_pos_raw = self._table_top_site.get("pos")
        self._table_offset_local = np.fromstring(
            table_top_pos_raw if table_top_pos_raw is not None else "0 0 0", sep=" "
        )
        self.table_offset = self._table_offset_local.copy()
        self.table_top_abs = self.table_offset

        self.set_table_friction(table_friction)

    def set_origin(self, offset):
        self.worldbody.set("pos", f"{offset[0]} {offset[1]} {offset[2]}")
        offset = np.array(offset, dtype=float)
        self.table_offset = self._table_offset_local + offset
        self.table_top_abs = self.table_offset

    def set_table_friction(self, table_friction: tuple[float, float, float]):
        self._table_friction = table_friction
        geom = self.table_collision_geom
        if geom is None:
            return
        geom.set(
            "friction", f"{table_friction[0]} {table_friction[1]} {table_friction[2]}"
        )

    def _add_floor(self):
        # Some VNB-Grasp arena XMLs already include a floor geom.
        if self.worldbody.find("geom[@name='floor']") is not None:
            return

        tex = new_element(
            tag="texture",
            name="texplane_rs",
            type="2d",
            builtin="checker",
            width=512,
            height=512,
            rgb1=(0.85, 0.85, 0.85),
            rgb2=(0.75, 0.75, 0.75),
        )
        mat = new_element(
            tag="material",
            name="matplane_rs",
            texture="texplane_rs",
            texrepeat=(30, 30),
            specular=0.05,
            reflectance=0.1,
        )
        self.asset.append(tex)
        self.asset.append(mat)

        floor = new_geom(
            name="floor_rs",
            type="plane",
            size=(100, 100, 1),
            pos=(0, 0, 0),
            material="matplane_rs",
            contype=1,
            conaffinity=1,
        )
        self.worldbody.append(floor)


class ZArmRealhandL6RightArena(Arena):
    def __init__(
        self,
        xml: str | None = None,
        *,
        table_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001),
        remove_robot: bool = True,
    ):
        if xml is None:
            xml = str(
                repo_root()
                / "arenas"
                / "zarm_realhand_l6_right_arena"
                / "zarm_realhand_l6_right_arena.xml"
            )

        super().__init__(xml)
        self._add_floor()

        if remove_robot:
            robot_body = self.worldbody.find("body[@name='zarm_realhand_l6_right']")
            if robot_body is not None:
                def _collect_body_names(elem):
                    names = set()
                    if elem.tag == "body":
                        n = elem.get("name")
                        if n:
                            names.add(n)
                    for child in list(elem):
                        if child.tag == "body":
                            names |= _collect_body_names(child)
                    return names

                removed_body_names = _collect_body_names(robot_body)
                self.worldbody.remove(robot_body)

                # Remove any contact excludes that refer to removed bodies.
                root = getattr(self, "root", None)
                if root is None:
                    root = getattr(self, "tree", None)
                    root = root.getroot() if root is not None else None
                if root is not None:
                    # Remove actuators that targeted the removed embedded robot.
                    # robosuite caches the <actuator> element as self.actuator and later merges its
                    # children, so we must clear the element itself.
                    for child in list(self.actuator):
                        self.actuator.remove(child)

                    actuator = root.find("actuator")
                    if actuator is not None and actuator is not self.actuator:
                        root.remove(actuator)

                    contact = root.find("contact")
                    if contact is not None:
                        for exclude in list(contact.findall("exclude")):
                            b1 = exclude.get("body1")
                            b2 = exclude.get("body2")
                            if (b1 in removed_body_names) or (b2 in removed_body_names):
                                contact.remove(exclude)

        self._table_friction = table_friction

        self.table_body = self.worldbody.find("body[@name='workspace_table']")
        self.table_collision_geom = self.worldbody.find(
            "body[@name='workspace_table']/geom[@name='workspace_table_collision']"
        )
        self._table_top_site = self.worldbody.find(
            "body[@name='workspace_table']/site[@name='workspace_table_site']"
        )
        if self._table_top_site is None:
            self._table_top_site = self.worldbody.find(
                "site[@name='workspace_table_site']"
            )
        if self._table_top_site is None:
            self._table_top_site = self.worldbody.find("site[@name='table_site']")

        if (
            self._table_top_site is None
            or self.table_body is None
            or self.table_collision_geom is None
        ):
            raise ValueError(
                "Arena XML must define body 'workspace_table', geom 'workspace_table_collision', and site 'workspace_table_site'"
            )

        table_body_pos_raw = self.table_body.get("pos")
        table_body_pos = np.fromstring(
            table_body_pos_raw if table_body_pos_raw is not None else "0 0 0", sep=" "
        )
        table_site_pos_raw = self._table_top_site.get("pos")
        table_site_pos = np.fromstring(
            table_site_pos_raw if table_site_pos_raw is not None else "0 0 0", sep=" "
        )
        self._table_offset_local = table_body_pos + table_site_pos
        self.table_offset = self._table_offset_local.copy()
        self.table_top_abs = self.table_offset

        self.set_table_friction(table_friction)

    def set_origin(self, offset):
        self.worldbody.set("pos", f"{offset[0]} {offset[1]} {offset[2]}")
        offset = np.array(offset, dtype=float)
        self.table_offset = self._table_offset_local + offset
        self.table_top_abs = self.table_offset

    def set_table_friction(self, table_friction: tuple[float, float, float]):
        self._table_friction = table_friction
        geom = self.table_collision_geom
        if geom is None:
            return
        geom.set(
            "friction", f"{table_friction[0]} {table_friction[1]} {table_friction[2]}"
        )

    def _add_floor(self):
        # Some VNB-Grasp arena XMLs already include a floor geom.
        if self.worldbody.find("geom[@name='floor']") is not None:
            return

        tex = new_element(
            tag="texture",
            name="texplane_rs",
            type="2d",
            builtin="checker",
            width=512,
            height=512,
            rgb1=(0.85, 0.85, 0.85),
            rgb2=(0.75, 0.75, 0.75),
        )
        mat = new_element(
            tag="material",
            name="matplane_rs",
            texture="texplane_rs",
            texrepeat=(30, 30),
            specular=0.05,
            reflectance=0.1,
        )
        self.asset.append(tex)
        self.asset.append(mat)

        floor = new_geom(
            name="floor_rs",
            type="plane",
            size=(100, 100, 1),
            pos=(0, 0, 0),
            material="matplane_rs",
            contype=1,
            conaffinity=1,
        )
        self.worldbody.append(floor)
