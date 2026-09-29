"""Record executed MuJoCo states for talk-demo review; never synthesize motion.

The historical runner is obtained from the local MuJoCoDex git history. Its
source, changes, commands and outcomes are retained beside each state archive.
This is an exploratory demo protocol, not a reproduction of a paper table.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import mujoco as mj
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=["cem", "variational"], required=True)
    p.add_argument("--runner", choices=["current", "historical", "release-historical"], default="current")
    p.add_argument("--object", default="cube")
    p.add_argument("--friction", type=float, default=.25)
    p.add_argument("--contact-friction", choices=["original", "matched"], default="original")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--beta", type=float, default=.9)
    p.add_argument("--shear-friction", type=float,
                   help="Optional, declared friction drop at the first shear pulse")
    p.add_argument("--stress-timing", choices=["original", "physical"], default="original",
                   help="physical accounts for all MuJoCo substeps when timing the stress test")
    p.add_argument("--stop-on-loss", action="store_true",
                   help="End shear at 30 mm motion or 10 mm drop, then settle without external force")
    p.add_argument("--restore-table-contacts", action="store_true",
                   help="Remove active-object exclusions against world/main table in this demo model")
    p.add_argument("--grasp-initialization", choices=["scripted", "firmgrasp"], default="scripted")
    p.add_argument("--object-offset-xy", type=float, nargs=2, default=[0., 0.])
    p.add_argument("--grasp-envelope", type=float,
                   help="Bound closure targets above the shared FirmGrasp pose; hold opposition yaw")
    p.add_argument("--lift-hold-seconds", type=float, default=.2)
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    source_root = (Path("/home/drce/ResearchProjects/MuJoCoDex_iros_release")
                   if a.runner == "release-historical" else ROOT)
    os.chdir(source_root)
    sys.path[:0] = [str(source_root), str(source_root / "examples")]
    torch.set_num_threads(1)
    np.random.seed(a.seed)
    provenance = {"command": sys.argv, "mujoco": mj.__version__, "torch": torch.__version__,
                  "standalone_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()}
    if a.runner in {"historical", "release-historical"}:
        rev = "cd9c77fa"
        source = subprocess.check_output(["git", "-C", "/home/drce/ResearchProjects/MuJoCoDex",
                  "show", rev + ":examples/run_variational_belief_experiments.py"], text=True)
        provenance["historical_revision"] = rev
        provenance["historical_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        if a.runner == "historical":
            source = source.replace("from mujocodex.", "from vnb_grasp.")
        source = source.replace("planner.plan(body_name, verbose=True, strategy=_strategy)",
                                "planner.plan(body_name, verbose=True)")
        provenance["source_adaptations"] = [
            "Use current planner plan(body_name, verbose=True); no strategy override"]
        if a.runner == "historical":
            provenance["source_adaptations"].append("mujocodex imports renamed to vnb_grasp")
    else:
        source = (ROOT / "examples/run_variational_belief_experiments.py").read_text()
        provenance["source_adaptations"] = []
    if a.stress_timing == "physical":
        start = source.index("def run_lift_and_shear(")
        end = source.index("\ndef ", start+1)
        stress = source[start:end]
        assert stress.count("dt = env.model.opt.timestep") == 1
        source = source[:start] + stress.replace("dt = env.model.opt.timestep",
            "dt = env.model.opt.timestep * env.n_substeps") + source[end:]
        provenance["source_adaptations"].append("Stress timing uses actual env.step duration (dt * n_substeps)")
    if a.lift_hold_seconds != .2:
        start = source.index("def run_lift_and_shear(")
        end = source.index("\ndef ", start+1)
        stress = source[start:end]
        assert stress.count('range(int(0.2 / dt))') == 1
        source = source[:start]+stress.replace('range(int(0.2 / dt))',
            f'range(int({a.lift_hold_seconds!r} / dt))')+source[end:]
        provenance['source_adaptations'].append(f'Unforced hold after lift: {a.lift_hold_seconds} s')
    source_path = a.output / "runner_snapshot.py"
    source_path.write_text(source)
    spec = importlib.util.spec_from_file_location("recorded_runner", source_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    last_hand_target = [None]
    source_pd = mod._pd_hand_ctrl
    def tracked_pd(target, env):
        last_hand_target[0] = np.array(target).copy()
        return source_pd(target, env)
    mod._pd_hand_ctrl = tracked_pd
    if a.grasp_envelope is not None:
        if a.grasp_initialization != "firmgrasp" or not 0 <= a.grasp_envelope <= .3:
            raise ValueError('Grasp envelope requires FirmGrasp initialization and 0..0.3 rad')
        from opposing_grasp_setup import BANK
        bank = json.loads(BANK.read_text())
        bank_q = np.array(bank['q'])
        original_pd = mod._pd_hand_ctrl
        def bounded_pd(target, env):
            names = [env.model.joint(i).name for i in range(6, 17)]
            ref = np.array([bank_q[bank['dof_order'].index(n)] for n in names])
            lower = env.model.jnt_range[6:17, 0].copy()
            upper = np.minimum(env.model.jnt_range[6:17, 1], ref+a.grasp_envelope)
            for name in ('thumb_cmc_yaw', 'index_mcp_pitch', 'index_dip'):
                idx = names.index(name)
                lower[idx] = upper[idx] = ref[idx]
            return original_pd(np.clip(target, lower, upper), env)
        mod._pd_hand_ctrl = bounded_pd
        provenance['source_adaptations'].append(
            'Shared FirmGrasp closure envelope applied to PD targets; thumb opposition yaw and retracted index held at bank pose')
    cfg = mod.OBJECT_CONFIGS[a.object]
    env = mod.make_env()
    if a.restore_table_contacts:
        scene = mj.MjSpec.from_file(str(source_root / "arenas/zarm_realhand_l6_right_arena_no_wrist_cam/scene.xml"))
        removed = []
        for exclusion in list(scene.excludes):
            pair = {exclusion.bodyname1, exclusion.bodyname2}
            if cfg["body"] in pair and pair.intersection({"world", "table"}):
                removed.append(sorted(pair))
                scene.delete(exclusion)
        env.model = scene.compile()
        env.data = mj.MjData(env.model)
        env.actmap = mod.ActuatorMap(env.model)
        provenance["removed_collision_exclusions"] = removed
        (a.output / "scene_snapshot.xml").write_text(scene.to_xml())
    original_position = mod.position_arm_and_object
    samples = []
    phase = ["initialization"]
    events = []
    start_sim = [None]
    next_sample = [0.]
    friction_changed = [False]
    obj_id = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_BODY, cfg["body"])
    palm_id = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_BODY, "palm_link")
    object_geoms = set(np.where(env.model.geom_bodyid == obj_id)[0].tolist())
    effective_friction = set()
    table_contacts = [0]
    shear_start = [None]
    lift_start = [None]
    contact_samples = []
    from opposing_grasp_setup import contact_audit, initialize

    class GraspLost(Exception):
        pass

    def record():
        if start_sim[0] is None:
            return
        t = env.data.time - start_sim[0]
        for c in env.data.contact:
            if c.geom1 in object_geoms or c.geom2 in object_geoms:
                other = c.geom2 if c.geom1 in object_geoms else c.geom1
                if other in env.fingertip_geoms:
                    effective_friction.add(round(float(c.friction[0]), 6))
                other_name = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_GEOM, other) or ""
                if "table" in other_name:
                    table_contacts[0] += 1
        if t + 1e-9 < next_sample[0]:
            return
        next_sample[0] += 1 / 30
        contact_samples.append(dict(time_s=float(t), phase=phase[0], **contact_audit(env.model, env.data, cfg['body'])))
        R = env.data.xmat[palm_id].reshape(3, 3)
        relative = R.T @ (env.data.xpos[obj_id] - env.data.xpos[palm_id])
        samples.append((t, env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(),
                        env.data.xfrc_applied[obj_id].copy(), relative.copy(), phase[0]))

    def position(*args, **kwargs):
        if a.grasp_initialization == "firmgrasp":
            ok = initialize(env, mod, cfg, a.friction, a.output, a.object_offset_xy)
        else:
            ok = original_position(*args, **kwargs)
        if a.contact_friction == "matched":
            for gid in object_geoms | set(env.fingertip_geoms):
                env.model.geom_friction[gid, 0] = a.friction
        start_sim[0] = float(env.data.time)
        phase[0] = "grasp"
        mj.mj_forward(env.model, env.data)
        mj.mj_saveModel(env.model, str(a.output / "model.mjb"))
        record()
        return ok

    mod.position_arm_and_object = position
    original_step = env.step

    def step(ctrl):
        force_active = np.linalg.norm(env.data.xfrc_applied[obj_id, :3]) > 0
        if force_active and shear_start[0] is None:
            shear_start[0] = (env.data.xpos[obj_id].copy(), env.data.xpos[palm_id].copy(), float(env.data.time))
        if (a.shear_friction is not None and not friction_changed[0]
                and np.linalg.norm(env.data.xfrc_applied[obj_id, :3]) > 0):
            for gid in object_geoms | set(env.fingertip_geoms):
                env.model.geom_friction[gid, 0] = a.shear_friction
            friction_changed[0] = True
            events.append({"phase": "friction_drop", "time_s": float(env.data.time-start_sim[0]),
                           "before": a.friction, "after": a.shear_friction})
        result = original_step(ctrl)
        record()
        if a.stop_on_loss and force_active:
            origin = shear_start[0][0]
            displacement = float(np.linalg.norm(env.data.xpos[obj_id]-origin))
            drop = float(origin[2]-env.data.xpos[obj_id, 2])
            if displacement > .03 or drop > .01:
                events.append({"phase": "grasp_lost", "time_s": float(env.data.time-start_sim[0]),
                               "displacement_m": displacement, "drop_m": drop})
                raise GraspLost
        return result

    env.step = step
    original_lift = mod.run_lift_and_shear

    def lift(*args, **kwargs):
        phase[0] = "lift_and_shear"
        lift_start[0] = (env.data.xpos[obj_id].copy(), env.data.xpos[palm_id].copy())
        events.append({"phase": phase[0], "time_s": float(env.data.time-start_sim[0])})
        try:
            result = original_lift(*args, **kwargs)
        except GraspLost:
            obj_lift = shear_start[0][0][2]-lift_start[0][0][2]
            palm_lift = shear_start[0][1][2]-lift_start[0][1][2]
            relative_lift = obj_lift / palm_lift if palm_lift > 1e-3 else 0.
            result = {"lift_success": relative_lift > .8, "lift_ratio": float(relative_lift),
                      "lift_height_achieved": float(obj_lift), "shear_success": False,
                      "pulses_survived": 0, "max_displacement": events[-1]["displacement_m"],
                      "peak_slip_distance": events[-1]["displacement_m"],
                      "time_to_slip": None, "termination": "30 mm motion or 10 mm drop"}
            env.data.xfrc_applied[:] = 0
            phase[0] = "force_removed_after_loss"
            for _ in range(round(1. / (env.model.opt.timestep*env.n_substeps))):
                ctrl = env.data.ctrl.copy()
                ctrl[6:17] = mod._pd_hand_ctrl(last_hand_target[0], env)
                original_step(ctrl)
                record()
        events.append({"phase": "end", "time_s": float(env.data.time-start_sim[0])})
        return result

    mod.run_lift_and_shear = lift
    # Omit the separate reset-based perturbation battery from this continuous clip.
    mod.run_perturbation_battery = lambda *args, **kwargs: {
        "n_survived": 0, "n_total": 0, "survival_rate": 0., "details": []}
    provenance["omitted"] = "Separate perturbation battery; robust_success from this run is not an evaluated metric"
    tick = time.time()
    fn = mod.run_cem_episode if a.method == "cem" else mod.run_variational_episode
    try:
        result = fn(env, cfg, beta=a.beta, seed=a.seed, friction=a.friction,
                    max_steps=a.max_steps, deadline=time.time()+600)
        (a.output / "result.json").write_text(json.dumps(mod._to_json(asdict(result)), indent=2))
    finally:
        provenance.update({"settings": vars(a) | {"output": str(a.output)}, "events": events,
                           "elapsed_s": time.time()-tick, "effective_hand_contact_friction": sorted(effective_friction),
                           "recorded_object_table_contacts": table_contacts[0]})
        if a.runner == "release-historical":
            provenance["release_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            provenance["release_path"] = str(source_root)
            modules = {}
            for name, module in list(sys.modules.items()):
                path = getattr(module, "__file__", None)
                if name.startswith("mujocodex") and path:
                    f = Path(path).resolve()
                    if not f.is_relative_to(source_root):
                        raise RuntimeError(f"Imported module outside release: {f}")
                    modules[name] = {"path": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
            provenance["imported_modules"] = modules
        if samples:
            provenance["model_sha256"] = hashlib.sha256((a.output / "model.mjb").read_bytes()).hexdigest()
        (a.output / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
        (a.output / "contact_audit.json").write_text(json.dumps(contact_samples))
        if samples:
            np.savez_compressed(a.output / "states.npz", time_s=[s[0] for s in samples],
                qpos=[s[1] for s in samples], qvel=[s[2] for s in samples], ctrl=[s[3] for s in samples],
                object_wrench=[s[4] for s in samples], object_in_palm=[s[5] for s in samples],
                phase=[s[6] for s in samples])
    print(json.dumps({"output": str(a.output), "result": mod._to_json(asdict(result))}, indent=2))


if __name__ == "__main__":
    main()
