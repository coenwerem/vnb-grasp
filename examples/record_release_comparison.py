"""Record actual VNB/CEM trajectories from the IROS release comparison runner.

Use MuJoCoDex/.venv Python (MuJoCo 3.9). The release checkout is read only;
snapshots and executed states are written under --output. No checkpoint is
loaded by the release runner. This is a qualitative rerun, not paper statistics.
"""
from __future__ import annotations
import argparse
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path, default=Path("/home/drce/ResearchProjects/MuJoCoDex_iros_release"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=["variational", "cem"], required=True)
    p.add_argument("--object", default="graspit_box")
    p.add_argument("--friction", type=float, default=.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta", type=float, default=.9)
    p.add_argument("--max-steps", type=int, default=40)
    a = p.parse_args()
    a.release = a.release.resolve()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    os.chdir(a.release)
    sys.path[:0] = [str(a.release / "examples"), str(a.release)]
    torch.set_num_threads(1)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    original = (a.release / "examples/capture_comparison_stills.py").read_text()
    # Correct the elapsed lift duration equally for both methods. Each env.step
    # executes ten substeps. Keep all method action/termination code unchanged.
    assert original.count("dt = env.model.opt.timestep") == 2
    source = original.replace("dt = env.model.opt.timestep", "dt = env.model.opt.timestep * env.n_substeps")
    source = source.replace("    # Lift\n", "    mark_lift()\n    # Lift\n")
    snapshot = a.output / "runner_snapshot.py"
    snapshot.write_text(source)
    spec = importlib.util.spec_from_file_location("release_comparison", snapshot)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    cfg = mod.OBJECT_CONFIGS[a.object]
    env = mod.make_env()
    scene = mj.MjSpec.from_file(str(a.release / "arenas/zarm_realhand_l6_right_arena_no_wrist_cam/scene.xml"))
    removed = []
    for ex in list(scene.excludes):
        pair = {ex.bodyname1, ex.bodyname2}
        if cfg["body"] in pair and pair.intersection({"world", "table"}):
            removed.append(sorted(pair))
            scene.delete(ex)
    env.model = scene.compile()
    env.data = mj.MjData(env.model)
    from mujocodex.control.actuator_map import ActuatorMap
    env.actmap = ActuatorMap(env.model)
    (a.output / "scene_snapshot.xml").write_text(scene.to_xml())
    obj_id = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_BODY, cfg["body"])
    palm_id = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_BODY, "palm_link")
    obj_geoms = set(np.where(env.model.geom_bodyid == obj_id)[0].tolist())
    samples, events, contacts_mu = [], [], set()
    start, next_t, phase = None, 0., "grasp"
    initial_state = None

    def record():
        nonlocal next_t
        if start is None:
            return
        t = float(env.data.time - start)
        for c in env.data.contact:
            if ((c.geom1 in obj_geoms and c.geom2 in env.fingertip_geoms) or
                    (c.geom2 in obj_geoms and c.geom1 in env.fingertip_geoms)):
                contacts_mu.add(round(float(c.friction[0]), 6))
        if t + 1e-9 < next_t:
            return
        next_t += 1 / 30
        R = env.data.xmat[palm_id].reshape(3, 3)
        relative = R.T @ (env.data.xpos[obj_id] - env.data.xpos[palm_id])
        samples.append((t, env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(),
                        env.data.xfrc_applied[obj_id].copy(), relative,
                        phase, env.data.xpos[obj_id].copy(), env.data.xpos[palm_id].copy()))

    original_position = mod.position_arm_and_object

    def position(*args, **kwargs):
        nonlocal start, initial_state
        ok = original_position(*args, **kwargs)
        for gid in obj_geoms | set(env.fingertip_geoms):
            env.model.geom_friction[gid, 0] = a.friction
        mj.mj_forward(env.model, env.data)
        start = float(env.data.time)
        # Store full integration state for exact repeatability, including solver warmstart.
        state_type = mj.mjtState.mjSTATE_INTEGRATION
        initial_state = np.empty(mj.mj_stateSize(env.model, state_type))
        mj.mj_getState(env.model, env.data, initial_state, state_type)
        mj.mj_saveModel(env.model, str(a.output / "model.mjb"))
        record()
        return ok

    mod.position_arm_and_object = position
    original_step = env.step

    def step(ctrl):
        result = original_step(ctrl)
        record()
        return result

    env.step = step

    def mark_lift():
        nonlocal phase
        phase = "lift_and_shear"  # Shared archive schema; this runner has lift only.
        events.append({"phase": phase, "time_s": float(env.data.time - start)})

    mod.mark_lift = mark_lift
    # The source's three still captures are unnecessary; all states are recorded.
    mod.render_hires = lambda *args, **kwargs: None
    mod.save_png = lambda *args, **kwargs: None
    tick = time.time()
    result = None
    try:
        fn = mod.run_vnb_episode if a.method == "variational" else mod.run_cem_episode
        result = fn(env, cfg, a.beta, a.seed, a.friction, a.max_steps,
                    "overview", 960, 780, a.output)
        (a.output / "result.json").write_text(json.dumps(result, indent=2))
    finally:
        modules = {}
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if path and (name.startswith("mujocodex") or name == "run_variational_belief_experiments"):
                f = Path(path).resolve()
                if not f.is_relative_to(a.release):
                    raise RuntimeError(f"Imported code outside IROS release: {f}")
                modules[name] = {"path": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        meta = dict(command=sys.argv, release_commit=commit, release_path=str(a.release),
                    mujoco=mj.__version__, torch=torch.__version__, imported_modules=modules,
                    runner_original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                    elapsed_s=time.time()-tick, events=events,
                    effective_hand_contact_friction=sorted(contacts_mu),
                    removed_collision_exclusions=removed,
                    source_adaptations=["Time lift/hold using timestep times n_substeps",
                        "Record executed states; replace three still captures with recording",
                        "Set both hand/object friction to the declared value",
                        "Restore active-object collisions with world/main table"],
                    settings=dict(method=a.method, object=a.object, seed=a.seed, beta=a.beta,
                        max_steps=a.max_steps, friction=a.friction, output=str(a.output),
                        runner="release-comparison", contact_friction="matched",
                        stress_timing="physical", restore_table_contacts=True))
        if samples:
            meta["model_sha256"] = hashlib.sha256((a.output / "model.mjb").read_bytes()).hexdigest()
            np.savez_compressed(a.output / "states.npz", time_s=[s[0] for s in samples],
                qpos=[s[1] for s in samples], qvel=[s[2] for s in samples], ctrl=[s[3] for s in samples],
                object_wrench=[s[4] for s in samples], object_in_palm=[s[5] for s in samples],
                phase=[s[6] for s in samples], object_pos=[s[7] for s in samples],
                palm_pos=[s[8] for s in samples], initial_integration_state=initial_state)
        (a.output / "provenance.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
